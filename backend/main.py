from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
import logging
from functools import lru_cache
import hashlib
from typing import List, Dict, Optional, Tuple
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["chrome-extension://*", "moz-extension://*"])

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.models = {
            'alephbert': {
                'name': 'onlplab/alephbert-base',
                'type': 'bert',
                'description': 'AlephBERT - Hebrew BERT model for embeddings'
            },
            'het5': {
                'name': 'avichr/heT5_small_summarization',
                'type': 't5',
                'description': 'HeT5 Small - Hebrew T5 model for abstractive summarization',
                'optional': True
            }
        }
        logger.info(f"ModelManager initialized. Models directory: {self.models_dir}")
    
    def get_model_path(self, model_key: str) -> str:
        if model_key not in self.models:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.models.keys())}")
        model_config = self.models[model_key]
        return model_config['name']
    
    def list_models(self) -> Dict[str, Dict]:
        model_status = {}
        for key, config in self.models.items():
            model_path = self.models_dir / key
            model_status[key] = {
                'name': config['name'],
                'type': config['type'],
                'description': config['description'],
                'downloaded': model_path.exists() if 'path' in config else None,
                'optional': config.get('optional', False)
            }
        return model_status
    
    def download_all_models(self) -> bool:
        success = True
        for key, config in self.models.items():
            if config.get('optional', False):
                logger.info(f"Skipping optional model: {key}")
                continue
            try:
                logger.info(f"Downloading {key} ({config['name']})...")
                logger.info(f"{key} will be downloaded on first use")
            except Exception as e:
                logger.error(f"Failed to download {key}: {e}")
                success = False
        return success
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        return self.models.get(model_key)

class HebrewSummarizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model_manager = ModelManager()
        self.models_loaded = False
        self.alephbert_tokenizer = None
        self.alephbert_model = None
        self.hebrew_stopwords = {
            'של', 'את', 'על', 'אל', 'עם', 'בין', 'לפני', 'אחרי', 'תחת', 'מעל',
            'זה', 'זו', 'זאת', 'אלה', 'אלו', 'הוא', 'היא', 'הם', 'הן', 'אני',
            'אתה', 'את', 'אנחנו', 'אתם', 'אתן', 'כל', 'כמה', 'איזה', 'איך',
            'מה', 'מי', 'איפה', 'מתי', 'למה', 'כן', 'לא', 'גם', 'רק', 'עוד'
        }
        self.embedding_cache = {}
        
    def load_models(self):
        if self.models_loaded:
            return
        try:
            logger.info("Loading AlephBERT model...")
            alephbert_path = self.model_manager.get_model_path('alephbert')
            self.alephbert_tokenizer = AutoTokenizer.from_pretrained(alephbert_path)
            self.alephbert_model = AutoModel.from_pretrained(alephbert_path).to(self.device)
            self.alephbert_model.eval()
            logger.info("AlephBERT model loaded successfully!")
            self.models_loaded = True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def preprocess_hebrew_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        noise_patterns = [
            r'\bקרא עוד\b',
            r'\bלחץ כאן\b',
            r'\bממומן\b',
            r'\bTaboola\b',
            r'\boutbrain\b',
            r'\bיתרון\b.*?\d+\.\d+',
            r'\bרגיל\b.*?\d+\.\d+',
            r'\bמעל/מתחת\b',
            r'\d+\.\d+X\d+\.\d+',
            r'מתוך \d+ משחקים',
            r'\bwww\.\S+',
            r'\bhttp\S+',
            r'\S+@\S+\.\S+'
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def split_sentences(self, text: str) -> List[str]:
        text = re.sub(r'(\w+)\.(\w+)', r'\1.\2', text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[א-ת])|(?<=[.!?])\s+(?=[A-Z])', text)
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (20 <= len(sentence) <= 500 and 
                not self._is_noise_sentence(sentence) and
                self._has_hebrew_content(sentence)):
                processed_sentences.append(sentence)
        return processed_sentences
    
    def _is_noise_sentence(self, sentence: str) -> bool:
        noise_patterns = [
            r'^\d+\.\s*$',
            r'^\W+$',
            r'^.{1,15}$',
            r'^(קרא עוד|לחץ כאן|שתף|לייק)$',
            r'^(taboola|outbrain|sponsored)$',
            r'^\S+@\S+\.\S+$',
            r'^(www\.|http)',
            r'^(.+?)\1+$'
        ]
        return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in noise_patterns)
    
    def _has_hebrew_content(self, text: str) -> bool:
        hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
        return hebrew_chars > len(text) * 0.3
    
    @lru_cache(maxsize=128)
    def get_sentence_embeddings(self, sentences_tuple: Tuple[str, ...]) -> np.ndarray:
        sentences = list(sentences_tuple)
        cache_key = hashlib.md5(''.join(sentences).encode()).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        embeddings = []
        batch_size = 4
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.alephbert_tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.alephbert_model(**inputs)
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                weighted_embeddings = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                embeddings.append(weighted_embeddings.cpu().numpy())
        result = np.vstack(embeddings)
        self.embedding_cache[cache_key] = result
        return result
    
    def _identify_text_sections(self, sentences: List[str]) -> Dict[str, List[int]]:
        total = len(sentences)
        opening_end = max(2, int(total * 0.25))
        closing_start = min(total - 2, int(total * 0.75))
        return {
            'opening': list(range(0, opening_end)),
            'middle': list(range(opening_end, closing_start)),
            'closing': list(range(closing_start, total))
        }
    
    def _identify_central_theme(self, opening_sentences: List[str], title: str = "") -> List[str]:
        theme_keywords = []
        combined_text = f"{title} {' '.join(opening_sentences[:3])}".lower()
        theme_indicators = [
            'משטרה', 'ממשלה', 'כנסת', 'בית משפט', 'חוק', 'מדיניות',
            'הפגנה', 'מחאה', 'עצרת', 'שביתה',
            'כלכלה', 'משק', 'תקציב', 'מס', 'שכר', 'מחירים', 'יוקר',
            'ביטחון', 'צבא', 'מלחמה', 'טרור', 'איום', 'פיגוע',
            'חברה', 'חינוך', 'בריאות', 'דיור', 'תחבורה', 'סביבה',
            'כדורגל', 'ספורט', 'משחק', 'קבוצה', 'שחקן', 'מאמן', 'ליגה'
        ]
        for keyword in theme_indicators:
            if keyword in combined_text:
                theme_keywords.append(keyword)
        if title:
            title_entities = re.findall(r'\b[א-ת][א-ת]+(?:\s+[א-ת][א-ת]+){0,1}\b', title.lower())
            for entity in title_entities:
                if 2 <= len(entity.split()) <= 3 and len(entity) > 3 and entity not in theme_keywords:
                    theme_keywords.append(entity)
        text_entities = re.findall(r'\b[א-ת][א-ת]+(?:\s+[א-ת][א-ת]+){0,1}\b', combined_text)
        for entity in text_entities:
            if len(entity.split()) <= 2 and len(entity) > 3 and entity not in theme_keywords:
                theme_keywords.append(entity)
        return theme_keywords[:7]
    
    def _theme_aware_sentence_scoring(self, sentences: List[str], embeddings: np.ndarray, 
                                     sections: Dict[str, List[int]], theme_keywords: List[str]) -> List[Tuple[float, str, int]]:
        scores = []
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0)
        threshold = np.percentile(similarity_matrix.flatten(), 70)
        similarity_matrix = np.where(similarity_matrix < threshold, 0, similarity_matrix)
        graph = nx.from_numpy_array(similarity_matrix)
        pagerank_scores = nx.pagerank(graph, alpha=0.85, max_iter=100)
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = 0
            score += pagerank_scores.get(i, 0) * 0.35
            theme_score = 0
            for keyword in theme_keywords:
                if keyword.lower() in sentence_lower:
                    theme_score += 1.0
            theme_score = min(theme_score / max(len(theme_keywords), 1), 1.0)
            score += theme_score * 0.25
            if i in sections['opening']:
                section_bonus = 1.2
            elif i in sections['closing']:
                section_bonus = 1.15
            else:
                section_bonus = 1.0
            position_score = (len(sentences) - i) / len(sentences)
            score += position_score * 0.10
            optimal_length = 150
            length_score = 1 - abs(len(sentence) - optimal_length) / optimal_length
            score += max(0, length_score) * 0.10
            hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', sentence))
            hebrew_ratio = hebrew_chars / len(sentence) if len(sentence) > 0 else 0
            score += hebrew_ratio * 0.05
            score *= section_bonus
            scores.append((score, sentence, i))
        return sorted(scores, key=lambda x: x[0], reverse=True)
    
    def _rephrase_sentence(self, sentence: str) -> str:
        if not sentence or len(sentence.strip()) < 10:
            return sentence
        paraphrased = sentence
        paraphrase_patterns = [
            (r'^זהו\s+(.+)', r'מדובר ב\1'),
            (r'בוצע\s+(.+)', r'נעשה \1'),
            (r'התקיים\s+(.+)', r'נערך \1'),
            (r'נמצא כי\s+(.+)', r'התברר ש\1'),
            (r'הוחלט כי\s+(.+)', r'החליטו ש\1'),
            (r'(.+)\s+ביצע\s+(.+)', r'\2 בוצע על ידי \1'),
            (r'(.+)\s+פיתח\s+(.+)', r'\2 פותח על ידי \1'),
            (r'הוא מכיל\s+(.+)', r'כולל \1'),
            (r'היא כוללת\s+(.+)', r'מכילה \1'),
            (r'המטרה היא\s+(.+)', r'נועד \1'),
            (r'הוא צריך\s+(.+)', r'נדרש \1'),
            (r'יש צורך\s+(.+)', r'נחוץ \1'),
            (r'בזמן האחרון', r'לאחרונה'),
            (r'בעתיד הקרוב', r'בקרוב'),
            (r'בעבר', r'קודם לכן'),
            (r'כיום', r'בימינו'),
            (r'בגלל\s+(.+)', r'עקב \1'),
            (r'כתוצאה מ(.+)', r'בעקבות \1'),
            (r'על מנת\s+(.+)', r'כדי \1'),
            (r'מספר רב של', r'רבים'),
            (r'כמות גדולה של', r'הרבה'),
            (r'חלק גדול מ', r'רוב'),
            (r'כמו כן,?\s*', 'בנוסף, '),
            (r'יתר על כן,?\s*', 'כמו כן, '),
            (r'בנוסף לכך,?\s*', 'בנוסף, '),
        ]
        transformations_applied = 0
        for pattern, replacement in paraphrase_patterns:
            new_paraphrased = re.sub(pattern, replacement, paraphrased, flags=re.IGNORECASE)
            if new_paraphrased != paraphrased:
                paraphrased = new_paraphrased
                transformations_applied += 1
                if transformations_applied >= 2:
                    break
        paraphrased = re.sub(r'\s+', ' ', paraphrased).strip()
        if paraphrased and paraphrased[0].islower():
            paraphrased = paraphrased[0].upper() + paraphrased[1:]
        return paraphrased if len(paraphrased) > 10 else sentence
    
    def _intelligent_rewrite(self, sentence: str, embeddings: np.ndarray, 
                            all_sentences: List[str], sentence_idx: int) -> str:
        if not sentence or len(sentence.split()) < 5:
            return sentence
        sentence_embedding = embeddings[sentence_idx]
        similarities = cosine_similarity([sentence_embedding], embeddings)[0]
        similar_indices = np.argsort(similarities)[-3:][::-1]
        similar_sentences = [all_sentences[i] for i in similar_indices if i != sentence_idx]
        words = sentence.split()
        key_words = [w for w in words if w not in self.hebrew_stopwords and len(w) > 2]
        paraphrased = self._rephrase_sentence(sentence)
        if self._calculate_change_ratio(sentence, paraphrased) > 0.15:
            return paraphrased
        return self._restructure_sentence(sentence, key_words)
    
    def _calculate_change_ratio(self, original: str, modified: str) -> float:
        orig_words = set(original.lower().split())
        mod_words = set(modified.lower().split())
        if len(orig_words) == 0:
            return 0.0
        changed_words = orig_words.symmetric_difference(mod_words)
        return len(changed_words) / len(orig_words)
    
    def _restructure_sentence(self, sentence: str, key_words: List[str]) -> str:
        if not key_words or len(key_words) < 3:
            return sentence
        sentence = re.sub(r'\b(כאמור|כפי שצוין|כידוע)\b,?\s*', '', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        return sentence
    
    def _select_representative_sentences(self, scored_sentences: List[Tuple[float, str, int]], 
                                       sections: Dict[str, List[int]], 
                                       target_ratio: float) -> List[Tuple[float, str, int]]:
        total_sentences = len(scored_sentences)
        target_count = max(2, int(total_sentences * target_ratio))
        logger.info(f"Selection: total_sentences={total_sentences}, target_ratio={target_ratio:.2f}, target_count={target_count}")
        if target_count <= 3:
            return scored_sentences[:target_count]
        min_opening = max(1, target_count // 5)
        min_closing = max(1, target_count // 6)
        min_middle = max(1, target_count - min_opening - min_closing)
        selected = []
        section_counts = {'opening': 0, 'middle': 0, 'closing': 0}
        for score, sentence, idx in scored_sentences:
            current_section = None
            if idx in sections['opening']:
                current_section = 'opening'
            elif idx in sections['middle']:
                current_section = 'middle'
            elif idx in sections['closing']:
                current_section = 'closing'
            if current_section:
                min_needed = {'opening': min_opening, 'middle': min_middle, 'closing': min_closing}
                if section_counts[current_section] < min_needed[current_section]:
                    selected.append((score, sentence, idx))
                    section_counts[current_section] += 1
                    if len(selected) >= target_count:
                        break
        remaining_slots = target_count - len(selected)
        selected_indices = {s[2] for s in selected}
        for score, sentence, idx in scored_sentences:
            if remaining_slots <= 0:
                break
            if idx not in selected_indices:
                selected.append((score, sentence, idx))
                remaining_slots -= 1
        logger.info(f"Selected {len(selected)} sentences (target was {target_count})")
        logger.info(f"Section distribution: opening={section_counts['opening']}, middle={section_counts['middle']}, closing={section_counts['closing']}")
        return selected
    
    def calculate_sentence_scores(self, sentences: List[str], embeddings: np.ndarray) -> List[Tuple[float, str, int]]:
        sections = self._identify_text_sections(sentences)
        theme_keywords = self._identify_central_theme(sentences[:3], "")
        return self._theme_aware_sentence_scoring(sentences, embeddings, sections, theme_keywords)
    
    def summarize(self, text: str, target_ratio: float = 0.25, title: str = "") -> Dict[str, any]:
        start_time = time.time()
        try:
            if not self.models_loaded:
                self.load_models()
            clean_text = self.preprocess_hebrew_text(text)
            sentences = self.split_sentences(clean_text)
            if len(sentences) < 3:
                return {
                    "summary": clean_text,
                    "metadata": {
                        "method": "no_summarization_needed",
                        "original_sentences": len(sentences),
                        "processing_time": time.time() - start_time
                    }
                }
            sections = self._identify_text_sections(sentences)
            logger.info(f"Text sections: opening={len(sections['opening'])}, middle={len(sections['middle'])}, closing={len(sections['closing'])}")
            theme_keywords = self._identify_central_theme(sentences[:3], title)
            logger.info(f"Identified central theme keywords: {theme_keywords}")
            embeddings = self.get_sentence_embeddings(tuple(sentences))
            scored_sentences = self._theme_aware_sentence_scoring(sentences, embeddings, sections, theme_keywords)
            selected = self._select_representative_sentences(scored_sentences, sections, target_ratio)
            selected.sort(key=lambda x: x[2])
            summary_sentences = [s[1] for s in selected]
            summary = ' '.join(summary_sentences)
            if summary and not summary.endswith('.'):
                summary += '.'
            section_counts = {
                'opening': len([s for s in selected if s[2] in sections['opening']]),
                'middle': len([s for s in selected if s[2] in sections['middle']]),
                'closing': len([s for s in selected if s[2] in sections['closing']])
            }
            return {
                "summary": summary,
                "metadata": {
                    "method": "ai_extractive_comprehensive",
                    "original_sentences": len(sentences),
                    "summary_sentences": len(selected),
                    "compression_ratio": len(selected) / len(sentences),
                    "sections_represented": section_counts,
                    "central_theme": theme_keywords,
                    "processing_time": time.time() - start_time,
                    "model": "AlephBERT + TextRank + Section-Aware + Theme"
                }
            }
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {
                "summary": text[:500] + "...",
                "metadata": {
                    "method": "fallback",
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
            }
    
    def abstractive_summarize(self, text: str, target_ratio: float = 0.25, title: str = "") -> Dict[str, any]:
        start_time = time.time()
        try:
            if not self.models_loaded:
                self.load_models()
            clean_text = self.preprocess_hebrew_text(text)
            sentences = self.split_sentences(clean_text)
            if len(sentences) < 3:
                return {
                    "summary": clean_text,
                    "metadata": {
                        "method": "no_summarization_needed",
                        "original_sentences": len(sentences),
                        "processing_time": time.time() - start_time
                    }
                }
            sections = self._identify_text_sections(sentences)
            theme_keywords = self._identify_central_theme(sentences[:3], title)
            embeddings = self.get_sentence_embeddings(tuple(sentences))
            scored_sentences = self._theme_aware_sentence_scoring(sentences, embeddings, sections, theme_keywords)
            target_count = max(2, int(len(sentences) * target_ratio * 0.7))
            selected = self._select_representative_sentences(scored_sentences, sections, target_ratio * 0.7)
            rewritten_sentences = []
            for score, sentence, idx in selected:
                rewritten = self._intelligent_rewrite(sentence, embeddings, sentences, idx)
                rewritten_sentences.append(rewritten)
            summary = self._connect_sentences_naturally(rewritten_sentences)
            section_counts = {
                'opening': len([s for s in selected if s[2] in sections['opening']]),
                'middle': len([s for s in selected if s[2] in sections['middle']]),
                'closing': len([s for s in selected if s[2] in sections['closing']])
            }
            return {
                "summary": summary,
                "metadata": {
                    "method": "ai_abstractive",
                    "original_sentences": len(sentences),
                    "summary_sentences": len(rewritten_sentences),
                    "compression_ratio": len(rewritten_sentences) / len(sentences),
                    "sections_represented": section_counts,
                    "central_theme": theme_keywords,
                    "processing_time": time.time() - start_time,
                    "model": "AlephBERT + Intelligent Rewriting"
                }
            }
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            logger.info("Falling back to extractive summarization")
            return self.summarize(text, target_ratio, title)
    
    def _connect_sentences_naturally(self, sentences: List[str]) -> str:
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0] + ('.' if not sentences[0].endswith('.') else '')
        connectors = ['בנוסף', 'כמו כן', 'יתר על כן', 'למעשה', 'לפיכך']
        result = sentences[0]
        for i, sentence in enumerate(sentences[1:], 1):
            if i < len(connectors):
                connector = connectors[i-1]
                clean_sentence = sentence
                for c in connectors:
                    if clean_sentence.startswith(c):
                        clean_sentence = clean_sentence[len(c):].strip()
                        if clean_sentence.startswith(','):
                            clean_sentence = clean_sentence[1:].strip()
                        break
                result += f'. {connector}, {clean_sentence.lower()}'
            else:
                result += f'. {sentence}'
        if not result.endswith(('.', '!', '?')):
            result += '.'
        return result
    
    def rephrase_text(self, text: str, intensity: str = "medium") -> Dict[str, any]:
        start_time = time.time()
        try:
            if not self.models_loaded:
                self.load_models()
            clean_text = self.preprocess_hebrew_text(text)
            sentences = self.split_sentences(clean_text)
            if len(sentences) == 0:
                return {
                    "rephrased": clean_text,
                    "metadata": {
                        "method": "no_rephrasing_needed",
                        "processing_time": time.time() - start_time
                    }
                }
            embeddings = self.get_sentence_embeddings(tuple(sentences))
            rephrased_sentences = []
            for i, sentence in enumerate(sentences):
                if intensity == "high":
                    rephrased = self._rephrase_sentence(sentence)
                    rephrased = self._rephrase_sentence(rephrased)
                elif intensity == "low":
                    rephrased = self._restructure_sentence(sentence, sentence.split())
                else:
                    rephrased = self._intelligent_rewrite(sentence, embeddings, sentences, i)
                rephrased_sentences.append(rephrased)
            rephrased_text = self._connect_sentences_naturally(rephrased_sentences)
            return {
                "rephrased": rephrased_text,
                "original": clean_text,
                "metadata": {
                    "method": "rephrasing",
                    "intensity": intensity,
                    "original_sentences": len(sentences),
                    "processing_time": time.time() - start_time,
                    "model": "AlephBERT + Paraphrasing"
                }
            }
        except Exception as e:
            logger.error(f"Rephrasing failed: {e}")
            return {
                "rephrased": text,
                "metadata": {
                    "method": "fallback",
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
            }

summarizer = HebrewSummarizer()

@app.route("/summarize", methods=["POST", "OPTIONS"])
def summarize_api():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        method = data.get("method", "extractive").lower()
        target_ratio = float(data.get("target_ratio", 0.25))
        title = data.get("title", "").strip()
        intensity = data.get("intensity", "medium").lower()
        if method == "abstractive":
            result = summarizer.abstractive_summarize(text, target_ratio, title)
        elif method == "rephrase":
            result = summarizer.rephrase_text(text, intensity)
            return jsonify({
                "rephrased": result["rephrased"],
                "original": result.get("original", text),
                "method": result["metadata"].get("method", "rephrasing"),
                "intensity": result["metadata"].get("intensity", intensity),
                "processing_time": result["metadata"].get("processing_time", 0),
                "model": result["metadata"].get("model", "AlephBERT + Paraphrasing")
            })
        else:
            result = summarizer.summarize(text, target_ratio, title)
        summary = result["summary"]
        original_length = len(text)
        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 0
        response = {
            "summary": summary,
            "original_length": original_length,
            "summary_length": summary_length,
            "compression_ratio": round(compression_ratio, 3),
            "method": result["metadata"].get("method", "ai_extractive_comprehensive"),
            "model": result["metadata"].get("model", "AlephBERT + TextRank + Section-Aware + Theme"),
            "is_abstractive": method == "abstractive"
        }
        if "processing_time" in result["metadata"]:
            response["processing_time"] = result["metadata"]["processing_time"]
        if "sections_represented" in result["metadata"]:
            response["sections_represented"] = result["metadata"]["sections_represented"]
        if "central_theme" in result["metadata"]:
            response["central_theme"] = result["metadata"]["central_theme"]
        return jsonify(response)
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": summarizer.models_loaded,
        "device": str(summarizer.device),
        "method": "ai_extractive",
        "version": "v4.0",
        "capabilities": ["extractive", "abstractive", "rephrase"]
    })

@app.route("/models/status", methods=["GET"])
def models_status():
    return jsonify({
        "models": summarizer.model_manager.list_models(),
        "models_loaded": summarizer.models_loaded,
        "device": str(summarizer.device)
    })

@app.route("/load_models", methods=["POST"])
def load_models():
    try:
        summarizer.load_models()
        return jsonify({"status": "Models loaded successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Hebrew Summarizer...")
    print("Server running on: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
