var article = document.querySelector('article') || document.querySelector('.article-body');
var nodes = article ? [...article.querySelectorAll("h1, h2, h3, p, span")] : [...document.body.querySelectorAll("h1, h2, h3, p, span")];

var data = nodes
    .map(el => el.innerText.trim())
    .filter(text => text.length > 50) // keep meaningful sentences
    .map(text => text
        .replace(/\| ממומן|ממומן|Taboola|by Taboola|מידע נוסף|אולי יעניין אותך גם/g, '') // remove common ad phrases
        .trim()
    )
    .filter(text => text.length > 50); // remove any lines that became short after cleaning

// Remove approximate duplicates
var seen = new Set();
data = data.filter(s => {
    let key = s.substring(0, 30);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
});

chrome.runtime.sendMessage({ type: "SCRAPED_DATA", data });
