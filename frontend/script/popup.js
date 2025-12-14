document.getElementById("scrape").addEventListener("click", () => {
    const para = document.getElementById("para");

    // Show loading animation
    para.innerHTML = '<div class="loading"></div>טוען...';

    // Execute content script
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            files: ["script/content.js"]
        }, () => {
        });
    });
});



chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "SCRAPED_DATA") {
        var s = "";
        for(var i=0; i<message.data.length;i++){
            if(message.data[i].length > 75){
                s = s.concat(message.data[i]) + '\n';
            }
        }
        if (s == "") {
            alert("Nothing To Summarize, Please Try A Different Page");
        } else {
            fetch("http://localhost:5000/summarize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: s })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Display the summary with additional info
                const summaryElement = document.getElementById("para");
                summaryElement.innerHTML = `
                    <div class="summary-content">
                        <h3>סיכום:</h3>
                        <p>${data.summary}</p>
                        <div class="summary-stats">
                            <small>אורך מקורי: ${data.original_length} תווים</small><br>
                            <small>אורך סיכום: ${data.summary_length} תווים</small><br>
                            <small>יחס דחיסה: ${data.compression_ratio}</small>
                        </div>
                    </div>
                `;
            })
            .catch(err => {
                console.error("Error calling backend:", err);
                document.getElementById("para").innerHTML = `
                    <div class="error">
                        <h3>שגיאה:</h3>
                        <p>${err.message}</p>
                        <small>ודא שהשרת פועל על http://localhost:5000</small>
                    </div>
                `;
            });
        }

}});