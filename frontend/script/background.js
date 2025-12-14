chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "SCRAPED_DATA") {
        console.log("Scraped Data:", message.data);
    }
});