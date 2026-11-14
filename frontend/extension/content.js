// Content script for LLM Context Memory Sync Extension
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "GET_SELECTION") {
    const selectedText = window.getSelection().toString().trim();
    sendResponse({
      title: document.title || window.location.hostname,
      url: window.location.href,
      selectedText: selectedText || document.body.innerText.substring(0, 1000),
      hasSelection: Boolean(selectedText),
    });
  }
  return true;
});
