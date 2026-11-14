// Service worker for LLM Context Memory Sync Extension
const API_BASE_URL = "http://127.0.0.1:8000";

// Create context menu item on extension install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "save-to-llm-memory",
    title: "Sync Selection to LLM External Memory",
    contexts: ["selection", "page"],
  });
});

// Handle context menu click
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "save-to-llm-memory" && tab?.id) {
    try {
      const [{ result }] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
          return {
            selectedText: window.getSelection().toString().trim(),
            title: document.title,
            url: window.location.href,
          };
        },
      });

      const selectedText = result.selectedText || "Captured page context reference.";
      const payload = {
        url: result.url,
        title: result.title || "Web Reference",
        selected_text: selectedText,
        tags: ["browser_extension", "context_menu"],
      };

      const response = await fetch(`${API_BASE_URL}/api/memory/capture-web`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        chrome.notifications?.create({
          type: "basic",
          iconUrl: "icon.png",
          title: "Memory Synced",
          message: "Excerpt successfully saved to LLM External Memory store.",
        });
      }
    } catch (err) {
      console.error("Failed to sync context to memory store:", err);
    }
  }
});
