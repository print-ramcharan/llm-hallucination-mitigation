// Popup logic for LLM Context Memory Sync Extension
const API_BASE_URL = "http://127.0.0.1:8000";

let currentTab = null;

document.addEventListener("DOMContentLoaded", async () => {
  const statusBadge = document.getElementById("backend-status");
  const sessionSelect = document.getElementById("session-select");
  const pageTitleInput = document.getElementById("page-title");
  const captureTextArea = document.getElementById("capture-text");
  const captureBtn = document.getElementById("capture-btn");
  const statusMsg = document.getElementById("status-msg");
  const readToggle = document.getElementById("read-toggle");
  const writeToggle = document.getElementById("write-toggle");

  // Check backend health
  try {
    const healthRes = await fetch(`${API_BASE_URL}/api/health`);
    if (healthRes.ok) {
      statusBadge.textContent = "Online";
      statusBadge.className = "badge";
    } else {
      throw new Error();
    }
  } catch {
    statusBadge.textContent = "Offline";
    statusBadge.className = "badge offline";
  }

  // Load active sessions
  try {
    const sessionsRes = await fetch(`${API_BASE_URL}/api/memory/sessions`);
    if (sessionsRes.ok) {
      const sessions = await sessionsRes.json();
      sessionSelect.innerHTML = "";
      if (sessions.length === 0) {
        sessionSelect.innerHTML = `<option value="">Default Session (Auto-creates)</option>`;
      } else {
        sessions.forEach((s) => {
          const opt = document.createElement("option");
          opt.value = s.session_id;
          opt.textContent = `${s.title} (${s.turn_count} turns)`;
          sessionSelect.appendChild(opt);
        });
      }
    }
  } catch (err) {
    console.error("Could not fetch sessions:", err);
  }

  // Query active tab and selection
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    currentTab = tab;
    if (tab?.id) {
      pageTitleInput.value = tab.title || tab.url || "";
      chrome.tabs.sendMessage(tab.id, { action: "GET_SELECTION" }, (response) => {
        if (chrome.runtime.lastError) return;
        if (response && response.selectedText) {
          captureTextArea.value = response.selectedText;
        }
      });
    }
  } catch (err) {
    console.warn("Could not query active tab:", err);
  }

  // Capture Button Handler
  captureBtn.addEventListener("click", async () => {
    const selectedText = captureTextArea.value.trim();
    if (!selectedText) {
      statusMsg.textContent = "Please highlight or paste text first.";
      statusMsg.className = "status-msg error";
      return;
    }

    captureBtn.disabled = true;
    captureBtn.textContent = "Syncing...";
    statusMsg.textContent = "";

    try {
      const payload = {
        url: currentTab?.url || "chrome-extension://local",
        title: pageTitleInput.value || "Web Context",
        selected_text: selectedText,
        session_id: sessionSelect.value || null,
        tags: ["browser_extension", "manual_capture"],
      };

      const res = await fetch(`${API_BASE_URL}/api/memory/capture-web`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Sync failed");

      statusMsg.textContent = "✓ Context synced to Memory!";
      statusMsg.className = "status-msg success";
      setTimeout(() => window.close(), 1500);
    } catch (err) {
      statusMsg.textContent = "Error syncing to backend.";
      statusMsg.className = "status-msg error";
    } finally {
      captureBtn.disabled = false;
      captureBtn.textContent = "⚡ Sync to Episodic Memory";
    }
  });

  // Permission Toggles Handler
  async function updatePermissions() {
    const sessionId = sessionSelect.value;
    if (!sessionId) return;

    try {
      await fetch(`${API_BASE_URL}/api/memory/sessions/${sessionId}/permissions`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          read_enabled: readToggle.checked,
          write_enabled: writeToggle.checked,
          auto_summarize: true,
        }),
      });
    } catch (err) {
      console.warn("Failed to update permissions:", err);
    }
  }

  readToggle.addEventListener("change", updatePermissions);
  writeToggle.addEventListener("change", updatePermissions);
});
