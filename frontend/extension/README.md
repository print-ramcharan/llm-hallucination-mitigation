# LLM Context Memory Sync — Browser Extension

This lightweight Chrome / Chromium (Edge, Brave, Opera) extension connects any active webpage with the **Cross-Session External Memory & Client Sync (Module 7)** backend.

## Features
- **Context Capture:** Highlight text on any documentation or webpage and click **⚡ Sync to Episodic Memory** or right-click to choose **"Sync Selection to LLM External Memory"**.
- **Multi-Session Routing:** Select which workspace session should receive the captured web context.
- **Read/Write Governance:** Toggle whether the LLM is allowed to read past episodic memories or write new conclusions from that tab.
- **Zero Bloat:** The backend vector store indexes the web excerpt into episodic memory without bloating your active document context window until semantically retrieved.

## Installation Instructions (Developer Mode)
1. Open your browser and navigate to:
   - Chrome: `chrome://extensions`
   - Edge: `edge://extensions`
   - Brave: `brave://extensions`
2. Enable the **"Developer mode"** toggle in the top-right corner.
3. Click the **"Load unpacked"** button.
4. Select this directory:
   ```
   c:\ProgFiles\Mitigating-Context-Degradation\frontend\extension
   ```
5. Ensure the backend FastAPI server is running at `http://127.0.0.1:8000`.
6. Click the extension icon in your browser toolbar or right-click text on any webpage to sync memory.
