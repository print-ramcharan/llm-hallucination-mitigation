"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  Brain,
  Download,
  FileCheck2,
  Globe,
  Lock,
  Plus,
  RefreshCw,
  Search,
  Sliders,
  Sparkles,
  Trash2,
  Unlock,
} from "lucide-react";
import {
  captureWebContent,
  clearSessionMemory,
  createSession,
  EpisodicMemoryItem,
  exportSessionMemoryMarkdown,
  fetchSessionDetails,
  fetchSessions,
  HierarchicalSummary,
  MemoryPermissions,
  searchMemory,
  Session,
  updateSessionPermissions,
} from "@/lib/api";

export function CrossSessionMemory() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [sessionDetails, setSessionDetails] = useState<{
    session: Session;
    permissions: MemoryPermissions;
    turns: EpisodicMemoryItem[];
    summaries: HierarchicalSummary[];
  } | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"turns" | "summaries" | "capture">("turns");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<EpisodicMemoryItem[] | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  // New Session Modal State
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [newSessionTitle, setNewSessionTitle] = useState("");

  // Web Context Capture State
  const [webUrl, setWebUrl] = useState("");
  const [webTitle, setWebTitle] = useState("");
  const [webText, setWebText] = useState("");
  const [isCapturing, setIsCapturing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Load Sessions
  const loadSessions = useCallback(async () => {
    try {
      setIsLoading(true);
      const sessList = await fetchSessions();
      setSessions(sessList);
      if (sessList.length > 0 && !selectedSessionId) {
        setSelectedSessionId(sessList[0].session_id);
      }
    } catch (err) {
      console.error("Failed to load sessions:", err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedSessionId]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Load Session Details
  const loadDetails = useCallback(async (sessionId: string) => {
    try {
      setIsLoading(true);
      const details = await fetchSessionDetails(sessionId);
      setSessionDetails(details);
      setSearchResults(null);
    } catch (err) {
      console.error("Failed to fetch session details:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedSessionId) {
      loadDetails(selectedSessionId);
    }
  }, [selectedSessionId, loadDetails]);

  // Toggle Permissions
  const handleTogglePermission = async (field: keyof MemoryPermissions) => {
    if (!selectedSessionId || !sessionDetails) return;
    const current = sessionDetails.permissions;
    const updated: MemoryPermissions = {
      ...current,
      [field]: !current[field],
    };

    try {
      const res = await updateSessionPermissions(selectedSessionId, updated);
      setSessionDetails({
        ...sessionDetails,
        permissions: res,
      });
      setStatusMessage("Permissions updated successfully.");
      setTimeout(() => setStatusMessage(null), 3000);
    } catch (err) {
      alert("Failed to update permissions: " + err);
    }
  };

  // Create Session
  const handleCreateSession = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSessionTitle.trim()) return;

    try {
      const created = await createSession(newSessionTitle.trim());
      setSessions((prev) => [created, ...prev]);
      setSelectedSessionId(created.session_id);
      setNewSessionTitle("");
      setIsCreatingSession(false);
      setStatusMessage(`Session '${created.title}' created.`);
      setTimeout(() => setStatusMessage(null), 3000);
    } catch (err) {
      alert("Failed to create session: " + err);
    }
  };

  // Clear Memory
  const handleClearMemory = async () => {
    if (!selectedSessionId) return;
    if (!confirm("Are you sure you want to clear all memories and summaries for this session?")) return;

    try {
      await clearSessionMemory(selectedSessionId);
      loadDetails(selectedSessionId);
      setStatusMessage("Session memory cleared.");
      setTimeout(() => setStatusMessage(null), 3000);
    } catch (err) {
      alert("Failed to clear memory: " + err);
    }
  };

  // Export Memory
  const handleExportMarkdown = async () => {
    if (!selectedSessionId) return;
    try {
      const md = await exportSessionMemoryMarkdown(selectedSessionId);
      const blob = new Blob([md], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `session_memory_${selectedSessionId}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert("Failed to export memory: " + err);
    }
  };

  // Semantic Memory Search
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      setSearchResults(null);
      return;
    }

    try {
      setIsSearching(true);
      const results = await searchMemory(searchQuery.trim(), selectedSessionId || undefined);
      setSearchResults(results.memories);
    } catch (err) {
      alert("Search failed: " + err);
    } finally {
      setIsSearching(false);
    }
  };

  // Capture Web Context
  const handleCaptureWeb = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!webUrl.trim() || !webText.trim()) return;

    try {
      setIsCapturing(true);
      await captureWebContent({
        url: webUrl.trim(),
        title: webTitle.trim() || "Web Excerpt",
        selected_text: webText.trim(),
        session_id: selectedSessionId || undefined,
      });

      setWebUrl("");
      setWebTitle("");
      setWebText("");
      setStatusMessage("Web context captured into episodic memory!");
      setTimeout(() => setStatusMessage(null), 3000);
      if (selectedSessionId) {
        loadDetails(selectedSessionId);
      }
    } catch (err) {
      alert("Failed to capture web context: " + err);
    } finally {
      setIsCapturing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner / Status */}
      {statusMessage && (
        <div className="p-3 bg-emerald-950/60 border border-emerald-500/30 rounded-xl text-emerald-300 text-sm flex items-center justify-between animate-in fade-in">
          <span>{statusMessage}</span>
          <button onClick={() => setStatusMessage(null)} className="text-emerald-400 hover:underline text-xs">
            Dismiss
          </button>
        </div>
      )}

      {/* Main Grid: Sidebar & Content */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Sessions & Permissions (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          {/* Session Selector Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-indigo-400" />
                <h2 className="text-sm font-semibold text-slate-100 uppercase tracking-wider">
                  Session Registry
                </h2>
                {isLoading && <RefreshCw className="w-3.5 h-3.5 text-indigo-400 animate-spin" />}
              </div>
              <button
                onClick={() => setIsCreatingSession(true)}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-lg bg-indigo-600/90 text-white hover:bg-indigo-500 transition"
              >
                <Plus className="w-3.5 h-3.5" />
                New
              </button>
            </div>

            {/* Modal for New Session */}
            {isCreatingSession && (
              <form onSubmit={handleCreateSession} className="mt-4 p-3 bg-slate-950/80 border border-slate-800 rounded-xl space-y-3">
                <label className="block text-xs font-medium text-slate-300">Session Title</label>
                <input
                  type="text"
                  value={newSessionTitle}
                  onChange={(e) => setNewSessionTitle(e.target.value)}
                  placeholder="e.g. Employee Benefits Investigation"
                  className="w-full text-xs px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-indigo-500"
                  autoFocus
                />
                <div className="flex justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => setIsCreatingSession(false)}
                    className="px-2.5 py-1 text-xs text-slate-400 hover:text-white"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-3 py-1 text-xs font-semibold rounded-lg bg-indigo-600 text-white hover:bg-indigo-500"
                  >
                    Create
                  </button>
                </div>
              </form>
            )}

            {/* Sessions List */}
            <div className="mt-4 space-y-2 max-h-64 overflow-y-auto pr-1">
              {sessions.length === 0 ? (
                <div className="text-center py-6 text-xs text-slate-500">
                  No sessions yet. Click New to create one.
                </div>
              ) : (
                sessions.map((s) => (
                  <button
                    key={s.session_id}
                    onClick={() => setSelectedSessionId(s.session_id)}
                    className={`w-full text-left p-3 rounded-xl transition border flex flex-col gap-1 ${
                      selectedSessionId === s.session_id
                        ? "bg-indigo-600/20 border-indigo-500/50 text-white"
                        : "bg-slate-950/50 border-slate-800/80 text-slate-300 hover:bg-slate-800/50"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-xs truncate">{s.title}</span>
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">
                        {s.turn_count} turns
                      </span>
                    </div>
                    <span className="text-[10px] text-slate-500 truncate font-mono">
                      ID: {s.session_id}
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Permissions & Governance Card */}
          {sessionDetails && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
                <Sliders className="w-4 h-4 text-emerald-400" />
                <h3 className="text-xs font-semibold text-slate-100 uppercase tracking-wider">
                  Read / Write Governance
                </h3>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <div>
                    <div className="font-medium text-slate-200">Episodic Memory Read</div>
                    <div className="text-[10px] text-slate-500">Inject past facts before inference</div>
                  </div>
                  <button
                    onClick={() => handleTogglePermission("read_enabled")}
                    className={`p-1.5 rounded-lg border transition ${
                      sessionDetails.permissions.read_enabled
                        ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"
                        : "bg-slate-800 text-slate-500 border-slate-700"
                    }`}
                  >
                    {sessionDetails.permissions.read_enabled ? <Unlock className="w-4 h-4" /> : <Lock className="w-4 h-4" />}
                  </button>
                </div>

                <div className="flex items-center justify-between text-xs">
                  <div>
                    <div className="font-medium text-slate-200">Episodic Memory Write</div>
                    <div className="text-[10px] text-slate-500">Store verified facts after inference</div>
                  </div>
                  <button
                    onClick={() => handleTogglePermission("write_enabled")}
                    className={`p-1.5 rounded-lg border transition ${
                      sessionDetails.permissions.write_enabled
                        ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"
                        : "bg-slate-800 text-slate-500 border-slate-700"
                    }`}
                  >
                    {sessionDetails.permissions.write_enabled ? <Unlock className="w-4 h-4" /> : <Lock className="w-4 h-4" />}
                  </button>
                </div>

                <div className="flex items-center justify-between text-xs">
                  <div>
                    <div className="font-medium text-slate-200">Auto-Summarization</div>
                    <div className="text-[10px] text-slate-500">Condense turns every 5 interactions</div>
                  </div>
                  <button
                    onClick={() => handleTogglePermission("auto_summarize")}
                    className={`p-1.5 rounded-lg border transition ${
                      sessionDetails.permissions.auto_summarize
                        ? "bg-indigo-500/20 text-indigo-400 border-indigo-500/40"
                        : "bg-slate-800 text-slate-500 border-slate-700"
                    }`}
                  >
                    <Sparkles className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-3 border-t border-slate-800 flex gap-2">
                <button
                  onClick={handleExportMarkdown}
                  className="flex-1 py-1.5 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center justify-center gap-1.5 transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  Export MD
                </button>
                <button
                  onClick={handleClearMemory}
                  className="py-1.5 px-3 rounded-xl bg-rose-950/40 hover:bg-rose-900/60 border border-rose-800/40 text-rose-300 text-xs font-medium flex items-center justify-center gap-1.5 transition"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Memory Inspection & Search (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {/* Search & Tabs Header Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
            {/* Search Input */}
            <form onSubmit={handleSearch} className="flex-1 relative">
              {isSearching ? (
                <RefreshCw className="w-4 h-4 absolute left-3 top-3 text-indigo-400 animate-spin" />
              ) : (
                <Search className="w-4 h-4 absolute left-3 top-3 text-slate-500" />
              )}
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search semantic memory across past turns..."
                className="w-full pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => {
                    setSearchQuery("");
                    setSearchResults(null);
                  }}
                  className="absolute right-3 top-2.5 text-xs text-slate-400 hover:text-white"
                >
                  ✕
                </button>
              )}
            </form>

            {/* Navigation Tabs */}
            <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 shrink-0">
              <button
                onClick={() => setActiveTab("turns")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-1.5 ${
                  activeTab === "turns" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                <FileCheck2 className="w-3.5 h-3.5" />
                Episodic Turns
              </button>
              <button
                onClick={() => setActiveTab("summaries")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-1.5 ${
                  activeTab === "summaries" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                Summaries
              </button>
              <button
                onClick={() => setActiveTab("capture")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-1.5 ${
                  activeTab === "capture" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                <Globe className="w-3.5 h-3.5" />
                Capture Web
              </button>
            </div>
          </div>

          {/* Tab 1: Episodic Turns */}
          {activeTab === "turns" && (
            <div className="space-y-4">
              {searchResults ? (
                <div className="p-3 bg-indigo-950/40 border border-indigo-500/30 rounded-xl text-xs text-indigo-300 flex items-center justify-between">
                  <span>Found {searchResults.length} matching semantic memories</span>
                  <button
                    onClick={() => {
                      setSearchResults(null);
                      setSearchQuery("");
                    }}
                    className="underline text-indigo-400"
                  >
                    Show all turns
                  </button>
                </div>
              ) : null}

              {(searchResults || sessionDetails?.turns || []).length === 0 ? (
                <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-12 text-center text-slate-500 space-y-2">
                  <Brain className="w-8 h-8 mx-auto text-slate-600 stroke-[1.5]" />
                  <p className="text-sm">No episodic memories recorded for this session yet.</p>
                  <p className="text-xs text-slate-600">
                    Interact via the Grounded Q&A tab with memory writing enabled or capture web snippets.
                  </p>
                </div>
              ) : (
                (searchResults || sessionDetails?.turns || []).map((t, idx) => (
                  <div
                    key={t.memory_id || idx}
                    className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-400 text-[10px] font-mono font-semibold">
                          Turn {t.turn_index + 1}
                        </span>
                        {t.intent && (
                          <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 text-[10px] font-mono">
                            {t.intent}
                          </span>
                        )}
                      </div>
                      <span className="text-[10px] text-slate-500 font-mono">
                        {new Date(t.timestamp).toLocaleTimeString()}
                      </span>
                    </div>

                    <div className="text-xs font-semibold text-slate-100 flex items-start gap-2">
                      <span className="text-indigo-400 shrink-0">Q:</span>
                      <span>{t.query}</span>
                    </div>

                    <div className="text-xs text-slate-300 pl-4 border-l-2 border-slate-800 leading-relaxed">
                      {t.answer}
                    </div>

                    {/* Verified Facts Badges */}
                    {t.verified_facts && t.verified_facts.length > 0 && (
                      <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
                        <span className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wider">
                          Verified Factual Claims:
                        </span>
                        <div className="flex flex-wrap gap-1.5">
                          {t.verified_facts.map((fact, fIdx) => (
                            <span
                              key={fIdx}
                              className="px-2 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-[11px] leading-snug"
                            >
                              ✓ {fact}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Citations */}
                    {t.citations && t.citations.length > 0 && (
                      <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
                        <span className="font-semibold text-slate-500">Citations:</span>
                        {t.citations.map((c, cIdx) => (
                          <span key={cIdx} className="px-1.5 py-0.5 rounded bg-slate-800 font-mono text-slate-300">
                            {c}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          )}

          {/* Tab 2: Persistent Hierarchical Summaries */}
          {activeTab === "summaries" && (
            <div className="space-y-4">
              {(sessionDetails?.summaries || []).length === 0 ? (
                <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-12 text-center text-slate-500 space-y-2">
                  <Sparkles className="w-8 h-8 mx-auto text-slate-600 stroke-[1.5]" />
                  <p className="text-sm">No hierarchical summaries generated yet.</p>
                  <p className="text-xs text-slate-600">
                    Summaries are generated automatically every 5 turns or can be triggered via API.
                  </p>
                </div>
              ) : (
                sessionDetails?.summaries.map((s) => (
                  <div
                    key={s.summary_id}
                    className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3"
                  >
                    <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-indigo-400" />
                        <h4 className="text-xs font-semibold text-white">{s.title}</h4>
                      </div>
                      <span className="text-[10px] text-slate-500 font-mono">
                        Condensed {s.turn_count} turns
                      </span>
                    </div>

                    <p className="text-xs text-slate-300 whitespace-pre-line leading-relaxed">
                      {s.summary_text}
                    </p>

                    {s.key_entities && s.key_entities.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1.5 pt-2">
                        <span className="text-[10px] font-semibold text-slate-500">Entities:</span>
                        {s.key_entities.map((e, eIdx) => (
                          <span
                            key={eIdx}
                            className="px-2 py-0.5 rounded-full bg-slate-800 text-[10px] text-slate-300 font-mono"
                          >
                            {e}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          )}

          {/* Tab 3: Capture Web Context */}
          {activeTab === "capture" && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
                <Globe className="w-5 h-5 text-indigo-400" />
                <div>
                  <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
                    Web Context Ingestion
                  </h3>
                  <p className="text-[11px] text-slate-400">
                    Ingest excerpts from web articles or documentation into your active memory session.
                  </p>
                </div>
              </div>

              <form onSubmit={handleCaptureWeb} className="space-y-4">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Source URL</label>
                  <input
                    type="url"
                    required
                    value={webUrl}
                    onChange={(e) => setWebUrl(e.target.value)}
                    placeholder="https://example.com/docs/policy"
                    className="w-full text-xs px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-white focus:outline-none focus:border-indigo-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Page Title</label>
                  <input
                    type="text"
                    value={webTitle}
                    onChange={(e) => setWebTitle(e.target.value)}
                    placeholder="e.g. Vacation & Sabbatical Guidelines 2024"
                    className="w-full text-xs px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-white focus:outline-none focus:border-indigo-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Selected Excerpt</label>
                  <textarea
                    required
                    rows={4}
                    value={webText}
                    onChange={(e) => setWebText(e.target.value)}
                    placeholder="Paste relevant text from the web page to store into episodic memory..."
                    className="w-full text-xs px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-white focus:outline-none focus:border-indigo-500"
                  />
                </div>

                <button
                  type="submit"
                  disabled={isCapturing}
                  className="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center justify-center gap-2 transition disabled:opacity-50"
                >
                  {isCapturing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Globe className="w-4 h-4" />}
                  Save Web Context to Memory Store
                </button>
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
