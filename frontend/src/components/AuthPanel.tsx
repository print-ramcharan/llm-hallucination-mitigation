"use client";

import React, { useState } from "react";
import { KeyRound, Layers3 } from "lucide-react";
import { AuthProfile, loginUser, registerUser } from "@/lib/api";

export function AuthPanel({ onAuthenticated }: { onAuthenticated: (profile: AuthProfile) => void }) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const profile = isRegister
        ? await registerUser(username, password)
        : await loginUser(username, password);
      onAuthenticated(profile);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-950 px-4 text-slate-100">
      <section className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900/80 p-7 shadow-2xl shadow-black/30">
        <div className="mb-6 flex items-center gap-3">
          <div className="rounded-xl bg-indigo-500/15 p-3 text-indigo-300"><Layers3 className="h-6 w-6" /></div>
          <div><p className="text-xs font-semibold uppercase tracking-widest text-cyan-300">Secure workspace</p><h1 className="mt-1 text-xl font-bold">Mitigating Context Degradation in LLMs</h1></div>
        </div>
        <h2 className="text-lg font-semibold">{isRegister ? "Create your account" : "Sign in"}</h2>
        <p className="mt-1 text-sm text-slate-400">Your documents and conversation memory are isolated to your account.</p>
        <form onSubmit={submit} className="mt-6 space-y-4">
          <label className="block text-sm text-slate-300">Username<input required minLength={3} maxLength={64} value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-white outline-none focus:border-indigo-400" /></label>
          <label className="block text-sm text-slate-300">Password<input required minLength={8} maxLength={256} type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={isRegister ? "new-password" : "current-password"} className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-white outline-none focus:border-indigo-400" /></label>
          {error && <p role="alert" className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-3 text-sm text-rose-300">{error}</p>}
          <button disabled={busy} className="flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-3 text-sm font-semibold hover:bg-indigo-500 disabled:opacity-60"><KeyRound className="h-4 w-4" />{busy ? "Please wait…" : isRegister ? "Create account" : "Sign in"}</button>
        </form>
        <button onClick={() => { setIsRegister((value) => !value); setError(null); }} className="mt-5 w-full text-center text-sm text-slate-400 hover:text-white">{isRegister ? "Already have an account? Sign in" : "New here? Create an account"}</button>
        <p className="mt-5 text-center text-[11px] leading-5 text-slate-500">Passwords are stored as salted PBKDF2 hashes. Bearer tokens expire after seven days.</p>
      </section>
    </main>
  );
}
