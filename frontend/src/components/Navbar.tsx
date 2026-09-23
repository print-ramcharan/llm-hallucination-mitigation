"use client";

import React from "react";
import { Activity, Layers } from "lucide-react";
import { HealthStatus } from "@/types/ingestion";

interface NavbarProps {
  backendHealth: HealthStatus | null;
  healthError: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ backendHealth, healthError }) => {
  return (
    <header className="sticky top-0 z-50 backdrop-blur-md bg-slate-900/80 border-b border-slate-800 px-6 py-3.5">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Project Brand */}
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-blue-500 to-cyan-400 p-0.5 shadow-lg shadow-indigo-500/20">
            <div className="h-full w-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Layers className="h-5 w-5 text-indigo-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-base font-bold tracking-tight text-white">
                Mitigating Context Degradation
              </span>
              <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                Module 01
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Document Knowledge Base & Context Retrieval
            </p>
          </div>
        </div>

        {/* Pipeline Progress Indicator */}
        <div className="hidden lg:flex items-center gap-1.5 text-xs font-medium bg-slate-950/60 border border-slate-800/80 px-3 py-1.5 rounded-lg">
          <span className="flex items-center gap-1 text-emerald-400 font-semibold">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
            1. Ingestion
          </span>
          <span className="text-slate-600">→</span>
          <span className="text-slate-500">2. Hybrid Retrieval</span>
          <span className="text-slate-600">→</span>
          <span className="text-slate-500">3. Rerank & Context Opt</span>
          <span className="text-slate-600">→</span>
          <span className="text-slate-500">4. Grounded Gen</span>
        </div>

        {/* Backend Connection Indicator */}
        <div className="flex items-center gap-2">
          {backendHealth ? (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 text-xs font-medium">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping"></span>
              FastAPI Online ({backendHealth.version})
            </div>
          ) : healthError ? (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-rose-950/40 border border-rose-500/30 text-rose-400 text-xs font-medium">
              <span className="h-2 w-2 rounded-full bg-rose-400"></span>
              FastAPI Disconnected
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800 text-slate-400 text-xs font-medium">
              <Activity className="h-3.5 w-3.5 animate-spin text-slate-400" />
              Connecting backend...
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
