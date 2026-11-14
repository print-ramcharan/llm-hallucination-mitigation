"use client";

import React from "react";
import { Activity } from "lucide-react";
import { HealthStatus } from "@/types/ingestion";

interface NavbarProps {
  backendHealth: HealthStatus | null;
  healthError: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ backendHealth, healthError }) => {
  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-900/90 px-4 py-3 backdrop-blur-md sm:px-6">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4">
        <h1 className="text-sm font-semibold tracking-tight text-white sm:text-base">
          Context Degradation Workspace
        </h1>
        <div className="flex items-center gap-2">
          {backendHealth ? (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 text-xs font-medium">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping"></span>
              Connected
            </div>
          ) : healthError ? (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-rose-950/40 border border-rose-500/30 text-rose-400 text-xs font-medium">
              <span className="h-2 w-2 rounded-full bg-rose-400"></span>
              Backend unavailable
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800 text-slate-400 text-xs font-medium">
              <Activity className="h-3.5 w-3.5 animate-spin text-slate-400" />
              Connecting...
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
