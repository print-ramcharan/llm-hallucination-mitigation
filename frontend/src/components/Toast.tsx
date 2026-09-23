"use client";

import React, { useEffect, useState } from "react";
import { CheckCircle2, Loader2, AlertCircle, Sparkles, X } from "lucide-react";

export type ToastType = "info" | "success" | "error";

export interface ToastMessage {
  id: string;
  title: string;
  description?: string;
  type: ToastType;
}

interface ToastProps {
  toast: ToastMessage | null;
  onDismiss: () => void;
}

export const Toast: React.FC<ToastProps> = ({ toast, onDismiss }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (toast) {
      setIsVisible(true);
      const timer = setTimeout(() => {
        setIsVisible(false);
        setTimeout(onDismiss, 300); // Allow exit animation to complete
      }, 3500);
      return () => clearTimeout(timer);
    } else {
      setIsVisible(false);
    }
  }, [toast, onDismiss]);

  if (!toast) return null;

  const getBorderAndBg = () => {
    switch (toast.type) {
      case "success":
        return "border-emerald-500/40 bg-slate-900/95 text-emerald-300 shadow-emerald-950/40";
      case "error":
        return "border-rose-500/40 bg-slate-900/95 text-rose-300 shadow-rose-950/40";
      default:
        return "border-indigo-500/40 bg-slate-900/95 text-indigo-300 shadow-indigo-950/40";
    }
  };

  const getIcon = () => {
    switch (toast.type) {
      case "success":
        return <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />;
      case "error":
        return <AlertCircle className="h-4 w-4 text-rose-400 shrink-0" />;
      default:
        return <Sparkles className="h-4 w-4 text-indigo-400 shrink-0 animate-pulse" />;
    }
  };

  return (
    <div
      className={`fixed bottom-6 right-6 z-50 transition-all duration-300 transform ${
        isVisible
          ? "translate-y-0 opacity-100 scale-100"
          : "translate-y-4 opacity-0 scale-95"
      } pointer-events-auto`}
    >
      <div
        className={`flex items-start gap-3 p-3.5 rounded-2xl border backdrop-blur-xl shadow-2xl max-w-md ${getBorderAndBg()}`}
      >
        <div className="mt-0.5">{getIcon()}</div>
        <div className="flex-1 min-w-0">
          <div className="text-xs font-semibold text-white tracking-tight">
            {toast.title}
          </div>
          {toast.description && (
            <div className="text-[11px] text-slate-400 mt-0.5 leading-relaxed">
              {toast.description}
            </div>
          )}
        </div>
        <button
          onClick={() => {
            setIsVisible(false);
            setTimeout(onDismiss, 300);
          }}
          className="text-slate-500 hover:text-slate-300 p-0.5 rounded transition-colors"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
};
