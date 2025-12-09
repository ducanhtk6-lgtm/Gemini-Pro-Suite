import React, { useEffect, useRef, useState } from 'react';
import { LogEntry } from '../types';

interface ActivityLogProps {
    logs: LogEntry[];
}

// Inline Lucide-style icons to ensure no external dependency issues
const TerminalIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>;
const CheckCircleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>;
const AlertTriangleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-yellow-500"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>;
const XCircleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-red-500"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>;
const InfoIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-400"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>;

export const ActivityLog: React.FC<ActivityLogProps> = ({ logs = [] }) => {
    const scrollRef = useRef<HTMLDivElement>(null);
    const [isNearBottom, setIsNearBottom] = useState(true);

    // Monitor scroll position
    const handleScroll = () => {
        if (!scrollRef.current) return;
        const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
        // Check if user is near bottom (within 50px)
        const isBottom = scrollHeight - scrollTop - clientHeight < 50;
        setIsNearBottom(isBottom);
    };

    // Auto-scroll to bottom only if user was already near bottom
    useEffect(() => {
        if (scrollRef.current && isNearBottom) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs, isNearBottom]);

    const formatTime = (timestamp: number) => {
        return new Date(timestamp).toLocaleTimeString('vi-VN', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    const getIcon = (type: LogEntry['type']) => {
        switch (type) {
            case 'success': return <CheckCircleIcon />;
            case 'error': return <XCircleIcon />;
            case 'warning': return <AlertTriangleIcon />;
            case 'info': default: return <InfoIcon />;
        }
    };

    return (
        <div className="flex flex-col h-full bg-slate-900 rounded-lg border border-slate-700 shadow-inner overflow-hidden font-mono text-xs">
            {/* Header */}
            <div className="flex items-center justify-between px-3 py-2 bg-slate-800 border-b border-slate-700 select-none">
                <div className="flex items-center gap-2 text-slate-300 font-bold tracking-wider">
                    <TerminalIcon />
                    <span>NHẬT KÝ HỆ THỐNG</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                    </span>
                    <span className="text-green-500 font-bold text-[10px]">LIVE</span>
                </div>
            </div>

            {/* Log Content */}
            <div 
                ref={scrollRef}
                onScroll={handleScroll}
                className="flex-1 overflow-y-auto p-3 space-y-1.5 custom-scrollbar bg-slate-950/50"
            >
                {logs.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-slate-600 italic opacity-50">
                        <TerminalIcon />
                        <span className="mt-2">Đang chờ sự kiện...</span>
                    </div>
                ) : (
                    logs.map((log) => (
                        <div key={log.id} className="flex items-start gap-3 group hover:bg-white/5 p-0.5 rounded transition-colors">
                            <span className="text-slate-500 shrink-0 select-none pt-0.5">
                                [{formatTime(log.timestamp)}]
                            </span>
                            <span className="pt-0.5 shrink-0">
                                {getIcon(log.type)}
                            </span>
                            <span className={`break-words leading-relaxed ${
                                log.type === 'error' ? 'text-red-400' :
                                log.type === 'warning' ? 'text-yellow-400' :
                                log.type === 'success' ? 'text-green-400' :
                                'text-slate-300'
                            }`}>
                                {log.message}
                            </span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};