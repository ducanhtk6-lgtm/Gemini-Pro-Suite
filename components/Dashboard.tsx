import React, { useState, useEffect, useMemo } from 'react';
import { Chunk, LogEntry, ProcessingStats, TranscriptionOutput, ImprovedTranscriptItem, RateLimitEvent, RemovalAuditResult, DetailedRemovalRow, Batch } from '../types';
import { ActivityLog } from './ActivityLog';
import { GoogleGenAI, HarmCategory, HarmBlockThreshold } from '@google/genai';
import { computeRemovalAudit } from '../utils/transcriptMetrics';

/**
 * @LOCKED_UI_COMPONENT
 * Giao diện Dashboard chính:
 * - Stats Bar (Thống kê)
 * - Progress Bar (Tiến độ) - Ẩn khi Manual Mode
 * - Grid View / Result View / Input View (Chế độ xem)
 * - Activity Log (Nhật ký)
 */

interface DashboardProps {
    file: File | null; // Nullable for Manual Mode
    onReset: () => void;
    // Props từ hook useChunkProcessor
    chunks: Chunk[];
    stats: ProcessingStats;
    logs: LogEntry[];
    result: TranscriptionOutput | null;
    isFinalizing?: boolean; 
    fileType: string;
    retryChunk: (id: string) => void;
    retryAllFailed: () => void;
    triggerStep2: () => void;
    manualAppendTranscript: (json: string) => void;
    // New Props for model switching
    step1Model: string;
    setStep1Model: (model: string) => void;
    step2Model: string;
    setStep2Model: (model: string) => void;
    rateLimitEvent: RateLimitEvent | null;
    clearCooldownNow: (reason: string) => void;
    // New Props for Step 2 Batching
    step2Batches: Batch[];
    retryBatch: (id: string) => void;
    retryAllFailedBatches: () => void;
}

// Icons
const FileTextIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>;
const ClockIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>;
const CheckCircleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>;
const LoaderIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="animate-spin"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>;
const AlertTriangleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>;
const RefreshCwIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/></svg>;
const CopyIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>;
const DownloadIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>;
const PlayIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3"/></svg>;
const PlusCircleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>;
const EditIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>;
const YoutubeIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2.5 17a24.12 24.12 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.56 49.56 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24.12 24.12 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.55 49.55 0 0 1-16.2 0A2 2 0 0 1 2.5 17"/><path d="m10 15 5-3-5-3z"/></svg>;
const CodeIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>;
const ZapIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>;
const SparklesIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9a2 2 0 0 1 2 2a6 6 0 0 0-9-9a2 2 0 0 1-2-2Z"/><path d="M3 12a6 6 0 0 0 9 9a2 2 0 0 1 2 2a6 6 0 0 0-9-9a2 2 0 0 1-2-2Z"/></svg>;
const BrainCircuitIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.993.142"/><path d="M12 5a3 3 0 1 1 5.993.142"/><path d="M15 12a3 3 0 1 0-5.993.142"/><path d="M15 12a3 3 0 1 1 5.993.142"/><path d="M9 12a3 3 0 1 0-5.993.142"/><path d="M9 12a3 3 0 1 1 5.993.142"/><path d="M12 19a3 3 0 1 0-5.993.142"/><path d="M12 19a3 3 0 1 1 5.993.142"/><path d="M16 8.27A3 3 0 0 1 14.23 11l-2.46 4a3 3 0 0 1-5.54.142"/><path d="M8 8.27A3 3 0 0 0 9.77 11l2.46 4a3 3 0 0 0 5.54.142"/></svg>;
const CheckBadgeIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3.85 8.62a4 4 0 0 1 4.78-4.78l1.21 1.21a2 2 0 0 0 2.82 0l1.21-1.21a4 4 0 0 1 4.78 4.78l-1.21 1.21a2 2 0 0 0 0 2.82l1.21 1.21a4 4 0 0 1-4.78 4.78l-1.21-1.21a2 2 0 0 0-2.82 0l-1.21 1.21a4 4 0 0 1-4.78-4.78l1.21-1.21a2 2 0 0 0 0-2.82z"/><path d="m9 12 2 2 4-4"/></svg>;


const formatDuration = (ms: number) => {
    if (ms <= 0) return "00:00";
    const seconds = Math.floor(ms / 1000);
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
};

const ApiOverloadRecoveryPanel = ({
    rateLimitEvent,
    step1Model,
    setStep1Model,
    step2Model,
    setStep2Model,
    clearCooldownNow,
    triggerStep2,
    hasStep1Data
}: {
    rateLimitEvent: RateLimitEvent,
    step1Model: string,
    setStep1Model: (m: string) => void,
    step2Model: string,
    setStep2Model: (m: string) => void,
    clearCooldownNow: (reason: string) => void,
    triggerStep2: () => void,
    hasStep1Data: boolean
}) => {
    const isStep1 = rateLimitEvent.step === 'STEP1';
    const currentModel = isStep1 ? step1Model : step2Model;
    const [selectedModel, setSelectedModel] = useState(currentModel);

    // Set a sensible default fallback model when the panel appears
    useEffect(() => {
        const lastFailedModel = rateLimitEvent.lastModel;
        if (lastFailedModel === 'gemini-2.5-pro') setSelectedModel('gemini-2.5-flash');
        else if (lastFailedModel === 'gemini-3-pro-preview') setSelectedModel('gemini-2.5-pro');
        else setSelectedModel('gemini-2.5-pro'); // default for flash or anything else
    }, [rateLimitEvent.lastModel]);

    const modelOptions = [
        { value: "gemini-2.5-flash", label: "Gemini 2.5 Flash" },
        { value: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
        { value: "gemini-3-pro-preview", label: "Gemini 3 Pro Preview" },
    ];

    const handleApply = () => {
        if (isStep1) {
            setStep1Model(selectedModel);
            clearCooldownNow(`User switched Step 1 model to ${selectedModel}. Resuming now.`);
        } else {
            setStep2Model(selectedModel);
            clearCooldownNow(`User switched Step 2 model to ${selectedModel}. Retrying Step 2 now.`);
            if (hasStep1Data) {
                triggerStep2();
            }
        }
    };
    
    const isStep2ButtonDisabled = !isStep1 && !hasStep1Data;

    return (
        <div className="bg-orange-900/40 border border-orange-700/60 p-3 rounded-lg flex flex-col sm:flex-row items-center justify-between gap-3 shadow-lg">
            <div className="flex-1">
                 <h4 className="font-bold text-orange-300 flex items-center gap-2"><AlertTriangleIcon /> <span>Quá tải API ở {isStep1 ? 'Step 1' : 'Step 2'}</span></h4>
                 <p className="text-xs text-orange-300/80 mt-1">
                     Model <code className="bg-black/30 px-1 py-0.5 rounded">{rateLimitEvent.lastModel}</code> đã bị giới hạn. Đổi model để tiếp tục ngay.
                 </p>
            </div>
            <div className="flex items-center gap-2 w-full sm:w-auto">
                 <select 
                     value={selectedModel}
                     onChange={(e) => setSelectedModel(e.target.value)}
                     className="block w-full bg-gray-900 border border-gray-600 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-orange-500 focus:border-orange-500 sm:text-sm"
                 >
                     {modelOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                 </select>
                 <button 
                     onClick={handleApply}
                     disabled={isStep2ButtonDisabled}
                     title={isStep2ButtonDisabled ? "Không có dữ liệu Step 1 để chạy lại" : ""}
                     className="px-4 py-2 bg-orange-600 hover:bg-orange-500 text-white rounded-lg font-bold text-sm flex items-center gap-2 transition-all shadow-lg shadow-orange-900/30 whitespace-nowrap disabled:bg-gray-600 disabled:cursor-not-allowed"
                 >
                    <ZapIcon /> {isStep1 ? 'Tiếp tục' : 'Thử lại'}
                 </button>
            </div>
        </div>
    );
};

export const Dashboard: React.FC<DashboardProps> = ({
    file, onReset, chunks, stats, logs, result, fileType, retryChunk, retryAllFailed, isFinalizing, triggerStep2, manualAppendTranscript,
    step1Model, setStep1Model, step2Model, setStep2Model, rateLimitEvent, clearCooldownNow,
    step2Batches, retryBatch, retryAllFailedBatches
}) => {
    const [elapsed, setElapsed] = useState(0);
    // If file is null, default to 'input' tab. Else default to 'grid'.
    const [activeTab, setActiveTab] = useState<'grid' | 'text' | 'input'>(file ? 'grid' : 'input');
    
    // Manual Input States
    const [manualInput, setManualInput] = useState("");
    const [inputType, setInputType] = useState<'json' | 'raw'>('json'); // 'json' or 'raw' (youtube text)
    const isManualMode = !file;
    const hasStep1Data = !!result?.improved_transcript && result.improved_transcript.length > 0;


    // Timer effect
    useEffect(() => {
        if (stats.startTime > 0 && !stats.endTime) {
            const timer = setInterval(() => {
                setElapsed(Date.now() - stats.startTime);
            }, 500);
            return () => clearInterval(timer);
        } else if (stats.endTime && stats.startTime) {
            setElapsed(stats.endTime - stats.startTime);
        } else {
            setElapsed(0);
        }
    }, [stats.startTime, stats.endTime]);

    const progressPercentage = useMemo(() => {
        if (stats.total === 0) return 0;
        return (stats.completed / stats.total) * 100;
    }, [stats.total, stats.completed]);

    // Auto-switch to text tab when done (Only in File Mode)
    useEffect(() => {
        if (!isManualMode && stats.total > 0 && stats.completed === stats.total) {
            setActiveTab('text');
        }
    }, [stats.completed, stats.total, isManualMode]);

    const parseRawTextToJSON = (rawText: string) => {
        // Regex to find timestamp at the start of a line (Supports 00:00, 0:00, [00:00], (00:00))
        // Group 1: Timestamp string
        // Group 2: Text content (optional)
        const lines = rawText.split('\n');
        const items: ImprovedTranscriptItem[] = [];
        const timestampRegex = /^(?:\[?\(?(\d{1,2}:\d{2}(?::\d{2})?)\)?\]?)\s*-?\s*(.*)/;

        lines.forEach(line => {
            const cleanLine = line.trim();
            if (!cleanLine) return;

            const match = cleanLine.match(timestampRegex);
            if (match) {
                let ts = match[1];
                let content = match[2].trim();
                
                // Normalize timestamp to [MM:SS] or [HH:MM:SS]
                if (!ts.startsWith('[')) ts = `[${ts}]`;

                // ALWAYS create a new item when a timestamp is found, even with empty content
                // (as content might be on the next line)
                items.push({
                    timestamp: ts,
                    original: content,
                    edited: content, 
                    speaker: "[??]",
                    uncertain: false
                });
            } else {
                // If the line has no timestamp, append it to the last item
                if (items.length > 0) {
                    const lastItem = items[items.length - 1];
                    // Add a space if the previous item already had content
                    const separator = lastItem.original ? " " : "";
                    lastItem.original += separator + cleanLine;
                    lastItem.edited += separator + cleanLine;
                }
            }
        });
        return items;
    };

    const handleManualAppend = () => {
        if (!manualInput.trim()) return;
        
        try {
            if (inputType === 'json') {
                manualAppendTranscript(manualInput);
            } else {
                // Parse Raw Text -> JSON -> Append
                const parsedItems = parseRawTextToJSON(manualInput);
                if (parsedItems.length === 0) {
                    alert("No valid timestamps found (e.g., 00:00 or 01:23). Please check the format.");
                    return;
                }
                const jsonString = JSON.stringify(parsedItems);
                manualAppendTranscript(jsonString);
            }
            setManualInput("");
        } catch (e: any) {
            alert("Data Error: " + e.message);
        }
    };

    return (
        <div className="w-full h-full flex flex-col gap-6 bg-gray-900 text-gray-100 p-4 rounded-xl">
            {/* 1) Top Stats Bar */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <StatCard 
                    icon={isManualMode ? <EditIcon /> : <FileTextIcon />} 
                    label={isManualMode ? "Mode" : "Document"} 
                    value={isManualMode ? "Manual Input" : file?.name} 
                    subValue={isManualMode ? "Step 2 Post-Edit" : (file!.size / 1024 / 1024).toFixed(2) + " MB"} 
                />
                
                {isManualMode ? (
                     <StatCard icon={<FileTextIcon />} label="Items Imported" value={(result?.improved_transcript?.length || 0).toString()} subValue="Dialogue lines" color="text-teal-400" />
                ) : (
                    <>
                        <StatCard icon={<ClockIcon />} label="Time" value={formatDuration(elapsed)} subValue={stats.endTime ? "Completed" : "Running..."} />
                        <StatCard icon={<CheckCircleIcon />} label="Completed" value={`${stats.completed}/${stats.total}`} color="text-green-400" />
                        <StatCard icon={<LoaderIcon />} label="Processing" value={stats.processing.toString()} color="text-blue-400" />
                    </>
                )}
                
                <div className="bg-gray-800 p-3 rounded-lg border border-gray-700 flex flex-col justify-between">
                    <div className="flex items-center gap-2 text-gray-400 text-xs uppercase font-bold tracking-wider">
                        <AlertTriangleIcon /> <span>Failed</span>
                    </div>
                    <div className="flex justify-between items-end mt-1">
                        <span className={`text-xl font-mono font-bold ${stats.failed > 0 ? 'text-red-400' : 'text-gray-500'}`}>{stats.failed}</span>
                        {!isManualMode && (
                            <button 
                                onClick={retryAllFailed}
                                disabled={stats.failed === 0}
                                className={`p-1 rounded transition-colors ${stats.failed > 0 ? 'bg-red-500/20 text-red-400 hover:bg-red-500/40' : 'text-gray-600 cursor-not-allowed'}`}
                                title="Retry all failed"
                            >
                                <RefreshCwIcon />
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* 2) Progress Bar (Only in File Mode) */}
            {!isManualMode && (
                <div className="relative w-full">
                    <div className="flex justify-between mb-1 text-xs font-medium text-gray-400">
                        <span>Processing Progress (Step 1)</span>
                        <span>{Math.round(progressPercentage)}%</span>
                    </div>
                    <div className={`w-full bg-gray-700 rounded-full h-2.5 overflow-hidden ${stats.isCoolingDown ? 'opacity-60 grayscale' : ''}`}>
                        <div 
                            className="bg-blue-500 h-2.5 rounded-full transition-all duration-500 ease-out" 
                            style={{ width: `${progressPercentage}%` }}
                        ></div>
                    </div>
                </div>
            )}
            
            {/* Step 2 Loading Indicator (Common) */}
            {isFinalizing && (
                <div className="w-full bg-purple-900/20 border border-purple-800 p-3 rounded flex items-center justify-center gap-3 animate-pulse">
                    <LoaderIcon />
                    <span className="text-purple-300 font-bold">Running Step 2: Refining and generating professional script...</span>
                </div>
            )}

            {/* 3) Cooldown Alert */}
            {stats.isCoolingDown && !rateLimitEvent?.active && (
                <div className="bg-yellow-900/30 border border-yellow-700/50 text-yellow-200 px-4 py-2 rounded-lg flex items-center justify-between animate-pulse">
                    <div className="flex items-center gap-2">
                        <ClockIcon />
                        <span className="text-sm font-medium">System is cooling down to avoid Rate Limit</span>
                    </div>
                    <span className="font-mono font-bold">{formatDuration(stats.cooldownSeconds * 1000)}</span>
                </div>
            )}

            {/* NEW: API Overload Recovery Panel */}
            {stats.isCoolingDown && rateLimitEvent?.active && (
                <ApiOverloadRecoveryPanel
                    rateLimitEvent={rateLimitEvent}
                    step1Model={step1Model}
                    setStep1Model={setStep1Model}
                    step2Model={step2Model}
                    setStep2Model={setStep2Model}
                    clearCooldownNow={clearCooldownNow}
                    triggerStep2={triggerStep2}
                    hasStep1Data={hasStep1Data}
                />
            )}


            {/* Main Content Area */}
            <div className="flex flex-col lg:flex-row gap-6 h-[600px] mt-2">
                {/* Left Column: Tabs & Content */}
                <div className="flex-1 flex flex-col bg-gray-800/50 rounded-lg border border-gray-700 overflow-hidden">
                    {/* Tabs */}
                    <div className="flex border-b border-gray-700">
                        {isManualMode && (
                            <button 
                                onClick={() => setActiveTab('input')}
                                className={`flex-1 py-3 text-sm font-medium text-center transition-colors ${activeTab === 'input' ? 'bg-gray-700 text-teal-300 border-b-2 border-teal-400' : 'text-gray-400 hover:bg-gray-700/50'}`}
                            >
                                Input Data
                            </button>
                        )}
                        {!isManualMode && (
                            <button 
                                onClick={() => setActiveTab('grid')}
                                className={`flex-1 py-3 text-sm font-medium text-center transition-colors ${activeTab === 'grid' ? 'bg-gray-700 text-blue-300 border-b-2 border-blue-400' : 'text-gray-400 hover:bg-gray-700/50'}`}
                            >
                                Chunk Grid
                            </button>
                        )}
                        <button 
                            onClick={() => setActiveTab('text')}
                            className={`flex-1 py-3 text-sm font-medium text-center transition-colors ${activeTab === 'text' ? 'bg-gray-700 text-teal-300 border-b-2 border-teal-400' : 'text-gray-400 hover:bg-gray-700/50'}`}
                        >
                            Results (Step 1 & 2)
                        </button>
                    </div>

                    {/* Tab Content */}
                    <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                        {activeTab === 'grid' && !isManualMode && (
                            <ChunkGrid chunks={chunks} onRetry={retryChunk} />
                        )}
                        
                        {activeTab === 'input' && isManualMode && (
                            <div className="flex flex-col h-full gap-4">
                                <div className="flex flex-col gap-2 h-full">
                                    <div className="flex justify-between items-center">
                                         <label className="text-xs font-bold text-gray-400 uppercase">Input Data</label>
                                         <div className="flex bg-gray-900 rounded-lg p-1 border border-gray-700">
                                            <button 
                                                onClick={() => setInputType('json')}
                                                className={`px-3 py-1 rounded text-xs font-bold flex items-center gap-2 transition-all ${inputType === 'json' ? 'bg-teal-700 text-white' : 'text-gray-400 hover:text-white'}`}
                                            >
                                                <CodeIcon /> JSON
                                            </button>
                                            <button 
                                                onClick={() => setInputType('raw')}
                                                className={`px-3 py-1 rounded text-xs font-bold flex items-center gap-2 transition-all ${inputType === 'raw' ? 'bg-red-700 text-white' : 'text-gray-400 hover:text-white'}`}
                                            >
                                                <YoutubeIcon /> YouTube / Raw Text
                                            </button>
                                         </div>
                                    </div>

                                    <textarea 
                                        className="w-full flex-1 min-h-[150px] bg-gray-900 border border-gray-700 rounded p-3 text-xs font-mono text-gray-300 focus:outline-none focus:border-teal-500 transition-colors"
                                        placeholder={inputType === 'json' 
                                            ? '[ { "timestamp": "...", "original": "...", "edited": "..." }, ... ]' 
                                            : "00:00 Hello everyone\n00:05 Today we will learn..."}
                                        value={manualInput}
                                        onChange={(e) => setManualInput(e.target.value)}
                                    />
                                    <div className="flex justify-between items-center mt-2">
                                        <div className="text-[10px] text-gray-500 italic">
                                            {inputType === 'raw' 
                                                ? "Auto-detects timestamps (e.g., 00:00, [12:30]) and converts to application format."
                                                : "Paste an array of ImprovedTranscriptItem[] directly."
                                            }
                                        </div>
                                        <button 
                                            onClick={handleManualAppend}
                                            className="px-4 py-2 bg-teal-700 hover:bg-teal-600 text-white text-sm rounded-lg font-bold flex items-center gap-2 shadow-lg"
                                        >
                                            <PlusCircleIcon /> 
                                            {inputType === 'raw' ? "Convert & Add" : "Append Data"}
                                        </button>
                                    </div>

                                    {/* Preview List */}
                                    <div className="border-t border-gray-700 pt-4 flex-1 flex flex-col overflow-hidden mt-2">
                                        <h4 className="text-xs font-bold text-gray-400 mb-2 uppercase">Current Data in Memory ({result?.improved_transcript?.length || 0} lines)</h4>
                                        <div className="flex-1 overflow-y-auto bg-gray-900/50 rounded p-2 border border-gray-800 custom-scrollbar">
                                            {result?.improved_transcript && result.improved_transcript.length > 0 ? (
                                                <RawTranscriptView items={result.improved_transcript} />
                                            ) : (
                                                <div className="text-gray-600 text-center italic mt-10">No data yet.</div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeTab === 'text' && (
                            <ResultView 
                                result={result} 
                                isFinalizing={isFinalizing} 
                                onTriggerStep2={triggerStep2}
                                fileName={file?.name || 'manual_transcript'}
                                step2Batches={step2Batches}
                                onRetryBatch={retryBatch}
                                onRetryAllFailedBatches={retryAllFailedBatches}
                            />
                        )}
                    </div>
                </div>

                {/* Right Column: Activity Log & Actions */}
                <div className="w-full lg:w-80 flex flex-col gap-4">
                    <div className="flex-1 overflow-hidden rounded-lg">
                        <ActivityLog logs={logs} />
                    </div>
                    
                    {/* Manual Trigger for Step 2 */}
                    {isManualMode && result?.improved_transcript && result.improved_transcript.length > 0 && !result.post_edit_result && (
                         <button 
                            onClick={triggerStep2}
                            disabled={isFinalizing}
                            className="w-full py-4 bg-teal-600 hover:bg-teal-500 text-white rounded-lg transition-colors font-bold shadow-lg shadow-teal-900/20 flex items-center justify-center gap-2"
                        >
                            {isFinalizing ? <LoaderIcon /> : <PlayIcon />}
                            RUN STEP 2 (POST-EDIT)
                        </button>
                    )}

                    <button 
                        onClick={onReset}
                        className="w-full py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors font-medium border border-gray-600 hover:border-gray-500"
                    >
                        {isManualMode ? "Exit Manual Mode" : "Start New Job"}
                    </button>
                </div>
            </div>
        </div>
    );
};

// --- Sub Components ---

const StatCard = ({ icon, label, value, subValue, color = "text-gray-200" }: any) => (
    <div className="bg-gray-800 p-3 rounded-lg border border-gray-700 flex flex-col justify-between overflow-hidden">
        <div className="flex items-center gap-2 text-gray-400 text-xs uppercase font-bold tracking-wider truncate">
            {icon} <span>{label}</span>
        </div>
        <div className="mt-2">
            <div className={`text-xl font-mono font-bold truncate ${color}`}>{value}</div>
            {subValue && <div className="text-xs text-gray-500 truncate" title={subValue}>{subValue}</div>}
        </div>
    </div>
);

const ChunkGrid = ({ chunks, onRetry }: { chunks: Chunk[], onRetry: (id: string) => void }) => {
    if (chunks.length === 0) return <div className="flex items-center justify-center h-full text-gray-500 italic">Initializing list...</div>;

    return (
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
            {chunks.map((chunk) => (
                <div 
                    key={chunk.id}
                    className={`
                        aspect-square rounded border flex items-center justify-center text-xs font-mono cursor-default relative group
                        ${chunk.status === 'completed' ? 'bg-green-900/30 border-green-700 text-green-400' : 
                          chunk.status === 'processing' ? 'bg-blue-900/30 border-blue-700 text-blue-400 animate-pulse' :
                          chunk.status === 'failed' ? 'bg-red-900/30 border-red-700 text-red-400 cursor-pointer hover:bg-red-900/50' :
                          'bg-gray-800 border-gray-700 text-gray-500'}
                    `}
                    onClick={() => chunk.status === 'failed' && onRetry(chunk.id)}
                    title={`Chunk #${chunk.index} - ${chunk.status}${chunk.error ? ': ' + chunk.error : ''}`}
                >
                    {chunk.index + 1}
                    {chunk.status === 'failed' && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity rounded">
                            <RefreshCwIcon />
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
};

const BatchGrid = ({ batches, onRetry }: { batches: Batch[], onRetry: (id: string) => void }) => {
    if (batches.length === 0) return null;

    return (
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
            {batches.map((batch) => (
                <div 
                    key={batch.id}
                    className={`
                        aspect-square rounded border flex items-center justify-center text-xs font-mono cursor-default relative group
                        ${batch.status === 'completed' ? 'bg-green-900/30 border-green-700 text-green-400' : 
                          batch.status === 'processing' ? 'bg-purple-900/30 border-purple-700 text-purple-400 animate-pulse' :
                          batch.status === 'failed' ? 'bg-red-900/30 border-red-700 text-red-400 cursor-pointer hover:bg-red-900/50' :
                          'bg-gray-800 border-gray-700 text-gray-500'}
                    `}
                    onClick={() => batch.status === 'failed' && onRetry(batch.id)}
                    title={`Batch #${batch.index} - ${batch.status}${batch.error ? ': ' + batch.error : ''}`}
                >
                    {batch.index + 1}
                    {batch.status === 'failed' && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity rounded">
                            <RefreshCwIcon />
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
};

const RawTranscriptView = ({ items }: { items: ImprovedTranscriptItem[] }) => {
     const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

    const handleCopy = (text: string, index: number) => {
        navigator.clipboard.writeText(text);
        setCopiedIndex(index);
        setTimeout(() => setCopiedIndex(null), 1500);
    };

    return (
        <div className="space-y-2 font-mono text-xs pr-2">
            {items.map((item, idx) => {
                let speakerLabel = item.speaker || "??";
                let speakerColor = "bg-gray-700 text-gray-300";
                if (speakerLabel.includes("GV")) speakerColor = "bg-blue-900/50 text-blue-300 border border-blue-800";
                else if (speakerLabel.includes("SV")) speakerColor = "bg-purple-900/50 text-purple-300 border border-purple-800";
                
                return (
                    <div key={idx} className={`flex gap-3 items-baseline group ${item.uncertain ? 'opacity-60' : ''}`}>
                        <span className="text-teal-600 w-[50px] shrink-0 text-right">{item.timestamp}</span>
                        <span className={`px-1 rounded text-[9px] font-bold uppercase w-[30px] text-center ${speakerColor}`}>
                            {speakerLabel.replace(/[\[\]]/g,'')}
                        </span>
                        <span className="text-gray-400 flex-1">{item.edited}</span>
                        <button 
                            onClick={() => handleCopy(item.edited, idx)}
                            className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-white transition-opacity p-1 rounded"
                        >
                            {copiedIndex === idx ? <CheckCircleIcon /> : <CopyIcon />}
                        </button>
                    </div>
                );
            })}
        </div>
    )
}

type AiAction = 'summarize' | 'key-points' | 'titles' | 'deep-analysis';

const AiActionsPanel = ({ transcriptText, onResult, onLoadingChange }: { transcriptText: string, onResult: (title: string, content: string) => void, onLoadingChange: (isLoading: boolean) => void }) => {
    
    const handleAction = async (action: AiAction) => {
        if (!transcriptText) return;

        onLoadingChange(true);
        let prompt = '';
        let model = 'gemini-2.5-flash';
        let config: any = {};
        let title = '';

        switch(action) {
            case 'summarize':
                title = 'Content Summary';
                prompt = `Summarize the following medical lecture into a short, concise paragraph focusing on the main ideas and key conclusions:\n\n---\n\n${transcriptText}`;
                break;
            case 'key-points':
                title = 'Key Points';
                prompt = `List the most important key points from the following medical lecture. Present them as bullet points:\n\n---\n\n${transcriptText}`;
                break;
            case 'titles':
                title = 'Suggested Titles';
                prompt = `Based on the content of the following medical lecture, suggest 5 engaging and relevant titles:\n\n---\n\n${transcriptText}`;
                break;
            case 'deep-analysis':
                title = 'In-depth Analysis (Thinking Mode)';
                model = 'gemini-3-pro-preview';
                config = { thinkingConfig: { thinkingBudget: 32768 } };
                prompt = `Perform an in-depth analysis of the following medical lecture. Focus on identifying complex concepts, relationships between ideas, points of potential confusion, and suggest related topics for further study. Present the results in a clear, structured format:\n\n---\n\n${transcriptText}`;
                break;
        }

        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
            // FIX: `safetySettings` must be a property of the `config` object.
            const response = await ai.models.generateContent({
                model,
                contents: prompt,
                config: {
                    ...config,
                    safetySettings: [
                        { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    ],
                },
            });
            onResult(title, response.text || 'No response.');
        } catch (error: any) {
            console.error(`AI Action (${action}) failed:`, error);
            onResult(`Error during ${title}`, error.message || 'An unknown error occurred.');
        } finally {
            onLoadingChange(false);
        }
    };

    return (
        <div className="bg-gray-900/50 p-4 rounded-lg border border-gray-700">
             <h3 className="font-bold text-violet-400 uppercase text-xs tracking-wider flex items-center gap-2 mb-3">
                <SparklesIcon />
                <span>AI Actions</span>
            </h3>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                <button onClick={() => handleAction('summarize')} className="px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded font-bold transition-colors text-center">Summarize</button>
                <button onClick={() => handleAction('key-points')} className="px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded font-bold transition-colors text-center">Key Points</button>
                <button onClick={() => handleAction('titles')} className="px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded font-bold transition-colors text-center">Titles</button>
                 <button onClick={() => handleAction('deep-analysis')} className="px-3 py-2 bg-violet-800 hover:bg-violet-700 text-white text-xs rounded font-bold transition-colors flex items-center justify-center gap-1.5" title="Uses Gemini 3 Pro with Thinking Mode">
                    <BrainCircuitIcon /> In-depth Analysis
                </button>
            </div>
        </div>
    );
};

const RemovalAuditReport = ({ metrics }: { metrics: RemovalAuditResult }) => {

    const getStatusChip = (status: DetailedRemovalRow['status']) => {
        switch (status) {
            case 'REPORTED': return <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-gray-600 text-gray-300">REPORTED</span>;
            case 'UNREPORTED': return <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-red-700 text-red-200">UNREPORTED!</span>;
            case 'USED_BUT_REPORTED': return <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-yellow-700 text-yellow-200">USED (Reported)</span>;
            case 'UNKNOWN_TIMESTAMP': return <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-orange-700 text-orange-200">UNKNOWN TS</span>;
            default: return null;
        }
    };
    
    return (
        <div className="mt-4 pt-4 border-t border-gray-700 text-xs text-gray-400 space-y-4">
            <h4 className="font-bold text-gray-300 uppercase tracking-wider text-sm flex items-center gap-2"><CheckBadgeIcon /> QA & Removal Audit</h4>
            
            {/* Summary Block */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2 text-center">
                <div className="bg-slate-800/50 p-2 rounded"><div className="font-bold text-lg text-white">{metrics.step1ItemCount}</div><div className="text-[10px] uppercase text-gray-500">Step 1 Items</div></div>
                <div className="bg-slate-800/50 p-2 rounded"><div className="font-bold text-lg text-green-400">{metrics.usedItemCount}</div><div className="text-[10px] uppercase text-gray-500">Used in Step 2</div></div>
                <div className="bg-slate-800/50 p-2 rounded"><div className="font-bold text-lg text-white">{metrics.actuallyRemovedCount}</div><div className="text-[10px] uppercase text-gray-500">Actually Removed</div></div>
                <div className={`bg-slate-800/50 p-2 rounded ${metrics.unreportedDropCount > 0 ? 'border border-red-600/50' : ''}`}><div className={`font-bold text-lg ${metrics.unreportedDropCount > 0 ? 'text-red-400' : 'text-white'}`}>{metrics.unreportedDropCount}</div><div className="text-[10px] uppercase text-gray-500">Unreported Drops</div></div>
                <div className={`bg-slate-800/50 p-2 rounded ${metrics.usedButReportedCount > 0 ? 'border border-yellow-600/50' : ''}`}><div className={`font-bold text-lg ${metrics.usedButReportedCount > 0 ? 'text-yellow-400' : 'text-white'}`}>{metrics.usedButReportedCount}</div><div className="text-[10px] uppercase text-gray-500">Used but Reported</div></div>
            </div>

            {/* Ranges & Reasons */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                     <h5 className="font-bold text-gray-400 mb-2">Removed Ranges (≥ 15s gap)</h5>
                     {metrics.removedRanges.length > 0 ? (
                        <div className="space-y-1">
                            {metrics.removedRanges.map((range, idx) => (
                                <div key={idx} className="text-xs font-mono bg-gray-950/50 p-1 rounded">
                                    <span className="text-amber-500">{range.start} - {range.end}</span> <span className="text-gray-500">({range.count} items)</span>
                                </div>
                            ))}
                        </div>
                     ) : <p className="italic text-gray-600">No significant ranges removed.</p>}
                </div>
                <div>
                     <h5 className="font-bold text-gray-400 mb-2">Reported Reasons</h5>
                     {Object.keys(metrics.reasonCounts).length > 0 ? (
                        <div className="space-y-1">
                             {Object.entries(metrics.reasonCounts).map(([reason, count]) => (
                                 <div key={reason} className="flex justify-between items-center text-xs bg-gray-950/50 p-1 rounded">
                                     <span className="text-gray-400">{reason}</span>
                                     <span className="font-bold text-white">{count}</span>
                                 </div>
                             ))}
                        </div>
                     ) : <p className="italic text-gray-600">No reasons reported.</p>}
                </div>
            </div>

            {/* Detailed List */}
            <details className="bg-gray-950/30 rounded-lg">
                <summary className="cursor-pointer p-2 font-bold text-gray-400 hover:bg-gray-950/50 rounded-t-lg">
                    Detailed Removal List ({metrics.detailedRows.length} items)
                </summary>
                <div className="p-2 border-t border-gray-800 max-h-80 overflow-y-auto custom-scrollbar">
                    <table className="w-full text-left text-xs">
                        <thead className="sticky top-0 bg-gray-900 z-10">
                            <tr>
                                <th className="p-1.5 w-20">Timestamp</th>
                                <th className="p-1.5 w-24">Status</th>
                                <th className="p-1.5">Excerpt</th>
                            </tr>
                        </thead>
                        <tbody>
                        {metrics.detailedRows.map((row, idx) => (
                            <tr key={idx} className="border-t border-gray-800/50 hover:bg-white/5">
                                <td className="p-1.5 font-mono text-teal-500 align-top">{row.timestamp}</td>
                                <td className="p-1.5 align-top">
                                    {getStatusChip(row.status)}
                                    {row.needsReview && <span className="block mt-1 text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-pink-700 text-pink-200">MED-RISK</span>}
                                </td>
                                <td className="p-1.5 text-gray-400 italic">"{row.excerpt}"</td>
                            </tr>
                        ))}
                        </tbody>
                    </table>
                </div>
            </details>
        </div>
    );
};


const ResultView = ({ 
    result, 
    isFinalizing, 
    onTriggerStep2,
    fileName,
    step2Batches,
    onRetryBatch,
    onRetryAllFailedBatches
}: { 
    result: TranscriptionOutput | null, 
    isFinalizing?: boolean,
    onTriggerStep2: () => void,
    fileName: string,
    step2Batches: Batch[],
    onRetryBatch: (id: string) => void,
    onRetryAllFailedBatches: () => void
}) => {
    const [aiActionResult, setAiActionResult] = useState<{title: string, content: string} | null>(null);
    const [isAiActionLoading, setIsAiActionLoading] = useState(false);

    const metrics = useMemo(() => {
        if (!result?.improved_transcript) return null;
        return computeRemovalAudit(
            result.improved_transcript,
            result.post_edit_result?.refined_script || [],
            result.post_edit_result?.removal_report
        );
    }, [result]);

    if (!result) return <div className="flex items-center justify-center h-full text-gray-500 italic">No processing results yet.</div>;
    
    const postEdit = result.post_edit_result;
    const hasRawData = result.improved_transcript && result.improved_transcript.length > 0;
    const transcriptText = useMemo(() => {
        if (postEdit?.refined_script) {
            return postEdit.refined_script.map(item => `${item.speaker}: ${item.text}`).join('\n');
        }
        if (hasRawData) {
            return result.improved_transcript.map(item => `${item.speaker || '[??]'}: ${item.edited}`).join('\n');
        }
        return '';
    }, [result]);
    
    const downloadFile = (content: string, type: string, extension: string) => {
        const safeFileName = fileName.replace(/[^a-z0-9]/gi, '_').toLowerCase();
        const blob = new Blob([content], { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${safeFileName}_${extension}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const handleDownloadStep1 = () => {
        if (!result.improved_transcript) return;
        const jsonContent = JSON.stringify(result.improved_transcript, null, 2);
        downloadFile(jsonContent, 'application/json', 'step1_raw.json');
    };

    const handleDownloadStep2 = () => {
        if (!postEdit?.refined_script) return;
        const textContent = postEdit.refined_script
            .map(item => `${item.start_timestamp}-${item.end_timestamp} ${item.speaker}:\n${item.text}`)
            .join('\n\n');
        downloadFile(textContent, 'text/plain', 'step2_refined.txt');
    };

    const failedBatchCount = useMemo(() => step2Batches.filter(b => b.status === 'failed').length, [step2Batches]);

    return (
        <div className="space-y-6">
             {hasRawData && !postEdit && !isFinalizing && (
                 <div className="bg-blue-900/20 border border-blue-800 p-4 rounded-lg flex flex-col sm:flex-row items-center justify-between gap-4">
                     <div>
                         <h4 className="font-bold text-blue-400">Raw data is ready ({result.improved_transcript.length} lines)</h4>
                         <p className="text-xs text-gray-400">You can review the raw transcript below before proceeding.</p>
                     </div>
                     <button 
                        onClick={onTriggerStep2}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold text-sm flex items-center gap-2 transition-all shadow-lg shadow-blue-900/20"
                     >
                         <PlayIcon /> Run Step 2 (Post-Edit)
                     </button>
                 </div>
             )}
            
            {transcriptText && (
                <AiActionsPanel 
                    transcriptText={transcriptText}
                    onResult={(title, content) => setAiActionResult({title, content})}
                    onLoadingChange={setIsAiActionLoading}
                />
            )}
            
            {(isAiActionLoading || aiActionResult) && (
                 <div className="bg-gray-900/50 p-4 rounded-lg border border-gray-700 relative group">
                     {isAiActionLoading ? (
                        <div className="flex items-center justify-center gap-3 text-violet-300 animate-pulse">
                            <LoaderIcon />
                            <span>AI is analyzing...</span>
                        </div>
                     ) : aiActionResult && (
                        <div>
                             <h3 className="font-bold text-violet-400 mb-2">{aiActionResult.title}</h3>
                             <pre className="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap font-sans bg-black/20 p-3 rounded">{aiActionResult.content}</pre>
                        </div>
                     )}
                 </div>
            )}

             {step2Batches.length > 0 && (
                 <div className="bg-gray-900/50 p-4 rounded-lg border border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="font-bold text-purple-400 uppercase text-xs tracking-wider">Step 2 Batch Processing</h3>
                        {failedBatchCount > 0 && (
                            <button onClick={onRetryAllFailedBatches} className="text-red-400 hover:text-white bg-red-500/20 hover:bg-red-500/40 transition-colors flex items-center gap-2 text-xs font-bold px-3 py-1 rounded-md">
                                <RefreshCwIcon /> Retry All ({failedBatchCount})
                            </button>
                        )}
                    </div>
                    <BatchGrid batches={step2Batches} onRetry={onRetryBatch} />
                 </div>
             )}


             <div className="bg-gray-900/50 p-4 rounded-lg border border-gray-700 relative group">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="font-bold text-teal-400 uppercase text-xs tracking-wider flex items-center gap-2">
                        <span>Professional Lecture Script (Step 2)</span>
                        {isFinalizing && <span className="animate-pulse text-purple-400 ml-2">- Generating...</span>}
                    </h3>
                    {postEdit && (
                         <button onClick={handleDownloadStep2} className="text-gray-500 hover:text-teal-400 transition-colors flex items-center gap-2 text-xs" title="Download as .txt">
                            <DownloadIcon /> Download TXT
                        </button>
                    )}
                </div>

                {metrics && postEdit && (
                    <div className="text-xs font-mono text-gray-500 bg-gray-950/50 p-1.5 rounded-md mb-3 flex flex-wrap items-center gap-x-3 gap-y-1">
                        <span>S1 Words: <b className="text-gray-300">{metrics.step1WordCount}</b></span>
                        <span className="text-gray-600">|</span>
                        <span>S2 Words: <b className="text-gray-300">{metrics.step2WordCount}</b></span>
                        <span className="text-gray-600">|</span>
                        <span>Δ: <b className={metrics.deltaWords >= 0 ? 'text-green-400' : 'text-red-400'}>{metrics.deltaWords}</b></span>
                        <span className="text-gray-600">|</span>
                        <span>%: <b className={metrics.deltaPercent >= 0 ? 'text-green-400' : 'text-red-400'}>{metrics.deltaPercent.toFixed(1)}%</b></span>
                    </div>
                )}
                
                {postEdit ? (
                    <div className="space-y-4">
                        {postEdit.refined_script?.map((item, idx) => (
                            <div key={idx} className={`p-3 rounded bg-gray-800/50 border border-gray-700 ${item.needs_review ? 'border-yellow-600/50 bg-yellow-900/10' : ''}`}>
                                <div className="flex items-center gap-2 mb-1">
                                    <span className="text-xs font-bold text-teal-500 bg-gray-900 px-1.5 py-0.5 rounded">
                                        {item.start_timestamp} - {item.end_timestamp}
                                    </span>
                                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase ${
                                        item.speaker.includes("GV") ? "bg-blue-900 text-blue-300" : "bg-purple-900 text-purple-300"
                                    }`}>
                                        {item.speaker}
                                    </span>
                                    {item.needs_review && <span className="text-[10px] text-yellow-500 font-bold uppercase ml-auto">Needs Review</span>}
                                </div>
                                <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap">{item.text}</p>
                            </div>
                        ))}

                        {metrics && <RemovalAuditReport metrics={metrics} />}
                        
                    </div>
                ) : (
                    <div className="text-gray-500 text-sm italic py-4 text-center">
                        {step2Batches.length > 0 ? "Processing batches..." : "Script will appear here after you run Step 2."}
                    </div>
                )}
            </div>

            <details className="bg-gray-900/50 p-4 rounded-lg border border-gray-700 opacity-80 hover:opacity-100 transition-opacity" open={!postEdit}>
                <summary className="flex items-center justify-between cursor-pointer">
                    <h3 className="font-bold text-blue-400 uppercase text-xs tracking-wider">Raw Transcript (Step 1 - Detailed)</h3>
                     {hasRawData && (
                        <div className="flex items-center gap-4">
                            {metrics && (
                                <span className="text-xs font-mono text-gray-500">Words: <b className="text-gray-300">{metrics.step1WordCount}</b></span>
                            )}
                            <button onClick={handleDownloadStep1} className="text-gray-500 hover:text-blue-400 transition-colors flex items-center gap-2 text-xs" title="Download as .json">
                                <DownloadIcon /> Download JSON
                            </button>
                        </div>
                    )}
                </summary>
                <div className="mt-3 pt-3 border-t border-gray-700/50">
                    {result.improved_transcript && <RawTranscriptView items={result.improved_transcript} />}
                </div>
            </details>
        </div>
    );
};