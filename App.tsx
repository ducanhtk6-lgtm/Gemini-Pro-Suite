import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { TranscriptionTurn } from './types';
import { createBlob } from './utils/audio';
import { useChunkProcessor } from './hooks/useChunkProcessor';
import { Dashboard } from './components/Dashboard';
import { ImageGenerator } from './components/ImageGenerator';

/**
 * @LOCKED_UI_ENTRY
 * Entry point của ứng dụng.
 * Quản lý 3 luồng chính:
 * 1. Live Transcription (Microphone) -> Giữ nguyên logic cũ.
 * 2. File Processing -> Upload File -> Dashboard.
 * 3. Manual Post-Edit -> No File -> Dashboard (Manual Mode).
 */

const MicIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-8 h-8">
        <path d="M8.25 4.5a3.75 3.75 0 1 1 7.5 0v8.25a3.75 3.75 0 1 1-7.5 0V4.5Z" />
        <path d="M6 10.5a.75.75 0 0 1 .75.75v1.5a5.25 5.25 0 1 0 10.5 0v-1.5a.75.75 0 0 1 1.5 0v1.5a6.75 6.75 0 1 1-13.5 0v-1.5a.75.75 0 0 1 .75-.75Z" />
    </svg>
);

const StopIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-8 h-8">
        <path fillRule="evenodd" d="M4.5 7.5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3-3h-9a3 3 0 0 1-3-3v-9Z" clipRule="evenodd" />
    </svg>
);

const UploadIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 mr-2">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
    </svg>
);

const EditIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 mr-2">
        <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
    </svg>
);

const BrainCircuitIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.993.142"/><path d="M12 5a3 3 0 1 1 5.993.142"/><path d="M15 12a3 3 0 1 0-5.993.142"/><path d="M15 12a3 3 0 1 1 5.993.142"/><path d="M9 12a3 3 0 1 0-5.993.142"/><path d="M9 12a3 3 0 1 1 5.993.142"/><path d="M12 19a3 3 0 1 0-5.993.142"/><path d="M12 19a3 3 0 1 1 5.993.142"/><path d="M16 8.27A3 3 0 0 1 14.23 11l-2.46 4a3 3 0 0 1-5.54.142"/><path d="M8 8.27A3 3 0 0 0 9.77 11l2.46 4a3 3 0 0 0 5.54.142"/></svg>;
const ImageIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>;
const AudioLinesIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/></svg>;


const MAX_FILE_SIZE_MB = 200;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
const ACCEPTED_AUDIO_TYPES = "audio/wav,audio/mpeg,audio/aac,audio/flac,audio/ogg,audio/aiff,audio/mp4";


const App: React.FC = () => {
    // App Mode
    const [appMode, setAppMode] = useState<'transcriber' | 'imageGenerator'>('transcriber');

    // State for real-time transcription
    const [isRecording, setIsRecording] = useState<boolean>(false);
    const [status, setStatus] = useState<string>('Click the button to start transcribing live.');
    const [transcriptionTurns, setTranscriptionTurns] = useState<TranscriptionTurn[]>([]);
    
    // State for file/manual transcription
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [showDashboard, setShowDashboard] = useState(false);
    const [step1Model, setStep1Model] = useState('gemini-2.5-pro');
    const [step2Model, setStep2Model] = useState('gemini-3-pro-preview');


    // Call the hook (always at top level)
    const chunkProcessor = useChunkProcessor(selectedFile, step1Model, step2Model);

    // Fix Race Condition: Trigger initialization when file is actually set in state
    useEffect(() => {
        if (selectedFile && showDashboard) {
            // Only initialize chunks if a file exists
            chunkProcessor.initializeChunks();
        }
    }, [selectedFile, showDashboard]); 

    // Refs for real-time transcription
    const sessionRef = useRef<any | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
    const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
    const recordingStartTimeRef = useRef<number | null>(null);
    const currentInputTranscriptionRef = useRef<string>('');
    const currentOutputTranscriptionRef = useRef<string>('');

    // --- Live Transcription Handlers (Unchanged logic) ---
    const handleMessage = (message: LiveServerMessage) => {
        if (message.serverContent?.inputTranscription) {
            const text = message.serverContent.inputTranscription.text;
            currentInputTranscriptionRef.current += text;
            setTranscriptionTurns(prev => {
                const newTurns = [...prev];
                const lastTurnIndex = newTurns.length - 1;
                const lastTurn = newTurns[lastTurnIndex];
                if (lastTurn && !lastTurn.isFinal) {
                    newTurns[lastTurnIndex] = { ...lastTurn, user: currentInputTranscriptionRef.current };
                } else {
                    const elapsedTime = Date.now() - (recordingStartTimeRef.current ?? Date.now());
                    const minutes = Math.floor(elapsedTime / 60000);
                    const seconds = Math.floor((elapsedTime % 60000) / 1000).toString().padStart(2, '0');
                    const timestamp = `${minutes}:${seconds}`;

                    newTurns.push({ user: currentInputTranscriptionRef.current, model: '', isFinal: false, timestamp });
                }
                return newTurns;
            });
        }
        if (message.serverContent?.outputTranscription) {
            const text = message.serverContent.outputTranscription.text;
            currentOutputTranscriptionRef.current += text;
             setTranscriptionTurns(prev => {
                const newTurns = [...prev];
                const lastTurnIndex = newTurns.length - 1;
                const lastTurn = newTurns[lastTurnIndex];
                if (lastTurn && !lastTurn.isFinal) {
                    newTurns[lastTurnIndex] = { ...lastTurn, model: currentOutputTranscriptionRef.current };
                }
                return newTurns;
            });
        }
        if (message.serverContent?.turnComplete) {
            setTranscriptionTurns(prev => {
                if (prev.length === 0) return prev;
                const newTurns = [...prev];
                const lastTurnIndex = newTurns.length - 1;
                const lastTurn = newTurns[lastTurnIndex];
                if (lastTurn && !lastTurn.isFinal) {
                    newTurns[lastTurnIndex] = { ...lastTurn, isFinal: true };
                }
                return newTurns;
            });
            currentInputTranscriptionRef.current = '';
            currentOutputTranscriptionRef.current = '';
        }
    };
    
    const stopRecording = useCallback(() => {
        if (!isRecording) return;
        setIsRecording(false);
        setStatus('Stopping session...');
        sessionRef.current?.close();
        sessionRef.current = null;
        streamRef.current?.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        if (scriptProcessorRef.current) {
            scriptProcessorRef.current.disconnect();
            scriptProcessorRef.current.onaudioprocess = null;
            scriptProcessorRef.current = null;
        }
        if(mediaStreamSourceRef.current) {
            mediaStreamSourceRef.current.disconnect();
            mediaStreamSourceRef.current = null;
        }
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close();
        }
        audioContextRef.current = null;
        recordingStartTimeRef.current = null;
        setStatus('Click the button to start transcribing live.');
    }, [isRecording]);

    const startRecording = useCallback(async () => {
        if (isRecording) return;
        try {
            setStatus('Initializing...');
            setTranscriptionTurns([]);
            recordingStartTimeRef.current = Date.now();
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;
            const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
            const context = new AudioContext({ sampleRate: 16000 });
            audioContextRef.current = context;
            const source = context.createMediaStreamSource(stream);
            mediaStreamSourceRef.current = source;
            const processor = context.createScriptProcessor(4096, 1, 1);
            scriptProcessorRef.current = processor;
            const sessionPromise = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                callbacks: {
                    onopen: () => { setIsRecording(true); setStatus('Recording... Speak now!'); },
                    onmessage: handleMessage,
                    onerror: (e: ErrorEvent) => { console.error('API Error:', e); setStatus(`Error: ${e.message}.`); stopRecording(); },
                    onclose: () => { if (isRecording) stopRecording(); },
                },
                config: { responseModalities: [Modality.AUDIO], inputAudioTranscription: {}, outputAudioTranscription: {} },
            });
            sessionPromise.then(session => { sessionRef.current = session; }).catch(error => { setStatus(`Connection failed: ${error.message}`); stopRecording(); });
            processor.onaudioprocess = (audioProcessingEvent) => {
                const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                const pcmBlob = createBlob(inputData);
                sessionPromise.then((session) => { session.sendRealtimeInput({ media: pcmBlob }); });
            };
            source.connect(processor);
            processor.connect(context.destination);
        } catch (error) { setStatus(`Failed to start: ${error instanceof Error ? error.message : String(error)}`); stopRecording(); }
    }, [isRecording, stopRecording]);

    const toggleRecording = () => { isRecording ? stopRecording() : startRecording(); };

    // --- File Handling ---
    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            if (file.size > MAX_FILE_SIZE_BYTES) {
                alert(`File is too large. Max ${MAX_FILE_SIZE_MB}MB.`);
                return;
            }
            if (file.type.includes('mp4') || file.name.toLowerCase().endsWith('.m4a')) {
                alert("Cảnh báo: File .m4a hoặc .mp4 (audio) có thể gây lỗi giải mã trên một số trình duyệt. Nếu gặp sự cố, vui lòng chuyển đổi file sang định dạng .mp3, .wav, hoặc .aac để đảm bảo hoạt động ổn định.");
            }
            setSelectedFile(file);
            setShowDashboard(true);
        }
    };

    const handleManualMode = () => {
        chunkProcessor.reset(); // CRITICAL: Clear any previous file data
        setSelectedFile(null); // No file
        setShowDashboard(true); // Enter dashboard
    };

    const handleDashboardReset = () => {
        chunkProcessor.reset();
        setSelectedFile(null);
        setShowDashboard(false);
    };

    const renderTranscriber = () => (
        <>
            {showDashboard ? (
                <Dashboard 
                    file={selectedFile}
                    onReset={handleDashboardReset}
                    step1Model={step1Model}
                    setStep1Model={setStep1Model}
                    step2Model={step2Model}
                    setStep2Model={setStep2Model}
                    {...chunkProcessor}
                />
            ) : (
                <div className="flex flex-col gap-8 max-w-5xl mx-auto w-full">
                    
                    {/* AI Model Config */}
                    <div className="bg-gray-900/50 border border-gray-800 p-4 rounded-2xl shadow-lg">
                        <h3 className="text-sm font-bold uppercase text-gray-400 mb-3 tracking-wider flex items-center gap-2"><BrainCircuitIcon /> Cấu hình AI Model</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {/* Step 1 Selector */}
                            <div>
                                <label htmlFor="step1-model" className="block text-sm font-medium text-gray-300">Step 1: Xử lý Chunk</label>
                                <select
                                    id="step1-model"
                                    value={step1Model}
                                    onChange={(e) => setStep1Model(e.target.value)}
                                    className="mt-1 block w-full bg-gray-900 border border-gray-600 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                                >
                                    <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                                    <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
                                    <option value="gemini-3-pro-preview">Gemini 3 Pro Preview</option>
                                </select>
                                <p className="text-xs text-gray-500 mt-1">Nhanh & rẻ (Flash), Cân bằng (2.5 Pro), Mạnh nhất (3 Pro).</p>
                            </div>
                            {/* Step 2 Selector */}
                            <div>
                                <label htmlFor="step2-model" className="block text-sm font-medium text-gray-300">Step 2: Tinh chỉnh (Post-Edit)</label>
                                <select
                                    id="step2-model"
                                    value={step2Model}
                                    onChange={(e) => setStep2Model(e.target.value)}
                                    className="mt-1 block w-full bg-gray-900 border border-gray-600 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-teal-500 focus:border-teal-500 sm:text-sm"
                                >
                                    <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                                    <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
                                    <option value="gemini-3-pro-preview">Gemini 3 Pro Preview</option>
                                </select>
                                <p className="text-xs text-gray-500 mt-1">Mạnh mẽ hơn, tạo script chuyên nghiệp, chi phí cao hơn.</p>
                            </div>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {/* Option 1: Upload File */}
                        <div className="p-8 bg-gray-900/50 border border-gray-800 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 hover:border-gray-700 group h-full">
                            <h2 className="text-2xl font-bold mb-6 text-center text-gray-200">Xử lý tài liệu (File)</h2>
                            <div className="flex flex-col items-center justify-center gap-6 h-48">
                                <label htmlFor="audio-upload" className="flex flex-col items-center justify-center w-full h-full border-2 border-dashed border-gray-700 rounded-xl cursor-pointer bg-gray-800/30 hover:bg-gray-800/60 transition-colors group-hover:border-blue-500/50">
                                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                        <UploadIcon />
                                        <p className="mb-2 text-sm text-gray-400"><span className="font-semibold text-blue-400">Click to upload</span> or drag and drop</p>
                                        <p className="text-xs text-gray-500">WAV, MP3, AAC, FLAC (Max {MAX_FILE_SIZE_MB}MB)</p>
                                    </div>
                                    <input id="audio-upload" type="file" className="hidden" accept={ACCEPTED_AUDIO_TYPES} onChange={handleFileSelect} />
                                </label>
                            </div>
                        </div>

                         {/* Option 2: Manual Post-Edit (New) */}
                         <div className="p-8 bg-gray-900/50 border border-gray-800 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 hover:border-gray-700 group h-full flex flex-col">
                            <h2 className="text-2xl font-bold mb-6 text-center text-gray-200">Tinh chỉnh Script (Manual)</h2>
                            <div className="flex flex-col items-center justify-center gap-4 flex-1">
                                <p className="text-sm text-gray-400 text-center px-4">
                                    Nhập thủ công dữ liệu JSON (ImprovedTranscript) để chạy quy trình <b>Step 2 (Post-Edit)</b> mà không cần file audio.
                                </p>
                                <button 
                                    onClick={handleManualMode}
                                    className="mt-2 px-6 py-4 bg-teal-900/30 hover:bg-teal-800/50 border border-teal-700/50 text-teal-300 rounded-xl font-bold flex items-center transition-all w-full justify-center group-hover:shadow-lg hover:border-teal-500"
                                >
                                    <EditIcon />
                                    Nhập liệu thủ công
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Divider */}
                    <div className="relative text-center">
                        <div className="absolute inset-0 flex items-center" aria-hidden="true"><div className="w-full border-t border-gray-800" /></div>
                        <div className="relative flex justify-center"><span className="bg-gray-950 px-4 text-sm font-medium text-gray-600">OR USE MICROPHONE</span></div>
                    </div>

                    {/* Option 3: Live Section */}
                    <div className="bg-gray-900/30 border border-gray-800 rounded-2xl p-6 flex flex-col items-center">
                         <h2 className="text-xl font-bold mb-4 text-gray-300">Hội thoại trực tiếp (Live)</h2>
                         <div className="w-full bg-black/40 rounded-xl p-4 mb-6 h-[200px] overflow-y-auto border border-gray-800 font-mono text-sm custom-scrollbar">
                            {transcriptionTurns.length === 0 ? (
                                <div className="flex items-center justify-center h-full text-gray-600 italic">
                                    Nội dung hội thoại sẽ hiển thị tại đây...
                                </div>
                            ) : (
                                transcriptionTurns.map((turn, index) => (
                                    <div key={index} className="mb-3">
                                        <span className="text-blue-500 font-bold mr-2">You ({turn.timestamp}):</span>
                                        <span className="text-gray-300">{turn.user}</span>
                                        {turn.model && (
                                            <div className="mt-1 ml-4 border-l-2 border-teal-800 pl-2">
                                                <span className="text-teal-500 font-bold mr-2">Gemini:</span>
                                                <span className="text-gray-400">{turn.model}</span>
                                            </div>
                                        )}
                                    </div>
                                ))
                            )}
                        </div>
                        
                        <p className="mb-4 text-sm text-gray-400 h-6">{status}</p>
                        <button
                            onClick={toggleRecording}
                            className={`
                                w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 shadow-lg
                                ${isRecording ? 'bg-red-600 hover:bg-red-500 animate-pulse' : 'bg-blue-600 hover:bg-blue-500'}
                            `}
                        >
                            {isRecording ? <StopIcon /> : <MicIcon />}
                        </button>
                    </div>
                </div>
            )}
        </>
    );

    return (
        <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col items-center p-4 sm:p-6 lg:p-8 font-sans">
            <div className="w-full max-w-7xl mx-auto flex flex-col h-full">
                <header className="text-center mb-8 flex flex-col items-center">
                     <div className="flex items-center gap-4">
                        <h1 className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-teal-300 tracking-tight">
                            Gemini Pro Suite
                        </h1>
                    </div>
                     <p className="text-gray-500 mt-2 text-sm uppercase tracking-widest font-medium">Powered by Gemini 2.5 & 3</p>

                     {/* App Mode Toggle */}
                    <div className="mt-6 flex p-1 bg-gray-800/50 border border-gray-700 rounded-lg">
                        <button
                            onClick={() => setAppMode('transcriber')}
                            className={`px-4 py-2 text-sm font-semibold rounded-md transition-colors flex items-center gap-2 ${appMode === 'transcriber' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:bg-gray-700'}`}
                        >
                           <AudioLinesIcon /> Audio Transcriber
                        </button>
                        <button
                            onClick={() => setAppMode('imageGenerator')}
                            className={`px-4 py-2 text-sm font-semibold rounded-md transition-colors flex items-center gap-2 ${appMode === 'imageGenerator' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:bg-gray-700'}`}
                        >
                            <ImageIcon /> Image Generation
                        </button>
                    </div>
                </header>
                
                {/* Main Content Area */}
                <main className="w-full">
                    {appMode === 'transcriber' ? renderTranscriber() : <ImageGenerator />}
                </main>
            </div>
        </div>
    );
};

export default App;