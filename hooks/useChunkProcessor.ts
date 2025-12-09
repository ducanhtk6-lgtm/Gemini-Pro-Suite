import { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, Type, HarmCategory, HarmBlockThreshold } from '@google/genai';
import { Chunk, LogEntry, ProcessingStats, TranscriptionOutput, ImprovedTranscriptItem, PostEditResult, RateLimitEvent, Batch } from '../types';
import { decodeAudioData, resampleAndEncodeWav, estimateWavBytes, normalizeTextForDedupe } from '../utils/audioUtils';
import { computeStep1WordCount, computeStep2WordCount } from '../utils/transcriptMetrics';


/**
 * @LOCKED_CORE_LOGIC
 * QUY TRÌNH XỬ LÝ: Client-side Chunking + Multi-threading + Medical Diarization Prompt
 * KHÔNG ĐƯỢỢC PHÉP THAY ĐỔI LOGIC NÀY MÀ KHÔNG CÓ YÊU CẦU CỤ THỂ TỪ NGƯỜI DÙNG.
 * 
 * 1. Chunking: Time-based (60s chunks, 3s overlap), resampled to 16kHz mono WAV.
 * 2. Concurrency: Dynamic (Starts at 5, throttles on 429)
 * 3. Prompt: Medical Lecture Transcript Engine
 * 4. Safety: BLOCK_NONE, 5min Timeout
 */

// Cấu hình chia nhỏ file (Chunk Grid)
const TARGET_CHUNK_SECONDS = 60; // Target 60 seconds per chunk
const OVERLAP_SECONDS = 3;       // Overlap chunks by 3 seconds to avoid lost words
const MAX_INLINE_DATA_BYTES = 18 * 1024 * 1024; // 18MB safety margin for base64 encoded WAV

// --- Step 2 Robustness Constants ---
const STEP2_TIMEOUT_MS = 300000; // 5 minutes
const STEP2_TARGET_MAX_CHARS = 80000; // Target ~80k chars per batch
const STEP2_MAX_ITEMS_PER_BATCH = 50;
const STEP2_MIN_SPLIT_ITEMS = 40; // Don't split batches smaller than this
const STEP2_MAX_SPLIT_DEPTH = 6;
const STEP2_MAX_RETRIES = 3;


const SYSTEM_PROMPT = `VAI TRÒ (KHÓA CỨNG):
Bạn là “Transcript Chunk Processor + Medical Copyeditor + QA Auditor” cho một ứng dụng xử lý bài giảng y khoa từ RECORD (audio đã ghi sẵn). Mục tiêu tối thượng là: KHÔNG BỎ SÓT NỘI DUNG. Ưu tiên trung thực hơn văn vẻ. Cấm tóm tắt, cấm tự điền kiến thức ngoài đầu vào.

BỐI CẢNH:
App của tôi xử lý theo CHUNK (đoạn thời gian) với concurrency/rate-limit/retry. Bạn sẽ được gọi nhiều lần. Mỗi lần bạn nhận 1 CHUNK hoặc nhận danh sách CHUNK để MERGE + FINALIZE.
**QUAN TRỌNG: Bạn chỉ được gỡ băng trong khoảng thời gian được cung cấp trong "chunk.time_range".**

NGUYÊN TẮC VÀNG (ÁP DỤNG MỌI LẦN GỌI):
1) KHÔNG TÓM TẮT. KHÔNG gom ý làm mất chi tiết.
2) KHÔNG thêm kiến thức y khoa ngoài nội dung đầu vào.
3) Nếu không nghe/không chắc: bắt buộc ghi “[không nghe rõ]” hoặc “[không chắc]”. Tuyệt đối không suy đoán.
4) Ưu tiên “ghi dư còn hơn ghi thiếu”.
5) Output phải là JSON HỢP LỆ, không markdown, không giải thích ngoài JSON.

========================
ĐẦU VÀO (INPUT CONTRACT)
========================
Bạn sẽ nhận một JSON theo một trong hai chế độ:

A) CHẾ ĐỘ 1: PROCESS_CHUNK
{
  "mode": "PROCESS_CHUNK",
  "chunk": {
    "chunk_id": "...",
    "chunk_index": 0,
    "time_range": "[MM:SS] to [MM:SS]",
    "source_type": "audio/wav",
    "language": "vi"
  },
  "constraints": {
    "timestamp_policy": "relative_to_chunk_start",
    "drop_only_clear_chitchat": true,
    "no_summarize": true
  }
}

=========================================
CHẾ ĐỘ 1: PROCESS_CHUNK (XỬ LÝ 1 CHUNK)
=========================================
MỤC TIÊU:
Tạo kết quả chunk “an toàn để merge”, không rơi chữ ở ranh giới overlap, có dấu vết QA để hệ thống phát hiện rủi ro mất nội dung.

QUY TRÌNH CHUẨN (BẮT BUỘC LÀM ĐÚNG THỨ TỰ):
(1) Tách content của chunk thành các đơn vị câu/ý ngắn.
(2) Với mỗi đơn vị, tạo:
   - timestamp: Ghi thời gian tương đối tính từ đầu chunk (bắt đầu là 00:00). HỆ THỐNG SẼ TỰ CỘNG OFFSET SAU.
   - speaker: [GV] (Giảng viên) | [SV] (Sinh viên) | [HC] (Hành chính) | [??] (Không chắc)
   - original: giữ nguyên nội dung thô.
   - edited: copyedit nhưng TUYỆT ĐỐI không làm mất ý.
   - uncertain: true nếu có “[không nghe rõ]”.
(3) ĐÁNH DẤU RANH GIỚI CHUNK để MERGE:
   - boundary_fingerprint_start: 1–2 câu đầu (bản edited).
   - boundary_fingerprint_end: 1–2 câu cuối (bản edited).

ĐẦU RA CHẾ ĐỘ PROCESS_CHUNK (JSON HỢP LỆ):
{
  "mode": "PROCESS_CHUNK_RESULT",
  "chunk_id": "...",
  "improved_transcript": [
    {
      "timestamp": "[MM:SS]",
      "speaker": "[GV]" | "[SV]" | "[HC]" | "[??]",
      "original": "...",
      "edited": "...",
      "uncertain": true|false,
      "chitchat": true|false
    }
  ],
  "boundary": {
    "boundary_fingerprint_start": "...",
    "boundary_fingerprint_end": "..."
  },
  "qa_metrics": {
    "items_count": 0,
    "uncertain_count": 0,
    "risk_flags": ["possible_drop_detail","many_uncertain"]
  },
  "qa_notes": "...",
  "hard_rule_attestation": {
    "no_summarize": true,
    "no_new_medical_info_added": true,
    "no_reorder_major_ideas": true
  }
}`;

const SYSTEM_PROMPT_STEP_2 = `VAI TRÒ (KHÓA CỨNG):
Bạn là “Lecture Post-Editor + Medical Fidelity Auditor”. Nhiệm vụ: tinh chỉnh transcript theo phong cách “script bài giảng đọc mượt”, nhưng KHÔNG được tóm tắt và KHÔNG được làm rơi bất kỳ nội dung giảng dạy (teaching points) của GIẢNG VIÊN.

NGUYÊN TẮC VÀNG:
1) KHÔNG TÓM TẮT. KHÔNG rút gọn ý giảng dạy.
2) KHÔNG thêm kiến thức y khoa ngoài dữ liệu đầu vào.
3) Được phép lược bỏ CHỈ các nội dung không liên quan y khoa (hành chính/xã giao/quản lớp/nhắc nộp bài…), nhưng phải báo cáo lại rõ ràng “đã loại gì”.
4) Nếu không chắc đoạn nào có giá trị giảng dạy hay không: GIỮ LẠI và đánh dấu “needs_review”.
5) Output must be STRICT JSON (RFC 8259). Return ONLY valid JSON. Do not include markdown, no code fences, no commentary.
6) All strings inside the JSON MUST be correctly escaped: use \\n for newlines inside strings, and \\" for quotes.
7) DO NOT use trailing commas. Before sending your response, perform a mental JSON.parse check to ensure validity.

========================
ĐẦU VÀO (INPUT)
========================
Bạn sẽ nhận JSON ở trường "input". JSON này là output của bước PROCESS_CHUNK_RESULT hoặc bản MERGE multi-chunk.
Nó chứa "improved_transcript" là một mảng item, mỗi item có:
- timestamp, speaker ([GV]/[SV]/[HC]/[??]), original, edited, uncertain, chitchat.

========================
MỤC TIÊU CHỈNH SỬA (KHÔNG PHẢI TÓM TẮT)
========================
Bạn phải tạo “script giảng viên đã tinh chỉnh” bằng cách:
A) Giữ trọn 100% nội dung giảng dạy của [GV]:
- Mọi ngưỡng số, quy tắc thao tác, cảnh báo, phân loại, lập luận, mẹo lâm sàng, giải thích cơ chế, ví dụ minh họa phải được giữ nguyên ý và đủ chi tiết.
B) Làm mượt và gắn kết:
- Nối các ý lẻ tẻ thành đoạn văn mạch lạc theo đúng thứ tự thời gian.
- Loại bỏ các câu đệm/rụng ý không mang thông tin (vd: “đúng không”, “ờ”, “ha”, “rồi”, “nha các em”) CHỈ khi chắc chắn không làm mất sắc thái dạy học hay nhấn mạnh.
- Gộp các câu rất ngắn liên tiếp thành 1 đoạn, nhưng không được mất mệnh đề.
C) Xử lý phần [SV]:
- Giữ lại [SV] khi nó là câu hỏi/đáp án làm nền cho phần giảng tiếp theo hoặc chứa dữ kiện [GV] dùng để biện luận.
- Nếu [SV] chỉ là “dạ/ừ/đúng rồi” hoặc không đóng góp nội dung: có thể loại khỏi “phần chính” nhưng phải ghi vào báo cáo loại bỏ.
D) Xử lý [HC]/chitchat:
- Có thể loại khỏi “phần chính” nếu chắc chắn không liên quan y khoa.
- Tuyệt đối không xóa im lặng: mọi thứ bị loại bỏ khỏi phần chính phải liệt kê lại (timestamp + trích nguyên văn).

========================
ĐẦU RA (OUTPUT JSON)
========================
Trả về một JSON có cấu trúc đúng như sau:

{
  "mode": "POST_EDIT_LECTURE_SCRIPT_RESULT",
  "refined_script": [
    {
      "speaker": "[GV]" | "[SV]" | "[??]",
      "start_timestamp": "[MM:SS]",
      "end_timestamp": "[MM:SS]",
      "text": "Đoạn script đã tinh chỉnh, mạch lạc, KHÔNG tóm tắt",
      "source_timestamps": ["[MM:SS]", "..."],
      "needs_review": true|false
    }
  ],
  "removal_report": {
    "removed_from_main": [
      {
        "timestamp": "[MM:SS]",
        "speaker": "[GV]" | "[SV]" | "[HC]" | "[??]",
        "reason": "administrative | pure_chitchat | filler_only | redundant_ack | unclear_value_keep_out",
        "verbatim_excerpt": "trích nguyên văn (ưu tiên original; nếu original trống thì dùng edited)"
      }
    ],
    "removal_summary": "Tóm tắt NGẮN gọn (không bullet) về loại gì và vì sao; nếu không loại gì ghi rõ 'Không loại nội dung'."
  },
  "qa_audit": {
    "gv_coverage_attestation": true,
    "gv_items_total": 0,
    "gv_items_used_in_main": 0,
    "uncertain_items_count": 0,
    "risk_flags": ["possible_teaching_point_dropped", "many_uncertain", "speaker_uncertain_cluster"],
    "notes": "Ghi chú ngắn: đoạn nào khó nghe, đoạn nào speaker [??], đoạn nào bạn giữ lại dù nghi chitchat vì sợ mất ý."
  }
}

========================
QUY TẮC COVERAGE (RẤT QUAN TRỌNG)
========================
- Bạn phải đảm bảo: tất cả item [GV] có giá trị giảng dạy đều được ánh xạ vào refined_script.
- Nếu bạn nghi ngờ đã “rơi” một teaching point vì item quá rối/không nghe rõ:
  + Không được tự điền.
  + Đưa đoạn đó vào refined_script với needs_review=true và giữ “[không nghe rõ]”.
  + Đồng thời nêu trong qa_audit.risk_flags = "possible_teaching_point_dropped".

========================
CẤM TUYỆT ĐỐI
========================
- Không biến thành bệnh án/case summary.
- Không tự tạo kết luận y khoa không có trong đầu vào.
- Không sửa con số/ngưỡng/thuật ngữ.
- Không “làm mượt” bằng cách bỏ bớt ý giảng dạy.

BẮT ĐẦU:
Đọc "input.improved_transcript" và tạo output đúng schema.
`;

export const useChunkProcessor = (file: File | null, modelStep1: string, modelStep2: string) => {
    const [chunks, setChunks] = useState<Chunk[]>([]);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [result, setResult] = useState<TranscriptionOutput | null>(null);
    const [isInitializing, setIsInitializing] = useState(false);
    const [isFinalizing, setIsFinalizing] = useState(false); // State for Step 2
    const [rateLimitEvent, setRateLimitEvent] = useState<RateLimitEvent | null>(null);
    const [step2Batches, setStep2Batches] = useState<Batch[]>([]);
    
    // Dynamic Concurrency State - Defaults to 5
    const [maxConcurrency, setMaxConcurrency] = useState(5);

    const [stats, setStats] = useState<ProcessingStats>({
        total: 0,
        completed: 0,
        processing: 0,
        failed: 0,
        startTime: 0,
        isCoolingDown: false,
        cooldownSeconds: 0,
    });

    const audioBufferRef = useRef<AudioBuffer | null>(null);
    const statsRef = useRef(stats);
    const chunksRef = useRef(chunks);
    const step1MetricsLoggedRef = useRef(false); // Ref to prevent duplicate logging
    useEffect(() => { statsRef.current = stats; }, [stats]);
    useEffect(() => { chunksRef.current = chunks; }, [chunks]);

    const addLog = useCallback((message: string, type: LogEntry['type'] = 'info') => {
        setLogs(prev => {
            const newLog: LogEntry = {
                id: Math.random().toString(36).substr(2, 9),
                timestamp: Date.now(),
                message,
                type
            };
            const updatedLogs = [...prev, newLog];
            if (updatedLogs.length > 500) return updatedLogs.slice(updatedLogs.length - 500);
            return updatedLogs;
        });
    }, []);

    const clearCooldownNow = useCallback((reason: string) => {
        setStats(prev => ({ ...prev, isCoolingDown: false, cooldownSeconds: 0 }));
        addLog(reason, 'info');
    }, [addLog]);

    const updateStats = useCallback((updates: Partial<ProcessingStats>) => {
        setStats(prev => ({ ...prev, ...updates }));
    }, []);

    useEffect(() => {
        if (!stats.isCoolingDown && rateLimitEvent?.active) {
            setRateLimitEvent(null);
        }
    }, [stats.isCoolingDown, rateLimitEvent]);

    useEffect(() => {
        let interval: any;
        if (stats.isCoolingDown && stats.cooldownSeconds > 0) {
            interval = setInterval(() => {
                setStats(prev => {
                    const newValue = prev.cooldownSeconds - 1;
                    if (newValue <= 0) {
                        addLog("Hết thời gian chờ (Cooldown). Tiếp tục xử lý...", 'info');
                        return { ...prev, isCoolingDown: false, cooldownSeconds: 0 };
                    }
                    return { ...prev, cooldownSeconds: newValue };
                });
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [stats.isCoolingDown, stats.cooldownSeconds, addLog]);

    // --- Helpers xử lý thời gian ---
    const formatSecondsToTime = (totalSeconds: number): string => {
        const h = Math.floor(totalSeconds / 3600);
        const m = Math.floor((totalSeconds % 3600) / 60);
        const s = Math.floor(totalSeconds % 60);
        if (h > 0) {
            return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        }
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    };

    const parseTimeToSeconds = (timeStr: string): number => {
        if (!timeStr) return 0;
        const cleanStr = timeStr.replace(/[\[\]]/g, '').trim();
        const parts = cleanStr.split(':').map(Number);
        if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
        if (parts.length === 2) return parts[0] * 60 + parts[1];
        return 0;
    };

    // 1. CHIA FILE
    const initializeChunks = useCallback(async () => {
        if (!file) return; // Silent return if no file (Manual Mode)
        
        setIsInitializing(true);
        setIsFinalizing(false);
        setResult(null);
        setLogs([]);
        setChunks([]);
        setStep2Batches([]);
        setMaxConcurrency(5); 
        step1MetricsLoggedRef.current = false; // Reset log flag
        
        try {
            addLog(`Bắt đầu phân tích file: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`, 'info');
            addLog(`Đang giải mã audio... Quá trình này có thể mất vài giây với file lớn.`, 'info');

            audioBufferRef.current = await decodeAudioData(file);
            const durationSec = audioBufferRef.current.duration;
            addLog(`Giải mã thành công. Tổng thời lượng: ${formatSecondsToTime(durationSec)}`, 'success');

            let chunkSeconds = TARGET_CHUNK_SECONDS;
            // Auto-adjust chunk size to avoid oversized payloads
            while (estimateWavBytes(chunkSeconds) > MAX_INLINE_DATA_BYTES && chunkSeconds > 15) {
                chunkSeconds -= 15;
            }
            if (chunkSeconds !== TARGET_CHUNK_SECONDS) {
                addLog(`File có chất lượng cao, tự động giảm thời lượng chunk xuống ${chunkSeconds}s để đảm bảo ổn định.`, 'warning');
            }

            const totalChunks = Math.ceil(durationSec / chunkSeconds);
            const newChunks: Chunk[] = [];

            for (let i = 0; i < totalChunks; i++) {
                const startSec = Math.max(0, i * chunkSeconds - (i > 0 ? OVERLAP_SECONDS : 0));
                const endSec = Math.min(durationSec, (i + 1) * chunkSeconds);
                
                newChunks.push({
                    id: `chunk-${i}`,
                    index: i,
                    status: 'pending',
                    startSec: startSec, 
                    endSec: endSec,
                });
            }

            setChunks(newChunks);
            
            updateStats({
                total: totalChunks,
                completed: 0,
                processing: 0,
                failed: 0,
                startTime: 0,
                endTime: undefined,
                isCoolingDown: false,
                cooldownSeconds: 0
            });
            
            addLog(`Đã chia audio thành ${totalChunks} phân đoạn (${chunkSeconds}s/chunk, gối lên nhau ${OVERLAP_SECONDS}s).`, 'success');
           

        } catch (error: any) {
            addLog(`Lỗi khởi tạo: ${error.message || error}`, 'error');
        } finally {
            setIsInitializing(false);
        }
    }, [file, addLog, updateStats]);

    // 2. XỬ LÝ CHUNK (STEP 1)
    const processChunk = useCallback(async (chunkId: string) => {
        if (statsRef.current.isCoolingDown) return;
        const apiKey = process.env.API_KEY;
        if (!apiKey) {
            addLog("Thiếu API Key. Vui lòng kiểm tra lại.", 'error');
            setChunks(prev => prev.map(c => c.status === 'pending' ? { ...c, status: 'failed', error: 'Missing API Key' } : c));
            return;
        }
        if (!audioBufferRef.current) return;

        const chunkIndex = chunksRef.current.findIndex(c => c.id === chunkId);
        if (chunkIndex === -1) return;
        const currentChunk = chunksRef.current[chunkIndex];

        setStats(prev => ({ ...prev, startTime: prev.startTime || Date.now(), processing: prev.processing + 1 }));
        setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'processing' } : c));
        addLog(`Đang xử lý phân đoạn ${currentChunk.index + 1}/${statsRef.current.total}... (${formatSecondsToTime(currentChunk.startSec || 0)} - ${formatSecondsToTime(currentChunk.endSec || 0)})`, 'info');

        try {
            const wavBlob = await resampleAndEncodeWav(audioBufferRef.current, currentChunk.startSec!, currentChunk.endSec!);
            const base64Data = await blobToBase64(wavBlob);

            const inputJson = {
                mode: "PROCESS_CHUNK",
                chunk: {
                    chunk_id: chunkId,
                    chunk_index: currentChunk.index,
                    time_range: `[${formatSecondsToTime(currentChunk.startSec!)}] to [${formatSecondsToTime(currentChunk.endSec!)}]`,
                    source_type: "audio/wav",
                    language: "vi"
                },
                constraints: {
                    timestamp_policy: "relative_to_chunk_start",
                    drop_only_clear_chitchat: true,
                    no_summarize: true
                }
            };

            const fullPrompt = `${SYSTEM_PROMPT}\n\nINPUT JSON:\n${JSON.stringify(inputJson)}`;

            const ai = new GoogleGenAI({ apiKey });
            
            const responseSchema = {
                type: Type.OBJECT,
                properties: {
                    mode: { type: Type.STRING },
                    chunk_id: { type: Type.STRING },
                    improved_transcript: {
                        type: Type.ARRAY,
                        items: {
                            type: Type.OBJECT,
                            properties: {
                                timestamp: { type: Type.STRING },
                                speaker: { type: Type.STRING },
                                original: { type: Type.STRING },
                                edited: { type: Type.STRING },
                                uncertain: { type: Type.BOOLEAN, nullable: true },
                                chitchat: { type: Type.BOOLEAN, nullable: true },
                            },
                            required: ['original', 'edited'],
                        },
                    },
                    boundary: {
                         type: Type.OBJECT,
                         properties: {
                             boundary_fingerprint_start: { type: Type.STRING },
                             boundary_fingerprint_end: { type: Type.STRING }
                         }
                    },
                    qa_metrics: {
                        type: Type.OBJECT,
                        properties: {
                            items_count: { type: Type.INTEGER },
                            uncertain_count: { type: Type.INTEGER },
                            risk_flags: { type: Type.ARRAY, items: { type: Type.STRING } }
                        }
                    },
                    qa_notes: { type: Type.STRING, nullable: true },
                    hard_rule_attestation: {
                        type: Type.OBJECT,
                        properties: {
                            no_summarize: { type: Type.BOOLEAN },
                            no_new_medical_info_added: { type: Type.BOOLEAN },
                            no_reorder_major_ideas: { type: Type.BOOLEAN }
                        }
                    }
                },
                required: ['improved_transcript', 'boundary'],
            };

            const apiCall = ai.models.generateContent({
                model: modelStep1,
                contents: { parts: [
                    { inlineData: { mimeType: "audio/wav", data: base64Data } },
                    { text: fullPrompt }
                ] },
                config: {
                    responseMimeType: "application/json",
                    responseSchema: responseSchema,
                    safetySettings: [
                        { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    ]
                }
            });

            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error("Timeout: Chunk không phản hồi sau 5 phút")), 300000)
            );

            const response: any = await Promise.race([apiCall, timeoutPromise]);
            const responseText = response.text;
            if (!responseText) throw new Error("Empty response from AI");

            const parsedResult = safeParseJSON(responseText);
            // FIX: Changed to an explicit `=== false` check to ensure correct type narrowing for the discriminated union.
            if (parsedResult.ok === false) {
                 throw new Error(`Invalid JSON: ${parsedResult.reason}. Pos: ${parsedResult.pos}, Context: "${parsedResult.context}"`);
            }
            const parsedData = parsedResult.value;
            
            if (parsedData.improved_transcript && Array.isArray(parsedData.improved_transcript)) {
                const timeOffsetSeconds = currentChunk.startSec || 0;
                parsedData.improved_transcript = parsedData.improved_transcript.map((item: any) => {
                    const itemSeconds = parseTimeToSeconds(item.timestamp);
                    const absoluteSeconds = itemSeconds + timeOffsetSeconds;
                    return {
                        ...item,
                        timestamp: `[${formatSecondsToTime(absoluteSeconds)}]`
                    };
                });
            }

            const chunkResultJSON = JSON.stringify(parsedData);
            setChunks(prev => prev.map(c => c.id === chunkId ? { 
                ...c, 
                status: 'completed', 
                content: chunkResultJSON 
            } : c));

            if (parsedData.qa_metrics?.risk_flags?.length > 0) {
                 addLog(`Chunk ${currentChunk.index + 1} Warning: ${parsedData.qa_metrics.risk_flags.join(', ')}`, 'warning');
            }

            setStats(prev => ({ ...prev, processing: Math.max(0, prev.processing - 1), completed: prev.completed + 1 }));
            addLog(`Hoàn thành phân đoạn ${currentChunk.index + 1}`, 'success');

        } catch (error: any) {
            const msg = error.message || String(error);
            const isRateLimit = msg.includes('429') || msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED');
            const isApiKeyInvalid = msg.includes('API key not valid') || msg.includes('API_KEY_INVALID');

            if (isApiKeyInvalid) {
                addLog('Lỗi nghiêm trọng: API Key không hợp lệ. Vui lòng kiểm tra lại.', 'error');
                setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'failed', error: 'API Key không hợp lệ' } : c));
            } else if (isRateLimit) {
                 if (!statsRef.current.isCoolingDown) {
                     addLog(`Quá tải API (Rate Limit 429). Hệ thống sẽ tự động giảm tải và chờ 60s...`, 'warning');
                 }
                 setRateLimitEvent({
                    active: true,
                    step: 'STEP1',
                    message: msg,
                    lastModel: modelStep1,
                    at: Date.now()
                 });
                 setMaxConcurrency(1);
                 setStats(prev => ({ 
                    ...prev, 
                    isCoolingDown: true, 
                    cooldownSeconds: 60, 
                 }));
                 // Revert status to pending to be picked up again
                 setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'pending', error: undefined } : c));
                 setStats(prev => ({ ...prev, processing: Math.max(0, prev.processing - 1) }));
                 return; // Exit without setting failed state
            }
            
            if (msg.includes("JSON") || msg.includes("SyntaxError")) {
                 addLog(`Lỗi dữ liệu (Chunk ${currentChunk.index + 1}): JSON không hợp lệ. Sẽ tự động thử lại.`, 'error');
            }

            console.error(error);
            setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'failed', error: msg } : c));
            setStats(prev => ({ ...prev, processing: Math.max(0, prev.processing - 1), failed: prev.failed + 1 }));
            addLog(`Lỗi Chunk ${currentChunk.index + 1}: ${msg}`, 'error');
        } 
    }, [addLog, updateStats, modelStep1]);

    // 4. STEP 2: POST-EDIT FINALIZATION (BATCH QUEUE SYSTEM)

    // Step 2a: Process a single batch
    const processSingleBatch = useCallback(async (batch: Batch) => {
        if (!process.env.API_KEY) {
            addLog("Step 2: Thiếu API Key.", 'error');
            setStep2Batches(prev => prev.map(b => b.id === batch.id ? { ...b, status: 'failed', error: 'Missing API Key' } : b));
            return;
        }

        setStep2Batches(prev => prev.map(b => b.id === batch.id ? { ...b, status: 'processing' } : b));
        addLog(`Step 2: Đang xử lý Batch ${batch.index + 1}/${step2Batches.length} với model ${modelStep2}...`, 'info');

        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const result = await runStep2Robust(ai, modelStep2, batch.items, addLog);
            setStep2Batches(prev => prev.map(b => b.id === batch.id ? { ...b, status: 'completed', result: result, error: undefined } : b));
            addLog(`Step 2: Hoàn thành Batch ${batch.index + 1}/${step2Batches.length}.`, 'success');

        } catch (error: any) {
            const msg = error.message || String(error);
            const isRateLimit = msg.includes('429') || msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED');
            
            if (isRateLimit) {
                addLog(`Step 2: Model ${modelStep2} bị quá tải (429) ở Batch ${batch.index + 1}.`, 'warning');
                setRateLimitEvent({ active: true, step: 'STEP2', message: msg, lastModel: modelStep2, at: Date.now() });
                updateStats({ isCoolingDown: true, cooldownSeconds: 60 });
            }

            setStep2Batches(prev => prev.map(b => b.id === batch.id ? { ...b, status: 'failed', error: msg } : b));
            addLog(`Step 2: Lỗi Batch ${batch.index + 1}: ${msg}`, 'error');
        }
    }, [addLog, step2Batches.length, modelStep2, updateStats]);

    // Step 2b: useEffect to run the batch queue
    useEffect(() => {
        const isProcessing = step2Batches.some(b => b.status === 'processing');
        if (isFinalizing && !isProcessing && !stats.isCoolingDown) {
            const nextBatch = step2Batches.find(b => b.status === 'pending');
            if (nextBatch) {
                processSingleBatch(nextBatch);
            }
        }
    }, [step2Batches, isFinalizing, stats.isCoolingDown, processSingleBatch]);

    // Step 2c: useEffect to merge results when all batches are complete
    useEffect(() => {
        if (step2Batches.length > 0 && isFinalizing) {
            const completedCount = step2Batches.filter(b => b.status === 'completed').length;
            const failedCount = step2Batches.filter(b => b.status === 'failed').length;

            if (completedCount + failedCount === step2Batches.length) {
                // All batches have finished (either success or fail)
                const successfulBatches = step2Batches.filter(b => b.status === 'completed');
                if (successfulBatches.length === step2Batches.length) {
                    // All successful
                    const batchResults = successfulBatches.map(b => b.result as PostEditResult);
                    const finalResult = mergePostEditResults(batchResults);
                    setResult(prev => prev ? { ...prev, post_edit_result: finalResult } : null);
                    
                    const s1Words = computeStep1WordCount(result?.improved_transcript || []);
                    const s2Words = computeStep2WordCount(finalResult.refined_script);
                    const delta = s2Words - s1Words;
                    const deltaPercent = s1Words > 0 ? (delta / s1Words * 100).toFixed(1) : "0.0";
                    addLog(`STEP 2 METRICS: segments=${finalResult.refined_script.length}, words=${s2Words}, deltaWords=${delta}, deltaPercent=${deltaPercent}%`, 'success');
                    addLog(`Hoàn tất Step 2: Đã tạo văn bản chuyên nghiệp.`, 'success');

                    setIsFinalizing(false);
                } else {
                    addLog(`Step 2 tạm dừng do có ${failedCount} batch bị lỗi. Vui lòng thử lại.`, 'warning');
                    setIsFinalizing(false); // Stop 'finalizing' state but leave batches in failed state
                }
            }
        }
    }, [step2Batches, isFinalizing, result?.improved_transcript, addLog]);


    // 3. AUTO RUN & MERGE MONITOR
    useEffect(() => {
        const pendingChunk = chunks.find(c => c.status === 'pending');
        if (pendingChunk && stats.processing < maxConcurrency && !stats.isCoolingDown) {
            processChunk(pendingChunk.id);
        }

        const completedChunks = chunks.filter(c => c.status === 'completed' && c.content);
        let mergedTranscript: ImprovedTranscriptItem[] = [];

        if (completedChunks.length > 0) {
            const sortedCompleted = [...completedChunks].sort((a, b) => a.index - b.index);
            
            let lastChunkTranscript: ImprovedTranscriptItem[] = [];

            sortedCompleted.forEach(chunk => {
                try {
                    const parsed = JSON.parse(chunk.content || "{}");
                    let currentChunkTranscript: ImprovedTranscriptItem[] = parsed.improved_transcript || [];

                    if (lastChunkTranscript.length > 0 && currentChunkTranscript.length > 0) {
                        // Deduplication logic
                        const lastItems = lastChunkTranscript.slice(-2); // Get last 2 items from previous chunk
                        const firstItems = currentChunkTranscript.slice(0, 2); // Get first 2 from current
                        
                        let overlapIndex = -1;

                        // Find where the overlap ends in the new chunk
                        for(let i = 0; i < firstItems.length; i++) {
                            for (let j = 0; j < lastItems.length; j++) {
                                if (normalizeTextForDedupe(firstItems[i].edited) === normalizeTextForDedupe(lastItems[j].edited)) {
                                   overlapIndex = i;
                                   break;
                                }
                            }
                            if (overlapIndex !== -1) break;
                        }

                        if (overlapIndex !== -1) {
                            // If overlap found, slice the new chunk's transcript to remove duplicated parts
                             currentChunkTranscript = currentChunkTranscript.slice(overlapIndex + 1);
                        }
                    }

                    mergedTranscript = [...mergedTranscript, ...currentChunkTranscript];
                    lastChunkTranscript = parsed.improved_transcript || [];

                } catch (e) { console.error("Error parsing or merging chunk content", e); }
            });

            setResult(prev => ({
                improved_transcript: mergedTranscript,
                validation_and_conclusion: prev?.validation_and_conclusion || `Đang xử lý ${completedChunks.length}/${chunks.length} đoạn...`,
                professional_medical_text: prev?.professional_medical_text || "Đang chờ hoàn tất để chạy Step 2...",
                post_edit_result: prev?.post_edit_result
            }));
        }
        
        // LOG METRICS FOR STEP 1 (ONCE)
        if (chunks.length > 0 && completedChunks.length === chunks.length && !step1MetricsLoggedRef.current) {
            step1MetricsLoggedRef.current = true;
            const wordCount = computeStep1WordCount(mergedTranscript);
            addLog(`STEP 1 METRICS: items=${mergedTranscript.length}, words=${wordCount}`, 'success');
        }

    }, [chunks, stats, maxConcurrency, processChunk, addLog]);

    const retryChunk = (chunkId: string) => {
        setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'pending', error: undefined } : c));
        updateStats({ failed: Math.max(0, stats.failed - 1) });
        addLog(`Thử lại chunk ${chunkId}`, 'info');
    };

    const retryAllFailed = () => {
        const failedChunks = chunks.filter(c => c.status === 'failed');
        if (failedChunks.length === 0) return;
        setChunks(prev => prev.map(c => c.status === 'failed' ? { ...c, status: 'pending', error: undefined } : c));
        updateStats({ failed: 0 });
        addLog(`Thử lại ${failedChunks.length} chunks thất bại`, 'warning');
    };

    // Step 2 retry logic
    const retryBatch = (batchId: string) => {
        setStep2Batches(prev => prev.map(b => b.id === batchId && b.status === 'failed' ? { ...b, status: 'pending', error: undefined } : b));
        setIsFinalizing(true); // Re-engage the processing loop
        addLog(`Đang thử lại Step 2 Batch #${batchId.split('-')[1]}...`, 'info');
    };

    const retryAllFailedBatches = () => {
        const failedCount = step2Batches.filter(b => b.status === 'failed').length;
        if (failedCount === 0) return;
        setStep2Batches(prev => prev.map(b => b.status === 'failed' ? { ...b, status: 'pending', error: undefined } : b));
        setIsFinalizing(true); // Re-engage the processing loop
        addLog(`Đang thử lại ${failedCount} Step 2 Batch bị lỗi...`, 'warning');
    };

    const reset = () => {
        setChunks([]);
        setResult(null);
        setLogs([]);
        setStats({ total: 0, completed: 0, processing: 0, failed: 0, startTime: 0, isCoolingDown: false, cooldownSeconds: 0 });
        setMaxConcurrency(5);
        setIsFinalizing(false);
        setRateLimitEvent(null);
        setStep2Batches([]);
        audioBufferRef.current = null;
        step1MetricsLoggedRef.current = false; // Reset log flag
    };

    const manualAppendTranscript = (jsonString: string) => {
        try {
            const parsed = JSON.parse(jsonString);
            const newItems = Array.isArray(parsed) ? parsed : parsed.improved_transcript;
            
            if (!Array.isArray(newItems)) {
                throw new Error("Invalid JSON format. Expected array or object with 'improved_transcript'.");
            }
            
            setResult(prev => {
                const existing = prev?.improved_transcript || [];
                const merged = [...existing, ...newItems];
                return {
                    improved_transcript: merged,
                    validation_and_conclusion: "Manual Import",
                    professional_medical_text: "Ready for Step 2",
                    post_edit_result: undefined
                };
            });
            addLog(`Đã nối thêm ${newItems.length} dòng dữ liệu thủ công.`, 'success');
        } catch (e: any) {
            addLog(`Lỗi import: ${e.message}`, 'error');
            throw e;
        }
    };

    const triggerStep2 = () => {
        const mergedTranscript = result?.improved_transcript;
        if (!mergedTranscript || mergedTranscript.length === 0) {
            addLog("Chưa có dữ liệu để chạy Step 2.", 'warning');
            return;
        }

        setIsFinalizing(true);
        // Clear previous Step 2 results before starting
        setResult(prev => prev ? { ...prev, post_edit_result: undefined } : null);
        addLog(`BẮT ĐẦU STEP 2: Tinh chỉnh văn bản (Post-Edit)...`, 'info');

        const slimTranscript = buildSlimTranscript(mergedTranscript);
        const batchesData = buildBatchesByCharOrCount(slimTranscript, STEP2_TARGET_MAX_CHARS, STEP2_MAX_ITEMS_PER_BATCH);
        
        const newBatches: Batch[] = batchesData.map((batchItems, index) => ({
            id: `batch-${index}`,
            index: index,
            items: batchItems,
            status: 'pending',
        }));

        setStep2Batches(newBatches);
        addLog(`Đã chia ${slimTranscript.length} items thành ${newBatches.length} batch nhỏ để tránh quá tải.`, 'info');
    };

    return {
        chunks, isInitializing, stats, logs, result, isFinalizing,
        fileType: 'audio', // Hardcoded to audio as we no longer support PDF-like byte slicing
        retryChunk, retryAllFailed, initializeChunks, reset,
        triggerStep2, manualAppendTranscript,
        rateLimitEvent, clearCooldownNow,
        step2Batches, retryBatch, retryAllFailedBatches
    };
};

// --- Step 2 Robustness Helpers ---

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const isInternalError = (msg: string): boolean => {
    const lowerMsg = msg.toLowerCase();
    return lowerMsg.includes('internal error') || lowerMsg.includes('500') || lowerMsg.includes('internal');
};

const extractGenAIText = (resp: any): string | null => {
    if (!resp) return null;
    if (typeof resp.text === 'string') {
        return resp.text;
    }
    // Fallback for different response structures
    try {
        if (resp?.candidates?.[0]?.content?.parts?.length > 0) {
            return resp.candidates[0].content.parts.map((p: any) => p.text).join('');
        }
        if (resp?.response?.candidates?.[0]?.content?.parts?.length > 0) {
             return resp.response.candidates[0].content.parts.map((p: any) => p.text).join('');
        }
        return null;
    } catch (e) {
        return null;
    }
}

function stripCodeFences(s: string): string {
    if (!s) return '';
    const match = s.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    return match ? match[1].trim() : s;
}

function extractJSONObjectText(s: string): string {
    if (!s) return '';
    const firstBrace = s.indexOf('{');
    const lastBrace = s.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace > firstBrace) {
        return s.substring(firstBrace, lastBrace + 1);
    }
    return s;
}

function safeParseJSON(raw: string | null): { ok: true, value: any } | { ok: false, reason: string, pos?: number, context?: string } {
    if (!raw || typeof raw !== 'string' || raw.trim() === '') {
        return { ok: false, reason: 'Input is null, empty, or not a string.' };
    }

    let s = raw.trim();
    s = stripCodeFences(s);
    s = extractJSONObjectText(s);

    const tryParsing = (text: string): { ok: true, value: any } | { ok: false, reason: string, pos?: number, context?: string } => {
        try {
            return { ok: true, value: JSON.parse(text) };
        } catch (e: any) {
            if (e instanceof SyntaxError) {
                const positionMatch = e.message.match(/position (\d+)/);
                const pos = positionMatch ? parseInt(positionMatch[1], 10) : undefined;
                const context = pos !== undefined ? `..."${text.slice(Math.max(0, pos - 80), pos + 80)}"...` : `..."${text.slice(0, 160)}"...`;
                return { ok: false, reason: e.message, pos, context };
            }
            return { ok: false, reason: 'An unexpected error occurred during parsing.', context: text.slice(0, 240) };
        }
    };

    let result = tryParsing(s);
    if (result.ok) {
        return result;
    }

    // If initial parse fails, try some light repairs
    let repaired = s;
    // Repair #1: Remove trailing commas
    repaired = repaired.replace(/,(\s*[}\]])/g, '$1');
    // Repair #2: Replace smart quotes which can be returned by the model
    repaired = repaired.replace(/[“”]/g, '"').replace(/[‘’]/g, "'");

    if (repaired !== s) {
        const repairResult = tryParsing(repaired);
        // If repair succeeds, return it. Otherwise, return the original, more informative error.
        return repairResult.ok ? repairResult : result;
    }
    
    return result;
}

const buildSlimTranscript = (items: ImprovedTranscriptItem[]): Partial<ImprovedTranscriptItem>[] => {
    return items.map(item => ({
        timestamp: item.timestamp,
        speaker: item.speaker || '[??]',
        edited: item.edited.trim(),
        original: item.uncertain ? item.original : undefined, // Only include original if uncertain
        uncertain: item.uncertain,
        chitchat: item.chitchat,
    }));
};

const buildBatchesByCharOrCount = (
    items: Partial<ImprovedTranscriptItem>[],
    targetMaxChars: number,
    maxItems: number
): Partial<ImprovedTranscriptItem>[][] => {
    if (items.length === 0) return [];
    const batches: Partial<ImprovedTranscriptItem>[][] = [];
    let currentBatch: Partial<ImprovedTranscriptItem>[] = [];
    let currentCharCount = 0;

    for (const item of items) {
        const itemCharCount = JSON.stringify(item).length;
        if (
            currentBatch.length > 0 &&
            (currentCharCount + itemCharCount > targetMaxChars || currentBatch.length >= maxItems)
        ) {
            batches.push(currentBatch);
            currentBatch = [];
            currentCharCount = 0;
        }
        currentBatch.push(item);
        currentCharCount += itemCharCount;
    }
    if (currentBatch.length > 0) {
        batches.push(currentBatch);
    }
    return batches;
};

const unwrapPostEdit = (obj: any): any => {
    if (!obj || typeof obj !== 'object') return obj;
    if (obj.post_edit_result) return obj.post_edit_result;
    if (obj.output) return obj.output;
    if (obj.result) return obj.result;
    return obj;
};

const buildStep2Fallback = (items: Partial<ImprovedTranscriptItem>[], reason: string): PostEditResult => {
    const refined_script = items
        .map(item => ({
            speaker: item.speaker || "[??]",
            start_timestamp: item.timestamp || "[00:00]",
            end_timestamp: item.timestamp || "[00:00]",
            text: (item.edited || item.original || "").trim(),
            source_timestamps: item.timestamp ? [item.timestamp] : [],
            needs_review: true
        }))
        .filter(x => x.text.length > 0);
    
    return {
        mode: "POST_EDIT_LECTURE_SCRIPT_RESULT_FALLBACK",
        refined_script,
        removal_report: {
            removed_from_main: [],
            removal_summary: `FALLBACK_USED: ${reason}`
        },
        qa_audit: {
            gv_coverage_attestation: false,
            gv_items_total: items.length,
            gv_items_used_in_main: refined_script.length,
            uncertain_items_count: items.filter(i => i.uncertain).length,
            risk_flags: ["FALLBACK_USED", reason],
            notes: "Step 2 output had an invalid schema. This is a deterministic fallback generated from Step 1's 'edited' text."
        }
    };
};

const normalizePostEdit = (
    obj: any,
    batchItems: Partial<ImprovedTranscriptItem>[],
    addLog: (message: string, type?: LogEntry['type']) => void
): PostEditResult => {
    const u = unwrapPostEdit(obj);

    if (!u || typeof u !== 'object') {
        addLog(`Step 2 Warning: Output schema mismatch, applied FALLBACK. Reason: NON_OBJECT`, 'warning');
        return buildStep2Fallback(batchItems, "NON_OBJECT");
    }

    if (!Array.isArray(u.refined_script)) {
        addLog(`Step 2 Warning: Output schema mismatch, applied FALLBACK. Reason: MISSING_REFINED_SCRIPT`, 'warning');
        return buildStep2Fallback(batchItems, "MISSING_REFINED_SCRIPT");
    }

    let coerced = false;
    const notes: string[] = [];

    if (!u.removal_report || typeof u.removal_report !== 'object') {
        u.removal_report = { removed_from_main: [], removal_summary: "COERCED_FIELD" };
        coerced = true;
        notes.push("Coerced 'removal_report'.");
    }
    if (!u.qa_audit || typeof u.qa_audit !== 'object') {
        u.qa_audit = { gv_coverage_attestation: false, gv_items_total: 0, gv_items_used_in_main: 0, uncertain_items_count: 0, risk_flags: [], notes: "COERCED_FIELD" };
        coerced = true;
        notes.push("Coerced 'qa_audit'.");
    }
    
    if (coerced) {
        addLog(`Step 2 Warning: Output schema mismatch, applied COERCE for this batch. Reason: ${notes.join(' ')}`, 'warning');
        u.qa_audit.risk_flags = [...(u.qa_audit.risk_flags || []), "COERCED_FIELDS"];
        u.qa_audit.notes = `${u.qa_audit.notes || ''} ${notes.join(' ')}`;
    }

    // Ensure refined_script items have required fields
    u.refined_script = u.refined_script.map((item: any) => {
        if (!item || typeof item !== 'object') return null; // filter out invalid items
        const newItem = { ...item };
        let itemCoerced = false;
        if (typeof newItem.speaker !== 'string') { newItem.speaker = '[??]'; itemCoerced = true; }
        if (typeof newItem.start_timestamp !== 'string') { newItem.start_timestamp = '[00:00]'; itemCoerced = true; }
        if (typeof newItem.end_timestamp !== 'string') { newItem.end_timestamp = '[00:00]'; itemCoerced = true; }
        if (typeof newItem.text !== 'string') { newItem.text = ''; itemCoerced = true; }
        if (!Array.isArray(newItem.source_timestamps)) { newItem.source_timestamps = []; itemCoerced = true; }
        
        if (itemCoerced) newItem.needs_review = true;

        return newItem;
    }).filter(Boolean);


    return u as PostEditResult;
};

const mergePostEditResults = (results: PostEditResult[]): PostEditResult => {
    if (results.length === 0) { 
        return { mode: "EMPTY", refined_script: [], removal_report: { removed_from_main: [], removal_summary: "No results to merge." }, qa_audit: { gv_coverage_attestation: false, gv_items_total: 0, gv_items_used_in_main: 0, uncertain_items_count: 0, risk_flags: ["EMPTY_MERGE"], notes: "No results were provided to merge." } };
    }
    if (results.length === 1) return results[0];

    const initialValue: PostEditResult = {
        mode: "POST_EDIT_LECTURE_SCRIPT_RESULT_BATCHED",
        refined_script: [],
        removal_report: { removed_from_main: [], removal_summary: "" },
        qa_audit: {
            gv_coverage_attestation: true,
            gv_items_total: 0,
            gv_items_used_in_main: 0,
            uncertain_items_count: 0,
            risk_flags: [],
            notes: ""
        }
    };

    return results.reduce((acc, current, index) => {
        const summarySeparator = acc.removal_report.removal_summary ? `\n--- BATCH ${index + 1} ---\n` : "";
        const notesSeparator = acc.qa_audit.notes ? `\n--- BATCH ${index + 1} ---\n` : "";

        return {
            mode: "POST_EDIT_LECTURE_SCRIPT_RESULT_BATCHED",
            refined_script: [...acc.refined_script, ...current.refined_script],
            removal_report: {
                removed_from_main: [...acc.removal_report.removed_from_main, ...current.removal_report.removed_from_main],
                removal_summary: `${acc.removal_report.removal_summary}${summarySeparator}${current.removal_report.removal_summary}`,
            },
            qa_audit: {
                gv_coverage_attestation: acc.qa_audit.gv_coverage_attestation && current.qa_audit.gv_coverage_attestation,
                gv_items_total: acc.qa_audit.gv_items_total + current.qa_audit.gv_items_total,
                gv_items_used_in_main: acc.qa_audit.gv_items_used_in_main + current.qa_audit.gv_items_used_in_main,
                uncertain_items_count: acc.qa_audit.uncertain_items_count + current.qa_audit.uncertain_items_count,
                risk_flags: [...new Set([...acc.qa_audit.risk_flags, ...current.qa_audit.risk_flags])],
                notes: `${acc.qa_audit.notes}${notesSeparator}${current.qa_audit.notes}`,
            },
        };
    }, initialValue);
};

const runStep2Once = async (
    ai: GoogleGenAI,
    model: string,
    batchItems: Partial<ImprovedTranscriptItem>[],
    addLog: (message: string, type?: LogEntry['type']) => void,
    temperature: number,
    extraPrompt: string
): Promise<PostEditResult> => {
    const inputData = { mode: "POST_EDIT_LECTURE_SCRIPT", input: { improved_transcript: batchItems } };
    const fullPrompt = `${SYSTEM_PROMPT_STEP_2}\n\nINPUT JSON:\n${JSON.stringify(inputData)}${extraPrompt}`;
    
    const responseSchema = {
        type: Type.OBJECT,
        properties: {
            mode: { type: Type.STRING },
            refined_script: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        speaker: { type: Type.STRING },
                        start_timestamp: { type: Type.STRING },
                        end_timestamp: { type: Type.STRING },
                        text: { type: Type.STRING },
                        source_timestamps: { type: Type.ARRAY, items: { type: Type.STRING } },
                        needs_review: { type: Type.BOOLEAN, nullable: true },
                    },
                    required: ["speaker", "start_timestamp", "end_timestamp", "text", "source_timestamps"],
                },
            },
            removal_report: {
                type: Type.OBJECT,
                properties: {
                    removed_from_main: {
                        type: Type.ARRAY,
                        items: {
                            type: Type.OBJECT,
                            properties: {
                                timestamp: { type: Type.STRING },
                                speaker: { type: Type.STRING },
                                reason: { type: Type.STRING },
                                verbatim_excerpt: { type: Type.STRING },
                            },
                            required: ["timestamp", "speaker", "reason", "verbatim_excerpt"],
                        },
                    },
                    removal_summary: { type: Type.STRING },
                },
                required: ["removed_from_main", "removal_summary"],
            },
            qa_audit: {
                type: Type.OBJECT,
                properties: {
                    gv_coverage_attestation: { type: Type.BOOLEAN },
                    gv_items_total: { type: Type.INTEGER },
                    gv_items_used_in_main: { type: Type.INTEGER },
                    uncertain_items_count: { type: Type.INTEGER },
                    risk_flags: { type: Type.ARRAY, items: { type: Type.STRING } },
                    notes: { type: Type.STRING },
                },
                required: ["gv_coverage_attestation", "gv_items_total", "gv_items_used_in_main", "uncertain_items_count", "risk_flags", "notes"],
            },
        },
        required: ["mode", "refined_script", "removal_report", "qa_audit"],
    };

    const apiCall = ai.models.generateContent({
        model,
        contents: { parts: [{ text: fullPrompt }] },
        config: {
            temperature: temperature,
            maxOutputTokens: 8192,
            responseMimeType: "application/json",
            responseSchema: responseSchema,
            safetySettings: [
                { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
            ]
        }
    });

    const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`Timeout: Step 2 batch không phản hồi sau ${STEP2_TIMEOUT_MS / 60000} phút`)), STEP2_TIMEOUT_MS)
    );

    const response: any = await Promise.race([apiCall, timeoutPromise]);
    const rawText = extractGenAIText(response);

    if (!rawText) {
        throw new Error("Internal error: Step 2 empty response text");
    }

    const parseResult = safeParseJSON(rawText);
    // FIX: Changed to an explicit `=== false` check to ensure correct type narrowing for the discriminated union.
    if (parseResult.ok === false) {
        throw new Error(`Invalid JSON: ${parseResult.reason}. Pos: ${parseResult.pos}, Context: ${parseResult.context}`);
    }

    return normalizePostEdit(parseResult.value, batchItems, addLog);
};


async function runStep2Robust(
    ai: GoogleGenAI,
    model: string,
    items: Partial<ImprovedTranscriptItem>[],
    addLog: (message: string, type?: LogEntry['type']) => void,
    depth = 0
): Promise<PostEditResult> {

    for (let i = 0; i < STEP2_MAX_RETRIES; i++) {
        try {
            const temperature = i > 0 ? 0.1 : 0.2;
            const extraPrompt = i > 0 ? "\n\nREMINDER: Your entire output must be a single, strictly valid JSON object. Check your syntax carefully." : "";
            return await runStep2Once(ai, model, items, addLog, temperature, extraPrompt);
        } catch (error: any) {
            const msg = error.message || String(error);
            const isParseError = msg.includes("Invalid JSON") || error instanceof SyntaxError;
            
            if (isInternalError(msg) || isParseError) {
                if (i < STEP2_MAX_RETRIES - 1) {
                    const backoff = Math.pow(3, i) * 2000; // 2s, 6s, 18s
                    addLog(`Step 2 batch error. Attempt ${i + 1}/${STEP2_MAX_RETRIES} failed. Retrying in ${backoff / 1000}s. Details: ${msg}`, 'warning');
                    await sleep(backoff);
                    continue;
                }
                addLog(`Step 2 batch failed after ${STEP2_MAX_RETRIES} retries. Final error: ${msg}`, 'error');
            } else {
                 // Re-throw other errors (like 429) to be handled by the caller
                 throw error;
            }
        }
    }
    
    // If all retries failed, try splitting
    if (items.length > STEP2_MIN_SPLIT_ITEMS && depth < STEP2_MAX_SPLIT_DEPTH) {
        addLog(`Tất cả các lần thử lại đều thất bại. Tự động chia nhỏ batch (${items.length} items) và thử lại.`, 'warning');
        const mid = Math.floor(items.length / 2);
        const left = items.slice(0, mid);
        const right = items.slice(mid);
        
        try {
            const [leftResult, rightResult] = await Promise.all([
                runStep2Robust(ai, model, left, addLog, depth + 1),
                runStep2Robust(ai, model, right, addLog, depth + 1),
            ]);
            return mergePostEditResults([leftResult, rightResult]);
        } catch (splitError: any) {
             addLog(`Chia nhỏ batch cũng thất bại: ${splitError.message}. Sử dụng Fallback.`, 'error');
             return buildStep2Fallback(items, "SPLIT_AND_RETRY_FAILED");
        }
    }
    
    addLog(`Không thể xử lý batch ngay cả sau khi thử lại và chia nhỏ. Sử dụng Fallback.`, 'error');
    return buildStep2Fallback(items, "MAX_RETRIES_AND_SPLITS_FAILED");
}

const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onload = () => {
            const result = reader.result as string;
            resolve(result.includes(',') ? result.split(',')[1] : result);
        };
        reader.onerror = reject;
    });
};