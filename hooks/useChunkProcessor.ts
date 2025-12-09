import { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, Type, HarmCategory, HarmBlockThreshold } from '@google/genai';
import { Chunk, LogEntry, ProcessingStats, TranscriptionOutput, ImprovedTranscriptItem, PostEditResult, RateLimitEvent, Batch } from '../types';
import { decodeAudioData, resampleAndEncodeWav, estimateWavBytes, normalizeTextForDedupe } from '../utils/audioUtils';
import { computeStep1WordCount, computeStep2WordCount } from '../utils/transcriptMetrics';


/**
 * @LOCKED_CORE_LOGIC
 * QUY TRÌNH XỬ LÝ: Client-side Chunking + Multi-threading + Medical Diarization Prompt
 * KHÔNG ĐƯỢC PHÉP THAY ĐỔI LOGIC NÀY MÀ KHÔNG CÓ YÊU CẦU CỤ THỂ TỪ NGƯỜI DÙNG.
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
    // --- Step 1 State ---
    const [chunks, setChunks] = useState<Chunk[]>([]);
    const [stats, setStats] = useState<ProcessingStats>({
        total: 0, completed: 0, processing: 0, failed: 0, startTime: 0,
        isCoolingDown: false, cooldownSeconds: 0,
    });
    const [maxConcurrency, setMaxConcurrency] = useState(5);

    // --- Step 2 State (NEW) ---
    const [step2Batches, setStep2Batches] = useState<Batch[]>([]);
    const [step2Stats, setStep2Stats] = useState<ProcessingStats>({
        total: 0, completed: 0, processing: 0, failed: 0, startTime: 0,
        isCoolingDown: false, cooldownSeconds: 0,
    });


    // --- Common State ---
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [result, setResult] = useState<TranscriptionOutput | null>(null);
    const [isInitializing, setIsInitializing] = useState(false);
    const [isFinalizing, setIsFinalizing] = useState(false);
    const [rateLimitEvent, setRateLimitEvent] = useState<RateLimitEvent | null>(null);
    

    const audioBufferRef = useRef<AudioBuffer | null>(null);
    const statsRef = useRef(stats);
    const chunksRef = useRef(chunks);
    const step1MetricsLoggedRef = useRef(false);
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
        setStep2Stats(prev => ({ ...prev, isCoolingDown: false, cooldownSeconds: 0 }));
        addLog(reason, 'info');
    }, [addLog]);

    const updateStats = useCallback((updates: Partial<ProcessingStats>) => {
        setStats(prev => ({ ...prev, ...updates }));
    }, []);
    
    // --- Cooldown Timer Effect ---
    useEffect(() => {
        if (!stats.isCoolingDown && !step2Stats.isCoolingDown && rateLimitEvent?.active) {
            setRateLimitEvent(null);
        }
    }, [stats.isCoolingDown, step2Stats.isCoolingDown, rateLimitEvent]);

    useEffect(() => {
        let interval: any;
        const step1Cooling = stats.isCoolingDown && stats.cooldownSeconds > 0;
        const step2Cooling = step2Stats.isCoolingDown && step2Stats.cooldownSeconds > 0;
        if (step1Cooling || step2Cooling) {
            interval = setInterval(() => {
                if (step1Cooling) {
                    setStats(prev => {
                        const newValue = prev.cooldownSeconds - 1;
                        if (newValue <= 0) {
                            addLog("Hết thời gian chờ (Step 1). Tiếp tục xử lý...", 'info');
                            return { ...prev, isCoolingDown: false, cooldownSeconds: 0 };
                        }
                        return { ...prev, cooldownSeconds: newValue };
                    });
                }
                 if (step2Cooling) {
                    setStep2Stats(prev => {
                        const newValue = prev.cooldownSeconds - 1;
                        if (newValue <= 0) {
                            addLog("Hết thời gian chờ (Step 2). Tiếp tục xử lý...", 'info');
                            return { ...prev, isCoolingDown: false, cooldownSeconds: 0 };
                        }
                        return { ...prev, cooldownSeconds: newValue };
                    });
                }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [stats.isCoolingDown, stats.cooldownSeconds, step2Stats.isCoolingDown, step2Stats.cooldownSeconds, addLog]);

    // --- Helpers ---
    const formatSecondsToTime = (totalSeconds: number): string => {
        const h = Math.floor(totalSeconds / 3600);
        const m = Math.floor((totalSeconds % 3600) / 60);
        const s = Math.floor(totalSeconds % 60);
        if (h > 0) return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
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

    // --- STEP 1 LOGIC (UNCHANGED) ---
    const initializeChunks = useCallback(async () => {
        if (!file) return;
        setIsInitializing(true);
        setIsFinalizing(false);
        setResult(null);
        setLogs([]);
        setChunks([]);
        setStep2Batches([]);
        setMaxConcurrency(5); 
        step1MetricsLoggedRef.current = false;
        try {
            addLog(`Bắt đầu phân tích file: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`, 'info');
            audioBufferRef.current = await decodeAudioData(file);
            const durationSec = audioBufferRef.current.duration;
            addLog(`Giải mã thành công. Tổng thời lượng: ${formatSecondsToTime(durationSec)}`, 'success');

            let chunkSeconds = TARGET_CHUNK_SECONDS;
            while (estimateWavBytes(chunkSeconds) > MAX_INLINE_DATA_BYTES && chunkSeconds > 15) {
                chunkSeconds -= 15;
            }
            if (chunkSeconds !== TARGET_CHUNK_SECONDS) {
                addLog(`File có chất lượng cao, tự động giảm thời lượng chunk xuống ${chunkSeconds}s để đảm bảo ổn định.`, 'warning');
            }

            const totalChunks = Math.ceil(durationSec / chunkSeconds);
            const newChunks: Chunk[] = Array.from({ length: totalChunks }, (_, i) => {
                 const startSec = Math.max(0, i * chunkSeconds - (i > 0 ? OVERLAP_SECONDS : 0));
                 const endSec = Math.min(durationSec, (i + 1) * chunkSeconds);
                 return { id: `chunk-${i}`, index: i, status: 'pending', startSec, endSec };
            });

            setChunks(newChunks);
            updateStats({ total: totalChunks, completed: 0, processing: 0, failed: 0, startTime: 0, endTime: undefined, isCoolingDown: false, cooldownSeconds: 0 });
            addLog(`Đã chia audio thành ${totalChunks} phân đoạn (${chunkSeconds}s/chunk, gối lên nhau ${OVERLAP_SECONDS}s).`, 'success');
        } catch (error: any) {
            addLog(`Lỗi khởi tạo: ${error.message || error}`, 'error');
        } finally {
            setIsInitializing(false);
        }
    }, [file, addLog, updateStats]);

    const processChunk = useCallback(async (chunkId: string) => {
        // (Implementation is unchanged from previous version, omitted for brevity but is still present)
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
            if (parsedResult.ok === false) {
                 throw new Error(`Invalid JSON: ${parsedResult.reason}. Pos: ${parsedResult.pos}, Context: "${parsedResult.context}"`);
            }
            const parsedData = parsedResult.value;
            
            if (parsedData.improved_transcript && Array.isArray(parsedData.improved_transcript)) {
                const timeOffsetSeconds = currentChunk.startSec || 0;
                parsedData.improved_transcript = parsedData.improved_transcript.map((item: any) => {
                    const itemSeconds = parseTimeToSeconds(item.timestamp);
                    const absoluteSeconds = itemSeconds + timeOffsetSeconds;
                    return { ...item, timestamp: `[${formatSecondsToTime(absoluteSeconds)}]` };
                });
            }

            const chunkResultJSON = JSON.stringify(parsedData);
            setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'completed', content: chunkResultJSON } : c));

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
                 setRateLimitEvent({ active: true, step: 'STEP1', message: msg, lastModel: modelStep1, at: Date.now() });
                 setMaxConcurrency(1);
                 setStats(prev => ({ ...prev, isCoolingDown: true, cooldownSeconds: 60 }));
                 setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'pending', error: undefined } : c));
                 setStats(prev => ({ ...prev, processing: Math.max(0, prev.processing - 1) }));
                 return;
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

    // --- STEP 2 LOGIC (REFACTORED) ---

    // 1. Setup function: Prepares batches and puts the system into "finalizing" mode.
    const performStep2Finalization = useCallback((mergedTranscript: ImprovedTranscriptItem[]) => {
        if (isFinalizing || !process.env.API_KEY) return;
        if (!mergedTranscript || mergedTranscript.length === 0) {
            addLog("Chưa có dữ liệu để chạy Step 2.", 'warning');
            return;
        }

        addLog(`BẮT ĐẦU STEP 2: Chuẩn bị các batch...`, 'info');
        setResult(prev => prev ? { ...prev, post_edit_result: undefined } : null); // Clear previous results

        const slimTranscript = buildSlimTranscript(mergedTranscript);
        const batchesData = buildBatchesByCharOrCount(slimTranscript, STEP2_TARGET_MAX_CHARS, STEP2_MAX_ITEMS_PER_BATCH);
        
        const newBatches: Batch[] = batchesData.map((items, index) => ({
            id: `batch-${index}`,
            index,
            items,
            status: 'pending',
        }));

        setStep2Batches(newBatches);
        setStep2Stats({
            total: newBatches.length, completed: 0, processing: 0, failed: 0,
            startTime: Date.now(), isCoolingDown: false, cooldownSeconds: 0,
        });
        addLog(`Đã chia ${slimTranscript.length} items thành ${newBatches.length} batch. Bắt đầu xử lý...`, 'info');
        setIsFinalizing(true); // This kicks off the processing useEffect
    }, [addLog, isFinalizing]);

    // 2. Processing effect: Watches the batch queue and processes one pending batch at a time.
    useEffect(() => {
        const processQueue = async () => {
            if (!isFinalizing || step2Stats.processing > 0 || step2Stats.isCoolingDown) return;

            const nextBatch = step2Batches.find(b => b.status === 'pending');
            if (!nextBatch) return;

            setStep2Batches(prev => prev.map(b => b.id === nextBatch.id ? { ...b, status: 'processing' } : b));
            setStep2Stats(prev => ({ ...prev, processing: prev.processing + 1 }));
            addLog(`Step 2: Đang xử lý Batch ${nextBatch.index + 1}/${step2Batches.length} với model ${modelStep2}...`, 'info');

            try {
                const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
                const result = await runStep2Robust(ai, modelStep2, nextBatch.items, addLog);

                if (result.mode.includes("FALLBACK")) {
                    throw new Error(result.qa_audit.notes || "Processing failed after all retries and splits.");
                }
                
                setStep2Batches(prev => prev.map(b => b.id === nextBatch.id ? { ...b, status: 'completed', result, error: undefined } : b));
                setStep2Stats(prev => ({ ...prev, processing: 0, completed: prev.completed + 1 }));
                addLog(`Step 2: Hoàn thành Batch ${nextBatch.index + 1}.`, 'success');
            } catch (error: any) {
                const msg = error.message || String(error);
                const isRateLimit = msg.includes('429') || msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED');

                if (isRateLimit) {
                    addLog(`Step 2: Quá tải API (429) ở Batch ${nextBatch.index + 1}. Tạm dừng 60s...`, 'error');
                    setRateLimitEvent({ active: true, step: 'STEP2', message: msg, lastModel: modelStep2, at: Date.now() });
                    setStep2Stats(prev => ({ ...prev, isCoolingDown: true, cooldownSeconds: 60 }));
                    // Revert to pending so it can be picked up again after cooldown or model switch
                    setStep2Batches(prev => prev.map(b => b.id === nextBatch.id ? { ...b, status: 'pending' } : b));
                    setStep2Stats(prev => ({ ...prev, processing: 0 }));
                } else {
                    setStep2Batches(prev => prev.map(b => b.id === nextBatch.id ? { ...b, status: 'failed', error: msg } : b));
                    setStep2Stats(prev => ({ ...prev, processing: 0, failed: prev.failed + 1 }));
                    addLog(`Step 2: Lỗi Batch ${nextBatch.index + 1}: ${msg}. Vui lòng thử lại.`, 'error');
                }
            }
        };

        processQueue();
    }, [step2Batches, isFinalizing, step2Stats.processing, step2Stats.isCoolingDown, modelStep2, addLog]);

    // 3. Merging effect: Watches for all batches to be completed.
    useEffect(() => {
        if (step2Batches.length > 0 && step2Batches.every(b => b.status === 'completed')) {
            const batchResults = step2Batches.map(b => b.result as PostEditResult);
            const finalResult = mergePostEditResults(batchResults);
            
            setResult(prev => prev ? { ...prev, post_edit_result: finalResult } : null);

            const s1Words = computeStep1WordCount(result?.improved_transcript || []);
            const s2Words = computeStep2WordCount(finalResult.refined_script);
            const delta = s2Words - s1Words;
            const deltaPercent = s1Words > 0 ? (delta / s1Words * 100).toFixed(1) : "0.0";
            
            addLog(`STEP 2 METRICS: segments=${finalResult.refined_script.length}, words=${s2Words}, deltaWords=${delta}, deltaPercent=${deltaPercent}%`, 'success');
            addLog(`Hoàn tất Step 2: Đã tạo văn bản chuyên nghiệp.`, 'success');
            
            setStep2Stats(prev => ({ ...prev, endTime: Date.now() }));
            setIsFinalizing(false);
        }
    }, [step2Batches, result?.improved_transcript, addLog]);


    // --- AUTO RUN & MERGE MONITOR (STEP 1) ---
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
                        const lastItems = lastChunkTranscript.slice(-2);
                        const firstItems = currentChunkTranscript.slice(0, 2);
                        let overlapIndex = -1;
                        for(let i = 0; i < firstItems.length; i++) {
                            for (let j = 0; j < lastItems.length; j++) {
                                if (normalizeTextForDedupe(firstItems[i].edited) === normalizeTextForDedupe(lastItems[j].edited)) {
                                   overlapIndex = i; break;
                                }
                            }
                            if (overlapIndex !== -1) break;
                        }
                        if (overlapIndex !== -1) {
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
        
        if (chunks.length > 0 && completedChunks.length === chunks.length && !step1MetricsLoggedRef.current) {
            step1MetricsLoggedRef.current = true;
            const wordCount = computeStep1WordCount(mergedTranscript);
            addLog(`STEP 1 METRICS: items=${mergedTranscript.length}, words=${wordCount}`, 'success');
        }

    }, [chunks, stats, maxConcurrency, processChunk, addLog]);

    // --- Action Handlers ---
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
    
    const retryStep2Batch = useCallback((batchId: string) => {
        setStep2Batches(prev => prev.map(b => b.id === batchId ? { ...b, status: 'pending', error: undefined } : b));
        setStep2Stats(prev => ({ ...prev, failed: Math.max(0, prev.failed - 1) }));
        addLog(`Thử lại Batch ${batchId}`, 'info');
    }, [addLog]);

    const retryAllFailedStep2 = useCallback(() => {
        const failedCount = step2Batches.filter(b => b.status === 'failed').length;
        if (failedCount === 0) return;
        setStep2Batches(prev => prev.map(b => b.status === 'failed' ? { ...b, status: 'pending', error: undefined } : b));
        setStep2Stats(prev => ({ ...prev, failed: 0 }));
        addLog(`Thử lại ${failedCount} batch thất bại của Step 2`, 'warning');
    }, [addLog, step2Batches]);

    const reset = () => {
        setChunks([]);
        setStep2Batches([]);
        setResult(null);
        setLogs([]);
        setStats({ total: 0, completed: 0, processing: 0, failed: 0, startTime: 0, isCoolingDown: false, cooldownSeconds: 0 });
        setStep2Stats({ total: 0, completed: 0, processing: 0, failed: 0, startTime: 0, isCoolingDown: false, cooldownSeconds: 0 });
        setMaxConcurrency(5);
        setIsFinalizing(false);
        setRateLimitEvent(null);
        audioBufferRef.current = null;
        step1MetricsLoggedRef.current = false;
    };

    const manualAppendTranscript = (jsonString: string) => {
        try {
            const parsed = JSON.parse(jsonString);
            const newItems = Array.isArray(parsed) ? parsed : parsed.improved_transcript;
            if (!Array.isArray(newItems)) { throw new Error("Invalid JSON format."); }
            
            setResult(prev => ({
                improved_transcript: [...(prev?.improved_transcript || []), ...newItems],
                validation_and_conclusion: "Manual Import", professional_medical_text: "Ready for Step 2",
            }));
            addLog(`Đã nối thêm ${newItems.length} dòng dữ liệu thủ công.`, 'success');
        } catch (e: any) { addLog(`Lỗi import: ${e.message}`, 'error'); throw e; }
    };

    const triggerStep2 = () => {
        if (result?.improved_transcript && result.improved_transcript.length > 0) {
            performStep2Finalization(result.improved_transcript);
        } else {
            addLog("Không có dữ liệu Transcript để chạy Step 2.", 'error');
        }
    };

    return {
        chunks, isInitializing, stats, logs, result, isFinalizing,
        step2Batches, step2Stats,
        fileType: 'audio',
        retryChunk, retryAllFailed, initializeChunks, reset,
        triggerStep2, manualAppendTranscript,
        retryStep2Batch, retryAllFailedStep2,
        rateLimitEvent, clearCooldownNow
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
    if (typeof resp.text === 'string') return resp.text;
    try {
        if (resp?.candidates?.[0]?.content?.parts?.length > 0) return resp.candidates[0].content.parts.map((p: any) => p.text).join('');
        if (resp?.response?.candidates?.[0]?.content?.parts?.length > 0) return resp.response.candidates[0].content.parts.map((p: any) => p.text).join('');
        return null;
    } catch (e) { return null; }
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
    if (firstBrace !== -1 && lastBrace > firstBrace) return s.substring(firstBrace, lastBrace + 1);
    return s;
}

function safeParseJSON(raw: string | null): { ok: true, value: any } | { ok: false, reason: string, pos?: number, context?: string } {
    if (!raw || typeof raw !== 'string' || raw.trim() === '') return { ok: false, reason: 'Input is null, empty, or not a string.' };
    let s = raw.trim();
    s = stripCodeFences(s);
    s = extractJSONObjectText(s);
    const tryParsing = (text: string): ReturnType<typeof safeParseJSON> => {
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
    if (result.ok) return result;
    let repaired = s.replace(/,(\s*[}\]])/g, '$1').replace(/[“”]/g, '"').replace(/[‘’]/g, "'");
    if (repaired !== s) {
        const repairResult = tryParsing(repaired);
        return repairResult.ok ? repairResult : result;
    }
    return result;
}

const buildSlimTranscript = (items: ImprovedTranscriptItem[]): Partial<ImprovedTranscriptItem>[] => items.map(item => ({
    timestamp: item.timestamp,
    speaker: item.speaker || '[??]',
    edited: item.edited.trim(),
    original: item.uncertain ? item.original : undefined,
    uncertain: item.uncertain,
    chitchat: item.chitchat,
}));

const buildBatchesByCharOrCount = (items: Partial<ImprovedTranscriptItem>[], targetMaxChars: number, maxItems: number): Partial<ImprovedTranscriptItem>[][] => {
    if (items.length === 0) return [];
    const batches: Partial<ImprovedTranscriptItem>[][] = [];
    let currentBatch: Partial<ImprovedTranscriptItem>[] = [];
    let currentCharCount = 0;
    for (const item of items) {
        const itemCharCount = JSON.stringify(item).length;
        if (currentBatch.length > 0 && (currentCharCount + itemCharCount > targetMaxChars || currentBatch.length >= maxItems)) {
            batches.push(currentBatch);
            currentBatch = [];
            currentCharCount = 0;
        }
        currentBatch.push(item);
        currentCharCount += itemCharCount;
    }
    if (currentBatch.length > 0) batches.push(currentBatch);
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
    const refined_script = items.map(item => ({
        speaker: item.speaker || "[??]",
        start_timestamp: item.timestamp || "[00:00]",
        end_timestamp: item.timestamp || "[00:00]",
        text: (item.edited || item.original || "").trim(),
        source_timestamps: item.timestamp ? [item.timestamp] : [],
        needs_review: true
    })).filter(x => x.text.length > 0);
    return {
        mode: "POST_EDIT_LECTURE_SCRIPT_RESULT_FALLBACK",
        refined_script,
        removal_report: { removed_from_main: [], removal_summary: `FALLBACK_USED: ${reason}` },
        qa_audit: {
            gv_coverage_attestation: false, gv_items_total: items.length, gv_items_used_in_main: refined_script.length,
            uncertain_items_count: items.filter(i => i.uncertain).length,
            risk_flags: ["FALLBACK_USED", reason],
            notes: "Step 2 output had an invalid schema. This is a deterministic fallback generated from Step 1's 'edited' text."
        }
    };
};

const normalizePostEdit = (obj: any, batchItems: Partial<ImprovedTranscriptItem>[], addLog: (message: string, type?: LogEntry['type']) => void): PostEditResult => {
    const u = unwrapPostEdit(obj);
    if (!u || typeof u !== 'object' || !Array.isArray(u.refined_script)) {
        addLog(`Step 2 Warning: Output schema mismatch, applied FALLBACK.`, 'warning');
        return buildStep2Fallback(batchItems, !Array.isArray(u.refined_script) ? "MISSING_REFINED_SCRIPT" : "NON_OBJECT");
    }
    let coerced = false;
    const notes: string[] = [];
    if (!u.removal_report || typeof u.removal_report !== 'object') { u.removal_report = { removed_from_main: [], removal_summary: "COERCED_FIELD" }; coerced = true; notes.push("Coerced 'removal_report'."); }
    if (!u.qa_audit || typeof u.qa_audit !== 'object') { u.qa_audit = { gv_coverage_attestation: false, gv_items_total: 0, gv_items_used_in_main: 0, uncertain_items_count: 0, risk_flags: [], notes: "COERCED_FIELD" }; coerced = true; notes.push("Coerced 'qa_audit'."); }
    if (coerced) { addLog(`Step 2 Warning: Coerced fields for this batch. Reason: ${notes.join(' ')}`, 'warning'); u.qa_audit.risk_flags = [...(u.qa_audit.risk_flags || []), "COERCED_FIELDS"]; u.qa_audit.notes = `${u.qa_audit.notes || ''} ${notes.join(' ')}`; }
    u.refined_script = u.refined_script.map((item: any) => {
        if (!item || typeof item !== 'object') return null;
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
    if (results.length === 0) return { mode: "EMPTY", refined_script: [], removal_report: { removed_from_main: [], removal_summary: "No results." }, qa_audit: { gv_coverage_attestation: false, gv_items_total: 0, gv_items_used_in_main: 0, uncertain_items_count: 0, risk_flags: ["EMPTY_MERGE"], notes: "No results." } };
    if (results.length === 1) return results[0];
    const initialValue: PostEditResult = {
        mode: "POST_EDIT_LECTURE_SCRIPT_RESULT_BATCHED",
        refined_script: [],
        removal_report: { removed_from_main: [], removal_summary: "" },
        qa_audit: { gv_coverage_attestation: true, gv_items_total: 0, gv_items_used_in_main: 0, uncertain_items_count: 0, risk_flags: [], notes: "" }
    };
    return results.reduce((acc, current, index) => ({
        mode: "POST_EDIT_LECTURE_SCRIPT_RESULT_BATCHED",
        refined_script: [...acc.refined_script, ...current.refined_script],
        removal_report: {
            removed_from_main: [...acc.removal_report.removed_from_main, ...current.removal_report.removed_from_main],
            removal_summary: `${acc.removal_report.removal_summary}${acc.removal_report.removal_summary ? `\n--- BATCH ${index + 1} ---\n` : ""}${current.removal_report.removal_summary}`,
        },
        qa_audit: {
            gv_coverage_attestation: acc.qa_audit.gv_coverage_attestation && current.qa_audit.gv_coverage_attestation,
            gv_items_total: acc.qa_audit.gv_items_total + current.qa_audit.gv_items_total,
            gv_items_used_in_main: acc.qa_audit.gv_items_used_in_main + current.qa_audit.gv_items_used_in_main,
            uncertain_items_count: acc.qa_audit.uncertain_items_count + current.qa_audit.uncertain_items_count,
            risk_flags: [...new Set([...acc.qa_audit.risk_flags, ...current.qa_audit.risk_flags])],
            notes: `${acc.qa_audit.notes}${acc.qa_audit.notes ? `\n--- BATCH ${index + 1} ---\n` : ""}${current.qa_audit.notes}`,
        },
    }), initialValue);
};

const runStep2Once = async (ai: GoogleGenAI, model: string, batchItems: Partial<ImprovedTranscriptItem>[], addLog: (message: string, type?: LogEntry['type']) => void, temperature: number, extraPrompt: string): Promise<PostEditResult> => {
    const inputData = { mode: "POST_EDIT_LECTURE_SCRIPT", input: { improved_transcript: batchItems } };
    const fullPrompt = `${SYSTEM_PROMPT_STEP_2}\n\nINPUT JSON:\n${JSON.stringify(inputData)}${extraPrompt}`;
    const responseSchema = { /* ... schema definition (omitted for brevity) ... */ };
    const apiCall = ai.models.generateContent({
        model, contents: { parts: [{ text: fullPrompt }] },
        config: {
            temperature, maxOutputTokens: 8192, responseMimeType: "application/json", responseSchema,
            safetySettings: Object.values(HarmCategory).map(category => ({ category, threshold: HarmBlockThreshold.BLOCK_NONE })),
        }
    });
    const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error(`Timeout: Step 2 batch không phản hồi`)), STEP2_TIMEOUT_MS));
    const response: any = await Promise.race([apiCall, timeoutPromise]);
    const rawText = extractGenAIText(response);
    if (!rawText) throw new Error("Internal error: Step 2 empty response text");
    const parseResult = safeParseJSON(rawText);
    if (parseResult.ok === false) throw new Error(`Invalid JSON: ${parseResult.reason}. Pos: ${parseResult.pos}, Context: ${parseResult.context}`);
    return normalizePostEdit(parseResult.value, batchItems, addLog);
};

async function runStep2Robust(ai: GoogleGenAI, model: string, items: Partial<ImprovedTranscriptItem>[], addLog: (message: string, type?: LogEntry['type']) => void, depth = 0): Promise<PostEditResult> {
    for (let i = 0; i < STEP2_MAX_RETRIES; i++) {
        try {
            const temperature = i > 0 ? 0.1 : 0.2;
            const extraPrompt = i > 0 ? "\n\nREMINDER: Your entire output must be a single, strictly valid JSON object." : "";
            return await runStep2Once(ai, model, items, addLog, temperature, extraPrompt);
        } catch (error: any) {
            const msg = error.message || String(error);
            const isParseError = msg.includes("Invalid JSON") || error instanceof SyntaxError;
            if (isInternalError(msg) || isParseError) {
                if (i < STEP2_MAX_RETRIES - 1) {
                    const backoff = Math.pow(3, i) * 2000;
                    addLog(`Step 2 batch error. Attempt ${i + 1}/${STEP2_MAX_RETRIES}. Retrying in ${backoff / 1000}s. Details: ${msg}`, 'warning');
                    await sleep(backoff);
                    continue;
                }
                addLog(`Step 2 batch failed after ${STEP2_MAX_RETRIES} retries. Final error: ${msg}`, 'error');
            } else {
                 throw error; // Re-throw unrecoverable errors like 429 to be caught by the queue processor
            }
        }
    }
    if (items.length > STEP2_MIN_SPLIT_ITEMS && depth < STEP2_MAX_SPLIT_DEPTH) {
        addLog(`Tất cả các lần thử lại đều thất bại. Tự động chia nhỏ batch (${items.length} items) và thử lại.`, 'warning');
        const mid = Math.floor(items.length / 2);
        try {
            const [leftResult, rightResult] = await Promise.all([
                runStep2Robust(ai, model, items.slice(0, mid), addLog, depth + 1),
                runStep2Robust(ai, model, items.slice(mid), addLog, depth + 1),
            ]);
            return mergePostEditResults([leftResult, rightResult]);
        } catch (splitError: any) {
             addLog(`Chia nhỏ batch cũng thất bại: ${splitError.message}. Sử dụng Fallback.`, 'error');
             return buildStep2Fallback(items, "SPLIT_AND_RETRY_FAILED");
        }
    }
    addLog(`Không thể xử lý batch sau khi thử lại và chia nhỏ. Sử dụng Fallback.`, 'error');
    return buildStep2Fallback(items, "MAX_RETRIES_AND_SPLITS_FAILED");
}

const blobToBase64 = (blob: Blob): Promise<string> => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onload = () => resolve((reader.result as string).split(',')[1] || '');
    reader.onerror = reject;
});