import { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, Type, HarmCategory, HarmBlockThreshold, Blob } from '@google/genai';
import { Chunk, LogEntry, ProcessingStats, TranscriptionOutput, ImprovedTranscriptItem, PostEditResult, RateLimitEvent, RefinedScriptItem, Step3Result } from '../types';
import { decodeAudioData, resampleAndEncodeWav, estimateWavBytes, normalizeTextForDedupe } from '../utils/audioUtils';
// FIX: Corrected import path.
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

const SYSTEM_PROMPT_STEP_2 = `VAI TRÒ (HARD CONSTRAINT):
Bạn là "Medical Textbook Editor & Fidelity Guardian" (Biên tập viên Sách giáo khoa Y học & Người bảo vệ tính trung thực).
Nhiệm vụ: Chuyển đổi transcript bài giảng/giao ban lâm sàng (dạng văn nói lộn xộn, nhiều từ thừa) thành "Văn bản Y học Chuyên khảo" (dạng văn viết gãy gọn, học thuật, súc tích).

MỤC TIÊU CỐT LÕI:
1. "Clean Verbatim": Giữ 100% dữ kiện y khoa (Triệu chứng, Thuốc, Liều lượng, Cơ chế, Chỉ số cận lâm sàng, Lập luận chẩn đoán, Teaching Points).
2. "Aggressive De-cluttering": Loại bỏ triệt để ngôn ngữ giao tiếp, từ đệm, thủ tục lớp học, và các đoạn dẫn dắt không mang thông tin chuyên môn.

QUY TẮC BIÊN TẬP (NGHIÊM NGẶT - PHẢI TUÂN THỦ):

1. XỬ LÝ NỘI DUNG [GV] (GIẢNG VIÊN):
   - GIỮ NGUYÊN: Mọi kiến thức y khoa, kinh nghiệm lâm sàng, phân tích phim X-quang, giải thích cơ chế, hướng dẫn điều trị.
   - XÓA TRIỆT ĐỂ (NOISE):
     + Các câu dẫn dắt/quản lý lớp: "Mấy em có hỏi gì không?", "Tiếp đi em", "Nhìn lên bảng", "Đoạn này không thi", "Các bạn nắm được chưa?", "Khúc này không nằm trong mục tiêu", "Các bạn có thắc mắc gì với ca này không".
     + Các từ đệm thừa thãi (Verbal tics) đặc trưng tiếng Việt: "Thì", "Là", "Cái", "Rằng", "Ở đây là", "Nói chung là", "Đại khái là", "Thật ra là", "Cái thứ hai là", "Cái này thì", "Cái mốc mà".
     + Các đoạn "Meta-talk": "Bây giờ tôi sẽ đọc phim này", "Em đọc diễn tiến cho các bạn nắm đi". -> CHUYỂN THÀNH hành động trực tiếp hoặc xóa bỏ nếu không chứa thông tin y khoa.
   - TÁI CẤU TRÚC (REPHRASE): Chuyển câu văn nói lủng củng thành câu khẳng định súc tích của sách giáo khoa.
     + Gốc: "Cái thứ hai là đối với thuốc ho thì thôi khỏi nói, thuốc ho thì dễ rồi."
     + Sửa: "Đối với thuốc ho, cách sử dụng đơn giản nên không cần bàn sâu."

2. XỬ LÝ NỘI DUNG [SV] (SINH VIÊN/BÁC SĨ TRÌNH BỆNH):
   - XÓA: Các từ "Dạ", "Thưa cô", "Em nghĩ là", "Tụi em thấy", "Hình như là" ở đầu câu.
   - CHUYỂN ĐỔI: Nếu [SV] trình bày bệnh sử/diễn tiến, hãy biên tập lại thành văn phong "Báo cáo ca lâm sàng" (Case Report) chuẩn mực.
     + Gốc: "Dạ, diễn tiến lúc nhập viện 5:30 ngày 11 tháng 11..."
     + Sửa: "Diễn tiến lúc nhập viện (05:30, 11/11):..."
   - GIỮ NGUYÊN: Các câu hỏi sai hoặc chẩn đoán sai của [SV] *NẾU VÀ CHỈ NẾU* sau đó [GV] sửa lại (đây là Teaching Point - Điểm giảng dạy). Nếu [SV] chỉ ậm ừ xác nhận ("Dạ", "Đúng ạ"), hãy xóa bỏ item đó và ghi vào removal_report.

3. XỬ LÝ SỐ LIỆU & THUẬT NGỮ (TUYỆT ĐỐI KHÔNG SAI LỆCH):
   - Giữ nguyên văn các con số (SpO2 98%, mạch 180, liều 9 mg/kg, liều 50-100 mg/kg). Không tự ý làm tròn.
   - Giữ nguyên tên thuốc (Solu-Medrol, Ventolin, Pulmicort, Amoxicillin/acid clavulanic...).
   - Nếu script gốc nói "1/4 ống" -> Giữ "1/4 ống". Không tự quy đổi ra ml nếu không chắc chắn.

========================
ĐẦU VÀO (INPUT)
========================
Bạn sẽ nhận JSON ở trường "input". Chứa "improved_transcript" (mảng item).

========================
ĐẦU RA (OUTPUT JSON - BẮT BUỘC ĐÚNG SCHEMA)
========================
Trả về một JSON có cấu trúc đúng như sau (STRICT JSON RFC 8259). Không thêm markdown, không thêm giải thích:

{
  "mode": "POST_EDIT_LECTURE_SCRIPT_RESULT",
  "refined_script": [
    {
      "speaker": "[GV]" | "[SV]" | "[??]",
      "start_timestamp": "[MM:SS]",
      "end_timestamp": "[MM:SS]",
      "text": "Nội dung đã biên tập thành văn viết y học súc tích, loại bỏ từ thừa.",
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
        "verbatim_excerpt": "trích nguyên văn đoạn bị xóa"
      }
    ],
    "removal_summary": "Tóm tắt NGẮN gọn loại nội dung đã xóa."
  },
  "qa_audit": {
    "gv_coverage_attestation": true,
    "gv_items_total": 0,
    "gv_items_used_in_main": 0,
    "uncertain_items_count": 0,
    "risk_flags": ["possible_teaching_point_dropped", "rephrased_complex_mechanism", "many_uncertain"],
    "notes": "Ghi chú về các đoạn khó nghe hoặc nghi ngờ."
  }
}

========================
HƯỚNG DẪN XỬ LÝ MỘT SỐ TÌNH HUỐNG CỤ THỂ (FEW-SHOT LEARNING TỪ DỮ LIỆU THỰC TẾ):
========================

Case 1: Xử lý từ đệm và câu hỏi thừa (Aggressive De-cluttering)
Input: "[GV]: Thì bây giờ làm sao? Những ca này thì nếu em bé đang đáp ứng với điều trị hen thuận lợi, em có thể theo dõi mà không sử dụng kháng sinh luôn."
Output Text: "Trong trường hợp bệnh nhi đáp ứng thuận lợi với điều trị hen, có thể tiếp tục theo dõi mà không cần chỉ định kháng sinh." (Loại bỏ "Thì bây giờ làm sao", "Những ca này thì").

Case 2: Xử lý thủ tục hành chính (Administrative Removal)
Input: "[GV]: Tiếp theo bé này sao em? Ở đây có bạn nào có thắc mắc gì không? Nếu không thì tiếp đi em."
Output: (Xóa bỏ hoàn toàn khỏi refined_script, ghi vào removal_report với reason="administrative").

Case 3: Xử lý trình bệnh của sinh viên (Student Case Report Style)
Input: "[SV]: Dạ, về công thức máu, bé có hemoglobin là 9,3 g/dL. Hồng cầu nhỏ... Do đó tụi em thấy bé này có thiếu máu hồng cầu nhỏ nhược sắc."
Output Text: "Công thức máu: Hemoglobin 9.3 g/dL, hồng cầu nhỏ, nhược sắc. Kết luận: Thiếu máu hồng cầu nhỏ nhược sắc mức độ nhẹ." (Bỏ "Dạ", "về", "là", "tụi em thấy").

Case 4: Xử lý văn nói giải thích (Syntactic Reconstruction)
Input: "[GV]: Cái thứ hai là đối với Corticoid, bé này là cơn nặng nên đã có chỉ định Corticoid toàn thân. Về lý thuyết, mình có 2 loại thuốc là..."
Output Text: "Về Corticoid: Bệnh nhi có chỉ định Corticoid toàn thân do cơn hen nặng. Về lý thuyết có 2 nhóm thuốc:..." (Loại bỏ "Cái thứ hai là", "là", "thì").

Case 5: Giữ nguyên tranh luận sai để sửa (Preserve Teaching Moment)
Input:
"[SV]: Dạ em nghĩ là dùng kháng sinh liều 50mg/kg ạ.
 [GV]: Không được, ca này phải dùng liều 80-90mg/kg vì nghi ngờ phế cầu kháng thuốc."
Output: (Giữ nguyên cả 2 câu thoại này vì đây là tình huống dạy học quan trọng).

BẮT ĐẦU:
Đọc "input.improved_transcript" và tạo output đúng schema JSON. Tuyệt đối không thêm lời bình luận.
`;

const SYSTEM_PROMPT_STEP_3 = `VAI TRÒ (HARD CONSTRAINT):
Bạn là "Medical Academic Rewriter + Fidelity Auditor + Self-Corrector".

NHIỆM VỤ:
Chuyển dữ liệu Step 2 (refined_script) thành văn bản y khoa liên tục (đoạn văn học thuật), đồng thời:
(1) tạo bản nháp (draft),
(2) chấm điểm rủi ro sai nghĩa so với input (1–5) kèm giải thích,
(3) tự kiểm tra theo checklist và tự sửa để tạo bản cuối (final), xuất report cho người dùng.

NGUYÊN TẮC BẤT DI BẤT DỊCH:
1) KHÔNG THÊM DỮ KIỆN MỚI. Mọi khẳng định trong output phải truy vết được về source_timestamps.
2) KHÔNG ĐỔI NGHĨA. Đặc biệt cấm: đổi phủ định, đổi trái/phải, đổi nguyên nhân-kết quả, đổi thời điểm, đổi liều/đơn vị/chỉ số.
3) BẢO TOÀN TOKEN NHẠY CẢM: mọi chuỗi dạng @@...@@, mọi số (kể cả dấu thập phân), đơn vị (mg, mmol/L...), dấu so sánh (> < >= <=), thời gian, tên thuốc, tên xét nghiệm.
4) GIỮ DẤU HIỆU KHÔNG CHẮC: nếu input có needs_review=true hoặc nội dung mơ hồ, phải ghi chú "(cần kiểm tra)" tại đoạn liên quan và tăng risk_level.
5) VĂN PHONG: tiếng Việt, học thuật, rõ ràng; chia đoạn hợp lý; ưu tiên 3–5 câu/đoạn; có ý lớn/ý nhỏ bằng heading/subheading (dạng dòng tiêu đề ngắn) nhưng tổng thể vẫn là văn bản liên tục.

THANG ĐIỂM RỦI RO SAI NGHĨA (1–5):
1 = Chỉ thay đổi văn phong, không ảnh hưởng nội dung.
2 = Diễn đạt lại có thể gây mơ hồ nhẹ nhưng vẫn đúng.
3 = Có nguy cơ hiểu sai nếu đọc nhanh (cần kiểm tra).
4 = Nguy cơ cao sai dữ kiện (số liệu/đơn vị/phủ định/laterality/thuật ngữ).
5 = Mâu thuẫn hoặc không truy vết được về input (SAI NGHIÊM TRỌNG).

CHECKLIST TỰ KIỂM (BẮT BUỘC THỰC HIỆN TRƯỚC KHI TRẢ OUTPUT):
- (C01) Bảo toàn số liệu/đơn vị/dấu so sánh/token @@...@@
- (C02) Không thêm dữ kiện/khẳng định ngoài input
- (C03) Bảo toàn phủ định (không/không có), điều kiện (nếu/thì), mức độ (có thể/chắc chắn)
- (C04) Bảo toàn laterality (trái/phải), vị trí, mốc thời gian
- (C05) Thuật ngữ y khoa chính xác, không tự suy đoán
- (C06) Mỗi đoạn phải có source_timestamps truy vết

ĐẦU VÀO:
Bạn sẽ nhận JSON có dạng:
{
  "mode": "STEP3_CONVERT_TO_CONTINUOUS_TEXT",
  "input": { "refined_script": [ { "speaker","start_timestamp","end_timestamp","text","source_timestamps","needs_review" } ] }
}

ĐẦU RA (JSON HỢP LỆ DUY NHẤT, KHÔNG THÊM CHỮ NGOÀI JSON):
{
  "mode": "STEP3_CONTINUOUS_TEXT_RESULT",
  "draft": { "text": "...", "paragraphs": [ ... ] },
  "meaning_drift_report": { "scale_definition": "...", "items": [ ... ], "summary": { ... } },
  "verification": { "overall_pass": true|false, "checks": [ ... ], "fixes_applied_count": 0, "remaining_risks": [], "notes": "..." },
  "final": { "text": "..." }
}

YÊU CẦU QUAN TRỌNG:
- Nếu overall_pass=false: final.text vẫn phải là bản tốt nhất có thể, và các đoạn liên quan phải gắn "(cần kiểm tra)" rõ ràng.
- paragraphs[].risk_level phải phản ánh rủi ro của đoạn đó.
- meaning_drift_report.items chỉ liệt kê các điểm có risk>=3 (để report gọn, đúng trọng tâm).
- Không dùng markdown, không dùng \`\`\`.

BẮT ĐẦU: đọc INPUT JSON và xuất đúng schema.
`;

// FIX: Changed blob type to `globalThis.Blob` to use the native browser Blob, which is expected by FileReader.
const blobToBase64 = (blob: globalThis.Blob): Promise<string> => {
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

export const useChunkProcessor = (file: File | null, modelStep1: string, modelStep2: string) => {
    const [chunks, setChunks] = useState<Chunk[]>([]);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [result, setResult] = useState<TranscriptionOutput | null>(null);
    const [isInitializing, setIsInitializing] = useState(false);
    const [isFinalizing, setIsFinalizing] = useState(false); // State for Step 2
    const [isStep3Running, setIsStep3Running] = useState(false); // State for Step 3
    const [rateLimitEvent, setRateLimitEvent] = useState<RateLimitEvent | null>(null);
    
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
// FIX: Added missing backticks to template literal.
            return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        }
// FIX: Added missing backticks to template literal.
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
        setIsStep3Running(false);
        setResult(null);
        setLogs([]);
        setChunks([]);
        setMaxConcurrency(5); 
        step1MetricsLoggedRef.current = false; // Reset log flag
        
        try {
// FIX: Added missing backticks to template literal.
            addLog(`Bắt đầu phân tích file: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`, 'info');
// FIX: Added missing backticks to template literal.
            addLog(`Đang giải mã audio... Quá trình này có thể mất vài giây với file lớn.`, 'info');

            audioBufferRef.current = await decodeAudioData(file);
            const durationSec = audioBufferRef.current.duration;
// FIX: Added missing backticks to template literal.
            addLog(`Giải mã thành công. Tổng thời lượng: ${formatSecondsToTime(durationSec)}`, 'success');

            let chunkSeconds = TARGET_CHUNK_SECONDS;
            // Auto-adjust chunk size to avoid oversized payloads
            while (estimateWavBytes(chunkSeconds) > MAX_INLINE_DATA_BYTES && chunkSeconds > 15) {
                chunkSeconds -= 15;
            }
            if (chunkSeconds !== TARGET_CHUNK_SECONDS) {
// FIX: Added missing backticks to template literal.
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
            
// FIX: Added missing backticks to template literal.
            addLog(`Đã chia audio thành ${totalChunks} phân đoạn (${chunkSeconds}s/chunk, gối lên nhau ${OVERLAP_SECONDS}s).`, 'success');
           

        } catch (error: any) {
// FIX: Added missing backticks to template literal.
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
// FIX: Added missing backticks to template literal.
        addLog(`Đang xử lý phân đoạn ${currentChunk.index + 1}/${statsRef.current.total}... (${formatSecondsToTime(currentChunk.startSec || 0)} - ${formatSecondsToTime(currentChunk.endSec || 0)})`, 'info');

        try {
            const wavBlob = await resampleAndEncodeWav(audioBufferRef.current, currentChunk.startSec!, currentChunk.endSec!);
            const base64Data = await blobToBase64(wavBlob);

            const inputJson = {
                mode: "PROCESS_CHUNK",
                chunk: {
                    chunk_id: chunkId,
                    chunk_index: currentChunk.index,
// FIX: Added missing backticks to template literal.
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
// FIX: Added missing backticks to template literal.
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
// FIX: Added missing backticks to template literal.
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
// FIX: Added missing backticks to template literal.
                 addLog(`Chunk ${currentChunk.index + 1} Warning: ${parsedData.qa_metrics.risk_flags.join(', ')}`, 'warning');
            }

            setStats(prev => ({ ...prev, processing: Math.max(0, prev.processing - 1), completed: prev.completed + 1 }));
// FIX: Added missing backticks to template literal.
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
// FIX: Added missing backticks to template literal.
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
// FIX: Added missing backticks to template literal.
                 addLog(`Lỗi dữ liệu (Chunk ${currentChunk.index + 1}): JSON không hợp lệ. Sẽ tự động thử lại.`, 'error');
            }

            console.error(error);
            setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'failed', error: msg } : c));
            setStats(prev => ({ ...prev, processing: Math.max(0, prev.processing - 1), failed: prev.failed + 1 }));
// FIX: Added missing backticks to template literal.
            addLog(`Lỗi Chunk ${currentChunk.index + 1}: ${msg}`, 'error');
        } 
    }, [addLog, updateStats, modelStep1]);

    // 4. STEP 2: POST-EDIT FINALIZATION (SMART FALLBACK VERSION)
    const performStep2Finalization = useCallback(async (mergedTranscript: ImprovedTranscriptItem[]) => {
        if (isFinalizing || !process.env.API_KEY) return;
        if (!mergedTranscript || mergedTranscript.length === 0) {
            addLog("Chưa có dữ liệu để chạy Step 2.", 'warning');
            return;
        }

        setIsFinalizing(true);
// FIX: Added missing backticks to template literal.
        addLog(`BẮT ĐẦU STEP 2: Tinh chỉnh văn bản (Post-Edit)...`, 'info');

        try {
            const slimTranscript = buildSlimTranscript(mergedTranscript);
            // Sử dụng Constants đã được giảm nhỏ ở bước 1
            const batches = buildBatchesByCharOrCount(slimTranscript, STEP2_TARGET_MAX_CHARS, STEP2_MAX_ITEMS_PER_BATCH);
// FIX: Added missing backticks to template literal.
            addLog(`Đã chia ${slimTranscript.length} items thành ${batches.length} batch nhỏ để tránh quá tải.`, 'info');

            const batchResults: PostEditResult[] = [];
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

            // Danh sách model dự phòng khi gặp lỗi 429
            const fallbackModels = [modelStep2, 'gemini-2.5-pro', 'gemini-2.5-flash'];
            // Loại bỏ trùng lặp nếu modelStep2 trùng với model trong danh sách
            const uniqueModels = [...new Set(fallbackModels)];

            for (let i = 0; i < batches.length; i++) {
                const batchItems = batches[i];
                let batchSuccess = false;
                
                // Thử lần lượt các model
                for (let m = 0; m < uniqueModels.length; m++) {
                    const currentModel = uniqueModels[m];
                    const isLastModel = m === uniqueModels.length - 1;

// FIX: Added missing backticks to template literal.
                    addLog(`Step 2: Đang xử lý Batch ${i + 1}/${batches.length} với model ${currentModel}...`, 'info');

                    try {
                        const result = await runStep2Robust(ai, currentModel, batchItems, addLog);
                        
                        // Kiểm tra nếu kết quả trả về là Fallback do lỗi không phục hồi được (trừ 429 đã catch bên dưới)
                        if (result.mode.includes("FALLBACK") && !result.mode.includes("BATCHED")) {
                             // Nếu fallback nhưng không phải do 429, có thể chấp nhận hoặc thử model khác tùy logic, 
                             // ở đây ta tạm chấp nhận để flow tiếp tục.
                        }

                        batchResults.push(result);
                        batchSuccess = true;
// FIX: Added missing backticks to template literal.
                        addLog(`Step 2: Hoàn thành Batch ${i + 1}/${batches.length}.`, 'success');
                        
                        // Nghỉ nhẹ 2s giữa các batch thành công để tránh spam
                        await sleep(2000); 
                        break; // Thành công thì thoát vòng lặp model, sang batch tiếp theo

                    } catch (error: any) {
                        const msg = error.message || String(error);
                        const isRateLimit = msg.includes('429') || msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED');

                        if (isRateLimit) {
                            if (!isLastModel) {
// FIX: Added missing backticks to template literal.
                                addLog(`Model ${currentModel} bị quá tải (429). Đang chuyển sang model nhẹ hơn: ${uniqueModels[m+1]}...`, 'warning');
                                await sleep(1000); // Nghỉ 1s trước khi switch
                                continue; // Thử model tiếp theo
                            } else {
                                // Nếu đây là model cuối cùng (Flash) mà vẫn lỗi -> Chờ 60s rồi thử lại chính nó
// FIX: Added missing backticks to template literal.
                                addLog(`Tất cả model đều quá tải. Hệ thống sẽ tạm dừng 60s để hồi phục quota...`, 'error');
                                setRateLimitEvent({
                                    active: true,
                                    step: 'STEP2',
                                    message: "All models exhausted. Cooling down...",
                                    lastModel: currentModel,
                                    at: Date.now()
                                });
                                updateStats({ isCoolingDown: true, cooldownSeconds: 60 });
                                await sleep(60000); // Chờ cứng 60s
                                
                                // Sau khi chờ, thử lại chính model này một lần nữa (bằng cách giảm index m đi 1 để vòng lặp lặp lại model này)
// FIX: Added missing backticks to template literal.
                                addLog(`Hết thời gian chờ. Thử lại Batch ${i + 1} với ${currentModel}...`, 'info');
                                m--; 
                                continue;
                            }
                        } else {
                            // Lỗi khác (không phải 429) -> Đã được handle trong runStep2Robust (trả về fallback), 
                            // nhưng nếu nó throw ra đây nghĩa là lỗi nghiêm trọng.
                            console.error("Critical Batch Error:", error);
                            // Nếu lỗi không phải rate limit, ta chấp nhận dùng fallback của runStep2Robust 
                            // (thực tế runStep2Robust không throw trừ khi crash, nó return fallback)
                            break; 
                        }
                    }
                }

                if (!batchSuccess) {
                     // Trường hợp cực đoan: batch này thất bại hoàn toàn dù đã đổi model
// FIX: Added missing backticks to template literal.
                     addLog(`Batch ${i+1} thất bại hoàn toàn. Sử dụng dữ liệu gốc làm fallback.`, 'error');
                     batchResults.push(buildStep2Fallback(batchItems, "ALL_MODELS_FAILED"));
                }
            }

            const finalResult = mergePostEditResults(batchResults);
            setResult(prev => prev ? { ...prev, post_edit_result: finalResult } : null);

            // LOG METRICS
            const s1Words = computeStep1WordCount(mergedTranscript);
            const s2Words = computeStep2WordCount(finalResult.refined_script);
            const delta = s2Words - s1Words;
            const deltaPercent = s1Words > 0 ? (delta / s1Words * 100).toFixed(1) : "0.0";
// FIX: Added missing backticks to template literal.
            addLog(`STEP 2 METRICS: segments=${finalResult.refined_script.length}, words=${s2Words}, deltaWords=${delta}, deltaPercent=${deltaPercent}%`, 'success');
// FIX: Added missing backticks to template literal.
            addLog(`Hoàn tất Step 2: Đã tạo văn bản chuyên nghiệp.`, 'success');

        } catch (error: any) {
            console.error("Step 2 Finalization Error:", error);
// FIX: Added missing backticks to template literal.
            addLog(`Lỗi không xác định ở Step 2: ${error.message}`, 'error');
        } finally {
            setIsFinalizing(false);
        }
    }, [addLog, isFinalizing, updateStats, modelStep2]);

    const performStep3Finalization = useCallback(async (refinedScript: RefinedScriptItem[]) => {
        if (isStep3Running || !process.env.API_KEY) return;
        if (!refinedScript || refinedScript.length === 0) {
            addLog("Chưa có dữ liệu Step 2 để chạy Step 3.", 'warning');
            return;
        }

        setIsStep3Running(true);
// FIX: Added missing backticks to template literal.
        addLog(`BẮT ĐẦU STEP 3: Tạo văn bản liên tục và kiểm định...`, 'info');
        
        try {
            // Simple batching for now, can be refined later.
            // Using a higher char limit as Step 3 is less complex than Step 2.
            const batches = buildBatchesByCharOrCount(refinedScript, 35000, 100);
// FIX: Added missing backticks to template literal.
            addLog(`Đã chia ${refinedScript.length} items thành ${batches.length} batch cho Step 3.`, 'info');

            const batchResults: Step3Result[] = [];
            for (let i = 0; i < batches.length; i++) {
// FIX: Added missing backticks to template literal.
                 addLog(`Step 3: Đang xử lý Batch ${i + 1}/${batches.length}...`, 'info');
                 const result = await runStep3Once(batches[i], modelStep2); // Re-use Step 2 model
                 batchResults.push(result);
// FIX: Added missing backticks to template literal.
                 addLog(`Step 3: Hoàn thành Batch ${i + 1}/${batches.length}.`, 'success');
            }

            const finalResult = mergeStep3Results(batchResults);
            setResult(prev => prev ? { ...prev, step3_result: finalResult } : null);
// FIX: Added missing backticks to template literal.
            addLog(`Hoàn tất Step 3. Fidelity Check: ${finalResult.verification.overall_pass ? 'PASS' : 'FAIL'}. High-risk items: ${finalResult.meaning_drift_report.summary.high_risk_count}.`, finalResult.verification.overall_pass ? 'success' : 'warning');
        
        } catch (error: any) {
            console.error("Step 3 Finalization Error:", error);
// FIX: Added missing backticks to template literal.
            addLog(`Lỗi không xác định ở Step 3: ${error.message}`, 'error');
             // Set a fallback result on error
            setResult(prev => prev ? { ...prev, step3_result: buildStep3Fallback(refinedScript, error.message) } : null);
        } finally {
            setIsStep3Running(false);
        }

    }, [addLog, isStep3Running, modelStep2]);


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
// FIX: Added missing backticks to template literal.
                validation_and_conclusion: prev?.validation_and_conclusion || `Đang xử lý ${completedChunks.length}/${chunks.length} đoạn...`,
                professional_medical_text: prev?.professional_medical_text || "Đang chờ hoàn tất để chạy Step 2...",
                post_edit_result: prev?.post_edit_result,
                step3_result: prev?.step3_result
            }));
        }
        
        // LOG METRICS FOR STEP 1 (ONCE)
        if (chunks.length > 0 && completedChunks.length === chunks.length && !step1MetricsLoggedRef.current) {
            step1MetricsLoggedRef.current = true;
            const wordCount = computeStep1WordCount(mergedTranscript);
// FIX: Added missing backticks to template literal.
            addLog(`STEP 1 METRICS: items=${mergedTranscript.length}, words=${wordCount}`, 'success');
        }

    }, [chunks, stats, maxConcurrency, processChunk, addLog]);

    const retryChunk = (chunkId: string) => {
        setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'pending', error: undefined } : c));
        updateStats({ failed: Math.max(0, stats.failed - 1) });
// FIX: Added missing backticks to template literal.
        addLog(`Thử lại chunk ${chunkId}`, 'info');
    };

    const retryAllFailed = () => {
        const failedChunks = chunks.filter(c => c.status === 'failed');
        if (failedChunks.length === 0) return;
        setChunks(prev => prev.map(c => c.status === 'failed' ? { ...c, status: 'pending', error: undefined } : c));
        updateStats({ failed: 0 });
// FIX: Added missing backticks to template literal.
        addLog(`Thử lại ${failedChunks.length} chunks thất bại`, 'warning');
    };

    const reset = () => {
        setChunks([]);
        setResult(null);
        setLogs([]);
        setStats({ total: 0, completed: 0, processing: 0, failed: 0, startTime: 0, isCoolingDown: false, cooldownSeconds: 0 });
        setMaxConcurrency(5);
        setIsFinalizing(false);
        setIsStep3Running(false);
        setRateLimitEvent(null);
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
                    post_edit_result: undefined,
                    step3_result: undefined // Invalidate Step 3 result
                };
            });
// FIX: Added missing backticks to template literal.
            addLog(`Đã nối thêm ${newItems.length} dòng dữ liệu thủ công.`, 'success');
        } catch (e: any) {
// FIX: Added missing backticks to template literal.
            addLog(`Lỗi import: ${e.message}`, 'error');
            throw e;
        }
    };

    const triggerStep2 = () => {
        if (result?.improved_transcript && result.improved_transcript.length > 0) {
            performStep2Finalization(result.improved_transcript);
        } else {
            addLog("Không có dữ liệu Transcript để chạy Step 2.", 'error');
        }
    };

    const triggerStep3 = () => {
        if (result?.post_edit_result?.refined_script && result.post_edit_result.refined_script.length > 0) {
            performStep3Finalization(result.post_edit_result.refined_script);
        } else {
            addLog("Không có dữ liệu Step 2 (Refined Script) để chạy Step 3.", 'error');
        }
    };

    return {
        chunks, isInitializing, stats, logs, result, isFinalizing,
        fileType: 'audio', // Hardcoded to audio as we no longer support PDF-like byte slicing
        retryChunk, retryAllFailed, initializeChunks, reset,
        triggerStep2, manualAppendTranscript,
        rateLimitEvent, clearCooldownNow,
        triggerStep3, isStep3Running
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
    items: any[],
    targetMaxChars: number,
    maxItems: number
): any[][] => {
    if (items.length === 0) return [];
    const batches: any[][] = [];
    let currentBatch: any[] = [];
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
// FIX: Added missing backticks to template literal.
        setTimeout(() => reject(new Error(`Timeout: Step 2 batch không phản hồi sau ${STEP2_TIMEOUT_MS / 60000} phút`)), STEP2_TIMEOUT_MS)
    );

    const response: any = await Promise.race([apiCall, timeoutPromise]);
    const rawText = extractGenAIText(response);

    if (!rawText) {
        throw new Error("Internal error: Step 2 empty response text");
    }

    const parseResult = safeParseJSON(rawText);
    if (parseResult.ok === false) {
// FIX: Added missing backticks and quotes to template literal for correct error message formatting.
        throw new Error(`Invalid JSON: ${parseResult.reason}. Pos: ${parseResult.pos}, Context: "${parseResult.context}"`);
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
// FIX: Added missing backticks to template literal.
                    addLog(`Step 2 batch error. Attempt ${i + 1}/${STEP2_MAX_RETRIES} failed. Retrying in ${backoff / 1000}s. Details: ${msg}`, 'warning');
                    await sleep(backoff);
                    continue;
                }
// FIX: Added missing backticks to template literal.
                addLog(`Step 2 batch failed after ${STEP2_MAX_RETRIES} retries. Final error: ${msg}`, 'error');
            } else {
// FIX: Added missing backticks to template literal.
                 addLog(`Lỗi không thể phục hồi ở Step 2 (batch): ${msg}`, 'error');
                 return buildStep2Fallback(items, "UNRECOVERABLE_ERROR");
            }
        }
    }
    
    // If all retries failed, try splitting
    if (items.length > STEP2_MIN_SPLIT_ITEMS && depth < STEP2_MAX_SPLIT_DEPTH) {
// FIX: Added missing backticks to template literal.
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
// FIX: Added missing backticks to template literal.
             addLog(`Chia nhỏ batch cũng thất bại: ${splitError.message}. Sử dụng Fallback.`, 'error');
             return buildStep2Fallback(items, "SPLIT_AND_RETRY_FAILED");
        }
    }
    
// FIX: Added missing backticks to template literal.
    addLog(`Không thể xử lý batch ngay cả sau khi thử lại và chia nhỏ. Sử dụng Fallback.`, 'error');
    return buildStep2Fallback(items, "MAX_RETRIES_AND_SPLITS_FAILED");
}

// --- Step 3 Helpers ---

const buildStep3Fallback = (refinedItems: RefinedScriptItem[], reason: string): Step3Result => {
  const joined = refinedItems
    .map(x => (x?.text || "").trim())
    .filter(Boolean)
    .join("\n\n");

  return {
    mode: "STEP3_CONTINUOUS_TEXT_RESULT_FALLBACK",
    draft: { text: joined, paragraphs: [] },
    meaning_drift_report: {
      scale_definition: "Fallback: không chấm rủi ro bằng AI do lỗi kỹ thuật.",
      items: [],
      summary: { total: 0, max_risk: 5, high_risk_count: 0 }
    },
    verification: {
      overall_pass: false,
      checks: [
        { id: "C00", name: "Fallback used", pass: false, notes: reason }
      ],
      fixes_applied_count: 0,
      remaining_risks: ["FALLBACK_USED"],
// FIX: Added missing backticks to template literal.
      notes: `Step 3 fallback do: ${reason}`
    },
    final: { text: joined }
  };
};

const runStep3Once = async (
    batchRefinedItems: RefinedScriptItem[],
    model: string,
): Promise<Step3Result> => {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const inputData = { mode: "STEP3_CONVERT_TO_CONTINUOUS_TEXT", input: { refined_script: batchRefinedItems } };
    const fullPrompt = `${SYSTEM_PROMPT_STEP_3}\n\nINPUT JSON:\n${JSON.stringify(inputData)}`;

    const step3Schema = {
      type: Type.OBJECT,
      properties: {
        mode: { type: Type.STRING },
        draft: {
          type: Type.OBJECT,
          properties: {
            text: { type: Type.STRING },
            paragraphs: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  heading: { type: Type.STRING, nullable: true },
                  subheading: { type: Type.STRING, nullable: true },
                  text: { type: Type.STRING },
                  source_timestamps: { type: Type.ARRAY, items: { type: Type.STRING } },
                  risk_level: { type: Type.NUMBER },
                  needs_review: { type: Type.BOOLEAN, nullable: true }
                },
                required: ["id","text","source_timestamps","risk_level"]
              }
            }
          },
          required: ["text","paragraphs"]
        },
        meaning_drift_report: {
          type: Type.OBJECT,
          properties: {
            scale_definition: { type: Type.STRING },
            items: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  risk_level: { type: Type.NUMBER },
                  type: { type: Type.STRING },
                  draft_excerpt: { type: Type.STRING },
                  issue: { type: Type.STRING },
                  suggested_fix: { type: Type.STRING },
                  applied_fix: { type: Type.STRING, nullable: true },
                  source_timestamps: { type: Type.ARRAY, items: { type: Type.STRING } }
                },
                required: ["id","risk_level","type","draft_excerpt","issue","suggested_fix","source_timestamps"]
              }
            },
            summary: {
              type: Type.OBJECT,
              properties: {
                total: { type: Type.NUMBER },
                max_risk: { type: Type.NUMBER },
                high_risk_count: { type: Type.NUMBER }
              },
              required: ["total","max_risk","high_risk_count"]
            }
          },
          required: ["scale_definition","items","summary"]
        },
        verification: {
          type: Type.OBJECT,
          properties: {
            overall_pass: { type: Type.BOOLEAN },
            checks: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  name: { type: Type.STRING },
                  pass: { type: Type.BOOLEAN },
                  notes: { type: Type.STRING, nullable: true }
                },
                required: ["id","name","pass"]
              }
            },
            fixes_applied_count: { type: Type.NUMBER },
            remaining_risks: { type: Type.ARRAY, items: { type: Type.STRING } },
            notes: { type: Type.STRING }
          },
          required: ["overall_pass","checks","fixes_applied_count","remaining_risks","notes"]
        },
        final: {
          type: Type.OBJECT,
          properties: { text: { type: Type.STRING } },
          required: ["text"]
        }
      },
      required: ["mode","draft","meaning_drift_report","verification","final"]
    };

    try {
        const response = await ai.models.generateContent({
            model,
            contents: { parts: [{ text: fullPrompt }] },
            config: {
                temperature: 0.2,
                responseMimeType: "application/json",
                responseSchema: step3Schema,
                safetySettings: [
                    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                ]
            }
        });

        const rawText = response.text;
        if (!rawText) throw new Error("Empty response from AI for Step 3");
        
        const parseResult = safeParseJSON(rawText);
        if (parseResult.ok === false) {
// FIX: Added missing backticks to template literal.
             throw new Error(`Invalid JSON for Step 3: ${parseResult.reason}.`);
        }
        return parseResult.value as Step3Result;

    } catch (e: any) {
        console.error("Error in runStep3Once", e);
        return buildStep3Fallback(batchRefinedItems, e.message);
    }
};

const mergeStep3Results = (results: Step3Result[]): Step3Result => {
    if (results.length === 0) return buildStep3Fallback([], "NO_RESULTS_TO_MERGE");
    if (results.length === 1) return results[0];

    const initial: Step3Result = {
        mode: "STEP3_CONTINUOUS_TEXT_RESULT_BATCHED",
        draft: { text: "", paragraphs: [] },
        meaning_drift_report: {
            scale_definition: results[0].meaning_drift_report.scale_definition,
            items: [],
            summary: { total: 0, max_risk: 1, high_risk_count: 0 }
        },
        verification: {
            overall_pass: true,
            checks: [], // Will take from the first result
            fixes_applied_count: 0,
            remaining_risks: [],
            notes: ""
        },
        final: { text: "" }
    };

    const merged = results.reduce((acc, current, index) => {
        acc.draft.text += (index > 0 ? "\n\n" : "") + current.draft.text;
        acc.final.text += (index > 0 ? "\n\n" : "") + current.final.text;

        current.draft.paragraphs.forEach(p => {
            acc.draft.paragraphs.push({ ...p, id: `B${index + 1}-${p.id}` });
        });

        current.meaning_drift_report.items.forEach(item => {
            acc.meaning_drift_report.items.push({ ...item, id: `B${index + 1}-${item.id}` });
        });
        
        acc.meaning_drift_report.summary.total += current.meaning_drift_report.summary.total;
        acc.meaning_drift_report.summary.max_risk = Math.max(acc.meaning_drift_report.summary.max_risk, current.meaning_drift_report.summary.max_risk) as 1|2|3|4|5;
        acc.meaning_drift_report.summary.high_risk_count += current.meaning_drift_report.summary.high_risk_count;

        acc.verification.overall_pass = acc.verification.overall_pass && current.verification.overall_pass;
        acc.verification.fixes_applied_count += current.verification.fixes_applied_count;
        acc.verification.remaining_risks = [...new Set([...acc.verification.remaining_risks, ...current.verification.remaining_risks])];
        acc.verification.notes += (index > 0 ? `\n--- BATCH ${index+1} ---\n` : "") + current.verification.notes;

        if (current.mode.includes("FALLBACK")) {
            acc.verification.overall_pass = false;
            acc.verification.remaining_risks.push("BATCH_FALLBACK_USED");
        }

        return acc;
    }, initial);

    merged.verification.checks = results[0].verification.checks; // Assume checks are the same

    return merged;
};