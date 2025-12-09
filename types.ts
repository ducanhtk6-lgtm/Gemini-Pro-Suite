/**
 * @LOCKED_DATA_CONTRACT
 * Cấu trúc dữ liệu cốt lõi cho ứng dụng Medical Transcript.
 * Bao gồm:
 * - ImprovedTranscriptItem: Chứa trường 'speaker' cho Diarization.
 * - ProcessingStats: Quản lý trạng thái xử lý đa luồng.
 * - TranscriptionOutput: Output tiêu chuẩn (Transcript, Validation, Medical Text).
 */

export interface TranscriptionTurn {
    user: string;
    model: string;
    isFinalizing?: boolean; // UI state indicator
    isFinal: boolean;
    timestamp: string;
}

export interface ImprovedTranscriptItem {
    timestamp: string;
    original: string;
    edited: string;
    uncertain?: boolean;
    chitchat?: boolean;
    speaker?: string; // [GV], [SV], [HC], [??]
}

// --- Step 2 Types (Post-Edit) ---

export interface RefinedScriptItem {
    speaker: string;
    start_timestamp: string;
    end_timestamp: string;
    text: string;
    source_timestamps: string[];
    needs_review?: boolean;
}

export interface RemovalItem {
    timestamp: string;
    speaker: string;
    reason: string;
    verbatim_excerpt: string;
}

export interface RemovalReport {
    removed_from_main: RemovalItem[];
    removal_summary: string;
}

export interface QAAudit {
    gv_coverage_attestation: boolean;
    gv_items_total: number;
    gv_items_used_in_main: number;
    uncertain_items_count: number;
    risk_flags: string[];
    notes: string;
}

export interface PostEditResult {
    mode: string;
    refined_script: RefinedScriptItem[];
    removal_report: RemovalReport;
    qa_audit: QAAudit;
}

// --- Step 3 Types (Continuous Text Generation & Audit) ---
export interface Step3Paragraph {
  id: string;                         // "P01", "P02"...
  heading?: string | null;            // ý lớn (nếu có)
  subheading?: string | null;         // ý nhỏ (nếu có)
  text: string;                       // đoạn văn đã viết
  source_timestamps: string[];        // truy vết: các timestamp từ Step 2
  risk_level: 1 | 2 | 3 | 4 | 5;      // rủi ro sai nghĩa của đoạn
  needs_review?: boolean | null;      // true nếu dựa trên input có needs_review
}

export type Step3DriftType =
  | "number_unit"
  | "negation"
  | "laterality"
  | "entity_term"
  | "temporal"
  | "causality"
  | "uncertainty"
  | "scope_overreach"
  | "other";

export interface Step3MeaningDriftItem {
  id: string;                         // "D01"...
  risk_level: 1 | 2 | 3 | 4 | 5;
  type: Step3DriftType;
  draft_excerpt: string;              // trích đoạn từ bản nháp
  issue: string;                      // mô tả nguy cơ sai nghĩa
  suggested_fix: string;              // gợi ý sửa
  applied_fix?: string | null;        // đã áp dụng sửa gì
  source_timestamps: string[];        // truy vết
}

export interface Step3VerificationCheck {
  id: string;                         // "C01"...
  name: string;                       // tên tiêu chí
  pass: boolean;
  notes?: string | null;
}

export interface Step3Result {
  mode: string; // "STEP3_CONTINUOUS_TEXT_RESULT" | "..._FALLBACK"
  draft: {
    text: string;
    paragraphs: Step3Paragraph[];
  };
  meaning_drift_report: {
    scale_definition: string;         // định nghĩa thang 1–5
    items: Step3MeaningDriftItem[];
    summary: {
      total: number;
      max_risk: 1 | 2 | 3 | 4 | 5;
      high_risk_count: number;        // count risk>=4
    };
  };
  verification: {
    overall_pass: boolean;
    checks: Step3VerificationCheck[];
    fixes_applied_count: number;
    remaining_risks: string[];
    notes: string;
  };
  final: {
    text: string;                     // bản đã sửa (hoặc = draft nếu pass)
  };
}

// --- Dashboard & Chunking Types ---

export type ChunkStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface Chunk {
    id: string;
    index: number;
    status: ChunkStatus;
    content?: string; // Preview nội dung (nếu có)
    startSec?: number; // Time-based start
    endSec?: number; // Time-based end
    error?: string;
}

export interface LogEntry {
    id: string;
    timestamp: number;
    message: string;
    type: 'info' | 'success' | 'error' | 'warning';
}

export interface ProcessingStats {
    total: number;
    completed: number;
    processing: number;
    failed: number;
    startTime: number;
    endTime?: number;
    isCoolingDown: boolean;
    cooldownSeconds: number;
}

export interface TranscriptionOutput {
    improved_transcript: ImprovedTranscriptItem[];
    validation_and_conclusion: string; // Legacy / Fallback
    professional_medical_text: string; // Legacy / Fallback
    post_edit_result?: PostEditResult; // Step 2 Output
    step3_result?: Step3Result; // Step 3 Output
}

// --- New Types for API Overload Recovery ---
export type RateLimitStep = 'STEP1' | 'STEP2';

export interface RateLimitEvent {
    active: boolean;
    step: RateLimitStep;
    message: string;
    lastModel: string;
    at: number;
}


// --- New Types for Metrics & QA Audit ---

export interface DetailedRemovalRow {
    timestamp: string;
    speaker: string;
    status: 'REPORTED' | 'UNREPORTED' | 'USED_BUT_REPORTED' | 'UNKNOWN_TIMESTAMP';
    reason?: string;
    excerpt: string;
    needsReview: boolean; // Medical Risk Flag
}

export interface TimeRange {
    start: string;
    end: string;
    count: number;
    startSec: number;
}

export interface RemovalAuditResult {
    // Word counts
    step1WordCount: number;
    step2WordCount: number;
    deltaWords: number;
    deltaPercent: number;

    // Summary counts
    step1ItemCount: number;
    usedItemCount: number;

    // Audit counts
    actuallyRemovedCount: number;
    reportedRemovedCount: number;
    unreportedDropCount: number;
    usedButReportedCount: number;
    unknownReportedCount: number;
    
    // For rendering the full list
    detailedRows: DetailedRemovalRow[];
    
    // Grouped ranges
    removedRanges: TimeRange[];

    // Reason breakdown
    reasonCounts: Record<string, number>;
}
