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

// --- New Types for Step 2 Batching ---
export type BatchStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface Batch {
    id: string;
    index: number;
    items: Partial<ImprovedTranscriptItem>[]; // The actual data for the batch
    status: BatchStatus;
    result?: PostEditResult; // The successful result from this batch
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