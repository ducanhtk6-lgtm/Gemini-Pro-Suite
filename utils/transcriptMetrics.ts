import { ImprovedTranscriptItem, RefinedScriptItem, RemovalReport, DetailedRemovalRow, RemovalAuditResult, TimeRange } from '../types';

/**
 * Chuẩn hóa văn bản: xóa khoảng trắng thừa và xuống dòng.
 */
export const normalizeText = (s: string): string => {
    if (!s) return "";
    return s.trim().replace(/\s+/g, ' ');
};

/**
 * Đếm số từ trong một chuỗi đã được chuẩn hóa.
 */
export const countWords = (s: string): number => {
    const normalized = normalizeText(s);
    if (normalized === "") return 0;
    return normalized.split(' ').length;
};

/**
 * Chuyển đổi timestamp dạng [HH:MM:SS] hoặc [MM:SS] thành giây.
 */
export const parseTimestampToSeconds = (ts: string): number | null => {
    if (!ts) return null;
    const cleanTs = ts.replace(/[\[\]]/g, '');
    const parts = cleanTs.split(':').map(Number);
    if (parts.some(isNaN)) return null;

    if (parts.length === 3) { // HH:MM:SS
        return parts[0] * 3600 + parts[1] * 60 + parts[2];
    }
    if (parts.length === 2) { // MM:SS
        return parts[0] * 60 + parts[1];
    }
    return null;
};

/**
 * Tính tổng số từ cho kết quả Step 1.
 */
export const computeStep1WordCount = (items: ImprovedTranscriptItem[]): number => {
    if (!items) return 0;
    return items.reduce((total, item) => total + countWords(item.edited || item.original), 0);
};

/**
 * Tính tổng số từ cho kết quả Step 2.
 */
export const computeStep2WordCount = (items: RefinedScriptItem[]): number => {
    if (!items) return 0;
    return items.reduce((total, item) => total + countWords(item.text), 0);
};

// Heuristic để xác định các điểm có rủi ro y khoa
const MED_RISK_REGEX = new RegExp(
    '\\b(\\d|mg|g|ml|mmhg|bpm|mmol\\/l|%|°c|chẩn đoán|điều trị|liều|triệu chứng|cơ chế|phác đồ|biến chứng|xét nghiệm|cận lâm sàng|lâm sàng|dấu hiệu)\\b',
    'i'
);

const checkMedicalRisk = (text: string): boolean => {
    return MED_RISK_REGEX.test(text);
};


/**
 * Thực hiện toàn bộ việc tính toán metrics và audit việc loại bỏ.
 */
export const computeRemovalAudit = (
    step1Items: ImprovedTranscriptItem[],
    step2Items: RefinedScriptItem[],
    removalReport?: RemovalReport
): RemovalAuditResult | null => {
    if (!step1Items || step1Items.length === 0) return null;

    // --- Word Count Metrics ---
    const step1WordCount = computeStep1WordCount(step1Items);
    const step2WordCount = computeStep2WordCount(step2Items);
    const deltaWords = step2WordCount - step1WordCount;
    const deltaPercent = step1WordCount > 0 ? (deltaWords / step1WordCount) * 100 : 0;

    // --- Timestamp Coverage Audit ---
    const step1Map = new Map<string, ImprovedTranscriptItem>(step1Items.map(item => [item.timestamp, item]));
    const reportedMap = new Map<string, any>(removalReport?.removed_from_main.map(item => [item.timestamp, item]) || []);
    
    const usedTimestamps = new Set<string>();
    step2Items.forEach(item => {
        item.source_timestamps?.forEach(ts => usedTimestamps.add(ts));
    });

    const detailedRows: DetailedRemovalRow[] = [];
    const actuallyRemovedTimestamps: string[] = [];
    const reasonCounts: Record<string, number> = {};

    // 1. Phân tích từng item trong Step 1
    for (const [timestamp, step1Item] of step1Map.entries()) {
        const isUsed = usedTimestamps.has(timestamp);
        const isReported = reportedMap.has(timestamp);

        if (!isUsed) { // Bị loại bỏ thực tế
            actuallyRemovedTimestamps.push(timestamp);
            const reportItem = reportedMap.get(timestamp);
            const excerpt = reportItem?.verbatim_excerpt || step1Item.edited || step1Item.original;
            detailedRows.push({
                timestamp,
                speaker: step1Item.speaker || reportItem?.speaker || '[??]',
                status: isReported ? 'REPORTED' : 'UNREPORTED',
                reason: reportItem?.reason,
                excerpt,
                needsReview: checkMedicalRisk(excerpt)
            });
        } else if (isUsed && isReported) { // Được sử dụng nhưng lại bị báo cáo loại bỏ (lỗi logic của model)
             const reportItem = reportedMap.get(timestamp);
             const excerpt = reportItem?.verbatim_excerpt || step1Item.edited || step1Item.original;
             detailedRows.push({
                timestamp,
                speaker: step1Item.speaker || reportItem?.speaker || '[??]',
                status: 'USED_BUT_REPORTED',
                reason: reportItem?.reason,
                excerpt,
                needsReview: checkMedicalRisk(excerpt)
            });
        }
    }

    // 2. Tìm các báo cáo loại bỏ không có trong Step 1 (model tự bịa ra timestamp)
    for (const [timestamp, reportItem] of reportedMap.entries()) {
        if (!step1Map.has(timestamp)) {
             detailedRows.push({
                timestamp,
                speaker: reportItem.speaker || '[??]',
                status: 'UNKNOWN_TIMESTAMP',
                reason: reportItem.reason,
                excerpt: reportItem.verbatim_excerpt,
                needsReview: checkMedicalRisk(reportItem.verbatim_excerpt)
            });
        }
        // Đếm reason
        if(reportItem.reason) {
            reasonCounts[reportItem.reason] = (reasonCounts[reportItem.reason] || 0) + 1;
        }
    }
    
    // Sắp xếp lại danh sách chi tiết
    detailedRows.sort((a, b) => (parseTimestampToSeconds(a.timestamp) || 0) - (parseTimestampToSeconds(b.timestamp) || 0));

    // 3. Nhóm các khoảng thời gian bị loại bỏ (Range Grouping)
    const removedRanges: TimeRange[] = [];
    if (actuallyRemovedTimestamps.length > 0) {
        const sortedSeconds = actuallyRemovedTimestamps
            .map(ts => ({ ts, sec: parseTimestampToSeconds(ts) || -1 }))
            .filter(item => item.sec !== -1)
            .sort((a, b) => a.sec - b.sec);
        
        if (sortedSeconds.length > 0) {
            let currentRange: { start: string, end: string, count: number, startSec: number, endSec: number } = {
                start: sortedSeconds[0].ts,
                end: sortedSeconds[0].ts,
                count: 1,
                startSec: sortedSeconds[0].sec,
                endSec: sortedSeconds[0].sec
            };

            for (let i = 1; i < sortedSeconds.length; i++) {
                if (sortedSeconds[i].sec - currentRange.endSec <= 15) { // Heuristic: 15 giây
                    currentRange.end = sortedSeconds[i].ts;
                    currentRange.endSec = sortedSeconds[i].sec;
                    currentRange.count++;
                } else {
                    removedRanges.push({start: currentRange.start, end: currentRange.end, count: currentRange.count, startSec: currentRange.startSec});
                    currentRange = {
                        start: sortedSeconds[i].ts,
                        end: sortedSeconds[i].ts,
                        count: 1,
                        startSec: sortedSeconds[i].sec,
                        endSec: sortedSeconds[i].sec
                    };
                }
            }
            removedRanges.push({start: currentRange.start, end: currentRange.end, count: currentRange.count, startSec: currentRange.startSec});
            removedRanges.sort((a, b) => a.startSec - b.startSec);
        }
    }

    const unreportedDrops = detailedRows.filter(r => r.status === 'UNREPORTED');
    const usedButReported = detailedRows.filter(r => r.status === 'USED_BUT_REPORTED');
    const unknownReported = detailedRows.filter(r => r.status === 'UNKNOWN_TIMESTAMP');

    return {
        step1WordCount,
        step2WordCount,
        deltaWords,
        deltaPercent,
        step1ItemCount: step1Items.length,
        usedItemCount: usedTimestamps.size,
        actuallyRemovedCount: actuallyRemovedTimestamps.length,
        reportedRemovedCount: reportedMap.size,
        unreportedDropCount: unreportedDrops.length,
        usedButReportedCount: usedButReported.length,
        unknownReportedCount: unknownReported.length,
        detailedRows,
        removedRanges,
        reasonCounts
    };
};