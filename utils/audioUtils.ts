/**
 * @LOCKED_AUDIO_UTILITY
 * Bộ công cụ xử lý audio phía client.
 * 1. Decode: Chuyển file thành AudioBuffer.
 * 2. Slice & Resample: Cắt AudioBuffer theo giây, resample về 16kHz mono bằng OfflineAudioContext.
 * 3. Encode WAV: Chuyển AudioBuffer đã resample thành file WAV 16-bit PCM hợp lệ.
 * 4. Dedupe Helper: Chuẩn hóa text để loại bỏ trùng lặp khi merge.
 */

/**
 * Ước tính kích thước file WAV (bytes) dựa trên các thông số.
 * Dùng để kiểm tra trước khi render để tránh tạo buffer quá lớn.
 */
export const estimateWavBytes = (durationSec: number, sampleRate = 16000, channels = 1, bitDepth = 16): number => {
    return durationSec * sampleRate * channels * (bitDepth / 8);
};

/**
 * Giải mã file audio thành AudioBuffer bằng Web Audio API.
 */
export const decodeAudioData = async (file: File): Promise<AudioBuffer> => {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();
    try {
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        return audioBuffer;
    } catch (error) {
        console.error("Error decoding audio data:", error);
        throw new Error("Không thể giải mã file audio. Vui lòng thử chuyển đổi file sang định dạng WAV hoặc MP3/AAC tiêu chuẩn.");
    } finally {
        audioContext.close();
    }
};

/**
 * Mã hóa một AudioBuffer thành định dạng WAV (PCM 16-bit).
 * @param buffer AudioBuffer (đã được resample về 16kHz, mono).
 * @returns Blob chứa dữ liệu file WAV.
 */
const encodeWav = (buffer: AudioBuffer): Blob => {
    const numOfChan = buffer.numberOfChannels;
    const length = buffer.length * numOfChan * 2; // 2 bytes per sample (16-bit)
    const bufferSize = 44 + length;
    const view = new DataView(new ArrayBuffer(bufferSize));
    const channels = [];
    let i, sample;
    let offset = 0;
    let pos = 0;

    // Helper to write string to DataView
    const writeString = (view: DataView, offset: number, string: string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    // RIFF header
    writeString(view, pos, 'RIFF'); pos += 4;
    view.setUint32(pos, 36 + length, true); pos += 4;
    writeString(view, pos, 'WAVE'); pos += 4;

    // fmt chunk
    writeString(view, pos, 'fmt '); pos += 4;
    view.setUint32(pos, 16, true); pos += 4; // Sub-chunk size
    view.setUint16(pos, 1, true); pos += 2;  // Audio format 1=PCM
    view.setUint16(pos, numOfChan, true); pos += 2;
    view.setUint32(pos, buffer.sampleRate, true); pos += 4;
    view.setUint32(pos, buffer.sampleRate * 2 * numOfChan, true); pos += 4; // Byte rate
    view.setUint16(pos, numOfChan * 2, true); pos += 2; // Block align
    view.setUint16(pos, 16, true); pos += 2; // Bits per sample

    // data chunk
    writeString(view, pos, 'data'); pos += 4;
    view.setUint32(pos, length, true); pos += 4;

    // Write the PCM data
    for (i = 0; i < buffer.numberOfChannels; i++) {
        channels.push(buffer.getChannelData(i));
    }

    while (pos < bufferSize) {
        for (i = 0; i < numOfChan; i++) {
            sample = Math.max(-1, Math.min(1, channels[i][offset])); // Clamp
            sample = (sample < 0 ? sample * 0x8000 : sample * 0x7FFF); // Scale to 16-bit
            view.setInt16(pos, sample, true);
            pos += 2;
        }
        offset++;
    }

    return new Blob([view], { type: 'audio/wav' });
};


/**
 * Cắt một đoạn từ AudioBuffer, resample về 16kHz mono, và mã hóa thành WAV.
 * @param fullBuffer AudioBuffer của toàn bộ file.
 * @param startSec Thời gian bắt đầu (giây).
 * @param endSec Thời gian kết thúc (giây).
 * @returns Blob file WAV đã xử lý.
 */
export const resampleAndEncodeWav = async (fullBuffer: AudioBuffer, startSec: number, endSec: number): Promise<Blob> => {
    const targetSampleRate = 16000;
    const duration = endSec - startSec;
    
    const frameCount = fullBuffer.length;
    const startOffset = Math.floor(startSec * fullBuffer.sampleRate);
    const endOffset = Math.floor(endSec * fullBuffer.sampleRate);
    const segmentFrameCount = endOffset - startOffset;

    if (segmentFrameCount <= 0) {
        throw new Error("Invalid audio segment duration.");
    }

    // Create a new buffer for the segment
    const segmentBuffer = new AudioContext().createBuffer(
        fullBuffer.numberOfChannels,
        segmentFrameCount,
        fullBuffer.sampleRate
    );

    // Copy data for each channel
    for (let i = 0; i < fullBuffer.numberOfChannels; i++) {
        const channelData = fullBuffer.getChannelData(i).slice(startOffset, endOffset);
        segmentBuffer.copyToChannel(channelData, i);
    }
    
    // Resample using OfflineAudioContext
    const offlineCtx = new OfflineAudioContext(1, Math.ceil(duration * targetSampleRate), targetSampleRate);
    const source = offlineCtx.createBufferSource();
    source.buffer = segmentBuffer;
    source.connect(offlineCtx.destination);
    source.start();

    const resampledBuffer = await offlineCtx.startRendering();
    
    return encodeWav(resampledBuffer);
};

/**
 * Chuẩn hóa văn bản để so sánh, loại bỏ trùng lặp.
 */
export const normalizeTextForDedupe = (text: string): string => {
    if (!text) return "";
    return text
        .toLowerCase()
        .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()\[\]]/g, "") // remove punctuation
        .replace(/\s{2,}/g, " ") // collapse whitespace
        .trim();
};
