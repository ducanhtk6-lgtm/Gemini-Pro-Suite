import React, { useState } from 'react';
import { GoogleGenAI } from '@google/genai';

const LoaderIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="animate-spin"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>;
const SparklesIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9a2 2 0 0 1 2 2a6 6 0 0 0-9-9a2 2 0 0 1-2-2Z"/><path d="M3 12a6 6 0 0 0 9 9a2 2 0 0 1 2 2a6 6 0 0 0-9-9a2 2 0 0 1-2-2Z"/></svg>;
const AlertTriangleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>;

type ImageSize = '1K' | '2K' | '4K';

export const ImageGenerator: React.FC = () => {
    const [prompt, setPrompt] = useState<string>('A photorealistic portrait of an astronaut in a field of surreal, glowing flowers on an alien planet, cinematic lighting.');
    const [imageSize, setImageSize] = useState<ImageSize>('1K');
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const handleGenerate = async () => {
        if (!prompt.trim() || !process.env.API_KEY) {
            setError('Please enter a prompt and ensure your API key is configured.');
            return;
        }

        setIsLoading(true);
        setError(null);
        setGeneratedImage(null);

        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const response = await ai.models.generateContent({
                model: 'gemini-3-pro-image-preview',
                contents: {
                    parts: [{ text: prompt }],
                },
                config: {
                    imageConfig: {
                        imageSize: imageSize,
                        aspectRatio: "1:1" // Keeping it fixed for simplicity
                    },
                },
            });
            
            let foundImage = false;
            if (response.candidates && response.candidates.length > 0) {
                 for (const part of response.candidates[0].content.parts) {
                    if (part.inlineData) {
                        const base64EncodeString: string = part.inlineData.data;
                        const imageUrl = `data:image/png;base64,${base64EncodeString}`;
                        setGeneratedImage(imageUrl);
                        foundImage = true;
                        break;
                    }
                }
            }

            if (!foundImage) {
                throw new Error("No image data found in the API response.");
            }

        } catch (err: any) {
            console.error("Image generation failed:", err);
            setError(err.message || 'An unexpected error occurred during image generation.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="w-full max-w-5xl mx-auto flex flex-col md:flex-row gap-8">
            {/* Left Column: Controls */}
            <div className="md:w-1/3 flex flex-col gap-6">
                <div className="bg-gray-900/50 border border-gray-800 p-4 rounded-2xl shadow-lg">
                    <label htmlFor="prompt-input" className="block text-sm font-bold uppercase text-gray-400 mb-3 tracking-wider">
                        Prompt
                    </label>
                    <textarea
                        id="prompt-input"
                        rows={6}
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-600 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-purple-500 focus:border-purple-500 sm:text-sm custom-scrollbar"
                        placeholder="e.g., A futuristic city skyline at sunset..."
                    />
                </div>

                <div className="bg-gray-900/50 border border-gray-800 p-4 rounded-2xl shadow-lg">
                    <label className="block text-sm font-bold uppercase text-gray-400 mb-3 tracking-wider">
                        Image Size
                    </label>
                    <div className="flex flex-col gap-2">
                        {(['1K', '2K', '4K'] as ImageSize[]).map((size) => (
                            <label key={size} className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg border border-gray-700 cursor-pointer hover:bg-gray-800 transition-colors">
                                <input
                                    type="radio"
                                    name="imageSize"
                                    value={size}
                                    checked={imageSize === size}
                                    onChange={() => setImageSize(size)}
                                    className="h-4 w-4 text-purple-600 bg-gray-700 border-gray-600 focus:ring-purple-500"
                                />
                                <span className="font-semibold">{size}</span>
                                <span className="text-xs text-gray-500 ml-auto">{size === '1K' ? '1024x1024' : size === '2K' ? '2048x2048' : '4096x4096'}</span>
                            </label>
                        ))}
                    </div>
                </div>

                 <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="w-full py-4 bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-all font-bold shadow-lg shadow-purple-900/30 flex items-center justify-center gap-2 disabled:bg-gray-600 disabled:cursor-not-allowed"
                >
                    {isLoading ? <LoaderIcon /> : <SparklesIcon />}
                    {isLoading ? 'Generating...' : 'Generate Image'}
                </button>
            </div>

            {/* Right Column: Display */}
            <div className="md:w-2/3 flex-1">
                 <div className="w-full aspect-square bg-black/40 rounded-2xl border-2 border-dashed border-gray-800 flex items-center justify-center p-4 relative overflow-hidden">
                    {isLoading && (
                        <div className="flex flex-col items-center gap-4 text-purple-300">
                            <LoaderIcon />
                            <span className="animate-pulse">Gemini is creating your image...</span>
                        </div>
                    )}
                    {error && !isLoading && (
                         <div className="text-center text-red-400 flex flex-col items-center gap-2 p-4">
                            <AlertTriangleIcon />
                            <h3 className="font-bold">Generation Failed</h3>
                            <p className="text-sm opacity-80">{error}</p>
                        </div>
                    )}
                    {!isLoading && !error && generatedImage && (
                        <img src={generatedImage} alt={prompt} className="w-full h-full object-contain rounded-lg" />
                    )}
                     {!isLoading && !error && !generatedImage && (
                        <div className="text-center text-gray-600">
                            <h3 className="text-lg font-bold mb-2">Image will appear here</h3>
                            <p className="text-sm">Enter your prompt and click "Generate Image" to begin.</p>
                        </div>
                    )}
                 </div>
            </div>
        </div>
    );
};
