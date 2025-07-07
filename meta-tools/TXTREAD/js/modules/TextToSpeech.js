// TextToSpeech.js - Text-to-Speech functionality with browser and Edge TTS support
export class TextToSpeech {
    constructor() {
        this.engine = 'browser'; // 'browser' or 'edge'
        this.rate = 1.0;
        this.pitch = 1.0;
        this.volume = 1.0;
        this.voice = null;
        this.utterance = null;
        this.isPlaying = false;
        this.isPaused = false;
        this.currentText = '';
        
        // Edge TTS specific
        this.edgeVoice = 'en-US-JennyNeural';
        this.edgeAudio = null;
        this.edgeAudioContext = null;
        this.isEdgePlaying = false;
        
        // Initialize voices
        this.voices = [];
        this.loadVoices();
    }

    loadVoices() {
        if ('speechSynthesis' in window) {
            // Load voices when they're ready
            const loadVoiceList = () => {
                this.voices = window.speechSynthesis.getVoices();
                if (this.voices.length === 0) {
                    // Voices might not be loaded yet, try again
                    setTimeout(loadVoiceList, 100);
                }
            };
            
            loadVoiceList();
            
            // Also listen for voice change event
            if (window.speechSynthesis.onvoiceschanged !== undefined) {
                window.speechSynthesis.onvoiceschanged = loadVoiceList;
            }
        }
    }

    setEngine(engine) {
        // Stop current playback
        this.stop();
        
        this.engine = engine;
        
        return this.engine;
    }

    setText(text) {
        this.currentText = text;
    }

    // Browser TTS methods
    async startBrowserTTS() {
        if (!this.currentText || !('speechSynthesis' in window)) {
            throw new Error('Text-to-speech is not supported in your browser');
        }

        this.stop(); // Stop any ongoing speech

        this.utterance = new SpeechSynthesisUtterance(this.currentText);
        this.utterance.rate = this.rate;
        this.utterance.pitch = this.pitch;
        this.utterance.volume = this.volume;

        if (this.voice) {
            this.utterance.voice = this.voice;
        }

        this.utterance.onend = () => {
            this.isPlaying = false;
            this.isPaused = false;
            if (this.onEnd) this.onEnd();
        };

        this.utterance.onerror = (event) => {
            console.error('TTS error:', event);
            this.isPlaying = false;
            this.isPaused = false;
            if (this.onError) this.onError(event);
        };

        this.utterance.onpause = () => {
            this.isPaused = true;
        };

        this.utterance.onresume = () => {
            this.isPaused = false;
        };

        window.speechSynthesis.speak(this.utterance);
        this.isPlaying = true;
    }

    pauseBrowserTTS() {
        if (window.speechSynthesis.speaking && !this.isPaused) {
            window.speechSynthesis.pause();
        }
    }

    resumeBrowserTTS() {
        if (window.speechSynthesis.paused) {
            window.speechSynthesis.resume();
        }
    }

    stopBrowserTTS() {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        this.isPlaying = false;
        this.isPaused = false;
    }

    // Edge TTS methods
    async startEdgeTTS() {
        if (!this.currentText) {
            throw new Error('No text to speak');
        }

        try {
            const audioData = await this.generateEdgeTTSAudio(this.currentText, this.edgeVoice);
            await this.playEdgeTTSAudio(audioData);
        } catch (error) {
            console.error('Edge TTS error:', error);
            // Fallback to browser TTS
            this.setEngine('browser');
            await this.startBrowserTTS();
        }
    }

    async generateEdgeTTSAudio(text, voice) {
        // This would normally call the Edge TTS API
        // For now, we'll create a mock implementation
        // In production, this would make an actual API call to Microsoft's service
        
        return new Promise((resolve, reject) => {
            // Simulate API call delay
            setTimeout(() => {
                // In a real implementation, this would return actual audio data
                // For now, we'll reject to trigger fallback to browser TTS
                reject(new Error('Edge TTS API not configured'));
            }, 100);
        });
    }

    async playEdgeTTSAudio(audioBlob) {
        try {
            // Create audio context if not exists
            if (!this.edgeAudioContext) {
                this.edgeAudioContext = new (window.AudioContext || window.webkitAudioContext)();
            }

            // Create audio element
            this.edgeAudio = new Audio(URL.createObjectURL(audioBlob));
            this.edgeAudio.playbackRate = this.rate;

            this.edgeAudio.onended = () => {
                this.isEdgePlaying = false;
                if (this.onEnd) this.onEnd();
            };

            this.edgeAudio.onerror = (e) => {
                console.error('Edge audio playback error:', e);
                this.isEdgePlaying = false;
                if (this.onError) this.onError(e);
            };

            await this.edgeAudio.play();
            this.isEdgePlaying = true;
        } catch (e) {
            console.error('Error playing Edge TTS audio:', e);
            throw e;
        }
    }

    pauseEdgeTTS() {
        if (this.edgeAudio && !this.edgeAudio.paused) {
            this.edgeAudio.pause();
        }
    }

    resumeEdgeTTS() {
        if (this.edgeAudio && this.edgeAudio.paused) {
            this.edgeAudio.play();
        }
    }

    stopEdgeTTS() {
        if (this.edgeAudio) {
            this.edgeAudio.pause();
            this.edgeAudio.currentTime = 0;
            this.isEdgePlaying = false;
        }
    }

    // Unified methods
    async start() {
        if (this.engine === 'browser') {
            await this.startBrowserTTS();
        } else {
            await this.startEdgeTTS();
        }
    }

    pause() {
        if (this.engine === 'browser') {
            this.pauseBrowserTTS();
        } else {
            this.pauseEdgeTTS();
        }
    }

    resume() {
        if (this.engine === 'browser') {
            this.resumeBrowserTTS();
        } else {
            this.resumeEdgeTTS();
        }
    }

    stop() {
        this.stopBrowserTTS();
        this.stopEdgeTTS();
    }

    togglePlayPause() {
        if (this.engine === 'browser') {
            if (window.speechSynthesis.speaking) {
                if (this.isPaused) {
                    this.resume();
                } else {
                    this.pause();
                }
            } else {
                this.start();
            }
        } else {
            if (this.isEdgePlaying) {
                if (this.edgeAudio && this.edgeAudio.paused) {
                    this.resume();
                } else {
                    this.pause();
                }
            } else {
                this.start();
            }
        }
    }

    // Settings methods
    setRate(rate) {
        this.rate = Math.max(0.5, Math.min(2.0, rate));
        
        if (this.utterance) {
            this.utterance.rate = this.rate;
        }
        if (this.edgeAudio) {
            this.edgeAudio.playbackRate = this.rate;
        }
        
        return this.rate;
    }

    setPitch(pitch) {
        this.pitch = Math.max(0.5, Math.min(2.0, pitch));
        
        if (this.utterance) {
            this.utterance.pitch = this.pitch;
        }
        
        return this.pitch;
    }

    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        
        if (this.utterance) {
            this.utterance.volume = this.volume;
        }
        if (this.edgeAudio) {
            this.edgeAudio.volume = this.volume;
        }
        
        return this.volume;
    }

    setVoice(voiceNameOrIndex) {
        if (typeof voiceNameOrIndex === 'number') {
            this.voice = this.voices[voiceNameOrIndex] || null;
        } else {
            this.voice = this.voices.find(v => v.name === voiceNameOrIndex) || null;
        }
        
        return this.voice;
    }

    setEdgeVoice(voiceName) {
        this.edgeVoice = voiceName;
        return this.edgeVoice;
    }

    getVoices() {
        return this.voices;
    }

    getEdgeVoices() {
        // List of available Edge voices
        return [
            { name: 'en-US-JennyNeural', lang: 'en-US', gender: 'Female' },
            { name: 'en-US-GuyNeural', lang: 'en-US', gender: 'Male' },
            { name: 'en-GB-SoniaNeural', lang: 'en-GB', gender: 'Female' },
            { name: 'en-GB-RyanNeural', lang: 'en-GB', gender: 'Male' },
            { name: 'en-AU-NatashaNeural', lang: 'en-AU', gender: 'Female' },
            { name: 'en-AU-WilliamNeural', lang: 'en-AU', gender: 'Male' },
            { name: 'en-CA-ClaraNeural', lang: 'en-CA', gender: 'Female' },
            { name: 'en-CA-LiamNeural', lang: 'en-CA', gender: 'Male' }
        ];
    }

    isSupported() {
        return 'speechSynthesis' in window;
    }

    getStatus() {
        if (this.engine === 'browser') {
            if (window.speechSynthesis.speaking) {
                return this.isPaused ? 'Paused' : 'Playing';
            }
        } else {
            if (this.isEdgePlaying) {
                return this.edgeAudio && this.edgeAudio.paused ? 'Paused' : 'Playing';
            }
        }
        return 'Stopped';
    }

    // Event handlers
    onEnd(callback) {
        this.onEnd = callback;
    }

    onError(callback) {
        this.onError = callback;
    }
}