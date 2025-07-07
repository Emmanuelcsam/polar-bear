// ReadingStats.js - Reading statistics and progress tracking
export class ReadingStats {
    constructor() {
        this.wordCount = 0;
        this.characterCount = 0;
        this.readingTime = 0;
        this.averageWPM = 250; // Average reading speed
        this.sessionStartTime = Date.now();
        this.sessionTime = 0;
        this.readingProgress = 0;
        this.interval = null;
    }

    startSession() {
        this.sessionStartTime = Date.now();
        this.sessionTime = 0;
        
        // Update session time every second
        this.interval = setInterval(() => {
            this.updateSessionTime();
        }, 1000);
    }

    stopSession() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
    }

    updateSessionTime() {
        const elapsed = Date.now() - this.sessionStartTime;
        this.sessionTime = Math.floor(elapsed / 1000); // Convert to seconds
    }

    analyzeText(text) {
        if (!text) {
            this.wordCount = 0;
            this.characterCount = 0;
            this.readingTime = 0;
            return;
        }

        // Clean text for accurate counting
        const cleanText = text.replace(/[#*`\[\]()]/g, '');
        
        // Count words
        const words = cleanText.split(/\s+/).filter(word => word.length > 0);
        this.wordCount = words.length;
        
        // Count characters (excluding spaces)
        this.characterCount = cleanText.replace(/\s/g, '').length;
        
        // Calculate reading time
        this.readingTime = Math.ceil(this.wordCount / this.averageWPM);
        
        return {
            wordCount: this.wordCount,
            characterCount: this.characterCount,
            readingTime: this.readingTime,
            words: words
        };
    }

    updateProgress(scrollPercentage) {
        this.readingProgress = Math.min(100, Math.max(0, scrollPercentage));
        return this.readingProgress;
    }

    calculateReadingSpeed() {
        if (this.sessionTime === 0 || this.wordCount === 0) return 0;
        
        const minutesElapsed = this.sessionTime / 60;
        const wordsRead = Math.floor((this.readingProgress / 100) * this.wordCount);
        
        return minutesElapsed > 0 ? Math.round(wordsRead / minutesElapsed) : 0;
    }

    getStats() {
        return {
            wordCount: this.wordCount,
            characterCount: this.characterCount,
            readingTime: this.readingTime,
            sessionTime: this.formatTime(this.sessionTime),
            readingProgress: this.readingProgress,
            currentSpeed: this.calculateReadingSpeed(),
            estimatedTimeRemaining: this.getTimeRemaining()
        };
    }

    getTimeRemaining() {
        if (this.readingProgress >= 100) return 0;
        
        const wordsRemaining = Math.floor(((100 - this.readingProgress) / 100) * this.wordCount);
        const currentSpeed = this.calculateReadingSpeed() || this.averageWPM;
        
        return Math.ceil(wordsRemaining / currentSpeed);
    }

    formatTime(seconds) {
        if (seconds < 60) {
            return `${seconds}s`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            return `${minutes}m ${remainingSeconds}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }
    }

    getReadingTimeEstimate() {
        return `${this.readingTime} min`;
    }

    getProgressPercentage() {
        return `${Math.round(this.readingProgress)}%`;
    }

    getWordCount() {
        return this.wordCount.toLocaleString();
    }

    getSessionTime() {
        return this.formatTime(this.sessionTime);
    }

    setAverageWPM(wpm) {
        this.averageWPM = Math.max(100, Math.min(500, wpm));
        // Recalculate reading time with new speed
        this.readingTime = Math.ceil(this.wordCount / this.averageWPM);
    }

    // Advanced statistics
    getDetailedStats() {
        const stats = this.getStats();
        
        return {
            ...stats,
            averageWordLength: this.characterCount > 0 ? 
                (this.characterCount / this.wordCount).toFixed(1) : 0,
            pagesEstimate: Math.ceil(this.wordCount / 250), // Assuming 250 words per page
            sentenceCount: this.estimateSentences(),
            paragraphCount: this.estimateParagraphs(),
            readabilityScore: this.calculateReadability()
        };
    }

    estimateSentences() {
        // This is a simple estimation - in production, use proper sentence detection
        return Math.ceil(this.wordCount / 15); // Average 15 words per sentence
    }

    estimateParagraphs() {
        // Simple estimation - in production, count actual paragraph breaks
        return Math.ceil(this.wordCount / 100); // Average 100 words per paragraph
    }

    calculateReadability() {
        // Simplified readability score (0-100, higher is easier)
        // In production, implement Flesch Reading Ease or similar
        const avgWordLength = this.characterCount / this.wordCount;
        const avgSentenceLength = 15; // Assumed average
        
        // Simple formula: shorter words and sentences = easier to read
        const score = Math.max(0, Math.min(100, 
            100 - (avgWordLength * 10) - (avgSentenceLength * 0.5)
        ));
        
        return Math.round(score);
    }

    exportStats() {
        const stats = this.getDetailedStats();
        
        return {
            ...stats,
            exportedAt: new Date().toISOString(),
            readingSpeed: this.averageWPM,
            sessionDuration: this.sessionTime
        };
    }
}