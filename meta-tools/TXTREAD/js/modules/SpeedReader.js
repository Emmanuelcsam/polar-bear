// SpeedReader.js - Speed reading functionality
export class SpeedReader {
    constructor() {
        this.words = [];
        this.currentWordIndex = 0;
        this.wordsPerMinute = 300;
        this.isReading = false;
        this.interval = null;
    }

    setWords(text) {
        // Remove markdown and special characters for cleaner speed reading
        const cleanText = text.replace(/[#*`\[\]()]/g, '');
        this.words = cleanText.split(/\s+/).filter(word => word.length > 0);
        this.currentWordIndex = 0;
    }

    start(onWordChange, onComplete) {
        if (this.words.length === 0) {
            throw new Error('No text loaded for speed reading');
        }

        if (this.currentWordIndex >= this.words.length) {
            this.currentWordIndex = 0;
        }

        this.isReading = true;
        const interval = 60000 / this.wordsPerMinute;

        this.interval = setInterval(() => {
            if (this.currentWordIndex < this.words.length) {
                onWordChange(this.words[this.currentWordIndex]);
                this.currentWordIndex++;
            } else {
                this.stop();
                if (onComplete) onComplete();
            }
        }, interval);
    }

    stop() {
        this.isReading = false;
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
    }

    pause() {
        this.stop();
    }

    resume(onWordChange, onComplete) {
        if (!this.isReading && this.currentWordIndex < this.words.length) {
            this.start(onWordChange, onComplete);
        }
    }

    changeSpeed(delta) {
        this.wordsPerMinute = Math.max(100, Math.min(800, this.wordsPerMinute + delta));
        
        // If currently reading, restart with new speed
        if (this.isReading) {
            const onWordChange = this._currentOnWordChange;
            const onComplete = this._currentOnComplete;
            this.stop();
            this.start(onWordChange, onComplete);
        }

        return this.wordsPerMinute;
    }

    setSpeed(wpm) {
        this.wordsPerMinute = Math.max(100, Math.min(800, wpm));
        return this.wordsPerMinute;
    }

    reset() {
        this.stop();
        this.currentWordIndex = 0;
    }

    getProgress() {
        if (this.words.length === 0) return 0;
        return (this.currentWordIndex / this.words.length) * 100;
    }

    getTimeRemaining() {
        if (this.words.length === 0 || this.wordsPerMinute === 0) return 0;
        const wordsRemaining = this.words.length - this.currentWordIndex;
        return Math.ceil(wordsRemaining / this.wordsPerMinute);
    }

    getCurrentWord() {
        if (this.currentWordIndex < this.words.length) {
            return this.words[this.currentWordIndex];
        }
        return '';
    }

    skipWords(count) {
        this.currentWordIndex = Math.max(0, Math.min(this.words.length - 1, this.currentWordIndex + count));
    }

    jumpToPosition(percentage) {
        const position = Math.floor((percentage / 100) * this.words.length);
        this.currentWordIndex = Math.max(0, Math.min(this.words.length - 1, position));
    }
}