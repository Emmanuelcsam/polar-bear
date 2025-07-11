// AutoScroller.js - Automatic scrolling functionality
export class AutoScroller {
    constructor() {
        this.isScrolling = false;
        this.isPaused = false;
        this.speed = 50; // pixels per second
        this.interval = null;
        this.targetElement = null;
    }

    setTarget(element) {
        this.targetElement = element;
    }

    start() {
        if (!this.targetElement) {
            throw new Error('No target element set for autoscrolling');
        }

        this.isScrolling = true;
        this.isPaused = false;

        // Calculate scroll increment based on speed (pixels per second)
        const scrollIncrement = this.speed / 60; // 60fps

        this.interval = setInterval(() => {
            if (!this.isPaused) {
                const currentScroll = this.targetElement.scrollTop;
                const maxScroll = this.targetElement.scrollHeight - this.targetElement.clientHeight;

                if (currentScroll < maxScroll) {
                    this.targetElement.scrollTop += scrollIncrement;
                } else {
                    // Reached the end
                    this.stop();
                    if (this.onComplete) this.onComplete();
                }
            }
        }, 1000 / 60); // 60fps for smooth scrolling
    }

    pause() {
        this.isPaused = true;
    }

    resume() {
        this.isPaused = false;
    }

    stop() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
        this.isScrolling = false;
        this.isPaused = false;
    }

    reset() {
        this.stop();
        if (this.targetElement) {
            this.targetElement.scrollTop = 0;
        }
    }

    setSpeed(pixelsPerSecond) {
        this.speed = Math.max(10, Math.min(200, pixelsPerSecond));
        
        // If currently scrolling, restart with new speed
        if (this.isScrolling && !this.isPaused) {
            this.stop();
            this.start();
        }

        return this.speed;
    }

    changeSpeed(delta) {
        return this.setSpeed(this.speed + delta);
    }

    getProgress() {
        if (!this.targetElement) return 0;
        
        const currentScroll = this.targetElement.scrollTop;
        const maxScroll = this.targetElement.scrollHeight - this.targetElement.clientHeight;
        
        if (maxScroll <= 0) return 100;
        return (currentScroll / maxScroll) * 100;
    }

    jumpToPosition(percentage) {
        if (!this.targetElement) return;
        
        const maxScroll = this.targetElement.scrollHeight - this.targetElement.clientHeight;
        const targetScroll = (percentage / 100) * maxScroll;
        
        this.targetElement.scrollTop = targetScroll;
    }

    scrollToElement(element) {
        if (!element || !this.targetElement) return;
        
        const elementTop = element.offsetTop;
        const containerTop = this.targetElement.offsetTop;
        const scrollPosition = elementTop - containerTop;
        
        this.targetElement.scrollTop = scrollPosition;
    }

    getTimeRemaining() {
        if (!this.targetElement || this.speed === 0) return 0;
        
        const currentScroll = this.targetElement.scrollTop;
        const maxScroll = this.targetElement.scrollHeight - this.targetElement.clientHeight;
        const remainingPixels = maxScroll - currentScroll;
        
        return Math.ceil(remainingPixels / this.speed);
    }

    isAtEnd() {
        if (!this.targetElement) return false;
        
        const currentScroll = this.targetElement.scrollTop;
        const maxScroll = this.targetElement.scrollHeight - this.targetElement.clientHeight;
        
        return currentScroll >= maxScroll - 1; // Allow 1px tolerance
    }

    isAtStart() {
        if (!this.targetElement) return true;
        return this.targetElement.scrollTop <= 1; // Allow 1px tolerance
    }
}