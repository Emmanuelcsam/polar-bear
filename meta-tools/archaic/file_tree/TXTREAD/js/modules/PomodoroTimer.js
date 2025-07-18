// PomodoroTimer.js - Pomodoro timer functionality
export class PomodoroTimer {
    constructor() {
        this.duration = 25 * 60; // 25 minutes in seconds
        this.breakDuration = 5 * 60; // 5 minutes in seconds
        this.longBreakDuration = 15 * 60; // 15 minutes in seconds
        this.currentTime = this.duration;
        this.isRunning = false;
        this.isBreak = false;
        this.sessionCount = 0;
        this.interval = null;
        this.onTick = null;
        this.onComplete = null;
    }

    start(onTick, onComplete) {
        if (this.isRunning) return;

        this.isRunning = true;
        this.onTick = onTick;
        this.onComplete = onComplete;

        this.interval = setInterval(() => {
            if (this.currentTime > 0) {
                this.currentTime--;
                if (this.onTick) {
                    this.onTick(this.getTimeDisplay(), this.getProgress());
                }
            } else {
                this.complete();
            }
        }, 1000);
    }

    pause() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
        this.isRunning = false;
    }

    resume() {
        if (!this.isRunning && this.currentTime > 0) {
            this.start(this.onTick, this.onComplete);
        }
    }

    reset() {
        this.pause();
        this.currentTime = this.isBreak ? this.getBreakDuration() : this.duration;
        if (this.onTick) {
            this.onTick(this.getTimeDisplay(), this.getProgress());
        }
    }

    complete() {
        this.pause();
        
        if (!this.isBreak) {
            this.sessionCount++;
        }

        if (this.onComplete) {
            const message = this.isBreak ? 
                'Break time is over! Ready to focus?' : 
                'Pomodoro session complete! Time for a break.';
            this.onComplete(message, this.isBreak);
        }

        // Switch between work and break
        this.isBreak = !this.isBreak;
        this.currentTime = this.isBreak ? this.getBreakDuration() : this.duration;
    }

    getBreakDuration() {
        // Every 4th session gets a long break
        return this.sessionCount > 0 && this.sessionCount % 4 === 0 ? 
            this.longBreakDuration : this.breakDuration;
    }

    setDuration(minutes) {
        this.duration = minutes * 60;
        if (!this.isBreak && !this.isRunning) {
            this.currentTime = this.duration;
        }
    }

    setBreakDuration(minutes) {
        this.breakDuration = minutes * 60;
    }

    setLongBreakDuration(minutes) {
        this.longBreakDuration = minutes * 60;
    }

    getTimeDisplay() {
        const minutes = Math.floor(this.currentTime / 60);
        const seconds = this.currentTime % 60;
        return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    getProgress() {
        const total = this.isBreak ? this.getBreakDuration() : this.duration;
        return ((total - this.currentTime) / total) * 100;
    }

    getSessionCount() {
        return this.sessionCount;
    }

    isWorkSession() {
        return !this.isBreak;
    }

    getStatus() {
        if (!this.isRunning && this.currentTime === this.duration) {
            return 'Ready to start';
        } else if (this.isRunning) {
            return this.isBreak ? 'Break time' : 'Focus time';
        } else {
            return 'Paused';
        }
    }

    skipToNext() {
        this.pause();
        this.currentTime = 0;
        this.complete();
    }

    getTotalFocusTime() {
        // Returns total focus time in minutes
        return (this.sessionCount * this.duration) / 60;
    }

    getSessionStats() {
        return {
            completedSessions: this.sessionCount,
            totalFocusMinutes: this.getTotalFocusTime(),
            currentSession: this.isBreak ? 'Break' : 'Focus',
            timeRemaining: this.getTimeDisplay(),
            progress: this.getProgress()
        };
    }
}