// PomodoroTimerTests.js - Tests for PomodoroTimer module
import { PomodoroTimer } from '../../js/modules/PomodoroTimer.js';
import { TestSuite } from '../TestSuite.js';

export class PomodoroTimerTests {
    constructor() {
        this.timer = new PomodoroTimer();
    }

    async runTests() {
        const results = [];

        // Test 1: Module initialization
        results.push(TestSuite.assert(
            this.timer instanceof PomodoroTimer,
            'PomodoroTimer instantiation',
            'Failed to create PomodoroTimer instance'
        ));

        // Test 2: Initial state
        results.push(TestSuite.assert(
            this.timer.duration === 25 * 60 &&
            this.timer.breakDuration === 5 * 60 &&
            this.timer.longBreakDuration === 15 * 60 &&
            !this.timer.isRunning &&
            !this.timer.isBreak,
            'Initial state',
            'Incorrect initial state values'
        ));

        // Test 3: Time display
        const timeDisplay = this.timer.getTimeDisplay();
        results.push(TestSuite.assert(
            timeDisplay === '25:00',
            'Time display format',
            `Expected 25:00, got ${timeDisplay}`
        ));

        // Test 4: Set durations
        this.timer.setDuration(30);
        results.push(TestSuite.assert(
            this.timer.duration === 30 * 60,
            'Set work duration',
            `Expected 1800 seconds, got ${this.timer.duration}`
        ));

        this.timer.setBreakDuration(10);
        results.push(TestSuite.assert(
            this.timer.breakDuration === 10 * 60,
            'Set break duration',
            `Expected 600 seconds, got ${this.timer.breakDuration}`
        ));

        this.timer.setLongBreakDuration(20);
        results.push(TestSuite.assert(
            this.timer.longBreakDuration === 20 * 60,
            'Set long break duration',
            `Expected 1200 seconds, got ${this.timer.longBreakDuration}`
        ));

        // Reset to defaults
        this.timer.setDuration(25);
        this.timer.setBreakDuration(5);
        this.timer.setLongBreakDuration(15);

        // Test 5: Progress calculation
        this.timer.currentTime = 20 * 60; // 5 minutes elapsed
        const progress = this.timer.getProgress();
        results.push(TestSuite.assert(
            progress === 20,
            'Progress calculation',
            `Expected 20%, got ${progress}%`
        ));

        // Test 6: Session status
        results.push(TestSuite.assert(
            this.timer.isWorkSession(),
            'Work session detection',
            'Should be in work session'
        ));

        const status = this.timer.getStatus();
        results.push(TestSuite.assert(
            status === 'Ready to start',
            'Status message',
            `Expected 'Ready to start', got '${status}'`
        ));

        // Test 7: Session count
        results.push(TestSuite.assert(
            this.timer.getSessionCount() === 0,
            'Initial session count',
            'Session count should start at 0'
        ));

        // Test 8: Break duration logic
        this.timer.sessionCount = 3;
        const breakDuration = this.timer.getBreakDuration();
        results.push(TestSuite.assert(
            breakDuration === 5 * 60,
            'Regular break duration',
            `Expected 300 seconds, got ${breakDuration}`
        ));

        this.timer.sessionCount = 4;
        const longBreak = this.timer.getBreakDuration();
        results.push(TestSuite.assert(
            longBreak === 15 * 60,
            'Long break after 4 sessions',
            `Expected 900 seconds, got ${longBreak}`
        ));

        // Test 9: Total focus time
        this.timer.sessionCount = 2;
        const totalFocus = this.timer.getTotalFocusTime();
        results.push(TestSuite.assert(
            totalFocus === 50,
            'Total focus time calculation',
            `Expected 50 minutes, got ${totalFocus}`
        ));

        // Test 10: Session stats
        const stats = this.timer.getSessionStats();
        results.push(TestSuite.assert(
            stats.completedSessions === 2 &&
            stats.totalFocusMinutes === 50 &&
            stats.currentSession === 'Focus' &&
            typeof stats.timeRemaining === 'string' &&
            typeof stats.progress === 'number',
            'Session stats structure',
            'Invalid session stats structure'
        ));

        // Test 11: Reset functionality
        this.timer.currentTime = 10 * 60;
        this.timer.reset();
        results.push(TestSuite.assert(
            this.timer.currentTime === this.timer.duration,
            'Reset timer',
            'Failed to reset timer to full duration'
        ));

        // Test 12: Skip to next
        let completeCalled = false;
        this.timer.onComplete = () => { completeCalled = true; };
        
        // Mock the complete method to avoid actual timer
        const originalComplete = this.timer.complete;
        this.timer.complete = function() {
            this.pause();
            if (this.onComplete) {
                this.onComplete('Test complete', this.isBreak);
            }
        };
        
        this.timer.skipToNext();
        
        results.push(TestSuite.assert(
            completeCalled,
            'Skip to next session',
            'Failed to trigger completion callback'
        ));
        
        // Restore original method
        this.timer.complete = originalComplete;

        return results;
    }
}