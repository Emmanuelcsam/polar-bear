// SpeedReaderTests.js - Tests for SpeedReader module
import { SpeedReader } from '../../js/modules/SpeedReader.js';
import { TestSuite } from '../TestSuite.js';

export class SpeedReaderTests {
    constructor() {
        this.speedReader = new SpeedReader();
    }

    async runTests() {
        const results = [];

        // Test 1: Module initialization
        results.push(TestSuite.assert(
            this.speedReader instanceof SpeedReader,
            'SpeedReader instantiation',
            'Failed to create SpeedReader instance'
        ));

        // Test 2: Initial state
        results.push(TestSuite.assert(
            this.speedReader.wordsPerMinute === 300 &&
            this.speedReader.currentWordIndex === 0 &&
            this.speedReader.isReading === false,
            'Initial state',
            'Incorrect initial state values'
        ));

        // Test 3: Set words
        const testText = 'The quick brown fox jumps over the lazy dog';
        this.speedReader.setWords(testText);
        results.push(TestSuite.assert(
            this.speedReader.words.length === 9,
            'Set words',
            `Expected 9 words, got ${this.speedReader.words.length}`
        ));

        // Test 4: Speed adjustment
        const newSpeed = this.speedReader.changeSpeed(50);
        results.push(TestSuite.assert(
            newSpeed === 350,
            'Speed increase',
            `Expected 350 WPM, got ${newSpeed}`
        ));

        const minSpeed = this.speedReader.setSpeed(50);
        results.push(TestSuite.assert(
            minSpeed === 100,
            'Speed minimum limit',
            `Expected 100 WPM minimum, got ${minSpeed}`
        ));

        const maxSpeed = this.speedReader.setSpeed(900);
        results.push(TestSuite.assert(
            maxSpeed === 800,
            'Speed maximum limit',
            `Expected 800 WPM maximum, got ${maxSpeed}`
        ));

        // Reset speed
        this.speedReader.setSpeed(300);

        // Test 5: Progress tracking
        this.speedReader.currentWordIndex = 5;
        const progress = this.speedReader.getProgress();
        results.push(TestSuite.assert(
            Math.round(progress) === 56,
            'Progress calculation',
            `Expected ~56%, got ${Math.round(progress)}%`
        ));

        // Test 6: Time remaining
        this.speedReader.currentWordIndex = 0;
        const timeRemaining = this.speedReader.getTimeRemaining();
        results.push(TestSuite.assert(
            timeRemaining === 1,
            'Time remaining calculation',
            `Expected 1 minute, got ${timeRemaining}`
        ));

        // Test 7: Current word
        const currentWord = this.speedReader.getCurrentWord();
        results.push(TestSuite.assert(
            currentWord === 'The',
            'Get current word',
            `Expected 'The', got '${currentWord}'`
        ));

        // Test 8: Skip words
        this.speedReader.skipWords(3);
        results.push(TestSuite.assert(
            this.speedReader.currentWordIndex === 3,
            'Skip words forward',
            `Expected index 3, got ${this.speedReader.currentWordIndex}`
        ));

        this.speedReader.skipWords(-2);
        results.push(TestSuite.assert(
            this.speedReader.currentWordIndex === 1,
            'Skip words backward',
            `Expected index 1, got ${this.speedReader.currentWordIndex}`
        ));

        // Test 9: Jump to position
        this.speedReader.jumpToPosition(50);
        const expectedIndex = Math.floor(9 * 0.5);
        results.push(TestSuite.assert(
            this.speedReader.currentWordIndex === expectedIndex,
            'Jump to position',
            `Expected index ${expectedIndex}, got ${this.speedReader.currentWordIndex}`
        ));

        // Test 10: Reset
        this.speedReader.reset();
        results.push(TestSuite.assert(
            this.speedReader.currentWordIndex === 0 && !this.speedReader.isReading,
            'Reset',
            'Failed to reset speed reader'
        ));

        // Test 11: Start/stop functionality (without actual interval)
        let wordChangeCount = 0;
        let completed = false;
        
        results.push(await TestSuite.assertAsync(
            async () => {
                this.speedReader.start(
                    (word) => { wordChangeCount++; },
                    () => { completed = true; }
                );
                
                const wasReading = this.speedReader.isReading;
                this.speedReader.stop();
                
                return wasReading && !this.speedReader.isReading;
            },
            'Start/stop functionality',
            'Failed to start/stop speed reader'
        ));

        // Test 12: Empty text handling
        this.speedReader.setWords('');
        results.push(TestSuite.assert(
            this.speedReader.words.length === 0,
            'Empty text handling',
            'Failed to handle empty text'
        ));

        let errorThrown = false;
        try {
            this.speedReader.start(() => {}, () => {});
        } catch (e) {
            errorThrown = true;
        }
        results.push(TestSuite.assert(
            errorThrown,
            'Error on empty text start',
            'Should throw error when starting with no text'
        ));

        return results;
    }
}