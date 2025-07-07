// ReadingStatsTests.js - Tests for ReadingStats module
import { ReadingStats } from '../../js/modules/ReadingStats.js';
import { TestSuite } from '../TestSuite.js';

export class ReadingStatsTests {
    constructor() {
        this.stats = new ReadingStats();
    }

    async runTests() {
        const results = [];

        // Test 1: Module initialization
        results.push(TestSuite.assert(
            this.stats instanceof ReadingStats,
            'ReadingStats instantiation',
            'Failed to create ReadingStats instance'
        ));

        // Test 2: Initial state
        results.push(TestSuite.assert(
            this.stats.wordCount === 0 &&
            this.stats.characterCount === 0 &&
            this.stats.readingTime === 0 &&
            this.stats.averageWPM === 250 &&
            this.stats.readingProgress === 0,
            'Initial state',
            'Incorrect initial state values'
        ));

        // Test 3: Text analysis
        const testText = 'The quick brown fox jumps over the lazy dog. This is a test sentence.';
        const analysis = this.stats.analyzeText(testText);
        
        results.push(TestSuite.assert(
            analysis.wordCount === 13,
            'Word count',
            `Expected 13 words, got ${analysis.wordCount}`
        ));

        results.push(TestSuite.assert(
            analysis.characterCount > 0,
            'Character count',
            'Character count should be greater than 0'
        ));

        results.push(TestSuite.assert(
            analysis.readingTime === 1,
            'Reading time calculation',
            `Expected 1 minute, got ${analysis.readingTime}`
        ));

        // Test 4: Progress update
        const progress = this.stats.updateProgress(50);
        results.push(TestSuite.assert(
            progress === 50,
            'Progress update',
            `Expected 50%, got ${progress}%`
        ));

        // Test 5: Reading speed calculation
        this.stats.sessionTime = 60; // 1 minute
        const speed = this.stats.calculateReadingSpeed();
        results.push(TestSuite.assert(
            speed > 0,
            'Reading speed calculation',
            'Reading speed should be calculated'
        ));

        // Test 6: Time formatting
        const timeFormats = [
            { seconds: 45, expected: '45s' },
            { seconds: 65, expected: '1m 5s' },
            { seconds: 3665, expected: '1h 1m' }
        ];

        timeFormats.forEach(({ seconds, expected }) => {
            const formatted = this.stats.formatTime(seconds);
            results.push(TestSuite.assert(
                formatted === expected,
                `Time format: ${seconds}s`,
                `Expected '${expected}', got '${formatted}'`
            ));
        });

        // Test 7: Time remaining calculation
        this.stats.readingProgress = 25;
        this.stats.wordCount = 100;
        const remaining = this.stats.getTimeRemaining();
        results.push(TestSuite.assert(
            remaining > 0,
            'Time remaining calculation',
            'Should calculate time remaining'
        ));

        // Test 8: Stats object structure
        const statsObj = this.stats.getStats();
        results.push(TestSuite.assert(
            statsObj.wordCount === 13 &&
            typeof statsObj.sessionTime === 'string' &&
            typeof statsObj.readingProgress === 'number' &&
            typeof statsObj.currentSpeed === 'number' &&
            typeof statsObj.estimatedTimeRemaining === 'number',
            'Stats object structure',
            'Invalid stats object structure'
        ));

        // Test 9: WPM adjustment
        this.stats.setAverageWPM(300);
        results.push(TestSuite.assert(
            this.stats.averageWPM === 300,
            'Set average WPM',
            `Expected 300 WPM, got ${this.stats.averageWPM}`
        ));

        // Test 10: Detailed stats
        const detailed = this.stats.getDetailedStats();
        results.push(TestSuite.assert(
            typeof detailed.averageWordLength === 'string' &&
            typeof detailed.pagesEstimate === 'number' &&
            typeof detailed.sentenceCount === 'number' &&
            typeof detailed.paragraphCount === 'number' &&
            typeof detailed.readabilityScore === 'number',
            'Detailed stats structure',
            'Invalid detailed stats structure'
        ));

        // Test 11: Export stats
        const exported = this.stats.exportStats();
        results.push(TestSuite.assert(
            exported.exportedAt &&
            exported.readingSpeed === 300 &&
            typeof exported.sessionDuration === 'number',
            'Export stats',
            'Invalid exported stats structure'
        ));

        // Test 12: Empty text handling
        this.stats.analyzeText('');
        results.push(TestSuite.assert(
            this.stats.wordCount === 0 &&
            this.stats.characterCount === 0 &&
            this.stats.readingTime === 0,
            'Empty text handling',
            'Failed to handle empty text'
        ));

        // Test 13: Session management
        this.stats.startSession();
        results.push(TestSuite.assert(
            this.stats.interval !== null,
            'Start session',
            'Failed to start session timer'
        ));

        this.stats.stopSession();
        results.push(TestSuite.assert(
            this.stats.interval === null,
            'Stop session',
            'Failed to stop session timer'
        ));

        // Test 14: Getters
        this.stats.wordCount = 1000;
        results.push(TestSuite.assert(
            this.stats.getWordCount() === '1,000',
            'Word count formatting',
            'Failed to format word count with comma'
        ));

        this.stats.readingTime = 5;
        results.push(TestSuite.assert(
            this.stats.getReadingTimeEstimate() === '5 min',
            'Reading time estimate',
            'Failed to format reading time estimate'
        ));

        this.stats.readingProgress = 75.4;
        results.push(TestSuite.assert(
            this.stats.getProgressPercentage() === '75%',
            'Progress percentage',
            'Failed to format progress percentage'
        ));

        return results;
    }
}