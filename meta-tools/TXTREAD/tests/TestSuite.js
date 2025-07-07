// TestSuite.js - Comprehensive test suite for all modules
import { FileHandlerTests } from './modules/FileHandlerTests.js';
import { TextFormatterTests } from './modules/TextFormatterTests.js';
import { SpeedReaderTests } from './modules/SpeedReaderTests.js';
import { AutoScrollerTests } from './modules/AutoScrollerTests.js';
import { PomodoroTimerTests } from './modules/PomodoroTimerTests.js';
import { TextToSpeechTests } from './modules/TextToSpeechTests.js';
import { AIAssistantTests } from './modules/AIAssistantTests.js';
import { ReadingStatsTests } from './modules/ReadingStatsTests.js';
import { UIControllerTests } from './modules/UIControllerTests.js';
import { IntegrationTests } from './modules/IntegrationTests.js';

export class TestSuite {
    constructor() {
        this.testModules = {
            FileHandler: new FileHandlerTests(),
            TextFormatter: new TextFormatterTests(),
            SpeedReader: new SpeedReaderTests(),
            AutoScroller: new AutoScrollerTests(),
            PomodoroTimer: new PomodoroTimerTests(),
            TextToSpeech: new TextToSpeechTests(),
            AIAssistant: new AIAssistantTests(),
            ReadingStats: new ReadingStatsTests(),
            UIController: new UIControllerTests(),
            Integration: new IntegrationTests()
        };
        
        this.results = {
            tests: {},
            total: 0,
            passed: 0,
            failed: 0,
            duration: 0
        };
    }

    async runAll() {
        console.log('🚀 Running all tests...');
        const startTime = Date.now();
        
        this.resetResults();
        
        for (const [name, module] of Object.entries(this.testModules)) {
            await this.runModuleTest(name, module);
        }
        
        this.results.duration = Date.now() - startTime;
        console.log(`✅ All tests completed in ${this.results.duration}ms`);
        
        return this.results;
    }

    async runModuleTests() {
        console.log('📦 Running module tests only...');
        const startTime = Date.now();
        
        this.resetResults();
        
        for (const [name, module] of Object.entries(this.testModules)) {
            if (name !== 'Integration') {
                await this.runModuleTest(name, module);
            }
        }
        
        this.results.duration = Date.now() - startTime;
        return this.results;
    }

    async runIntegrationTests() {
        console.log('🔗 Running integration tests only...');
        const startTime = Date.now();
        
        this.resetResults();
        
        await this.runModuleTest('Integration', this.testModules.Integration);
        
        this.results.duration = Date.now() - startTime;
        return this.results;
    }

    async runModuleTest(name, module) {
        console.log(`\n📋 Testing ${name}...`);
        
        this.results.tests[name] = [];
        
        try {
            const testResults = await module.runTests();
            
            testResults.forEach(result => {
                this.results.total++;
                if (result.passed) {
                    this.results.passed++;
                } else {
                    this.results.failed++;
                }
                
                this.results.tests[name].push(result);
                
                const icon = result.passed ? '✅' : '❌';
                console.log(`${icon} ${result.name}`);
                if (!result.passed && result.error) {
                    console.error(`   Error: ${result.error}`);
                }
            });
        } catch (error) {
            console.error(`Failed to run tests for ${name}:`, error);
            this.results.tests[name].push({
                name: 'Module Test Execution',
                passed: false,
                error: error.message
            });
            this.results.total++;
            this.results.failed++;
        }
    }

    resetResults() {
        this.results = {
            tests: {},
            total: 0,
            passed: 0,
            failed: 0,
            duration: 0
        };
    }

    // Helper method for assertions
    static assert(condition, testName, errorMessage = '') {
        return {
            name: testName,
            passed: condition,
            error: condition ? null : errorMessage
        };
    }

    // Helper method for async assertions
    static async assertAsync(asyncFn, testName, errorMessage = '') {
        try {
            const result = await asyncFn();
            return {
                name: testName,
                passed: result,
                error: result ? null : errorMessage
            };
        } catch (error) {
            return {
                name: testName,
                passed: false,
                error: error.message || errorMessage
            };
        }
    }

    // Mock helper for testing
    static createMock(methods = {}) {
        const mock = {
            calls: {},
            ...methods
        };
        
        Object.keys(methods).forEach(method => {
            const original = mock[method];
            mock.calls[method] = [];
            
            mock[method] = function(...args) {
                mock.calls[method].push(args);
                return typeof original === 'function' ? original.apply(this, args) : original;
            };
        });
        
        return mock;
    }

    // DOM helper for testing
    static createTestDOM() {
        const container = document.createElement('div');
        container.id = 'test-container';
        container.style.display = 'none';
        document.body.appendChild(container);
        
        return {
            container,
            cleanup: () => {
                container.remove();
            },
            createElement: (tag, props = {}) => {
                const element = document.createElement(tag);
                Object.assign(element, props);
                container.appendChild(element);
                return element;
            }
        };
    }

    // Delay helper for async testing
    static delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Performance testing helper
    static async measurePerformance(fn, iterations = 100) {
        const times = [];
        
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await fn();
            const end = performance.now();
            times.push(end - start);
        }
        
        const average = times.reduce((a, b) => a + b, 0) / times.length;
        const min = Math.min(...times);
        const max = Math.max(...times);
        
        return { average, min, max, iterations };
    }
}