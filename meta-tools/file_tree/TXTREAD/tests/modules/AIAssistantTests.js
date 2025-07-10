// AIAssistantTests.js - Tests for AIAssistant module
import { AIAssistant } from '../../js/modules/AIAssistant.js';
import { TestSuite } from '../TestSuite.js';

export class AIAssistantTests {
    constructor() {
        this.assistant = new AIAssistant();
    }

    async runTests() {
        const results = [];

        // Test 1: Module initialization
        results.push(TestSuite.assert(
            this.assistant instanceof AIAssistant,
            'AIAssistant instantiation',
            'Failed to create AIAssistant instance'
        ));

        // Test 2: Initial state
        results.push(TestSuite.assert(
            this.assistant.apiKey === '' &&
            this.assistant.currentText === '' &&
            Array.isArray(this.assistant.messages) &&
            this.assistant.maxContextLength === 8000,
            'Initial state',
            'Incorrect initial state values'
        ));

        // Test 3: Set API key
        const testKey = 'test-api-key-1234567890';
        const keySet = this.assistant.setApiKey(testKey);
        results.push(TestSuite.assert(
            keySet && this.assistant.apiKey === testKey,
            'Set API key',
            'Failed to set API key'
        ));

        // Test 4: Set current text
        const testText = 'This is a test document for AI analysis.';
        this.assistant.setCurrentText(testText);
        results.push(TestSuite.assert(
            this.assistant.currentText === testText,
            'Set current text',
            'Failed to set current text'
        ));

        // Test 5: Add message
        const message = this.assistant.addMessage('user', 'Test message');
        results.push(TestSuite.assert(
            message.type === 'user' &&
            message.content === 'Test message' &&
            message.timestamp instanceof Date,
            'Add message',
            'Failed to add message correctly'
        ));

        results.push(TestSuite.assert(
            this.assistant.messages.length === 1,
            'Message stored',
            'Message not stored in messages array'
        ));

        // Test 6: Build prompts
        const prompts = [
            { input: 'quiz', expectedPhrase: 'multiple-choice questions' },
            { input: 'summarize', expectedPhrase: 'comprehensive summary' },
            { input: 'key points', expectedPhrase: 'key points' },
            { input: 'explain', expectedPhrase: 'simple' },
            { input: 'analyze', expectedPhrase: 'analysis' },
            { input: 'custom question', expectedPhrase: 'User question:' }
        ];

        prompts.forEach(({ input, expectedPhrase }) => {
            const prompt = this.assistant.buildPrompt(input, 'Test context');
            results.push(TestSuite.assert(
                prompt.includes(expectedPhrase),
                `Build prompt: ${input}`,
                `Expected prompt to contain '${expectedPhrase}'`
            ));
        });

        // Test 7: Error parsing
        const errorCodes = [
            { code: 400, expected: 'Bad request' },
            { code: 401, expected: 'Authentication failed' },
            { code: 403, expected: 'Access forbidden' },
            { code: 429, expected: 'Too many requests' }
        ];

        errorCodes.forEach(({ code, expected }) => {
            const errorMsg = this.assistant.getGenericErrorMessage(code);
            results.push(TestSuite.assert(
                errorMsg.includes(expected),
                `Error message for ${code}`,
                `Expected error to contain '${expected}'`
            ));
        });

        // Test 8: API error parsing
        const apiError = this.assistant.parseApiError('API_KEY_INVALID', 401);
        results.push(TestSuite.assert(
            apiError.includes('Invalid API key'),
            'Parse API key error',
            'Failed to parse API key error'
        ));

        // Test 9: Clear messages
        this.assistant.clearMessages();
        results.push(TestSuite.assert(
            this.assistant.messages.length === 0,
            'Clear messages',
            'Failed to clear messages'
        ));

        // Test 10: Debug mode
        this.assistant.setDebugMode(true);
        results.push(TestSuite.assert(
            this.assistant.debugMode === true,
            'Set debug mode',
            'Failed to set debug mode'
        ));

        // Test 11: Quick actions
        results.push(TestSuite.assert(
            typeof this.assistant.generateQuiz === 'function' &&
            typeof this.assistant.generateSummary === 'function' &&
            typeof this.assistant.extractKeyPoints === 'function' &&
            typeof this.assistant.explainConcepts === 'function' &&
            typeof this.assistant.analyzeText === 'function',
            'Quick action methods exist',
            'Missing quick action methods'
        ));

        // Test 12: Send message without API key
        this.assistant.setApiKey('');
        let errorThrown = false;
        try {
            await this.assistant.sendMessage('Test');
        } catch (e) {
            errorThrown = e.message === 'API key is required';
        }
        results.push(TestSuite.assert(
            errorThrown,
            'Error on missing API key',
            'Should throw error when API key is missing'
        ));

        // Test 13: Send message without text
        this.assistant.setApiKey('test-key');
        this.assistant.setCurrentText('');
        errorThrown = false;
        try {
            await this.assistant.sendMessage('Test');
        } catch (e) {
            errorThrown = e.message === 'No text loaded for analysis';
        }
        results.push(TestSuite.assert(
            errorThrown,
            'Error on missing text',
            'Should throw error when no text is loaded'
        ));

        // Test 14: Context truncation
        const longText = 'a'.repeat(10000);
        this.assistant.setCurrentText(longText);
        const prompt = this.assistant.buildPrompt('test', longText);
        results.push(TestSuite.assert(
            prompt.includes('...') && !prompt.includes('a'.repeat(9000)),
            'Context truncation',
            'Failed to truncate long context'
        ));

        return results;
    }
}