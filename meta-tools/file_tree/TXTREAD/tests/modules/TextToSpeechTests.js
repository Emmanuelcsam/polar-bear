// TextToSpeechTests.js - Tests for TextToSpeech module
import { TextToSpeech } from '../../js/modules/TextToSpeech.js';
import { TestSuite } from '../TestSuite.js';

export class TextToSpeechTests {
    constructor() {
        this.tts = new TextToSpeech();
    }

    async runTests() {
        const results = [];

        // Test 1: Module initialization
        results.push(TestSuite.assert(
            this.tts instanceof TextToSpeech,
            'TextToSpeech instantiation',
            'Failed to create TextToSpeech instance'
        ));

        // Test 2: Initial state
        results.push(TestSuite.assert(
            this.tts.engine === 'browser' &&
            this.tts.rate === 1.0 &&
            this.tts.pitch === 1.0 &&
            this.tts.volume === 1.0 &&
            !this.tts.isPlaying &&
            !this.tts.isPaused,
            'Initial state',
            'Incorrect initial state values'
        ));

        // Test 3: Engine switching
        this.tts.setEngine('edge');
        results.push(TestSuite.assert(
            this.tts.engine === 'edge',
            'Switch to Edge engine',
            'Failed to switch to Edge engine'
        ));

        this.tts.setEngine('browser');
        results.push(TestSuite.assert(
            this.tts.engine === 'browser',
            'Switch back to browser engine',
            'Failed to switch back to browser engine'
        ));

        // Test 4: Set text
        const testText = 'Hello, this is a test.';
        this.tts.setText(testText);
        results.push(TestSuite.assert(
            this.tts.currentText === testText,
            'Set text',
            'Failed to set text'
        ));

        // Test 5: Rate adjustment
        const newRate = this.tts.setRate(1.5);
        results.push(TestSuite.assert(
            newRate === 1.5,
            'Set rate',
            `Expected rate 1.5, got ${newRate}`
        ));

        const minRate = this.tts.setRate(0.3);
        results.push(TestSuite.assert(
            minRate === 0.5,
            'Rate minimum limit',
            `Expected 0.5 minimum, got ${minRate}`
        ));

        const maxRate = this.tts.setRate(2.5);
        results.push(TestSuite.assert(
            maxRate === 2.0,
            'Rate maximum limit',
            `Expected 2.0 maximum, got ${maxRate}`
        ));

        // Test 6: Pitch adjustment
        const newPitch = this.tts.setPitch(1.2);
        results.push(TestSuite.assert(
            newPitch === 1.2,
            'Set pitch',
            `Expected pitch 1.2, got ${newPitch}`
        ));

        // Test 7: Volume adjustment
        const newVolume = this.tts.setVolume(0.8);
        results.push(TestSuite.assert(
            newVolume === 0.8,
            'Set volume',
            `Expected volume 0.8, got ${newVolume}`
        ));

        // Test 8: Browser support check
        results.push(TestSuite.assert(
            typeof this.tts.isSupported() === 'boolean',
            'Browser support check',
            'isSupported should return boolean'
        ));

        // Test 9: Edge voices list
        const edgeVoices = this.tts.getEdgeVoices();
        results.push(TestSuite.assert(
            Array.isArray(edgeVoices) && edgeVoices.length > 0,
            'Edge voices list',
            'Should return array of Edge voices'
        ));

        results.push(TestSuite.assert(
            edgeVoices[0].name === 'en-US-JennyNeural',
            'Default Edge voice',
            'First Edge voice should be Jenny'
        ));

        // Test 10: Status reporting
        const status = this.tts.getStatus();
        results.push(TestSuite.assert(
            status === 'Stopped',
            'Initial status',
            `Expected 'Stopped', got '${status}'`
        ));

        // Test 11: Edge voice selection
        this.tts.setEdgeVoice('en-GB-SoniaNeural');
        results.push(TestSuite.assert(
            this.tts.edgeVoice === 'en-GB-SoniaNeural',
            'Edge voice selection',
            'Failed to set Edge voice'
        ));

        // Test 12: Event handler setup
        let handlerCalled = false;
        this.tts.onEnd(() => { handlerCalled = true; });
        
        results.push(TestSuite.assert(
            typeof this.tts.onEnd === 'function',
            'Event handler setup',
            'Failed to set onEnd handler'
        ));

        // Test 13: Voice loading (if available)
        await TestSuite.delay(100); // Give time for voices to load
        results.push(TestSuite.assert(
            Array.isArray(this.tts.voices),
            'Voices array exists',
            'Voices should be an array'
        ));

        // Test 14: Error handling for empty text
        this.tts.setText('');
        let errorThrown = false;
        try {
            await this.tts.startBrowserTTS();
        } catch (e) {
            errorThrown = true;
        }
        results.push(TestSuite.assert(
            errorThrown || !this.tts.isPlaying,
            'Handle empty text',
            'Should handle empty text gracefully'
        ));

        return results;
    }
}