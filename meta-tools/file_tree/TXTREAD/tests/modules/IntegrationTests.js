// IntegrationTests.js - Integration tests for module interactions
import { FileHandler } from '../../js/modules/FileHandler.js';
import { TextFormatter } from '../../js/modules/TextFormatter.js';
import { SpeedReader } from '../../js/modules/SpeedReader.js';
import { ReadingStats } from '../../js/modules/ReadingStats.js';
import { AIAssistant } from '../../js/modules/AIAssistant.js';
import { TestSuite } from '../TestSuite.js';

export class IntegrationTests {
    constructor() {
        this.fileHandler = new FileHandler();
        this.textFormatter = new TextFormatter();
        this.speedReader = new SpeedReader();
        this.readingStats = new ReadingStats();
        this.aiAssistant = new AIAssistant();
    }

    async runTests() {
        const results = [];

        // Test 1: File reading to text formatting pipeline
        results.push(await TestSuite.assertAsync(
            async () => {
                const markdownContent = '# Test\n**Bold** text with *italics*';
                const mockFile = new File([markdownContent], 'test.md', { type: 'text/markdown' });
                
                const fileResult = await this.fileHandler.readFile(mockFile);
                const formatted = this.textFormatter.formatText(fileResult.text, {
                    isMarkdown: fileResult.isMarkdown
                });
                
                return formatted.includes('<h1>') && 
                       formatted.includes('<strong>') && 
                       formatted.includes('<em>');
            },
            'File to formatter pipeline',
            'Failed to process markdown file through formatter'
        ));

        // Test 2: Text analysis to speed reader setup
        const testText = 'The quick brown fox jumps over the lazy dog. This is a test sentence for speed reading.';
        const stats = this.readingStats.analyzeText(testText);
        this.speedReader.setWords(testText);
        
        results.push(TestSuite.assert(
            stats.words.length === this.speedReader.words.length,
            'Stats to speed reader consistency',
            'Word count mismatch between stats and speed reader'
        ));

        // Test 3: Reading stats and speed reader progress sync
        this.speedReader.currentWordIndex = 5;
        const speedProgress = this.speedReader.getProgress();
        this.readingStats.updateProgress(speedProgress);
        
        results.push(TestSuite.assert(
            Math.abs(this.readingStats.readingProgress - speedProgress) < 1,
            'Progress synchronization',
            'Progress not synchronized between modules'
        ));

        // Test 4: AI assistant with formatted text
        this.aiAssistant.setCurrentText(testText);
        const prompt = this.aiAssistant.buildPrompt('summarize', testText);
        
        results.push(TestSuite.assert(
            prompt.includes(testText) && prompt.includes('summary'),
            'AI assistant text processing',
            'AI assistant failed to build prompt with text'
        ));

        // Test 5: Python file handling integration
        results.push(await TestSuite.assertAsync(
            async () => {
                const pythonCode = 'def hello():\n    print("Hello, World!")';
                const mockPyFile = new File([pythonCode], 'test.py', { type: 'text/plain' });
                
                const result = await this.fileHandler.readFile(mockPyFile);
                return result.fileType === 'python' && result.text === pythonCode;
            },
            'Python file handling',
            'Failed to handle Python file correctly'
        ));

        // Test 6: Jupyter notebook to formatted output
        results.push(await TestSuite.assertAsync(
            async () => {
                const notebook = {
                    cells: [
                        { cell_type: 'markdown', source: ['# Title'] },
                        { cell_type: 'code', source: ['print("test")'] }
                    ]
                };
                const mockNotebook = new File([JSON.stringify(notebook)], 'test.ipynb', { type: 'application/json' });
                
                const result = await this.fileHandler.readFile(mockNotebook);
                return result.fileType === 'notebook' && 
                       result.text.includes('[Markdown Cell 1]') &&
                       result.text.includes('[Code Cell 2]');
            },
            'Jupyter notebook processing',
            'Failed to process Jupyter notebook'
        ));

        // Test 7: Bionic reading with stats
        this.textFormatter.bionicMode = true;
        const bionicFormatted = this.textFormatter.formatText(testText, { bionicMode: true });
        
        results.push(TestSuite.assert(
            bionicFormatted.includes('class="word"') && bionicFormatted.includes('class="bold"'),
            'Bionic reading formatting',
            'Failed to apply bionic reading'
        ));

        // Test 8: Reading time calculation consistency
        const wordCount = testText.split(/\s+/).length;
        const statsTime = this.readingStats.readingTime;
        const speedReaderTime = this.speedReader.getTimeRemaining();
        
        results.push(TestSuite.assert(
            typeof statsTime === 'number' && typeof speedReaderTime === 'number',
            'Reading time calculations',
            'Reading time calculations failed'
        ));

        // Test 9: LaTeX detection and processing flag
        const latexText = 'This is $x^2 + y^2 = z^2$ inline math';
        const latexFile = new File([latexText], 'math.txt', { type: 'text/plain' });
        
        results.push(await TestSuite.assertAsync(
            async () => {
                const result = await this.fileHandler.readFile(latexFile);
                return result.hasLaTeX === true;
            },
            'LaTeX detection',
            'Failed to detect LaTeX content'
        ));

        // Test 10: Syntax highlighting with code content
        this.textFormatter.syntaxHighlightingEnabled = true;
        const codeText = '```javascript\nfunction test() { return true; }\n```';
        const highlighted = this.textFormatter.formatText(codeText, { isMarkdown: true });
        
        results.push(TestSuite.assert(
            highlighted.includes('language-javascript'),
            'Code syntax highlighting',
            'Failed to apply syntax highlighting to code'
        ));

        // Test 11: Multi-format support verification
        const formats = this.fileHandler.getSupportedFormats();
        results.push(TestSuite.assert(
            formats.includes('.py') && 
            formats.includes('.ipynb') && 
            formats.includes('.pdf') &&
            formats.includes('.md'),
            'Multi-format support',
            'Missing expected file format support'
        ));

        // Test 12: Performance test - large text processing
        results.push(await TestSuite.assertAsync(
            async () => {
                const largeText = 'word '.repeat(10000);
                const startTime = performance.now();
                
                this.readingStats.analyzeText(largeText);
                this.speedReader.setWords(largeText);
                const formatted = this.textFormatter.formatText(largeText);
                
                const endTime = performance.now();
                const processingTime = endTime - startTime;
                
                return processingTime < 1000; // Should process in under 1 second
            },
            'Performance - large text',
            'Large text processing too slow'
        ));

        return results;
    }
}