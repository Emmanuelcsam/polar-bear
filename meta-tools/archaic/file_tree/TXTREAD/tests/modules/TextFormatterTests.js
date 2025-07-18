// TextFormatterTests.js - Tests for TextFormatter module
import { TextFormatter } from '../../js/modules/TextFormatter.js';
import { TestSuite } from '../TestSuite.js';

export class TextFormatterTests {
    constructor() {
        this.formatter = new TextFormatter();
    }

    async runTests() {
        const results = [];

        // Test 1: Module initialization
        results.push(TestSuite.assert(
            this.formatter instanceof TextFormatter,
            'TextFormatter instantiation',
            'Failed to create TextFormatter instance'
        ));

        // Test 2: Markdown parsing - headers
        const headerTests = [
            { input: '# Header 1', expected: '<h1>Header 1</h1>' },
            { input: '## Header 2', expected: '<h2>Header 2</h2>' },
            { input: '### Header 3', expected: '<h3>Header 3</h3>' }
        ];

        headerTests.forEach(({ input, expected }) => {
            const result = this.formatter.parseMarkdown(input);
            results.push(TestSuite.assert(
                result.includes(expected),
                `Markdown parsing: ${input}`,
                `Expected ${expected}, got ${result}`
            ));
        });

        // Test 3: Markdown parsing - formatting
        const formatTests = [
            { input: '**bold text**', expected: '<strong>bold text</strong>' },
            { input: '*italic text*', expected: '<em>italic text</em>' },
            { input: '`code`', expected: '<code>code</code>' },
            { input: '~~strikethrough~~', expected: '<del>strikethrough</del>' }
        ];

        formatTests.forEach(({ input, expected }) => {
            const result = this.formatter.parseMarkdown(input);
            results.push(TestSuite.assert(
                result.includes(expected),
                `Markdown formatting: ${input}`,
                `Expected ${expected} in result`
            ));
        });

        // Test 4: Bionic reading
        const bionicTests = [
            { word: 'the', boldLength: 3 }, // Short word, all bold
            { word: 'reading', boldLength: 3 }, // Longer word, partial bold
            { word: 'comprehension', boldLength: 5 } // Long word
        ];

        bionicTests.forEach(({ word, boldLength }) => {
            const result = this.formatter.applyBionicReading(word);
            results.push(TestSuite.assert(
                result.includes('class="word"') && result.includes('class="bold"'),
                `Bionic reading: ${word}`,
                'Failed to apply bionic reading formatting'
            ));
        });

        // Test 5: Syntax highlighting toggle
        const initialState = this.formatter.syntaxHighlightingEnabled;
        this.formatter.toggleSyntaxHighlighting();
        results.push(TestSuite.assert(
            this.formatter.syntaxHighlightingEnabled !== initialState,
            'Syntax highlighting toggle',
            'Failed to toggle syntax highlighting'
        ));
        this.formatter.toggleSyntaxHighlighting(); // Reset

        // Test 6: Code highlighting
        const codeTests = [
            {
                code: 'function test() { return true; }',
                lang: 'javascript',
                expectedClasses: ['syntax-keyword', 'syntax-function']
            },
            {
                code: 'def test(): return True',
                lang: 'python',
                expectedClasses: ['syntax-keyword', 'syntax-function']
            }
        ];

        codeTests.forEach(({ code, lang, expectedClasses }) => {
            const result = this.formatter.highlightCode(code, lang);
            const hasExpectedClasses = expectedClasses.every(cls => result.includes(cls));
            results.push(TestSuite.assert(
                hasExpectedClasses,
                `Code highlighting: ${lang}`,
                `Missing expected syntax classes in ${lang} highlighting`
            ));
        });

        // Test 7: Natural language highlighting
        const nlTest = 'The quick brown fox jumps over the lazy dog.';
        const nlResult = this.formatter.highlightNaturalLanguage(nlTest);
        results.push(TestSuite.assert(
            nlResult.includes('syntax-'),
            'Natural language highlighting',
            'Failed to apply natural language syntax highlighting'
        ));

        // Test 8: HTML escaping
        const xssTest = '<script>alert("XSS")</script>';
        const escaped = this.formatter.parseMarkdown(xssTest);
        results.push(TestSuite.assert(
            !escaped.includes('<script>') && escaped.includes('&lt;script&gt;'),
            'XSS prevention',
            'Failed to escape HTML in markdown'
        ));

        // Test 9: Toggle states
        results.push(TestSuite.assert(
            typeof this.formatter.toggleBionicMode === 'function' &&
            typeof this.formatter.toggleFocusGradient === 'function',
            'Toggle methods exist',
            'Missing toggle methods'
        ));

        // Test 10: Format text with options
        const testText = '# Test\n**Bold** and *italic*';
        const formatted = this.formatter.formatText(testText, { isMarkdown: true });
        results.push(TestSuite.assert(
            formatted.includes('<h1>') && formatted.includes('<strong>') && formatted.includes('<em>'),
            'Format text with options',
            'Failed to format text with markdown option'
        ));

        // Test 11: Link parsing
        const linkTest = '[Google](https://google.com)';
        const linkResult = this.formatter.parseMarkdown(linkTest);
        results.push(TestSuite.assert(
            linkResult.includes('<a href="https://google.com"') && linkResult.includes('target="_blank"'),
            'Link parsing',
            'Failed to parse markdown link'
        ));

        // Test 12: List parsing
        const listTest = '- Item 1\n- Item 2\n- Item 3';
        const listResult = this.formatter.parseMarkdown(listTest);
        results.push(TestSuite.assert(
            listResult.includes('<ul>') && listResult.includes('<li>Item 1</li>'),
            'List parsing',
            'Failed to parse markdown list'
        ));

        return results;
    }
}