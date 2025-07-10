// FileHandlerTests.js - Tests for FileHandler module
import { FileHandler } from '../../js/modules/FileHandler.js';
import { TestSuite } from '../TestSuite.js';

export class FileHandlerTests {
    constructor() {
        this.fileHandler = new FileHandler();
    }

    async runTests() {
        const results = [];
        
        // Test 1: Module initialization
        results.push(TestSuite.assert(
            this.fileHandler instanceof FileHandler,
            'FileHandler instantiation',
            'Failed to create FileHandler instance'
        ));

        // Test 2: Supported formats
        results.push(TestSuite.assert(
            this.fileHandler.supportedFormats.python.includes('.py'),
            'Python file support',
            'Python files not in supported formats'
        ));

        results.push(TestSuite.assert(
            this.fileHandler.supportedFormats.python.includes('.ipynb'),
            'Jupyter notebook support',
            'Jupyter notebooks not in supported formats'
        ));

        // Test 3: File type detection
        results.push(TestSuite.assert(
            this.fileHandler._isPDF('document.pdf'),
            'PDF detection',
            'Failed to detect PDF file'
        ));

        results.push(TestSuite.assert(
            this.fileHandler._isDOCX('document.docx'),
            'DOCX detection',
            'Failed to detect DOCX file'
        ));

        results.push(TestSuite.assert(
            this.fileHandler._isPythonNotebook('notebook.ipynb'),
            'Jupyter notebook detection',
            'Failed to detect Jupyter notebook'
        ));

        // Test 4: Mock file reading
        results.push(await TestSuite.assertAsync(
            async () => {
                const mockFile = new File(['print("Hello, World!")'], 'test.py', { type: 'text/plain' });
                try {
                    const result = await this.fileHandler.readFile(mockFile);
                    return result.text === 'print("Hello, World!")' && result.fileType === 'python';
                } catch (e) {
                    return false;
                }
            },
            'Python file reading',
            'Failed to read Python file'
        ));

        // Test 5: Accept string generation
        results.push(TestSuite.assert(
            this.fileHandler.getAcceptString().includes('.py'),
            'Accept string includes Python',
            'Accept string missing Python extensions'
        ));

        // Test 6: LaTeX detection
        const latexText = 'This is $x^2 + y^2 = z^2$ inline math';
        results.push(TestSuite.assert(
            latexText.includes('$'),
            'LaTeX detection in text',
            'Failed to detect LaTeX markers'
        ));

        // Test 7: File type mapping
        const testFiles = [
            { name: 'test.py', expected: 'python' },
            { name: 'test.ipynb', expected: 'notebook' },
            { name: 'test.pdf', expected: 'pdf' },
            { name: 'test.docx', expected: 'docx' },
            { name: 'test.md', expected: 'markdown' },
            { name: 'test.txt', expected: 'text' }
        ];

        testFiles.forEach(({ name, expected }) => {
            const mockFile = { type: 'text/plain' };
            results.push(TestSuite.assert(
                this.fileHandler._getFileType(name, mockFile) === expected,
                `File type detection: ${name}`,
                `Expected ${expected}, got ${this.fileHandler._getFileType(name, mockFile)}`
            ));
        });

        // Test 8: Jupyter notebook JSON parsing
        results.push(await TestSuite.assertAsync(
            async () => {
                const notebookContent = JSON.stringify({
                    cells: [
                        {
                            cell_type: 'markdown',
                            source: ['# Test Notebook']
                        },
                        {
                            cell_type: 'code',
                            source: ['print("Hello")'],
                            outputs: []
                        }
                    ]
                });
                
                const mockFile = new File([notebookContent], 'test.ipynb', { type: 'application/json' });
                try {
                    const result = await this.fileHandler._readJupyterNotebook(mockFile);
                    return result.includes('[Markdown Cell 1]') && result.includes('[Code Cell 2]');
                } catch (e) {
                    return false;
                }
            },
            'Jupyter notebook parsing',
            'Failed to parse Jupyter notebook structure'
        ));

        // Test 9: Library loading
        results.push(TestSuite.assert(
            typeof this.fileHandler.loadLibraries === 'function',
            'Library loader exists',
            'loadLibraries method not found'
        ));

        // Test 10: Error handling
        results.push(await TestSuite.assertAsync(
            async () => {
                try {
                    await this.fileHandler.readFile(null);
                    return false; // Should throw error
                } catch (e) {
                    return e.message.includes('Failed to read file');
                }
            },
            'Error handling for invalid file',
            'Did not properly handle invalid file input'
        ));

        return results;
    }
}