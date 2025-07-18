// FileHandler.js - Handles file reading for various formats including Python files
export class FileHandler {
    constructor() {
        this.currentText = '';
        this.isMarkdown = false;
        this.supportedFormats = {
            text: ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log'],
            pdf: ['.pdf'],
            docx: ['.docx', '.doc'],
            image: ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
            python: ['.py', '.pyw', '.ipynb']
        };
    }

    async loadLibraries() {
        // Lazy load required libraries
        const loaders = {
            loadPDFJS: () => this._loadScript(
                'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js',
                () => {
                    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
                }
            ),
            loadTesseract: () => this._loadScript('https://unpkg.com/tesseract.js@4.1.1/dist/tesseract.min.js'),
            loadMammoth: () => this._loadScript('https://cdn.jsdelivr.net/npm/mammoth@1.6.0/mammoth.browser.min.js'),
            loadMathJax: () => {
                window.MathJax = {
                    tex: {
                        inlineMath: [['$', '$'], ['\\(', '\\)']],
                        displayMath: [['$$', '$$'], ['\\[', '\\]']]
                    },
                    svg: { fontCache: 'global' }
                };
                return this._loadScript('https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js');
            }
        };
        return loaders;
    }

    _loadScript(src, onLoad) {
        return new Promise((resolve, reject) => {
            const existingScript = document.querySelector(`script[src="${src}"]`);
            if (existingScript || (src.includes('pdf.js') && window.pdfjsLib) || 
                (src.includes('tesseract') && window.Tesseract) || 
                (src.includes('mammoth') && window.mammoth) ||
                (src.includes('mathjax') && window.MathJax)) {
                resolve();
                return;
            }
            
            const script = document.createElement('script');
            script.src = src;
            script.async = true;
            script.onload = () => {
                if (onLoad) onLoad();
                resolve();
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    async readFile(file, progressCallback) {
        const fileName = file.name.toLowerCase();
        let text = '';
        
        try {
            if (progressCallback) progressCallback(`📖 Loading ${file.name}...`);
            
            if (this._isPDF(fileName)) {
                text = await this._readPDF(file);
            } else if (this._isDOCX(fileName)) {
                text = await this._readDOCX(file);
            } else if (this._isImage(file)) {
                text = await this._performOCR(file, progressCallback);
            } else if (this._isPythonNotebook(fileName)) {
                text = await this._readJupyterNotebook(file);
            } else {
                text = await this._readTextFile(file);
            }
            
            this.currentText = text;
            this.isMarkdown = fileName.endsWith('.md');
            
            // Check for LaTeX content
            const hasLaTeX = text.includes('$') || text.includes('\\[') || text.includes('\\(');
            
            return {
                text,
                isMarkdown: this.isMarkdown,
                hasLaTeX,
                fileType: this._getFileType(fileName, file)
            };
        } catch (error) {
            throw new Error(`Failed to read file: ${error.message}`);
        }
    }

    async _readPDF(file) {
        const loaders = await this.loadLibraries();
        await loaders.loadPDFJS();
        
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        
        let fullText = '';
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            const pageText = textContent.items.map(item => item.str).join(' ');
            fullText += pageText + '\n\n';
        }
        
        return fullText;
    }

    async _readDOCX(file) {
        const loaders = await this.loadLibraries();
        await loaders.loadMammoth();
        
        const arrayBuffer = await file.arrayBuffer();
        const result = await mammoth.extractRawText({ arrayBuffer });
        
        if (result.messages.length > 0) {
            console.warn('DOCX conversion warnings:', result.messages);
        }
        
        return result.value;
    }

    async _performOCR(file, progressCallback) {
        const loaders = await this.loadLibraries();
        await loaders.loadTesseract();
        
        if (progressCallback) progressCallback('🔍 Performing OCR on image... This may take a moment.');
        
        const worker = await Tesseract.createWorker({
            logger: m => console.log(m)
        });
        
        await worker.loadLanguage('eng');
        await worker.initialize('eng');
        
        const { data: { text } } = await worker.recognize(file);
        await worker.terminate();
        
        if (!text || text.trim().length === 0) {
            throw new Error('No text found in image');
        }
        
        return text;
    }

    async _readJupyterNotebook(file) {
        const text = await this._readTextFile(file);
        try {
            const notebook = JSON.parse(text);
            let extractedText = '';
            
            // Extract text from notebook cells
            if (notebook.cells && Array.isArray(notebook.cells)) {
                notebook.cells.forEach((cell, index) => {
                    if (cell.cell_type === 'markdown') {
                        extractedText += `[Markdown Cell ${index + 1}]\n`;
                        extractedText += (cell.source.join ? cell.source.join('') : cell.source) + '\n\n';
                    } else if (cell.cell_type === 'code') {
                        extractedText += `[Code Cell ${index + 1}]\n`;
                        extractedText += '```python\n';
                        extractedText += (cell.source.join ? cell.source.join('') : cell.source);
                        extractedText += '\n```\n\n';
                        
                        // Include output if available
                        if (cell.outputs && cell.outputs.length > 0) {
                            extractedText += '[Output]\n';
                            cell.outputs.forEach(output => {
                                if (output.text) {
                                    extractedText += output.text.join ? output.text.join('') : output.text;
                                } else if (output.data && output.data['text/plain']) {
                                    extractedText += output.data['text/plain'].join ? 
                                        output.data['text/plain'].join('') : output.data['text/plain'];
                                }
                            });
                            extractedText += '\n\n';
                        }
                    }
                });
            }
            
            return extractedText || text; // Fallback to raw JSON if extraction fails
        } catch (e) {
            // If not valid JSON, treat as regular Python file
            return text;
        }
    }

    async _readTextFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }

    _isPDF(fileName) {
        return fileName.endsWith('.pdf');
    }

    _isDOCX(fileName) {
        return fileName.endsWith('.docx') || fileName.endsWith('.doc');
    }

    _isImage(file) {
        return file.type.startsWith('image/');
    }

    _isPythonNotebook(fileName) {
        return fileName.endsWith('.ipynb');
    }

    _getFileType(fileName, file) {
        if (this._isPDF(fileName)) return 'pdf';
        if (this._isDOCX(fileName)) return 'docx';
        if (this._isImage(file)) return 'image';
        if (this._isPythonNotebook(fileName)) return 'notebook';
        if (fileName.endsWith('.py')) return 'python';
        if (fileName.endsWith('.md')) return 'markdown';
        return 'text';
    }

    async processLaTeX(text) {
        const loaders = await this.loadLibraries();
        await loaders.loadMathJax();
        
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = text;
        
        await MathJax.typesetPromise([tempDiv]);
        
        return tempDiv.innerHTML;
    }

    getSupportedFormats() {
        const allFormats = [];
        Object.values(this.supportedFormats).forEach(formats => {
            allFormats.push(...formats);
        });
        return [...new Set(allFormats)].join(', ');
    }

    getAcceptString() {
        return this.getSupportedFormats() + ', image/*';
    }
}