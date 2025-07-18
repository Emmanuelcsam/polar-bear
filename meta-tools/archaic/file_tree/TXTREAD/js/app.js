// app.js - Main application file that coordinates all modules
import { FileHandler } from './modules/FileHandler.js';
import { TextFormatter } from './modules/TextFormatter.js';
import { SpeedReader } from './modules/SpeedReader.js';
import { AutoScroller } from './modules/AutoScroller.js';
import { PomodoroTimer } from './modules/PomodoroTimer.js';
import { TextToSpeech } from './modules/TextToSpeech.js';
import { AIAssistant } from './modules/AIAssistant.js';
import { ReadingStats } from './modules/ReadingStats.js';
import { UIController } from './modules/UIController.js';

class ZenReaderApp {
    constructor() {
        // Initialize all modules
        this.fileHandler = new FileHandler();
        this.textFormatter = new TextFormatter();
        this.speedReader = new SpeedReader();
        this.autoScroller = new AutoScroller();
        this.pomodoroTimer = new PomodoroTimer();
        this.tts = new TextToSpeech();
        this.aiAssistant = new AIAssistant();
        this.readingStats = new ReadingStats();
        this.uiController = new UIController();

        // Application state
        this.currentText = '';
        this.isMarkdown = false;
        this.debugMode = false;

        // Initialize the app
        this.init();
    }

    async init() {
        // Start reading session
        this.readingStats.startSession();

        // Initialize UI
        this.uiController.init();

        // Setup event listeners
        this.setupEventListeners();

        // Setup keyboard shortcuts
        this.setupShortcuts();

        // Load saved settings
        this.loadSettings();

        // Initialize features
        this.initializeFeatures();

        console.log('Zen Reader Pro initialized successfully');
    }

    setupEventListeners() {
        // File upload
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');

        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }

        if (uploadArea) {
            uploadArea.addEventListener('click', () => fileInput?.click());
            uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
            uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
            uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        }

        // Features menu
        const featuresBtn = document.getElementById('featuresBtn');
        if (featuresBtn) {
            featuresBtn.addEventListener('click', () => this.uiController.toggleFeatureMenu());
        }

        // Feature items
        document.querySelectorAll('.feature-item').forEach((item, index) => {
            item.addEventListener('click', () => this.handleFeatureClick(index));
        });

        // AI chat
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');

        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }

        // API key
        const apiKeyInput = document.getElementById('apiKey');
        const saveApiKeyBtn = document.getElementById('saveApiKey');

        if (saveApiKeyBtn) {
            saveApiKeyBtn.addEventListener('click', () => this.saveApiKey());
        }

        // Quick actions
        document.querySelectorAll('.quick-action').forEach(action => {
            action.addEventListener('click', (e) => {
                const text = e.target.textContent;
                this.handleQuickAction(text);
            });
        });

        // Reading progress tracking
        const textDisplay = document.getElementById('textDisplay');
        if (textDisplay) {
            textDisplay.addEventListener('scroll', () => this.updateReadingProgress());
        }

        // Speed reader controls
        document.getElementById('speedPlayPause')?.addEventListener('click', () => {
            this.toggleSpeedReading();
        });

        document.getElementById('speedDecrease')?.addEventListener('click', () => {
            const newSpeed = this.speedReader.changeSpeed(-50);
            document.getElementById('wpmDisplay').textContent = `${newSpeed} WPM`;
        });

        document.getElementById('speedIncrease')?.addEventListener('click', () => {
            const newSpeed = this.speedReader.changeSpeed(50);
            document.getElementById('wpmDisplay').textContent = `${newSpeed} WPM`;
        });

        // Pomodoro controls
        document.getElementById('pomodoroStart')?.addEventListener('click', () => {
            this.startPomodoro();
        });

        document.getElementById('pomodoroPause')?.addEventListener('click', () => {
            this.pomodoroTimer.pause();
        });

        document.getElementById('pomodoroReset')?.addEventListener('click', () => {
            this.pomodoroTimer.reset();
            this.updatePomodoroDisplay();
        });

        // TTS controls
        document.getElementById('ttsPlayPause')?.addEventListener('click', () => {
            this.tts.togglePlayPause();
        });

        document.getElementById('ttsStop')?.addEventListener('click', () => {
            this.tts.stop();
        });

        document.getElementById('ttsRate')?.addEventListener('input', (e) => {
            const rate = parseFloat(e.target.value);
            this.tts.setRate(rate);
            document.getElementById('ttsRateValue').textContent = `${rate}x`;
        });

        // TTS engine switching
        document.getElementById('browserTTSBtn')?.addEventListener('click', () => {
            this.switchTTSEngine('browser');
        });

        document.getElementById('edgeTTSBtn')?.addEventListener('click', () => {
            this.switchTTSEngine('edge');
        });

        document.getElementById('edgeVoiceSelector')?.addEventListener('change', (e) => {
            this.tts.setEdgeVoice(e.target.value);
        });

        // Autoscroll controls
        document.getElementById('autoscrollPlayPause')?.addEventListener('click', () => {
            this.toggleAutoscroll();
        });

        document.getElementById('autoscrollStop')?.addEventListener('click', () => {
            this.autoScroller.stop();
            document.getElementById('autoscrollPlayPause').innerHTML = '▶️ Start';
            document.getElementById('autoscrollStatus').textContent = 'Stopped';
        });

        document.getElementById('autoscrollReset')?.addEventListener('click', () => {
            this.autoScroller.reset();
            this.updateReadingProgress();
        });

        document.getElementById('autoscrollSpeed')?.addEventListener('input', (e) => {
            const speed = parseInt(e.target.value);
            this.autoScroller.setSpeed(speed);
            document.getElementById('autoscrollSpeedValue').textContent = `${speed} px/s`;
        });

        // Exit fullscreen button
        document.getElementById('exitFullscreen')?.addEventListener('click', () => {
            this.uiController.toggleFeature('fullscreen');
        });

        // AI fullscreen button
        document.getElementById('aiFullscreenBtn')?.addEventListener('click', () => {
            this.uiController.toggleAIFullscreen();
        });
    }

    setupShortcuts() {
        this.uiController.setShortcutHandlers({
            toggleFullscreen: () => this.toggleFullscreen(),
            startSpeedReader: () => this.startSpeedReader(),
            toggleBionicMode: () => this.toggleBionicMode(),
            togglePomodoro: () => this.togglePomodoro(),
            toggleTTS: () => this.toggleTTS(),
            toggleSyntaxHighlighting: () => this.toggleSyntaxHighlighting(),
            toggleAIFullscreen: () => this.uiController.toggleAIFullscreen(),
            toggleAutoscroll: () => this.toggleAutoscroll(),
            exitAll: () => this.uiController.exitAll()
        });
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            await this.loadFile(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        event.target.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.target.classList.remove('dragover');
    }

    async handleDrop(event) {
        event.preventDefault();
        event.target.classList.remove('dragover');
        
        const file = event.dataTransfer.files[0];
        if (file) {
            await this.loadFile(file);
        }
    }

    async loadFile(file) {
        try {
            const result = await this.fileHandler.readFile(file, (message) => {
                this.addMessage('ai', message);
            });

            this.currentText = result.text;
            this.isMarkdown = result.isMarkdown;
            
            // Update AI assistant with new text
            this.aiAssistant.setCurrentText(this.currentText);
            
            // Update reading stats
            const stats = this.readingStats.analyzeText(this.currentText);
            
            // Update speed reader
            this.speedReader.setWords(this.currentText);
            
            // Update TTS
            this.tts.setText(this.currentText);
            
            // Display the text
            await this.displayText(result);
            
            // Show content area
            document.getElementById('contentArea')?.classList.add('active');
            
            // Update UI stats
            this.updateStats();
            
            // Show appropriate AI message
            if (this.aiAssistant.apiKey) {
                const messages = {
                    'pdf': '📄 PDF loaded successfully! I can help you understand the content, create summaries, or quiz you on key concepts.',
                    'docx': '📝 Word document loaded! Feel free to ask questions about the content.',
                    'image': '✅ OCR complete! I extracted the text from your image. Ask me about the content or request a summary.',
                    'notebook': '📓 Jupyter notebook loaded! I can help explain the code or summarize the content.',
                    'python': '🐍 Python file loaded! I can help explain the code or analyze its functionality.',
                    'markdown': '📝 Markdown file loaded! The formatting has been preserved. Ask me anything about the content.',
                    'text': 'Text loaded! I can help you understand it better. Try asking for a summary, quiz yourself, or ask about specific concepts.'
                };
                
                this.addMessage('ai', messages[result.fileType] || messages.text);
            } else {
                this.addMessage('ai', 'File loaded! Add your Gemini API key to enable AI assistance for summaries, quizzes, and explanations.');
            }
            
        } catch (error) {
            console.error('Error loading file:', error);
            alert('Failed to load file: ' + error.message);
            this.addMessage('ai', `❌ Error loading file: ${error.message}`);
        }
    }

    async displayText(result) {
        const textContent = document.getElementById('textContent');
        if (!textContent) return;

        let html = result.text;
        
        // Process LaTeX if needed
        if (result.hasLaTeX) {
            this.addMessage('ai', '🔢 LaTeX content detected. Processing mathematical expressions...');
            html = await this.fileHandler.processLaTeX(html);
        }
        
        // Apply formatting
        html = this.textFormatter.formatText(html, {
            isMarkdown: result.isMarkdown,
            bionicMode: this.uiController.isFeatureActive('bionic'),
            syntaxHighlighting: this.uiController.isFeatureActive('syntax')
        });
        
        textContent.innerHTML = html;
        
        // Show reading progress bar
        document.querySelector('.reading-progress')?.classList.add('active');
    }

    handleFeatureClick(index) {
        const features = [
            'fullscreen',
            'speed',
            'bionic', 
            'focus',
            'pomodoro',
            'tts',
            'theme',
            'increase',
            'decrease',
            'syntax',
            'autoscroll'
        ];

        const feature = features[index];
        
        switch (feature) {
            case 'fullscreen':
                this.toggleFullscreen();
                break;
            case 'speed':
                this.startSpeedReader();
                break;
            case 'bionic':
                this.toggleBionicMode();
                break;
            case 'focus':
                this.toggleFocusGradient();
                break;
            case 'pomodoro':
                this.togglePomodoro();
                break;
            case 'tts':
                this.toggleTTS();
                break;
            case 'theme':
                this.toggleTheme();
                break;
            case 'increase':
                this.uiController.increaseFontSize();
                break;
            case 'decrease':
                this.uiController.decreaseFontSize();
                break;
            case 'syntax':
                this.toggleSyntaxHighlighting();
                break;
            case 'autoscroll':
                this.toggleAutoscroll();
                break;
        }
    }

    toggleFullscreen() {
        const isActive = this.uiController.toggleFeature('fullscreen');
        return isActive;
    }

    startSpeedReader() {
        if (!this.currentText) {
            alert('Please load a text file first');
            return;
        }
        
        this.uiController.showSpeedReaderModal();
        document.getElementById('wpmDisplay').textContent = `${this.speedReader.wordsPerMinute} WPM`;
    }

    toggleSpeedReading() {
        if (this.speedReader.isReading) {
            this.speedReader.stop();
            document.getElementById('speedPlayPause').textContent = 'Start';
        } else {
            this.speedReader.start(
                (word) => {
                    document.getElementById('speedWord').textContent = word;
                },
                () => {
                    document.getElementById('speedWord').textContent = 'Complete!';
                    document.getElementById('speedPlayPause').textContent = 'Start';
                }
            );
            document.getElementById('speedPlayPause').textContent = 'Pause';
        }
    }

    toggleBionicMode() {
        this.textFormatter.toggleBionicMode();
        const isActive = this.uiController.toggleFeature('bionic');
        
        if (this.currentText) {
            this.redisplayText();
        }
        
        return isActive;
    }

    toggleFocusGradient() {
        this.textFormatter.toggleFocusGradient();
        const isActive = this.uiController.toggleFeature('focus');
        return isActive;
    }

    togglePomodoro() {
        const isActive = this.uiController.toggleFeature('pomodoro');
        
        if (isActive) {
            this.updatePomodoroDisplay();
        }
        
        return isActive;
    }

    startPomodoro() {
        this.pomodoroTimer.start(
            (time, progress) => {
                document.getElementById('pomodoroTime').textContent = time;
            },
            (message, wasBreak) => {
                alert(message);
                this.pomodoroTimer.reset();
                this.updatePomodoroDisplay();
            }
        );
    }

    updatePomodoroDisplay() {
        document.getElementById('pomodoroTime').textContent = this.pomodoroTimer.getTimeDisplay();
    }

    toggleTTS() {
        const isActive = this.uiController.toggleFeature('tts');
        
        if (!isActive) {
            this.tts.stop();
        } else if (this.currentText) {
            this.tts.start();
        }
        
        return isActive;
    }

    switchTTSEngine(engine) {
        this.tts.stop();
        this.tts.setEngine(engine);
        
        // Update UI
        document.getElementById('browserTTSBtn')?.classList.toggle('active', engine === 'browser');
        document.getElementById('edgeTTSBtn')?.classList.toggle('active', engine === 'edge');
        document.getElementById('edgeTTSOptions')?.classList.toggle('active', engine === 'edge');
    }

    toggleTheme() {
        const newTheme = this.uiController.toggleTheme();
        this.uiController.updateFeatureUI('theme', newTheme === 'light');
        return newTheme;
    }

    toggleSyntaxHighlighting() {
        this.textFormatter.toggleSyntaxHighlighting();
        const isActive = this.uiController.toggleFeature('syntax');
        
        // Re-render AI messages
        const aiMessages = document.querySelectorAll('.message.ai');
        aiMessages.forEach(messageDiv => {
            const bubble = messageDiv.querySelector('.message-bubble');
            const originalText = messageDiv.dataset.originalText;
            if (originalText && bubble) {
                bubble.innerHTML = this.parseAIMarkdown(originalText);
            }
        });
        
        // Re-render main text if needed
        if (this.currentText) {
            this.redisplayText();
        }
        
        return isActive;
    }

    toggleAutoscroll() {
        const isActive = this.uiController.toggleFeature('autoscroll');
        
        if (isActive) {
            const textDisplay = document.getElementById('textDisplay');
            this.autoScroller.setTarget(textDisplay);
        }
        
        return isActive;
    }

    toggleAutoscrollPlayPause() {
        if (this.autoScroller.isScrolling) {
            if (this.autoScroller.isPaused) {
                this.autoScroller.resume();
                document.getElementById('autoscrollPlayPause').innerHTML = '⏸️ Pause';
                document.getElementById('autoscrollStatus').textContent = 'Scrolling';
            } else {
                this.autoScroller.pause();
                document.getElementById('autoscrollPlayPause').innerHTML = '▶️ Resume';
                document.getElementById('autoscrollStatus').textContent = 'Paused';
            }
        } else {
            this.autoScroller.start();
            this.autoScroller.onComplete = () => {
                alert('Reached the end of the document');
                document.getElementById('autoscrollPlayPause').innerHTML = '▶️ Start';
                document.getElementById('autoscrollStatus').textContent = 'Completed';
            };
            document.getElementById('autoscrollPlayPause').innerHTML = '⏸️ Pause';
            document.getElementById('autoscrollStatus').textContent = 'Scrolling';
        }
    }

    async redisplayText() {
        const result = {
            text: this.currentText,
            isMarkdown: this.isMarkdown,
            hasLaTeX: false
        };
        
        await this.displayText(result);
    }

    updateReadingProgress() {
        const textDisplay = document.getElementById('textDisplay');
        if (!textDisplay) return;
        
        const scrollPercentage = (textDisplay.scrollTop / 
            (textDisplay.scrollHeight - textDisplay.clientHeight)) * 100;
        
        this.readingStats.updateProgress(scrollPercentage);
        this.uiController.updateReadingProgress(scrollPercentage);
        this.updateStats();
    }

    updateStats() {
        const stats = this.readingStats.getStats();
        this.uiController.updateStats({
            wordCount: this.readingStats.getWordCount(),
            readingTime: this.readingStats.getReadingTimeEstimate(),
            readingProgress: Math.round(stats.readingProgress),
            sessionTime: stats.sessionTime
        });
    }

    async sendMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message) return;
        
        if (!this.aiAssistant.apiKey) {
            this.addMessage('ai', 'Please add your Gemini API key first and click Save.');
            return;
        }
        
        if (!this.currentText) {
            this.addMessage('ai', 'Please upload a text file first.');
            return;
        }
        
        input.value = '';
        
        const sendBtn = document.getElementById('sendBtn');
        sendBtn.disabled = true;
        sendBtn.innerHTML = '<span class="loading"></span>';
        
        try {
            const response = await this.aiAssistant.sendMessage(message);
            this.addMessage('ai', response);
        } catch (error) {
            console.error('API error:', error);
            this.addMessage('ai', `Sorry, I encountered an error: ${error.message}`);
        } finally {
            sendBtn.disabled = false;
            sendBtn.innerHTML = 'Send';
        }
    }

    handleQuickAction(action) {
        const actions = {
            '📊 Summarize': () => this.aiAssistant.generateSummary(),
            '❓ Quiz Me': () => this.aiAssistant.generateQuiz(),
            '🔑 Key Points': () => this.aiAssistant.extractKeyPoints(),
            '💡 Explain': () => this.aiAssistant.explainConcepts()
        };
        
        const handler = Object.entries(actions).find(([key]) => key.includes(action))?.[1];
        if (handler) {
            handler().then(response => {
                this.addMessage('ai', response);
            }).catch(error => {
                this.addMessage('ai', `Error: ${error.message}`);
            });
        }
    }

    addMessage(type, content) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        
        if (type === 'ai') {
            messageDiv.dataset.originalText = content;
            bubbleDiv.innerHTML = this.parseAIMarkdown(content);
        } else {
            bubbleDiv.textContent = content;
        }
        
        messageDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    parseAIMarkdown(text) {
        // Use a simplified version of the markdown parser for AI messages
        // This could be enhanced to use the full TextFormatter
        let html = text;
        
        // Basic markdown parsing
        html = html.replace(/&/g, '&amp;')
                  .replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;');
        
        // Apply basic formatting
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        html = html.replace(/`(.+?)`/g, '<code>$1</code>');
        html = html.replace(/\n/g, '<br>');
        
        return html;
    }

    saveApiKey() {
        const apiKeyInput = document.getElementById('apiKey');
        if (!apiKeyInput) return;
        
        const apiKey = apiKeyInput.value.trim();
        if (apiKey) {
            this.aiAssistant.setApiKey(apiKey);
            localStorage.setItem('zenReaderApiKey', apiKey);
            this.uiController.showNotification('API key saved successfully', 'success');
        }
    }

    loadSettings() {
        // Load API key
        const savedApiKey = localStorage.getItem('zenReaderApiKey');
        if (savedApiKey) {
            this.aiAssistant.setApiKey(savedApiKey);
            const apiKeyInput = document.getElementById('apiKey');
            if (apiKeyInput) {
                apiKeyInput.value = savedApiKey;
            }
        }
        
        // Load other settings as needed
    }

    initializeFeatures() {
        // Set autoscroller target
        const textDisplay = document.getElementById('textDisplay');
        if (textDisplay) {
            this.autoScroller.setTarget(textDisplay);
        }
        
        // Initialize file input accept attribute
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.accept = this.fileHandler.getAcceptString();
        }
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.zenReader = new ZenReaderApp();
});

// Export for testing
export { ZenReaderApp };