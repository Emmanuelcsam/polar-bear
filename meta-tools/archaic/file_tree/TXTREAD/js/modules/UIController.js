// UIController.js - UI state management and interactions
export class UIController {
    constructor() {
        this.isFullscreen = false;
        this.isAIFullscreen = false;
        this.theme = 'dark';
        this.activeFeatures = new Set();
        this.keyboardShortcuts = {
            'ctrl+f': 'toggleFullscreen',
            'ctrl+s': 'startSpeedReader',
            'ctrl+b': 'toggleBionicMode',
            'ctrl+p': 'togglePomodoro',
            'ctrl+l': 'toggleTTS',
            'ctrl+h': 'toggleSyntaxHighlighting',
            'ctrl+a': 'toggleAIFullscreen',
            'ctrl+r': 'toggleAutoscroll',
            'escape': 'exitAll'
        };
    }

    init() {
        this.setupKeyboardShortcuts();
        this.setupTheme();
        this.initializeFeatures();
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Don't trigger shortcuts when typing in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            const key = this.getKeyCombo(e);
            const action = this.keyboardShortcuts[key];

            if (action) {
                e.preventDefault();
                this.handleShortcut(action);
            }
        });
    }

    getKeyCombo(event) {
        const keys = [];
        if (event.ctrlKey || event.metaKey) keys.push('ctrl');
        if (event.altKey) keys.push('alt');
        if (event.shiftKey) keys.push('shift');
        
        if (event.key && event.key !== 'Control' && event.key !== 'Alt' && 
            event.key !== 'Shift' && event.key !== 'Meta') {
            keys.push(event.key.toLowerCase());
        }
        
        return keys.join('+');
    }

    handleShortcut(action) {
        if (this.shortcuts && this.shortcuts[action]) {
            this.shortcuts[action]();
        }
    }

    setShortcutHandlers(handlers) {
        this.shortcuts = handlers;
    }

    setupTheme() {
        // Check for saved theme preference
        const savedTheme = localStorage.getItem('zenReaderTheme') || 'dark';
        this.setTheme(savedTheme);
    }

    setTheme(theme) {
        this.theme = theme;
        document.body.classList.toggle('light-theme', theme === 'light');
        localStorage.setItem('zenReaderTheme', theme);
        return this.theme;
    }

    toggleTheme() {
        return this.setTheme(this.theme === 'dark' ? 'light' : 'dark');
    }

    initializeFeatures() {
        // Initialize feature states
        this.activeFeatures.clear();
    }

    toggleFeature(featureName) {
        if (this.activeFeatures.has(featureName)) {
            this.activeFeatures.delete(featureName);
            this.updateFeatureUI(featureName, false);
            return false;
        } else {
            this.activeFeatures.add(featureName);
            this.updateFeatureUI(featureName, true);
            return true;
        }
    }

    updateFeatureUI(feature, isActive) {
        // Update feature menu item
        const featureMap = {
            'fullscreen': 0,
            'speed': 1,
            'bionic': 2,
            'focus': 3,
            'pomodoro': 4,
            'tts': 5,
            'theme': 6,
            'increase': 7,
            'decrease': 8,
            'syntax': 9,
            'autoscroll': 10
        };

        const items = document.querySelectorAll('.feature-item');
        const index = featureMap[feature];

        if (index !== undefined && items[index]) {
            items[index].classList.toggle('active', isActive);
        }

        // Update specific UI elements
        switch (feature) {
            case 'fullscreen':
                this.updateFullscreenUI(isActive);
                break;
            case 'pomodoro':
                this.updatePomodoroUI(isActive);
                break;
            case 'tts':
                this.updateTTSUI(isActive);
                break;
            case 'autoscroll':
                this.updateAutoscrollUI(isActive);
                break;
            case 'focus':
                this.updateFocusGradientUI(isActive);
                break;
        }
    }

    updateFullscreenUI(isActive) {
        const textDisplay = document.getElementById('textDisplay');
        if (textDisplay) {
            textDisplay.classList.toggle('fullscreen', isActive);
        }
        this.isFullscreen = isActive;
    }

    updatePomodoroUI(isActive) {
        const widget = document.getElementById('pomodoroWidget');
        if (widget) {
            widget.classList.toggle('active', isActive);
        }
    }

    updateTTSUI(isActive) {
        const controls = document.getElementById('ttsControls');
        if (controls) {
            controls.classList.toggle('active', isActive);
        }
    }

    updateAutoscrollUI(isActive) {
        const controls = document.getElementById('autoscrollControls');
        if (controls) {
            controls.classList.toggle('active', isActive);
        }
    }

    updateFocusGradientUI(isActive) {
        const textContent = document.getElementById('textContent');
        if (textContent) {
            textContent.classList.toggle('focus-gradient', isActive);
        }
    }

    toggleAIFullscreen() {
        const aiPanel = document.getElementById('aiPanel');
        const btn = document.getElementById('aiFullscreenBtn');
        const contentArea = document.querySelector('.content-area');
        const textDisplay = document.getElementById('textDisplay');
        
        this.isAIFullscreen = !this.isAIFullscreen;

        if (this.isAIFullscreen) {
            aiPanel.classList.add('fullscreen');
            btn.innerHTML = '✕';
            btn.title = 'Exit fullscreen';
            contentArea.style.gridTemplateColumns = '1fr';
            textDisplay.style.display = 'none';
        } else {
            aiPanel.classList.remove('fullscreen');
            btn.innerHTML = '⛶';
            btn.title = 'Toggle fullscreen';
            contentArea.style.gridTemplateColumns = '1fr 400px';
            textDisplay.style.display = 'block';
        }

        return this.isAIFullscreen;
    }

    showSpeedReaderModal() {
        const modal = document.getElementById('speedReaderModal');
        if (modal) {
            modal.classList.add('active');
        }
    }

    hideSpeedReaderModal() {
        const modal = document.getElementById('speedReaderModal');
        if (modal) {
            modal.classList.remove('active');
        }
    }

    toggleFeatureMenu() {
        const menu = document.getElementById('featuresMenu');
        if (menu) {
            menu.classList.toggle('active');
        }
    }

    updateReadingProgress(percentage) {
        const progressBar = document.querySelector('.reading-progress-bar');
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
        }

        // Show/hide progress bar
        const progressContainer = document.querySelector('.reading-progress');
        if (progressContainer) {
            progressContainer.classList.toggle('active', percentage > 0);
        }
    }

    updateStats(stats) {
        const elements = {
            wordCount: stats.wordCount,
            readingTime: stats.readingTime,
            progressPercent: stats.readingProgress + '%',
            sessionTime: stats.sessionTime
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }

    showNotification(message, type = 'info', duration = 3000) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;

        // Add to body
        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => notification.classList.add('show'), 10);

        // Remove after duration
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }

    createLoadingSpinner(text = 'Loading...') {
        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';
        spinner.innerHTML = `
            <div class="spinner"></div>
            <p>${text}</p>
        `;
        return spinner;
    }

    increaseFontSize() {
        const textContent = document.getElementById('textContent');
        if (textContent) {
            const currentSize = parseFloat(window.getComputedStyle(textContent).fontSize);
            textContent.style.fontSize = `${currentSize * 1.1}px`;
        }
    }

    decreaseFontSize() {
        const textContent = document.getElementById('textContent');
        if (textContent) {
            const currentSize = parseFloat(window.getComputedStyle(textContent).fontSize);
            textContent.style.fontSize = `${currentSize * 0.9}px`;
        }
    }

    resetFontSize() {
        const textContent = document.getElementById('textContent');
        if (textContent) {
            textContent.style.fontSize = '';
        }
    }

    exitAll() {
        // Exit fullscreen modes
        if (this.isFullscreen) {
            this.toggleFeature('fullscreen');
        }
        if (this.isAIFullscreen) {
            this.toggleAIFullscreen();
        }
        
        // Close modals
        this.hideSpeedReaderModal();
        
        // Close feature panels
        const panels = ['autoscrollControls', 'ttsControls'];
        panels.forEach(id => {
            const panel = document.getElementById(id);
            if (panel && panel.classList.contains('active')) {
                panel.classList.remove('active');
            }
        });
    }

    getActiveFeatures() {
        return Array.from(this.activeFeatures);
    }

    isFeatureActive(feature) {
        return this.activeFeatures.has(feature);
    }
}