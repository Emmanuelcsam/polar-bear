// UIControllerTests.js - Tests for UIController module
import { UIController } from '../../js/modules/UIController.js';
import { TestSuite } from '../TestSuite.js';

export class UIControllerTests {
    constructor() {
        this.uiController = new UIController();
    }

    async runTests() {
        const results = [];
        const testDOM = TestSuite.createTestDOM();

        // Test 1: Module initialization
        results.push(TestSuite.assert(
            this.uiController instanceof UIController,
            'UIController instantiation',
            'Failed to create UIController instance'
        ));

        // Test 2: Initial state
        results.push(TestSuite.assert(
            !this.uiController.isFullscreen &&
            !this.uiController.isAIFullscreen &&
            this.uiController.theme === 'dark' &&
            this.uiController.activeFeatures instanceof Set,
            'Initial state',
            'Incorrect initial state values'
        ));

        // Test 3: Keyboard shortcuts mapping
        results.push(TestSuite.assert(
            this.uiController.keyboardShortcuts['ctrl+f'] === 'toggleFullscreen' &&
            this.uiController.keyboardShortcuts['ctrl+s'] === 'startSpeedReader' &&
            this.uiController.keyboardShortcuts['escape'] === 'exitAll',
            'Keyboard shortcuts mapping',
            'Incorrect keyboard shortcuts mapping'
        ));

        // Test 4: Theme management
        const savedTheme = localStorage.getItem('zenReaderTheme');
        
        this.uiController.setTheme('light');
        results.push(TestSuite.assert(
            this.uiController.theme === 'light' &&
            document.body.classList.contains('light-theme'),
            'Set light theme',
            'Failed to set light theme'
        ));

        const toggledTheme = this.uiController.toggleTheme();
        results.push(TestSuite.assert(
            toggledTheme === 'dark' &&
            !document.body.classList.contains('light-theme'),
            'Toggle theme',
            'Failed to toggle theme'
        ));

        // Restore saved theme
        if (savedTheme) {
            localStorage.setItem('zenReaderTheme', savedTheme);
        }

        // Test 5: Feature management
        const isActive = this.uiController.toggleFeature('bionic');
        results.push(TestSuite.assert(
            isActive && this.uiController.activeFeatures.has('bionic'),
            'Activate feature',
            'Failed to activate feature'
        ));

        const isInactive = this.uiController.toggleFeature('bionic');
        results.push(TestSuite.assert(
            !isInactive && !this.uiController.activeFeatures.has('bionic'),
            'Deactivate feature',
            'Failed to deactivate feature'
        ));

        // Test 6: Key combo parsing
        const mockEvent = {
            ctrlKey: true,
            key: 'f',
            preventDefault: () => {}
        };
        const combo = this.uiController.getKeyCombo(mockEvent);
        results.push(TestSuite.assert(
            combo === 'ctrl+f',
            'Key combo parsing',
            `Expected 'ctrl+f', got '${combo}'`
        ));

        // Test 7: Feature state checking
        this.uiController.activeFeatures.add('syntax');
        results.push(TestSuite.assert(
            this.uiController.isFeatureActive('syntax'),
            'Check active feature',
            'Failed to detect active feature'
        ));

        results.push(TestSuite.assert(
            !this.uiController.isFeatureActive('pomodoro'),
            'Check inactive feature',
            'Failed to detect inactive feature'
        ));

        // Test 8: Get active features
        this.uiController.activeFeatures.clear();
        this.uiController.activeFeatures.add('fullscreen');
        this.uiController.activeFeatures.add('bionic');
        
        const activeFeatures = this.uiController.getActiveFeatures();
        results.push(TestSuite.assert(
            Array.isArray(activeFeatures) &&
            activeFeatures.length === 2 &&
            activeFeatures.includes('fullscreen') &&
            activeFeatures.includes('bionic'),
            'Get active features',
            'Failed to get active features list'
        ));

        // Test 9: Notification creation
        const notification = this.uiController.createLoadingSpinner('Testing...');
        results.push(TestSuite.assert(
            notification.className === 'loading-spinner' &&
            notification.innerHTML.includes('Testing...'),
            'Create loading spinner',
            'Failed to create loading spinner'
        ));

        // Test 10: Font size controls (with mock elements)
        const textContent = testDOM.createElement('div', {
            id: 'textContent',
            style: 'font-size: 16px;'
        });
        
        // Mock getElementById
        const originalGetElementById = document.getElementById;
        document.getElementById = (id) => {
            if (id === 'textContent') return textContent;
            return originalGetElementById.call(document, id);
        };

        this.uiController.increaseFontSize();
        const increasedSize = parseFloat(textContent.style.fontSize);
        results.push(TestSuite.assert(
            increasedSize > 16,
            'Increase font size',
            `Expected > 16px, got ${increasedSize}px`
        ));

        this.uiController.decreaseFontSize();
        const decreasedSize = parseFloat(textContent.style.fontSize);
        results.push(TestSuite.assert(
            decreasedSize < increasedSize,
            'Decrease font size',
            'Failed to decrease font size'
        ));

        this.uiController.resetFontSize();
        results.push(TestSuite.assert(
            textContent.style.fontSize === '',
            'Reset font size',
            'Failed to reset font size'
        ));

        // Restore getElementById
        document.getElementById = originalGetElementById;

        // Test 11: Exit all functionality
        this.uiController.isFullscreen = true;
        this.uiController.isAIFullscreen = true;
        this.uiController.exitAll();
        
        results.push(TestSuite.assert(
            !this.uiController.isFullscreen && !this.uiController.isAIFullscreen,
            'Exit all modes',
            'Failed to exit all modes'
        ));

        // Test 12: Update feature UI (mock test)
        let updateCalled = false;
        this.uiController.updateFullscreenUI = (isActive) => {
            updateCalled = true;
        };
        
        this.uiController.updateFeatureUI('fullscreen', true);
        results.push(TestSuite.assert(
            updateCalled,
            'Update feature UI',
            'Failed to call update method'
        ));

        // Cleanup
        testDOM.cleanup();
        document.body.classList.remove('light-theme');

        return results;
    }
}