// AutoScrollerTests.js - Tests for AutoScroller module
import { AutoScroller } from '../../js/modules/AutoScroller.js';
import { TestSuite } from '../TestSuite.js';

export class AutoScrollerTests {
    constructor() {
        this.autoScroller = new AutoScroller();
    }

    async runTests() {
        const results = [];
        const testDOM = TestSuite.createTestDOM();

        // Test 1: Module initialization
        results.push(TestSuite.assert(
            this.autoScroller instanceof AutoScroller,
            'AutoScroller instantiation',
            'Failed to create AutoScroller instance'
        ));

        // Test 2: Initial state
        results.push(TestSuite.assert(
            !this.autoScroller.isScrolling &&
            !this.autoScroller.isPaused &&
            this.autoScroller.speed === 50,
            'Initial state',
            'Incorrect initial state values'
        ));

        // Test 3: Set target element
        const testElement = testDOM.createElement('div', {
            id: 'test-scroll-container',
            style: 'height: 100px; overflow-y: auto;'
        });
        testElement.innerHTML = '<div style="height: 500px;">Content</div>';
        
        this.autoScroller.setTarget(testElement);
        results.push(TestSuite.assert(
            this.autoScroller.targetElement === testElement,
            'Set target element',
            'Failed to set target element'
        ));

        // Test 4: Speed adjustment
        const newSpeed = this.autoScroller.setSpeed(100);
        results.push(TestSuite.assert(
            newSpeed === 100,
            'Speed adjustment',
            `Expected 100 px/s, got ${newSpeed}`
        ));

        const minSpeed = this.autoScroller.setSpeed(5);
        results.push(TestSuite.assert(
            minSpeed === 10,
            'Speed minimum limit',
            `Expected 10 px/s minimum, got ${minSpeed}`
        ));

        const maxSpeed = this.autoScroller.setSpeed(250);
        results.push(TestSuite.assert(
            maxSpeed === 200,
            'Speed maximum limit',
            `Expected 200 px/s maximum, got ${maxSpeed}`
        ));

        // Test 5: Change speed delta
        this.autoScroller.setSpeed(50);
        const deltaSpeed = this.autoScroller.changeSpeed(20);
        results.push(TestSuite.assert(
            deltaSpeed === 70,
            'Change speed by delta',
            `Expected 70 px/s, got ${deltaSpeed}`
        ));

        // Test 6: Progress calculation
        testElement.scrollTop = 200;
        const progress = this.autoScroller.getProgress();
        results.push(TestSuite.assert(
            progress === 50,
            'Progress calculation',
            `Expected 50%, got ${progress}%`
        ));

        // Test 7: Jump to position
        this.autoScroller.jumpToPosition(75);
        results.push(TestSuite.assert(
            testElement.scrollTop === 300,
            'Jump to position',
            `Expected scrollTop 300, got ${testElement.scrollTop}`
        ));

        // Test 8: Reset
        this.autoScroller.reset();
        results.push(TestSuite.assert(
            testElement.scrollTop === 0,
            'Reset scroll position',
            'Failed to reset scroll position'
        ));

        // Test 9: Time remaining calculation
        testElement.scrollTop = 0;
        this.autoScroller.setSpeed(100);
        const timeRemaining = this.autoScroller.getTimeRemaining();
        results.push(TestSuite.assert(
            timeRemaining === 4,
            'Time remaining calculation',
            `Expected 4 seconds, got ${timeRemaining}`
        ));

        // Test 10: Position detection
        testElement.scrollTop = 0;
        results.push(TestSuite.assert(
            this.autoScroller.isAtStart(),
            'Detect at start',
            'Failed to detect scroll at start'
        ));

        testElement.scrollTop = 400;
        results.push(TestSuite.assert(
            this.autoScroller.isAtEnd(),
            'Detect at end',
            'Failed to detect scroll at end'
        ));

        // Test 11: Start without target
        const tempScroller = new AutoScroller();
        let errorThrown = false;
        try {
            tempScroller.start();
        } catch (e) {
            errorThrown = true;
        }
        results.push(TestSuite.assert(
            errorThrown,
            'Error on start without target',
            'Should throw error when starting without target element'
        ));

        // Test 12: Scroll to element
        const targetChild = testDOM.createElement('div', {
            id: 'scroll-target',
            style: 'margin-top: 250px; height: 50px;'
        });
        testElement.appendChild(targetChild);
        
        this.autoScroller.scrollToElement(targetChild);
        results.push(TestSuite.assert(
            Math.abs(testElement.scrollTop - 250) < 10,
            'Scroll to element',
            `Expected scrollTop ~250, got ${testElement.scrollTop}`
        ));

        // Cleanup
        testDOM.cleanup();

        return results;
    }
}