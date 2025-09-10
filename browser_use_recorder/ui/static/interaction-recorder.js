/**
 * Interaction Recorder for browser-use UI
 * 
 * Captures user interactions and sends them to the backend for recording.
 * Handles click, typing, scroll, and navigation events with intelligent deduplication.
 */
(function() {
    'use strict';

    const CONFIG = {
        TYPING_DEBOUNCE_MS: 800,
        KEY_DEBOUNCE_MS: 200,
        CLICK_DEBOUNCE_MS: 200,
        SCROLL_DEBOUNCE_MS: 300,
        MAX_TEXT_LENGTH: 200,
        SPECIAL_KEYS: ['Enter']
    };

    const state = {
        typingBuffer: { element: null, value: null, timeoutId: null },
        lastEvent: { signature: null, timestamp: 0 },
        scrollTimeoutId: null,
    };

    function getElementXPath(element) {
        if (!element || element.nodeType !== Node.ELEMENT_NODE) return null;
        
        // Prefer unique ID
        if (element.id && document.querySelectorAll(`#${element.id}`).length === 1) {
            return `id("${element.id}")`;
        }

        // Build structural XPath
        const parts = [];
        let current = element;
        while (current && current.nodeType === Node.ELEMENT_NODE && current.tagName !== 'BODY' && current.tagName !== 'HTML') {
            let index = 1;
            for (let sibling = current.previousElementSibling; sibling; sibling = sibling.previousElementSibling) {
                if (sibling.tagName === current.tagName) index++;
            }
            parts.unshift(`${current.tagName.toLowerCase()}[${index}]`);
            current = current.parentElement;
        }
        return parts.length ? `//${parts.join('/')}` : null;
    }

    function getElementContext(element) {
        return {
            tag: element.tagName,
            inputType: element.type || null,
            name: element.name || null,
            id: element.id || null,
            text: element.textContent ? element.textContent.slice(0, CONFIG.MAX_TEXT_LENGTH).trim() : null,
            value: element.value,
            xpath: getElementXPath(element),
        };
    }

    function logInteraction(data) {
        console.log('__browser_use_interaction__', JSON.stringify(data));
    }

    function shouldDebounce(signature, debounceMs) {
        const now = Date.now();
        if (state.lastEvent.signature === signature && (now - state.lastEvent.timestamp) < debounceMs) {
            return true;
        }
        state.lastEvent.signature = signature;
        state.lastEvent.timestamp = now;
        return false;
    }

    function commitTyping() {
        const { element, value, timeoutId } = state.typingBuffer;
        if (!element) return;

        clearTimeout(timeoutId);
        logInteraction({
            type: 'type',
            timestamp: Date.now(),
            ...getElementContext(element),
            value: value,
        });

        // Reset buffer
        state.typingBuffer.element = null;
        state.typingBuffer.value = null;
        state.typingBuffer.timeoutId = null;
    }

    function handleClick(event) {
        commitTyping(); // Commit any pending typing
        const element = event.target;
        const signature = `click:${getElementXPath(element)}`;
        if (shouldDebounce(signature, CONFIG.CLICK_DEBOUNCE_MS)) return;

        logInteraction({
            type: 'click',
            timestamp: Date.now(),
            ...getElementContext(element),
        });
    }

    function handleInput(event) {
        const element = event.target;
        if (!['INPUT', 'TEXTAREA'].includes(element.tagName)) return;

        // If typing in a new element, commit previous typing
        if (state.typingBuffer.element && state.typingBuffer.element !== element) {
            commitTyping();
        }

        clearTimeout(state.typingBuffer.timeoutId);
        state.typingBuffer.element = element;
        state.typingBuffer.value = element.value;
        state.typingBuffer.timeoutId = setTimeout(commitTyping, CONFIG.TYPING_DEBOUNCE_MS);
    }

    function handleChange(event) {
        const element = event.target;
        if (['INPUT', 'TEXTAREA'].includes(element.tagName)) {
            commitTyping();
        }
    }

    function handleKeydown(event) {
        if (event.key === 'Enter') {
            commitTyping();
        }

        if (!CONFIG.SPECIAL_KEYS.includes(event.key)) return;

        const element = event.target;
        const signature = `key:${getElementXPath(element)}:${event.key}`;
        if (shouldDebounce(signature, CONFIG.KEY_DEBOUNCE_MS)) return;

        logInteraction({
            type: 'key',
            timestamp: Date.now(),
            ...getElementContext(element),
            key: event.key,
        });
    }

    function handleScroll() {
        clearTimeout(state.scrollTimeoutId);
        state.scrollTimeoutId = setTimeout(() => {
            const element = document.scrollingElement || document.documentElement;
            logInteraction({
                type: 'scroll',
                timestamp: Date.now(),
                ...getElementContext(element),
                scrollTop: element.scrollTop,
                scrollLeft: element.scrollLeft,
            });
        }, CONFIG.SCROLL_DEBOUNCE_MS);
    }

    function initialize() {
        // Clear typing on page unload
        window.addEventListener('beforeunload', () => commitTyping());

        // Add event listeners
        document.addEventListener('click', handleClick, { capture: true });
        document.addEventListener('input', handleInput, { capture: true });
        document.addEventListener('change', handleChange, { capture: true });
        document.addEventListener('keydown', handleKeydown, { capture: true });
        document.addEventListener('scroll', handleScroll, { capture: true, passive: true });
        
        console.log('🎯 Interaction Recorder initialized');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
})();
