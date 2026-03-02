/**
 * Web Accessibility Example JavaScript
 * - Accordion
 * - Tab component
 * - Modal
 * - Custom listbox
 * - Focus management
 */

// ============================================
// Accordion
// ============================================
function initAccordion() {
    const button = document.getElementById('accordion-btn');
    const content = document.getElementById('accordion-content');

    if (!button || !content) return;

    // Why: Toggling aria-expanded tells assistive technology whether content is visible,
    // while the hidden attribute handles visual show/hide - both must stay in sync
    button.addEventListener('click', () => {
        const isExpanded = button.getAttribute('aria-expanded') === 'true';

        button.setAttribute('aria-expanded', !isExpanded);
        content.hidden = isExpanded;
    });
}

// ============================================
// Tab Component
// ============================================
function initTabs() {
    const tablist = document.querySelector('[role="tablist"]');
    if (!tablist) return;

    const tabs = tablist.querySelectorAll('[role="tab"]');
    const panels = document.querySelectorAll('[role="tabpanel"]');

    // Tab click events
    tabs.forEach(tab => {
        tab.addEventListener('click', () => activateTab(tab, tabs, panels));
    });

    // Why: Arrow key navigation follows the WAI-ARIA tabs pattern, where left/right arrows move
    // between tabs and Home/End jump to first/last - matching user expectations from native widgets
    // Keyboard navigation
    tablist.addEventListener('keydown', (e) => {
        const currentTab = document.activeElement;
        const currentIndex = Array.from(tabs).indexOf(currentTab);

        let newIndex;

        switch (e.key) {
            case 'ArrowLeft':
                newIndex = currentIndex - 1;
                if (newIndex < 0) newIndex = tabs.length - 1;
                break;
            case 'ArrowRight':
                newIndex = currentIndex + 1;
                if (newIndex >= tabs.length) newIndex = 0;
                break;
            case 'Home':
                newIndex = 0;
                break;
            case 'End':
                newIndex = tabs.length - 1;
                break;
            default:
                return;
        }

        e.preventDefault();
        tabs[newIndex].focus();
        activateTab(tabs[newIndex], tabs, panels);
    });
}

function activateTab(selectedTab, tabs, panels) {
    // Deactivate all tabs
    tabs.forEach(tab => {
        tab.setAttribute('aria-selected', 'false');
        tab.setAttribute('tabindex', '-1');
    });

    // Hide all panels
    panels.forEach(panel => {
        panel.hidden = true;
    });

    // Activate selected tab
    selectedTab.setAttribute('aria-selected', 'true');
    selectedTab.setAttribute('tabindex', '0');

    // Show corresponding panel
    const panelId = selectedTab.getAttribute('aria-controls');
    const panel = document.getElementById(panelId);
    if (panel) {
        panel.hidden = false;
    }
}

// ============================================
// Live Region Demo
// ============================================
function initLiveRegion() {
    const button = document.getElementById('update-live-btn');
    const liveRegion = document.getElementById('live-region');

    if (!button || !liveRegion) return;

    const messages = [
        'A new notification has arrived.',
        'Task completed successfully.',
        'You have 3 new messages.',
        'File uploaded successfully.',
        'Settings have been saved.'
    ];

    let messageIndex = 0;

    button.addEventListener('click', () => {
        liveRegion.textContent = messages[messageIndex];
        messageIndex = (messageIndex + 1) % messages.length;
    });
}

// ============================================
// Modal
// ============================================
let previousFocusElement = null;

function initModal() {
    const openBtn = document.getElementById('open-modal-btn');
    const modal = document.getElementById('modal');
    const overlay = document.getElementById('modal-overlay');
    const closeBtn = document.getElementById('modal-close');
    const confirmBtn = document.getElementById('modal-confirm');

    if (!openBtn || !modal) return;

    // Open
    openBtn.addEventListener('click', () => openModal(modal, overlay));

    // Close button
    closeBtn?.addEventListener('click', () => closeModal(modal, overlay));
    confirmBtn?.addEventListener('click', () => {
        alert('Confirmed!');
        closeModal(modal, overlay);
    });

    // Overlay click
    overlay?.addEventListener('click', () => closeModal(modal, overlay));

    // ESC key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.hidden) {
            closeModal(modal, overlay);
        }
    });

    // Why: Focus trap keeps keyboard navigation confined within the modal, preventing users
    // from tabbing into obscured background content - a WCAG 2.1 requirement for modal dialogs
    // Focus trap
    modal.addEventListener('keydown', (e) => {
        if (e.key !== 'Tab') return;

        const focusableElements = modal.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        if (e.shiftKey && document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
        }
    });
}

// Why: Saving and restoring focus ensures the user returns to their previous position after
// closing the modal, maintaining context and meeting WCAG focus management guidelines
function openModal(modal, overlay) {
    // Save current focus
    previousFocusElement = document.activeElement;

    // Show modal
    modal.hidden = false;
    overlay.hidden = false;

    // Prevent background scrolling
    document.body.style.overflow = 'hidden';

    // Focus on first focusable element
    const firstFocusable = modal.querySelector('button, [href], input');
    if (firstFocusable) {
        firstFocusable.focus();
    }
}

function closeModal(modal, overlay) {
    modal.hidden = true;
    overlay.hidden = true;

    // Restore background scrolling
    document.body.style.overflow = '';

    // Restore previous focus
    if (previousFocusElement) {
        previousFocusElement.focus();
    }
}

// ============================================
// Custom Listbox
// ============================================
function initListbox() {
    const listbox = document.getElementById('custom-listbox');
    const output = document.getElementById('listbox-output');

    if (!listbox) return;

    const options = listbox.querySelectorAll('[role="option"]');
    let currentIndex = 0;

    // Set initial selection state
    updateSelection(options, currentIndex);

    listbox.addEventListener('keydown', (e) => {
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                currentIndex = Math.min(currentIndex + 1, options.length - 1);
                updateSelection(options, currentIndex);
                break;
            case 'ArrowUp':
                e.preventDefault();
                currentIndex = Math.max(currentIndex - 1, 0);
                updateSelection(options, currentIndex);
                break;
            case 'Home':
                e.preventDefault();
                currentIndex = 0;
                updateSelection(options, currentIndex);
                break;
            case 'End':
                e.preventDefault();
                currentIndex = options.length - 1;
                updateSelection(options, currentIndex);
                break;
            case 'Enter':
            case ' ':
                e.preventDefault();
                selectOption(options[currentIndex], output);
                break;
        }
    });

    // Select with click
    options.forEach((option, index) => {
        option.addEventListener('click', () => {
            currentIndex = index;
            updateSelection(options, currentIndex);
            selectOption(option, output);
        });
    });
}

function updateSelection(options, index) {
    options.forEach((option, i) => {
        option.setAttribute('aria-selected', i === index);
    });

    // Update aria-activedescendant on listbox
    const listbox = options[0]?.parentElement;
    if (listbox) {
        listbox.setAttribute('aria-activedescendant', options[index].id);
    }
}

function selectOption(option, output) {
    const text = option.textContent.trim();
    const itemName = text.replace(/^.+\s/, ''); // Remove emoji
    output.textContent = `Selected: ${itemName}`;
}

// ============================================
// Form Validation
// ============================================
function initFormValidation() {
    const form = document.getElementById('accessible-form');
    if (!form) return;

    const emailInput = document.getElementById('email');
    const emailError = document.getElementById('email-error');

    // Why: Validating on blur (not every keystroke) avoids premature error messages,
    // while re-validating on input only when already invalid gives instant feedback during correction
    // Real-time email validation
    emailInput?.addEventListener('blur', () => {
        validateEmail(emailInput, emailError);
    });

    emailInput?.addEventListener('input', () => {
        if (emailInput.getAttribute('aria-invalid') === 'true') {
            validateEmail(emailInput, emailError);
        }
    });

    // Form submission
    form.addEventListener('submit', (e) => {
        e.preventDefault();

        let isValid = true;

        // Email validation
        if (!validateEmail(emailInput, emailError)) {
            isValid = false;
        }

        // Required field validation
        const requiredFields = form.querySelectorAll('[required]');
        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                isValid = false;
                field.setAttribute('aria-invalid', 'true');
            } else {
                field.setAttribute('aria-invalid', 'false');
            }
        });

        if (isValid) {
            alert('Form submitted successfully!');
            form.reset();
        } else {
            // Focus on first error field
            const firstError = form.querySelector('[aria-invalid="true"]');
            if (firstError) {
                firstError.focus();
            }
        }
    });
}

function validateEmail(input, errorElement) {
    if (!input || !errorElement) return true;

    const email = input.value.trim();
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!email) {
        showError(input, errorElement, 'Please enter an email address.');
        return false;
    }

    if (!emailRegex.test(email)) {
        showError(input, errorElement, 'Please enter a valid email format.');
        return false;
    }

    hideError(input, errorElement);
    return true;
}

function showError(input, errorElement, message) {
    input.setAttribute('aria-invalid', 'true');
    errorElement.textContent = message;
    errorElement.hidden = false;
}

function hideError(input, errorElement) {
    input.setAttribute('aria-invalid', 'false');
    errorElement.textContent = '';
    errorElement.hidden = true;
}

// ============================================
// Focus Trap Demo
// ============================================
function initFocusTrap() {
    const trapArea = document.getElementById('focus-trap-demo');
    if (!trapArea) return;

    const focusableElements = trapArea.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    trapArea.addEventListener('keydown', (e) => {
        if (e.key !== 'Tab') return;

        if (e.shiftKey && document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
        }
    });
}

// ============================================
// Initialize
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initAccordion();
    initTabs();
    initLiveRegion();
    initModal();
    initListbox();
    initFormValidation();
    initFocusTrap();

    console.log('Web accessibility examples loaded.');
    console.log('Try testing with a screen reader or keyboard!');
});
