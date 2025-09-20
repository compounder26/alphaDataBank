/* Alpha Dashboard Custom JavaScript
 * Custom client-side functionality for the dashboard
 */

// Utility functions for dashboard interactions
window.alphaDashboard = {

    // Initialize dashboard-specific functionality
    init: function() {
        console.log('Alpha Dashboard initialized');
        this.setupKeyboardShortcuts();
        this.setupAccessibility();
    },

    // Keyboard shortcuts
    setupKeyboardShortcuts: function() {
        document.addEventListener('keydown', function(e) {
            // ESC to close modals
            if (e.key === 'Escape') {
                const modals = document.querySelectorAll('.modal.show');
                modals.forEach(modal => {
                    const closeBtn = modal.querySelector('.btn-close, .modal-close');
                    if (closeBtn) closeBtn.click();
                });
            }

            // Ctrl+F to focus search
            if (e.ctrlKey && e.key === 'f') {
                e.preventDefault();
                const searchInputs = document.querySelectorAll('input[type="search"], .dash-dropdown input');
                if (searchInputs.length > 0) {
                    searchInputs[0].focus();
                }
            }
        });
    },

    // Accessibility improvements
    setupAccessibility: function() {
        // Add ARIA labels to interactive elements
        const badges = document.querySelectorAll('.badge[title]');
        badges.forEach(badge => {
            if (!badge.getAttribute('aria-label')) {
                badge.setAttribute('aria-label', badge.getAttribute('title'));
            }
        });

        // Improve focus visibility
        const style = document.createElement('style');
        style.textContent = `
            *:focus {
                outline: 2px solid #0066cc !important;
                outline-offset: 2px !important;
            }
        `;
        document.head.appendChild(style);
    },

    // Copy text to clipboard
    copyToClipboard: function(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                this.showToast('Copied to clipboard');
            });
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            this.showToast('Copied to clipboard');
        }
    },

    // Simple toast notification
    showToast: function(message, duration = 3000) {
        const toast = document.createElement('div');
        toast.className = 'position-fixed top-0 end-0 m-3 alert alert-success';
        toast.style.zIndex = '9999';
        toast.textContent = message;

        document.body.appendChild(toast);

        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, duration);
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    window.alphaDashboard.init();
});