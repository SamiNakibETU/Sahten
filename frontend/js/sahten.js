/**
 * SAHTEN CLIENT (MVP)
 * Persona: Editorial Chef from L'Orient-Le Jour
 * 
 * Features:
 * - Model selection (nano/mini/auto)
 * - A/B testing support
 * - Response model tracking
 * 
 * Configuration:
 * - Set window.SAHTEN_API_BASE before loading to override API URL
 * - Or pass { apiBase: "https://..." } to constructor
 * - Pass { modelSelector: element } for model dropdown support
 */

export class SahtenChat {
    constructor(config = {}) {
        // API URL Resolution
        const host = window.location.hostname;
        const protocol = window.location.protocol;
        
        const isLocalDev = (host === 'localhost' || host === '127.0.0.1');
        const isFile = (protocol === 'file:');
        
        let defaultApiBase;
        
        if (window.SAHTEN_API_BASE) {
            defaultApiBase = window.SAHTEN_API_BASE;
        } else if (isLocalDev || isFile) {
            const apiHost = (host === '127.0.0.1') ? '127.0.0.1' : 'localhost';
            defaultApiBase = `http://${apiHost}:8000/api`;
        } else {
            defaultApiBase = '/api';
        }

        this.config = {
            apiBase: defaultApiBase,
            modelSelector: null,  // Optional model dropdown element
            ...config
        };

        this.state = {
            isOpen: false,
            isLoading: false,
            size: 'window',
            touchStartY: 0,
            lastModelUsed: null,  // Track model used in last response
        };

        this.dom = {
            container: document.querySelector('.sahten-widget-container'),
            backdrop: document.querySelector('.sahten-backdrop'),
            trigger: document.querySelector('.sahten-trigger'),
            closeBtn: document.querySelector('.sahten-close-btn'),
            body: document.querySelector('.sahten-body'),
            form: document.querySelector('#sahten-form'),
            input: document.querySelector('#sahten-input'),
            sendBtn: document.querySelector('.send-btn'),
            sizeBtns: document.querySelectorAll('.sahten-size-btn'),
            dragHandle: document.querySelector('.sahten-drag-handle'),
            modelSelector: this.config.modelSelector,
        };

        this.init();
    }

    init() {
        if (!this.dom.container) return; 
        this.bindEvents();
        this.loadModels();
        console.log(`Sahten MVP initialized. Backend: ${this.config.apiBase}`);
    }

    async loadModels() {
        /**
         * Fetch available models from API and update dropdown.
         */
        if (!this.dom.modelSelector) return;
        
        try {
            const response = await fetch(`${this.config.apiBase}/models`);
            if (response.ok) {
                const data = await response.json();
                console.log('Available models:', data.models);
                console.log('Default model:', data.default);
                console.log('A/B testing:', data.ab_testing_enabled ? 'ON' : 'OFF');
                
                // Update dropdown options if needed
                // (keeping static options for now as they match the API)
            }
        } catch (e) {
            console.log('Could not fetch models (will use static list)');
        }
    }

    getSelectedModel() {
        /**
         * Get currently selected model from dropdown.
         */
        if (!this.dom.modelSelector) return null;
        const value = this.dom.modelSelector.value;
        return value === 'auto' ? null : value;
    }

    bindEvents() {
        // Open/Close
        this.dom.trigger.addEventListener('click', () => this.toggle(true));
        this.dom.closeBtn.addEventListener('click', () => this.toggle(false));
        this.dom.backdrop.addEventListener('click', () => this.toggle(false));

        // Form Submit
        this.dom.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });

        // Size Switching (Desktop)
        this.dom.sizeBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.setSize(action);
            });
        });

        // Mobile Drag/Slide Logic
        if (this.dom.dragHandle) {
            this.dom.dragHandle.addEventListener('touchstart', (e) => {
                this.state.touchStartY = e.touches[0].clientY;
            }, { passive: true });

            this.dom.dragHandle.addEventListener('touchend', (e) => {
                const endY = e.changedTouches[0].clientY;
                const diff = endY - this.state.touchStartY;

                if (diff > 50) {
                    if (this.state.size === 'full') this.setSize('mid');
                    else if (this.state.size === 'mid') this.setSize('window');
                    else this.toggle(false);
                } 
                else if (diff < -50) {
                    if (this.state.size === 'window') this.setSize('mid');
                    else if (this.state.size === 'mid') this.setSize('full');
                }
            }, { passive: true });
        }
    }

    toggle(forceState) {
        this.state.isOpen = forceState !== undefined ? forceState : !this.state.isOpen;
        
        if (this.state.isOpen) {
            this.dom.container.setAttribute('data-state', 'open');
            this.dom.backdrop.classList.add('visible');
            this.dom.trigger.style.opacity = '0';
            this.dom.trigger.style.pointerEvents = 'none';
            setTimeout(() => this.dom.input.focus(), 100);
            
            // Welcome Message
            if (this.dom.body.children.length === 0) {
                this.appendBotMessage({
                    html: `<div class="sahten-narrative">
                        <p>Bienvenue à la table de <strong>L'Orient-Le Jour</strong>.</p>
                        <p>Je suis <em>Sahten</em>, votre chef dévoué.</p>
                        <p>Quelle saveur ou quel souvenir culinaire souhaitez-vous raviver aujourd'hui ?</p>
                    </div>`
                });
            }
        } else {
            this.dom.container.removeAttribute('data-state');
            this.dom.backdrop.classList.remove('visible');
            this.dom.trigger.style.opacity = '1';
            this.dom.trigger.style.pointerEvents = 'auto';
        }
    }

    setSize(size) {
        this.state.size = size;
        this.dom.container.setAttribute('data-size', size);
        
        this.dom.sizeBtns.forEach(btn => {
            if (btn.dataset.action === size) btn.classList.add('active');
            else btn.classList.remove('active');
        });
    }

    async sendMessage() {
        const text = this.dom.input.value.trim();
        if (!text || this.state.isLoading) return;

        this.dom.input.value = '';
        this.appendUserMessage(text);
        this.setLoading(true);

        try {
            const model = this.getSelectedModel();
            const payload = { 
                message: text, 
                debug: false,
            };
            
            // Add model if specified (not auto)
            if (model) {
                payload.model = model;
            }
            
            console.log(`Sending to ${this.config.apiBase}/chat`, payload);
            
            const response = await fetch(`${this.config.apiBase}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            
            // Track model used
            this.state.lastModelUsed = data.model_used;
            
            // Add model indicator to response if in debug/testing mode
            if (data.model_used) {
                data.html = this.addModelIndicator(data.html, data.model_used);
            }
            
            this.appendBotMessage(data);

        } catch (error) {
            console.error(error);
            this.appendBotMessage({
                html: `<div class="sahten-narrative" style="color: var(--color-accent);">
                    <p><em>Mille excuses, un petit incident en cuisine...</em></p>
                    <p>Pourriez-vous répéter votre demande ?</p>
                    <p style="font-size: 11px; opacity: 0.7; margin-top: 10px;">
                        (Debug: ${this.config.apiBase} - ${error.message || 'Réseau'})
                    </p>
                </div>`
            });
        } finally {
            this.setLoading(false);
        }
    }

    addModelIndicator(html, modelUsed) {
        /**
         * Add a small model indicator to the response HTML (for testing).
         */
        const modelName = modelUsed.includes('nano') ? '⚡ nano' : '✨ mini';
        const indicator = `<div style="font-size: 9px; opacity: 0.5; text-align: right; margin-top: 8px; font-family: monospace;">${modelName}</div>`;
        return html + indicator;
    }

    appendUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'msg msg-user';
        div.textContent = text;
        this.dom.body.appendChild(div);
        this.scrollToBottom();
    }

    appendBotMessage(data) {
        const div = document.createElement('div');
        div.className = 'msg msg-bot';
        div.innerHTML = data.html; 
        this.dom.body.appendChild(div);
        this.scrollToBottom();
    }

    setLoading(loading) {
        this.state.isLoading = loading;
        this.dom.sendBtn.disabled = loading;
        this.dom.input.disabled = loading;
        
        if (loading) {
            const loader = document.createElement('div');
            loader.className = 'msg msg-bot loading-indicator';
            loader.innerHTML = `
                <div class="loading-dots">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            `;
            this.dom.body.appendChild(loader);
            this.scrollToBottom();
            this._currentLoader = loader;
        } else {
            if (this._currentLoader) {
                this._currentLoader.remove();
                this._currentLoader = null;
            }
            this.dom.input.disabled = false;
            this.dom.input.focus();
        }
    }

    scrollToBottom() {
        this.dom.body.scrollTop = this.dom.body.scrollHeight;
    }
}
