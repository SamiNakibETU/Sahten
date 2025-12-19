/**
 * SAHTEN CLIENT (V3.1 Responsive)
 * Persona: Editorial Chef from L'Orient-Le Jour
 */

export class SahtenChat {
    constructor(config) {
        // Durable URL logic
        const host = window.location.hostname;
        const protocol = window.location.protocol;
        
        let apiHost = 'localhost';
        if (host === '127.0.0.1') apiHost = '127.0.0.1';

        const isLocalDev = (host === 'localhost' || host === '127.0.0.1');
        const isFile = (protocol === 'file:');
        
        const defaultApiBase = (isLocalDev || isFile) 
            ? `http://${apiHost}:8000/api` 
            : '/api';

        this.config = {
            apiBase: defaultApiBase, 
            ...config
        };

        this.state = {
            isOpen: false,
            isLoading: false,
            size: 'window', // window | mid | full
            touchStartY: 0
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
            dragHandle: document.querySelector('.sahten-drag-handle')
        };

        this.init();
    }

    init() {
        if (!this.dom.container) return; 
        this.bindEvents();
        console.log(`Sahten initialized. Backend URL: ${this.config.apiBase}`);
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

                // Swipe Down (> 50px) -> Minimize or Close
                if (diff > 50) {
                    if (this.state.size === 'full') this.setSize('mid');
                    else if (this.state.size === 'mid') this.setSize('window');
                    else this.toggle(false);
                } 
                // Swipe Up (> 50px) -> Expand
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
            
            // Welcome Message - Editorial Chef Persona
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
            // Reset size on close for consistency? Optional.
            // this.setSize('window'); 
        }
    }

    setSize(size) {
        this.state.size = size;
        this.dom.container.setAttribute('data-size', size);
        
        // Update active button state
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
            console.log(`Sending message to: ${this.config.apiBase}/chat`);
            const response = await fetch(`${this.config.apiBase}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, debug: false })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            this.appendBotMessage(data);

        } catch (error) {
            console.error(error);
            // Polite, in-character error message with DEBUG info
            this.appendBotMessage({
                html: `<div class="sahten-narrative" style="color: var(--color-accent);">
                    <p><em>Mille excuses, un petit incident en cuisine...</em></p>
                    <p>Pourriez-vous répéter votre demande ?</p>
                    <p style="font-size: 11px; opacity: 0.7; margin-top: 10px;">
                        (Debug: Impossible de joindre <code>${this.config.apiBase}</code>.<br>
                        Page: <code>${window.location.origin}</code><br>
                        Erreur: ${error.message || 'Réseau'})
                    </p>
                </div>`
            });
        } finally {
            this.setLoading(false);
        }
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
        this.dom.input.disabled = loading; // Prevent double submit
        
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
