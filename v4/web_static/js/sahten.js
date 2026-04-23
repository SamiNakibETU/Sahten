/**
 * Sahteïn — client widget (v2.2)
 * L’Orient-Le Jour · assistant recettes
 * 
 * Features:
 * - Model selection (nano/mini/auto)
 * - A/B testing support
 * - Response model tracking
 * - XSS Protection via DOMPurify
 * - Feedback collection (👍/👎)
 * 
 * Configuration:
 * - Set window.SAHTEN_API_BASE before loading to override API URL
 * - Or pass { apiBase: "https://..." } to constructor
 * - Pass { modelSelector: element } for model dropdown support
 * 
 * Security:
 * - All HTML responses are sanitized via DOMPurify before rendering
 * - Only whitelisted tags/attributes are allowed
 */

// Allowed HTML elements for sanitization
const ALLOWED_TAGS = [
    'div', 'p', 'span', 'strong', 'em', 'u', 'br',
    'a', 'article', 'section', 'h2', 'h3', 'header',
    'ul', 'li', 'ol',
    'button',
    'blockquote',  // For recipe citations/grounding
    'img',  // For OLJ logo in welcome
];

const ALLOWED_ATTR = [
    'id',
    'class',
    'href',
    'target',
    'style',
    'lang',
    'aria-label',
    'aria-labelledby',
    'aria-describedby',
    'aria-hidden',
    'role',
    'type',
    'title',
    'src',
    'alt',
    'width',
    'height',
    'decoding',
    'data-markers',
    'data-center',
];

/**
 * Sanitize HTML using DOMPurify (if available) or basic sanitization
 */
function sanitizeHTML(html) {
    if (typeof DOMPurify !== 'undefined') {
        return DOMPurify.sanitize(html, {
            ALLOWED_TAGS: ALLOWED_TAGS,
            ALLOWED_ATTR: ALLOWED_ATTR,
            ALLOW_DATA_ATTR: false,
        });
    }
    // Fallback: basic sanitization using browser's DOM parser
    // This preserves HTML structure while removing scripts
    // DOMPurify not loaded, using basic sanitization
    const temp = document.createElement('div');
    temp.innerHTML = html;
    
    // Remove potentially dangerous elements
    /* Pas de suppression des <button> : accueil (exemples de questions) ; handlers on* retirés ci-dessous */
    const dangerous = temp.querySelectorAll('script, iframe, object, embed, form, input');
    dangerous.forEach(el => el.remove());
    
    // Remove event handlers from all elements
    temp.querySelectorAll('*').forEach(el => {
        Array.from(el.attributes).forEach(attr => {
            if (attr.name.startsWith('on')) {
                el.removeAttribute(attr.name);
            }
        });
    });
    
    return temp.innerHTML;
}

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
            // Streaming désactivé tant que POST /api/chat/stream n'est pas implémenté (évite 404 inutiles en prod).
            useStream: false,
            ...config
        };

        this.state = {
            isOpen: false,
            isLoading: false,
            size: 'window',
            touchStartY: 0,
            lastModelUsed: null,  // Track model used in last response
            sessionId: this.generateSessionId(),  // For conversation memory
            debugMode: new URLSearchParams(window.location.search).get('debug') === '1',
        };

        this.dom = {
            container: document.querySelector('.sahten-widget-container'),
            backdrop: document.querySelector('.sahten-backdrop'),
            trigger: document.querySelector('.sahten-trigger'),
            triggerWrap: null,
            triggerBubble: null,
            closeBtn: document.querySelector('.sahten-close-btn'),
            body: document.querySelector('.sahten-body'),
            form: document.querySelector('#sahten-form'),
            input: document.querySelector('#sahten-input'),
            sendBtn: document.querySelector('.send-btn'),
            sizeBtns: document.querySelectorAll('.sahten-size-btn'),
            dragHandle: document.querySelector('.sahten-drag-handle'),
            modelSelector: this.config.modelSelector,
        };

        this._a11yLive = null;
        /** Texte affiché à gauche du bouton flottant (sans survol). */
        this.TRIGGER_HINT_TEXT = 'Une idée recette ?';
        /** Sauvegarde des derniers échanges côté client (complète la session serveur / Redis). */
        this._localTurnsKey = 'sahten_v4_conversation';
        this._maxLocalTurns = 5;

        this.init();
    }

    init() {
        if (!this.dom.container) return;
        this.setupTriggerShell();
        this.bindEvents();
        this.ensureA11yLiveRegion();
        const initialSize = this.dom.container.dataset.size || this.state.size;
        this.setSize(initialSize);
        this._restoreLocalHistoryIfAny();
        this.loadModels();
    }

    /**
     * Bulle fixe à gauche du logo (texte permanent, pas seulement au survol).
     */
    setupTriggerShell() {
        const btn = this.dom.trigger;
        if (!btn) return;
        const existing = btn.closest('.sahten-trigger-wrap');
        if (existing) {
            this.dom.triggerWrap = existing;
            existing.setAttribute('title', 'Ouvrir Sahteïn');
            this.dom.triggerBubble = existing.querySelector('.sahten-trigger-bubble');
            if (this.dom.triggerBubble) {
                this.dom.triggerBubble.textContent = this.TRIGGER_HINT_TEXT;
                this.dom.triggerBubble.removeAttribute('aria-hidden');
            }
            return;
        }
        const wrap = document.createElement('div');
        wrap.className = 'sahten-trigger-wrap';
        const bubble = document.createElement('div');
        bubble.className = 'sahten-trigger-bubble';
        bubble.textContent = this.TRIGGER_HINT_TEXT;
        btn.parentNode.insertBefore(wrap, btn);
        wrap.appendChild(bubble);
        wrap.appendChild(btn);
        wrap.setAttribute('title', 'Ouvrir Sahteïn');
        this.dom.triggerWrap = wrap;
        this.dom.triggerBubble = bubble;
    }

    /** Zone lue par les lecteurs d'écran à l'ouverture du panneau. */
    ensureA11yLiveRegion() {
        if (this._a11yLive || !this.dom.container) return;
        const el = document.createElement('div');
        el.className = 'sahten-sr-only';
        el.setAttribute('role', 'status');
        el.setAttribute('aria-live', 'polite');
        el.setAttribute('aria-atomic', 'true');
        this.dom.container.appendChild(el);
        this._a11yLive = el;
    }

    generateSessionId() {
        /**
         * Generate a unique session ID for conversation memory.
         * Uses sessionStorage (per browser tab) so that:
         *   - "une autre" still works within the same tab across page reloads.
         *   - A new tab / new browser session starts fresh with no exclusions.
         * This avoids the permanent accumulation of exclusions that would prevent
         * recipes from ever being suggested again after repeated testing.
         */
        const storageKey = 'sahten_session_id';
        let storage;
        try {
            storage = window.sessionStorage;
            storage.setItem('__sahten_test__', '1');
            storage.removeItem('__sahten_test__');
        } catch (_) {
            // sessionStorage unavailable (unlikely) — fall back to in-memory (no persistence)
            storage = { _m: {}, getItem(k) { return this._m[k] ?? null; }, setItem(k, v) { this._m[k] = v; } };
        }
        let sessionId = storage.getItem(storageKey);
        
        if (!sessionId) {
            sessionId = 'ses_' + Date.now().toString(36) + Math.random().toString(36).substring(2, 8);
            storage.setItem(storageKey, sessionId);
        }
        
        return sessionId;
    }

    _restoreLocalHistoryIfAny() {
        if (!this.dom.body) return;
        try {
            const raw = localStorage.getItem(this._localTurnsKey);
            if (!raw) return;
            const data = JSON.parse(raw);
            if (
                !data ||
                data.sessionId !== this.state.sessionId ||
                !Array.isArray(data.turns) ||
                data.turns.length === 0
            ) {
                return;
            }
            this.dom.body.innerHTML = '';
            for (const t of data.turns) {
                if (!t || typeof t.user !== 'string' || typeof t.html !== 'string') continue;
                this.appendUserMessage(t.user);
                this.appendBotMessage(
                    { html: t.html, request_id: t.request_id ?? null },
                    { skipPersist: true }
                );
            }
        } catch {
            /* quota / parse */
        }
    }

    _persistLocalHistory() {
        if (!this.dom.body) return;
        try {
            const turns = [];
            const userNodes = this.dom.body.querySelectorAll('.msg-user');
            userNodes.forEach((un) => {
                const ub = un.querySelector('.msg-user-bubble');
                if (!ub) return;
                const uText = ub.textContent.replace(/\s+/g, ' ').trim();
                let next = un.nextElementSibling;
                while (next && next.classList.contains('loading-indicator')) {
                    next = next.nextElementSibling;
                }
                if (!next || !next.classList.contains('msg-bot')) return;
                let html = next.innerHTML;
                if (html.length > 120_000) {
                    html = html.slice(0, 120_000);
                }
                const rid = next.dataset.requestId || null;
                turns.push({ user: uText, html, request_id: rid });
            });
            const last = turns.slice(-this._maxLocalTurns);
            localStorage.setItem(
                this._localTurnsKey,
                JSON.stringify({ sessionId: this.state.sessionId, turns: last })
            );
        } catch {
            /* private mode / quota */
        }
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
                // Available models loaded from API
                // (keeping static options for now as they match the API)
            }
        } catch (e) {
            // Could not fetch models (will use static list)
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


    // ── Focus trap for full-screen mode ──────────────────────────────────────
    _focusTrap(e) {
        if (!this.dom.container || this.state.size !== 'full') return;
        const focusable = Array.from(this.dom.container.querySelectorAll(
            'button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
        )).filter(el => !el.closest('[aria-hidden="true"]'));
        if (!focusable.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey) {
            if (document.activeElement === first) {
                e.preventDefault();
                last.focus();
            }
        } else {
            if (document.activeElement === last) {
                e.preventDefault();
                first.focus();
            }
        }
    }

    bindEvents() {
        // Open/Close
        if (this.dom.triggerWrap && this.dom.trigger) {
            this.dom.triggerWrap.addEventListener('click', (e) => {
                let el = e.target;
                if (el && el.nodeType === Node.TEXT_NODE) el = el.parentElement;
                if (!(el instanceof Element)) return;
                if (el.closest('.sahten-trigger')) return;
                if (this.state.isOpen) return;  // widget already open
                e.preventDefault();
                this.toggle(true);
            });
        }
        if (this.dom.trigger) {
            this.dom.trigger.addEventListener('click', () => this.toggle(true));
        }
        if (this.dom.closeBtn) {
            this.dom.closeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggle(false);
            });
        }
        // Delegated close: click anywhere on header controls area also works
        const headerEl = document.querySelector('.sahten-header');
        if (headerEl) {
            headerEl.addEventListener('click', (e) => {
                if (e.target.closest('.sahten-close-btn')) {
                    e.stopPropagation();
                    this.toggle(false);
                }
            });
        }
        if (this.dom.backdrop) {
            this.dom.backdrop.addEventListener('click', () => this.toggle(false));
        }

        // Escape key closes the widget; Tab key traps focus in full mode
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.state.isOpen) {
                this.toggle(false);
            } else if (e.key === 'Tab' && this.state.isOpen) {
                this._focusTrap(e);
            }
        });

        // Form Submit
        if (this.dom.form) {
            this.dom.form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.sendMessage();
            });
        }

        // Accueil : exemples de questions → insertion dans le champ
        if (this.dom.body) {
            this.dom.body.addEventListener('click', (e) => {
                const btn = e.target.closest('.welcome-example-prompt');
                if (!btn || !this.dom.body.contains(btn)) return;
                e.preventDefault();
                const text = btn.textContent.replace(/\s+/g, ' ').trim();
                if (!text || !this.dom.input) return;
                this.dom.input.value = text;
                this.dom.input.focus();
                if (this.dom.input.tagName === 'TEXTAREA') this.autoResizeInput();
            });
        }

        // Textarea: Enter = send, Shift+Enter = newline; auto-resize
        if (this.dom.input && this.dom.input.tagName === 'TEXTAREA') {
            this.dom.input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
            this.dom.input.addEventListener('input', () => this.autoResizeInput());
        }

        // Size Switching (Desktop)
        this.dom.sizeBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.setSize(action);
            });
        });

        // Mobile Drag/Slide Logic
        if (this.dom.dragHandle) {
            let startY = 0;

            this.dom.dragHandle.addEventListener('touchstart', (e) => {
                startY = e.touches[0].clientY;
            }, { passive: true });

            this.dom.dragHandle.addEventListener('touchend', (e) => {
                const diff = e.changedTouches[0].clientY - startY;

                if (diff > 40) {
                    // Swipe down -> smaller or close
                    if (this.state.size === 'full') this.setSize('mid');
                    else if (this.state.size === 'mid') this.setSize('window');
                    else this.toggle(false);
                } else if (diff < -40) {
                    // Swipe up -> bigger
                    if (this.state.size === 'window') this.setSize('mid');
                    else if (this.state.size === 'mid') this.setSize('full');
                }
            }, { passive: true });

            // Also allow tapping the handle area to cycle sizes
            this.dom.dragHandle.addEventListener('click', () => {
                if (this.state.size === 'window') this.setSize('mid');
                else if (this.state.size === 'mid') this.setSize('full');
                else this.setSize('window');
            });
        }
    }

    toggle(forceState) {
        this.state.isOpen = forceState !== undefined ? forceState : !this.state.isOpen;
        
        if (this.state.isOpen) {
            this.dom.container.setAttribute('data-state', 'open');
            if (this.dom.backdrop) {
                this.dom.backdrop.classList.add('visible');
                this.dom.backdrop.setAttribute('data-size', this.state.size);
            }
            if (this.dom.triggerWrap) {
                this.dom.triggerWrap.style.opacity = '0';
                this.dom.triggerWrap.style.pointerEvents = 'none';
            } else if (this.dom.trigger) {
                this.dom.trigger.style.opacity = '0';
                this.dom.trigger.style.pointerEvents = 'none';
            }
            if (this.dom.trigger) {
                this.dom.trigger.setAttribute('aria-expanded', 'true');
            }
            this.dom.body.scrollTop = 0;
            if (this._a11yLive) {
                this._a11yLive.textContent =
                    "Sahteïn est ouvert. Proposez une recette, un ingrédient ou un nom de chef.";
            }
            setTimeout(() => {
                if (this.dom.input) {
                    this.dom.input.focus();
                    if (this.dom.input.tagName === 'TEXTAREA') this.autoResizeInput();
                }
            }, 100);
            
            // Welcome Message
            if (this.dom.body.children.length === 0) {
                this.appendBotMessage({
                    html: `<article class="welcome-editorial welcome-editorial--stanza welcome-editorial--atable" lang="fr">
                        <div class="welcome-stanza">
                            <p class="welcome-lead">👋 Vous cherchez une recette libanaise (traditionnelle ou revisitée), arménienne ou encore des saveurs méditerranéennes ?</p>
                            <p class="welcome-sub">🌿🍋 Dites-moi tout, et je vous proposerai une recette répondant à vos envies.</p>
                            <p class="welcome-note">Soyez indulgents avec moi, je vais certainement faire des erreurs, mais je viens de me lancer. Et avec le temps, je vais certainement m’améliorer.</p>
                        </div>
                        <section class="welcome-examples welcome-examples--reference" role="region" aria-labelledby="welcome-examples-title">
                            <p class="welcome-examples-label" id="welcome-examples-title">Pour commencer</p>
                            <ul class="welcome-examples-list">
                                <li><button type="button" class="welcome-example-prompt">Léger et rapide ce soir ?</button></li>
                                <li><button type="button" class="welcome-example-prompt">Un menu pour six ?</button></li>
                                <li><button type="button" class="welcome-example-prompt">Trois idées au poulet ?</button></li>
                            </ul>
                        </section>
                    </article>`
                });
            }
        } else {
            this.dom.container.removeAttribute('data-state');
            if (this.dom.backdrop) {
                this.dom.backdrop.classList.remove('visible');
            }
            if (this.dom.triggerWrap) {
                this.dom.triggerWrap.style.opacity = '1';
                this.dom.triggerWrap.style.pointerEvents = 'auto';
            } else if (this.dom.trigger) {
                this.dom.trigger.style.opacity = '1';
                this.dom.trigger.style.pointerEvents = 'auto';
            }
            if (this.dom.trigger) {
                this.dom.trigger.setAttribute('aria-expanded', 'false');
            }
            if (this._a11yLive) {
                this._a11yLive.textContent = '';
            }
        }
    }

    autoResizeInput() {
        const el = this.dom.input;
        if (!el || el.tagName !== 'TEXTAREA') return;
        el.style.height = '1px';
        const h = Math.max(44, Math.min(el.scrollHeight, 200));
        el.style.height = h + 'px';
    }

    setSize(size) {
        const allowedSizes = new Set(['window', 'mid', 'full']);
        const nextSize = allowedSizes.has(size) ? size : 'window';
        this.state.size = nextSize;
        this.dom.container.setAttribute('data-size', nextSize);
        if (this.dom.backdrop) this.dom.backdrop.setAttribute('data-size', nextSize);
        
        this.dom.sizeBtns.forEach(btn => {
            const isActive = btn.dataset.action === nextSize;
            btn.classList.toggle('active', isActive);
            btn.setAttribute('aria-pressed', String(isActive));
        });
    }

    /**
     * Consume SSE from POST /chat/stream; returns final "done" payload or null.
     */
    async sendMessageStream(text, model) {
        try {
            const payload = {
                message: text,
                debug: this.state.debugMode,
                session_id: this.state.sessionId,
            };
            if (model) payload.model = model;
            const res = await fetch(`${this.config.apiBase}/chat/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Accept: 'text/event-stream',
                },
                body: JSON.stringify(payload),
            });
            if (!res.ok || !res.body) return null;
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';
            let donePayload = null;
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                let sep;
                while ((sep = buf.indexOf('\n\n')) !== -1) {
                    const frame = buf.slice(0, sep);
                    buf = buf.slice(sep + 2);
                    for (const line of frame.split('\n')) {
                        if (!line.startsWith('data:')) continue;
                        const raw = line.startsWith('data: ') ? line.slice(6) : line.slice(5);
                        try {
                            const evt = JSON.parse(raw.trim());
                            if (evt.type === 'done') donePayload = evt;
                        } catch (_) {
                            /* ignore partial JSON */
                        }
                    }
                }
            }
            return donePayload && donePayload.html ? donePayload : null;
        } catch (e) {
            return null;
        }
    }

    async sendMessage() {
        const text = this.dom.input.value.trim();
        if (!text || this.state.isLoading) return;

        this.dom.input.value = '';
        if (this.dom.input.tagName === 'TEXTAREA') this.autoResizeInput();
        this.appendUserMessage(text);
        this.setLoading(true);

        try {
            const model = this.getSelectedModel();
            let data = null;

            if (this.config.useStream !== false) {
                data = await this.sendMessageStream(text, model);
            }

            if (!data) {
                const payload = { 
                    message: text, 
                    debug: this.state.debugMode,
                    session_id: this.state.sessionId,
                };
                if (model) payload.model = model;
                const response = await fetch(`${this.config.apiBase}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const rawBody = await response.text();
                let parsed = null;
                try {
                    parsed = rawBody ? JSON.parse(rawBody) : null;
                } catch {
                    parsed = null;
                }
                if (!response.ok) {
                    let detail = `HTTP ${response.status}`;
                    if (parsed && parsed.detail !== undefined) {
                        detail = typeof parsed.detail === 'string'
                            ? parsed.detail
                            : JSON.stringify(parsed.detail);
                    }
                    throw new Error(detail);
                }
                data = parsed;
            }

            this.state.lastModelUsed = data.model_used;

            if (this.state.debugMode && data.model_used) {
                data.html = this.addModelIndicator(data.html, data.model_used);
            }

            this.appendBotMessage(data);

        } catch (error) {
            // API error occurred — en debug, afficher le détail serveur (FastAPI `detail`)
            const msg = String(error.message || 'Erreur réseau');
            const safe = msg.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            this.appendBotMessage({
                html: `<div class="sahten-narrative">
                    <p><em>Mille excuses, un petit incident en cuisine...</em></p>
                    <p>Pourriez-vous répéter votre demande ?</p>
                    ${this.state.debugMode ? `<p style="font-size:11px;opacity:0.6;margin-top:8px;word-break:break-word;">` + this.config.apiBase + ` — ` + safe + `</p>` : `<p style="font-size:11px;opacity:0.55;margin-top:8px;">Si le problème continue, vérifiez les journaux du serveur (erreur API).</p>`}
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
        const modelName = modelUsed.includes('nano') ? 'nano' : 'mini';
        const indicator = `<div style="font-size: 9px; opacity: 0.5; text-align: right; margin-top: 8px; font-family: monospace;">${modelName}</div>`;
        return html + indicator;
    }

    appendUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'msg msg-user';
        div.setAttribute('data-user-msg', '');
        const bubble = document.createElement('span');
        bubble.className = 'msg-user-bubble';
        bubble.textContent = text;
        div.appendChild(bubble);
        this.dom.body.appendChild(div);
        this.scrollToBottom();
    }

    appendBotMessage(data, options = {}) {
        const skipPersist = options.skipPersist === true;
        const div = document.createElement('div');
        div.className = 'msg msg-bot';
        div.innerHTML = sanitizeHTML(data.html);
        if (data.request_id) {
            div.dataset.requestId = data.request_id;
        }

        if (data.request_id) {
            this.trackImpressions(div, data);
            this.addClickTracking(div, data);
            div.appendChild(this.createFeedbackButtons(data.request_id));
        }

        this.dom.body.appendChild(div);
        this.initializeRestaurantMap(div);
        this.scrollToResponseStart();
        if (!skipPersist) {
            this._persistLocalHistory();
        }
    }

    initializeRestaurantMap(container) {
        const mapEl = container.querySelector('#sahten-map');
        if (!mapEl || typeof L === 'undefined') {
            return;
        }

        let markers = [];
        let center = null;
        try {
            const markersRaw = mapEl.dataset.markers || '[]';
            markers = JSON.parse(markersRaw);
        } catch (e) {
            markers = [];
        }
        try {
            const centerRaw = mapEl.dataset.center;
            center = centerRaw ? JSON.parse(centerRaw) : null;
        } catch (e) {
            center = null;
        }

        const geocoded = markers.filter(m => typeof m.lat === 'number' && typeof m.lon === 'number');
        const hasCenter = center && typeof center.lat === 'number' && typeof center.lon === 'number';
        if (!geocoded.length && !hasCenter) {
            return;
        }

        const initialLat = center?.lat ?? geocoded[0]?.lat;
        const initialLon = center?.lon ?? geocoded[0]?.lon;
        const initialZoom = center?.zoom ?? 13;
        const map = L.map(mapEl).setView([initialLat, initialLon], initialZoom);

        // Carte personnalisée Sahten : CartoDB Positron (fond épuré, moderne)
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 19,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
        }).addTo(map);

        // Marqueur personnalisé aux couleurs Sahten (#94C2AE)
        const sahtenIcon = L.divIcon({
            className: 'sahten-map-marker',
            html: '<div class="sahten-marker-pin"></div>',
            iconSize: [24, 36],
            iconAnchor: [12, 36],
        });

        geocoded.forEach(marker => {
            const pin = L.marker([marker.lat, marker.lon], { icon: sahtenIcon }).addTo(map);
            const popup = [marker.title, marker.address].filter(Boolean).join('<br>');
            if (popup) {
                pin.bindPopup(popup);
            }
        });
    }

    trackImpressions(container, data) {
        /**
         * Track impression events for each recipe card displayed.
         */
        const recipeCards = container.querySelectorAll(
            '.recipe-card, .sahten-recipe-card--preview'
        );
        recipeCards.forEach(card => {
            const link = card.querySelector('a');
            const titleEl =
                card.querySelector('.recipe-title') ||
                card.querySelector('.sahten-recipe-card__title');
            if (link && titleEl) {
                this.trackEvent('impression', {
                    request_id: data.request_id,
                    recipe_url: link.href,
                    recipe_title: titleEl.textContent,
                    intent: data.intent,
                    model_used: data.model_used,
                });
            }
        });
    }

    addClickTracking(container, data) {
        /**
         * Add click tracking to recipe card links.
         */
        const recipeLinks = container.querySelectorAll(
            '.recipe-card a, .sahten-recipe-card__link'
        );
        recipeLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const titleEl =
                    link.querySelector('.recipe-title') ||
                    link.querySelector('.sahten-recipe-card__title');
                this.trackEvent('click', {
                    request_id: data.request_id,
                    recipe_url: link.href,
                    recipe_title: titleEl ? titleEl.textContent : '',
                    intent: data.intent,
                    model_used: data.model_used,
                });
            });
        });
    }

    async trackEvent(eventType, eventData) {
        /**
         * Send event to the analytics API.
         */
        try {
            const payload = {
                event_type: eventType,
                session_id: this.state.sessionId,
                ...eventData,
            };
            
            // Fire and forget (don't wait for response)
            fetch(`${this.config.apiBase}/events`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            }).catch(() => { /* Event tracking failed silently */ });
            
        } catch (error) {
            // Failed to track event silently
        }
    }

    createFeedbackButtons(requestId) {
        /**
         * Create feedback buttons (👍/👎) for a response.
         */
        const container = document.createElement('div');
        container.className = 'feedback-container';
        container.innerHTML = `
            <div class="feedback-buttons" data-request-id="${requestId}">
                <span class="feedback-prompt">Cette réponse vous a-t-elle aidé ?</span>
                <button class="feedback-btn feedback-positive" data-rating="positive" title="Utile">Oui</button>
                <button class="feedback-btn feedback-negative" data-rating="negative" title="Pas utile">Non</button>
            </div>
            <div class="feedback-reason" style="display: none;">
                <input type="text" class="feedback-reason-input" placeholder="Pourquoi ? (optionnel)" maxlength="200">
                <button class="feedback-submit-btn">Envoyer</button>
            </div>
            <div class="feedback-thanks" style="display: none;">
                <span>Merci pour votre retour !</span>
            </div>
        `;
        
        // Add event listeners
        const buttons = container.querySelectorAll('.feedback-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', (e) => this.handleFeedbackClick(e, container, requestId));
        });
        
        const submitBtn = container.querySelector('.feedback-submit-btn');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => this.submitFeedbackWithReason(container, requestId));
        }
        
        return container;
    }

    handleFeedbackClick(event, container, requestId) {
        const rating = event.target.dataset.rating;
        const buttonsDiv = container.querySelector('.feedback-buttons');
        const reasonDiv = container.querySelector('.feedback-reason');
        const thanksDiv = container.querySelector('.feedback-thanks');
        
        // Highlight selected button
        container.querySelectorAll('.feedback-btn').forEach(btn => btn.classList.remove('selected'));
        event.target.classList.add('selected');
        
        if (rating === 'negative') {
            // Show reason input for negative feedback
            reasonDiv.style.display = 'flex';
            this.pendingFeedback = { requestId, rating };
        } else {
            // Submit positive feedback immediately
            this.submitFeedback(requestId, rating, null);
            buttonsDiv.style.display = 'none';
            thanksDiv.style.display = 'block';
        }
    }

    submitFeedbackWithReason(container, requestId) {
        const reasonInput = container.querySelector('.feedback-reason-input');
        const reason = reasonInput ? reasonInput.value.trim() : null;
        const rating = this.pendingFeedback?.rating || 'negative';
        
        this.submitFeedback(requestId, rating, reason);
        
        // Update UI
        container.querySelector('.feedback-buttons').style.display = 'none';
        container.querySelector('.feedback-reason').style.display = 'none';
        container.querySelector('.feedback-thanks').style.display = 'block';
    }

    async submitFeedback(requestId, rating, reason) {
        /**
         * Submit feedback to the API.
         */
        try {
            const payload = {
                request_id: requestId,
                rating: rating,
                reason: reason,
                session_id: this.state.sessionId,
            };
            
            await fetch(`${this.config.apiBase}/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        } catch (error) {
            // Failed to submit feedback silently
        }
    }

    setLoading(loading) {
        this.state.isLoading = loading;
        if (this.dom.sendBtn) this.dom.sendBtn.disabled = loading;
        if (this.dom.input) this.dom.input.disabled = loading;

        if (loading) {
            const loader = document.createElement('div');
            loader.className = 'msg msg-bot loading-indicator sahten-thinking-loader';
            loader.innerHTML = `
                <div class="sahten-thinking-block">
                    <img class="sahten-thinking-mascot" src="assets/v7_logo_sahten.svg" alt="" width="40" height="40" decoding="async" />
                    <div class="sahten-thinking-main">
                        <p class="sahten-thinking-line">Sahteïn parcourt les recettes…</p>
                        <div class="loading-dots" aria-hidden="true">
                            <div class="dot"></div>
                            <div class="dot"></div>
                            <div class="dot"></div>
                        </div>
                    </div>
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
            if (this.dom.input) {
                this.dom.input.disabled = false;
                this.dom.input.focus();
            }
        }
    }

    scrollToBottom() {
        this.dom.body.scrollTop = this.dom.body.scrollHeight;
    }

    scrollToResponseStart() {
        // Scroll so the user's last question + start of response are visible (not jump to bottom)
        const userMsgs = this.dom.body.querySelectorAll('[data-user-msg]');
        const lastUserMsg = userMsgs[userMsgs.length - 1];
        const botMsgs = this.dom.body.querySelectorAll('.msg-bot');
        const lastBotMsg = botMsgs[botMsgs.length - 1];
        if (lastUserMsg && lastBotMsg) {
            const scrollTarget = lastUserMsg.offsetTop - 20;
            this.dom.body.scrollTo({ top: Math.max(0, scrollTarget), behavior: 'smooth' });
        } else {
            this.scrollToBottom();
        }
    }
}
