/**
 * SAHTEN CLIENT (v2.1)
 * Persona: Editorial Chef from L'Orient-Le Jour
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
        const initialSize = this.dom.container.dataset.size || this.state.size;
        this.setSize(initialSize);
        this.loadModels();
        // Sahten v2.1 initialized
    }

    generateSessionId() {
        /**
         * Generate a unique session ID for conversation memory.
         * Persists in sessionStorage so refreshing page keeps the session.
         */
        const storageKey = 'sahten_session_id';
        let sessionId = sessionStorage.getItem(storageKey);
        
        if (!sessionId) {
            sessionId = 'ses_' + Date.now().toString(36) + Math.random().toString(36).substring(2, 8);
            sessionStorage.setItem(storageKey, sessionId);
        }
        
        return sessionId;
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
            this.dom.backdrop.classList.add('visible');
            this.dom.backdrop.setAttribute('data-size', this.state.size);
            this.dom.trigger.style.opacity = '0';
            this.dom.trigger.style.pointerEvents = 'none';
            this.dom.body.scrollTop = 0;
            setTimeout(() => {
                this.dom.input.focus();
                if (this.dom.input.tagName === 'TEXTAREA') this.autoResizeInput();
            }, 100);
            
            // Welcome Message
            if (this.dom.body.children.length === 0) {
                this.appendBotMessage({
                    html: `<article class="welcome-editorial welcome-editorial--mockup" lang="fr">
                        <div class="welcome-intro">
                            <p class="welcome-mockup-line welcome-mockup-line--lead">
                                Bonjour, je suis <span class="welcome-lead-lockup"><img class="welcome-inline-mascot" src="assets/sahten_logo_v2.svg" alt="" width="32" height="32" /><span class="welcome-highlight">Sahten</span></span>.
                            </p>
                            <p class="welcome-mockup-line">
                                Je suis le robot culinaire de
                                <img src="assets/logo_2_olj.svg" alt="L'Orient-Le Jour" class="welcome-olj-logotype" width="168" height="18" />
                            </p>
                            <p class="welcome-mockup-line">
                                J'ai plein de recettes dans mes carnets, et je serais ravi de te faire découvrir la
                                <span class="welcome-highlight">table</span>
                                libanaise.
                            </p>
                        </div>
                        <div class="welcome-examples" role="region" aria-labelledby="welcome-examples-heading">
                            <div class="welcome-examples-head">
                                <span id="welcome-examples-heading" class="welcome-examples-label">Exemples de questions</span>
                                <span id="welcome-examples-tip" class="welcome-examples-tip">cliquer pour insérer dans le champ</span>
                            </div>
                            <ul class="welcome-examples-list" aria-describedby="welcome-examples-tip">
                                <li>
                                    <button type="button" class="welcome-example-prompt" title="Insérer cette question dans le champ">Le taboulé de Kamal Mouzawak&nbsp;?</button>
                                </li>
                                <li>
                                    <button type="button" class="welcome-example-prompt" title="Insérer cette question dans le champ">Un plat au poulet pour ce soir&nbsp;?</button>
                                </li>
                                <li>
                                    <button type="button" class="welcome-example-prompt" title="Insérer cette question dans le champ">Une idée avec du boulgour et des herbes&nbsp;?</button>
                                </li>
                            </ul>
                        </div>
                    </article>`
                });
            }
        } else {
            this.dom.container.removeAttribute('data-state');
            this.dom.backdrop.classList.remove('visible');
            this.dom.trigger.style.opacity = '1';
            this.dom.trigger.style.pointerEvents = 'auto';
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
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                data = await response.json();
            }

            this.state.lastModelUsed = data.model_used;

            if (this.state.debugMode && data.model_used) {
                data.html = this.addModelIndicator(data.html, data.model_used);
            }

            this.appendBotMessage(data);

        } catch (error) {
            // API error occurred
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
        const modelName = modelUsed.includes('nano') ? 'nano' : 'mini';
        const indicator = `<div style="font-size: 9px; opacity: 0.5; text-align: right; margin-top: 8px; font-family: monospace;">${modelName}</div>`;
        return html + indicator;
    }

    appendUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'msg msg-user';
        div.textContent = text;
        div.setAttribute('data-user-msg', '');
        this.dom.body.appendChild(div);
        this.scrollToBottom();
    }

    appendBotMessage(data) {
        const div = document.createElement('div');
        div.className = 'msg msg-bot';
        div.innerHTML = sanitizeHTML(data.html);

        if (data.request_id) {
            this.trackImpressions(div, data);
            this.addClickTracking(div, data);
            div.appendChild(this.createFeedbackButtons(data.request_id));
        }

        this.dom.body.appendChild(div);
        this.initializeRestaurantMap(div);
        this.scrollToResponseStart();
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
        const recipeCards = container.querySelectorAll('.recipe-card');
        recipeCards.forEach(card => {
            const link = card.querySelector('a');
            const titleEl = card.querySelector('.recipe-title');
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
        const recipeLinks = container.querySelectorAll('.recipe-card a');
        recipeLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const titleEl = link.querySelector('.recipe-title');
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
