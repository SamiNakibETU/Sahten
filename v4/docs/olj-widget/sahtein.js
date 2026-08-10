// Sahteïn — widget de chat recettes (L'Orient-Le Jour)
// Remplacement direct de /assets/js/sahtein.js. MÊME structure HTML/DOM que
// la version en ligne : un seul fichier à échanger, aucune modification de page.
//
// Corrections vs version du 21 avril 2026 :
//   1. session_id persistant (localStorage) envoyé à chaque requête
//      -> la mémoire de conversation fonctionne (« une autre recette », relances).
//   2. Coquille du message d'accueil corrigée.
//   3. Double envoi supprimé (l'ancien 2e listener sur #sendButton relançait un
//      appel API à chaque clic = 2 requêtes payantes par message).
//   4. Modèle codé en dur retiré : le serveur choisit son modèle.
//   5. Code mort retiré (réponse « tabboulé » factice jamais appelée).

// === SESSION (mémoire de conversation) ===
// Un identifiant stable par navigateur : le backend rattache l'historique et
// peut proposer « une autre recette », suivre les relances, éviter les répétitions.
const SAHTEN_API = 'https://sahtenbotapi.lorientlejour.com/api/chat';

function getSessionId() {
    try {
        let sid = localStorage.getItem('sahten_session_id');
        if (!sid) {
            const rand = (crypto && crypto.randomUUID)
                ? crypto.randomUUID().replace(/-/g, '')
                : (Date.now().toString(36) + Math.random().toString(36).slice(2));
            sid = 'ses_' + rand.slice(0, 12);
            localStorage.setItem('sahten_session_id', sid);
        }
        return sid;
    } catch (e) {
        // localStorage indisponible (mode privé strict) : session éphémère.
        if (!window._sahtenSid) {
            window._sahtenSid = 'ses_' + Math.random().toString(36).slice(2, 14);
        }
        return window._sahtenSid;
    }
}

// Appel API unique, partagé par tous les points d'entrée (barre + chat).
// Appel API avec délai maximal : une requête figée (serveur lent, réseau qui
// décroche) échoue proprement au lieu de laisser le spinner tourner sans fin.
// Lève une erreur portant le code HTTP (utile pour distinguer 429 = trop de
// demandes) ; l'appelant affiche le message adéquat.
async function askSahten(query) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 45000);
    try {
        const res = await fetch(SAHTEN_API, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, session_id: getSessionId() }),
            signal: controller.signal
        });
        if (!res.ok) {
            const err = new Error('HTTP ' + res.status);
            err.status = res.status;
            throw err;
        }
        return res.json();
    } finally {
        clearTimeout(timer);
    }
}

// === STATE & ELEMENTS ===
let first_message_sent = false;
const elements = {
    blob: document.getElementById('blob'),
    speechBubble: document.getElementById('speechBubble'),
    searchBar: document.getElementById('searchBar'),
    searchInput: document.getElementById('searchInput'),
    sendButton: document.getElementById('sendButton'),
    chatWindow: document.getElementById('chatWindow'),
    chatHeader: document.getElementById('chatHeader'),
    chatMessages: document.getElementById('chatMessages'),
    chatInput: document.getElementById('chatInput'),
    goToBottomBtn: document.getElementById('goToBottomBtn'),
    controls: {
        window: document.getElementById('windowBtn'),
        half: document.getElementById('halfBtn'),
        full: document.getElementById('fullBtn'),
        close: document.getElementById('closeBtn')
    }
};

const state = {
    isSearchVisible: false,
    isChatVisible: false,
    isDragging: false,
    isSending: false,  // verrou anti-double-envoi (une requête à la fois)
    dragStart: { x: 0, y: 0 },
    windowStart: { x: 0, y: 0 }
};

// === UTILITIES ===
const utils = {
    autoResize: (textarea) => {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    },

    updateBlobVisibility: () => {
        const shouldHide = window.innerWidth <= 768 && (state.isSearchVisible || state.isChatVisible);
        elements.blob.classList.toggle('hidden', shouldHide);
    },

    checkScrollPosition: () => {
        const chatMessages = elements.chatMessages;
        const isNearBottom = chatMessages.scrollTop + chatMessages.clientHeight >= chatMessages.scrollHeight - 20;
        elements.goToBottomBtn.classList.toggle('visible', !isNearBottom);
    },

    scrollToBottom: () => {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
        elements.goToBottomBtn.classList.remove('visible');
    },

    // Sanitisation défense en profondeur : le serveur échappe déjà le HTML, mais
    // on nettoie aussi côté client. DOMPurify si présent, sinon repli inerte.
    sanitize: (html) => {
        if (window.DOMPurify) {
            return window.DOMPurify.sanitize(html, { ADD_ATTR: ['target'] });
        }
        // Repli : parser dans un document INERTE (aucun script/handler exécuté),
        // retirer les éléments et attributs dangereux.
        const doc = new DOMParser().parseFromString(html, 'text/html');
        doc.querySelectorAll('script, iframe, object, embed, link, style').forEach(n => n.remove());
        doc.querySelectorAll('*').forEach(el => {
            [...el.attributes].forEach(attr => {
                const name = attr.name.toLowerCase();
                const val = (attr.value || '').toLowerCase().trim();
                if (name.startsWith('on') || val.startsWith('javascript:') || val.startsWith('data:') || val.startsWith('vbscript:')) {
                    el.removeAttribute(attr.name);
                }
            });
        });
        return doc.body.innerHTML;
    }
};

// === MESSAGE SYSTEM ===
const messageSystem = {
    add: (type, content) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (type === 'loading') {
            contentDiv.innerHTML = '<span></span><span></span><span></span>';
        } else if (type === 'assistant-html') {
            contentDiv.innerHTML = utils.sanitize(content);
            messageDiv.classList.add('assistant');
            messageSystem.addCopyButton(messageDiv, contentDiv);
        } else if (type.startsWith('assistant')) {
            if (content.includes('<')) {
                contentDiv.innerHTML = utils.sanitize(content);
            } else {
                contentDiv.textContent = content;
            }
            if (type === 'assistant-error') {
                messageDiv.classList.add('assistant', 'error');
            } else {
                messageDiv.classList.add('assistant');
                messageSystem.addCopyButton(messageDiv, contentDiv);
            }
        } else {
            contentDiv.textContent = content;
            messageDiv.classList.add(type);
        }

        messageDiv.appendChild(contentDiv);
        elements.chatMessages.appendChild(messageDiv);

        // Auto-scroll seulement si l'utilisateur est déjà en bas.
        const isNearBottom = elements.chatMessages.scrollTop + elements.chatMessages.clientHeight >= elements.chatMessages.scrollHeight - 100;
        if (isNearBottom) {
            setTimeout(() => {
                elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
            }, 10);
        }
        setTimeout(() => utils.checkScrollPosition(), 10);
        return messageDiv;
    },

    addCopyButton: (messageDiv, contentDiv) => {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.title = 'Copier le texte';
        copyBtn.innerHTML = '📄';
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(contentDiv.innerText).then(() => {
                copyBtn.innerHTML = '✅';
                copyBtn.classList.add('copied');
                setTimeout(() => {
                    copyBtn.innerHTML = '📄';
                    copyBtn.classList.remove('copied');
                }, 2000);
            }).catch(err => console.error('Erreur de copie : ', err));
        });
        messageDiv.appendChild(copyBtn);
    },

    // Rend la réponse de l'API, quelle que soit sa forme (html ou answer_sentences).
    renderAnswer: (data) => {
        if (data && data.html) {
            messageSystem.add('assistant-html', data.html);
            return;
        }
        if (data && Array.isArray(data.answer_sentences) && data.answer_sentences.length) {
            const text = data.answer_sentences.map(s => s.text || '').join(' ').trim();
            messageSystem.add('assistant', text || 'Aucune réponse trouvée.');
            return;
        }
        messageSystem.add('assistant', (data && data.reply) || 'Aucune réponse trouvée.');
    }
};

// === UI CONTROLS ===
const ui = {
    showSearch: () => {
        state.isSearchVisible = true;
        elements.searchBar.classList.add('visible');
        utils.updateBlobVisibility();
        setTimeout(() => elements.searchInput.focus(), 100);
    },

    hideSearch: () => {
        state.isSearchVisible = false;
        elements.searchBar.classList.remove('visible');
        elements.searchInput.value = '';
        elements.searchInput.style.height = 'auto';
        utils.updateBlobVisibility();
    },

    showChat: async (query = '') => {
        state.isChatVisible = true;
        ui.hideSearch();
        elements.chatWindow.classList.add('visible');

        if (first_message_sent === false) {
            messageSystem.add('assistant', "👋 Vous cherchez une recette libanaise (traditionnelle ou revisitée), arménienne ou encore des saveurs méditerranéennes ? 🌿🍋 Dites-moi tout, et je vous proposerai une recette répondant à vos envies.<br /><br />Soyez indulgents avec moi, je vais certainement faire des erreurs, mais je viens de me lancer. Et avec le temps, je vais certainement m'améliorer.");
            first_message_sent = true;
        }

        if (query) {
            await ui.handleUserQuery(query);
        }

        utils.updateBlobVisibility();
        setTimeout(() => {
            elements.chatInput.focus();
            utils.checkScrollPosition();
        }, 100);
    },

    hideChat: () => {
        state.isChatVisible = false;
        elements.chatWindow.classList.remove('visible');
        const messages = elements.chatMessages.querySelectorAll('.message');
        messages.forEach(message => message.remove());
        elements.chatInput.value = '';
        elements.chatInput.style.height = 'auto';
        elements.goToBottomBtn.classList.remove('visible');
        utils.updateBlobVisibility();
    },

    handleUserQuery: async (query) => {
        if (!query) return;
        // Une requête à la fois : évite le double envoi (donc le double coût)
        // quand l'utilisateur, face à un bot lent, renvoie sa question.
        if (state.isSending) return;
        state.isSending = true;
        messageSystem.add('user', query);
        const loadingMsg = messageSystem.add('loading');
        try {
            const data = await askSahten(query);
            loadingMsg.remove();
            messageSystem.renderAnswer(data);
        } catch (err) {
            console.error(err);
            loadingMsg.remove();
            const msg = (err && err.status === 429)
                ? "Beaucoup de demandes en ce moment. Merci de réessayer dans un instant."
                : "Oups. Il semblerait que Sahteïn soit totalement débordé en cuisine. Pourriez-vous renvoyer votre requête ?";
            messageSystem.add('assistant-error', msg);
        } finally {
            state.isSending = false;
        }
    }
};

// === DRAG SYSTEM ===
const dragSystem = {
    start: (e) => {
        if (e.target.closest('.chat-controls')) return;
        if (window.innerWidth <= 768) return;
        if (elements.chatWindow.classList.contains('full-screen') ||
            elements.chatWindow.classList.contains('half-screen')) return;

        state.isDragging = true;
        state.dragStart = { x: e.clientX, y: e.clientY };
        const rect = elements.chatWindow.getBoundingClientRect();
        state.windowStart = { x: rect.left, y: rect.top };
        document.addEventListener('mousemove', dragSystem.move);
        document.addEventListener('mouseup', dragSystem.stop);
        e.preventDefault();
    },

    move: (e) => {
        if (!state.isDragging) return;
        const deltaX = e.clientX - state.dragStart.x;
        const deltaY = e.clientY - state.dragStart.y;
        const newX = Math.max(0, Math.min(state.windowStart.x + deltaX,
            window.innerWidth - elements.chatWindow.offsetWidth));
        const newY = Math.max(0, Math.min(state.windowStart.y + deltaY,
            window.innerHeight - elements.chatWindow.offsetHeight));
        elements.chatWindow.style.left = newX + 'px';
        elements.chatWindow.style.top = newY + 'px';
        elements.chatWindow.style.right = 'auto';
    },

    stop: () => {
        state.isDragging = false;
        document.removeEventListener('mousemove', dragSystem.move);
        document.removeEventListener('mouseup', dragSystem.stop);
    }
};

// === EVENT LISTENERS ===
const setupEvents = () => {
    elements.blob.addEventListener('click', () => {
        if (state.isSearchVisible) {
            ui.hideSearch();
            state.isSearchVisible = false;
        } else {
            ui.showSearch();
            elements.speechBubble.classList.remove('visible');
            state.isSearchVisible = true;
        }
    });

    document.addEventListener('click', (e) => {
        if (!state.isSearchVisible) return;
        if (elements.searchBar.contains(e.target) || elements.blob.contains(e.target)) return;
        ui.hideSearch();
        state.isSearchVisible = false;
    });

    elements.blob.addEventListener('mouseenter', () => {
        if (!state.isSearchVisible && !state.isChatVisible) {
            elements.speechBubble.classList.add('visible');
        }
    });

    elements.blob.addEventListener('mouseleave', () => {
        setTimeout(() => elements.speechBubble.classList.remove('visible'), 1500);
    });

    // Barre de recherche : un SEUL point d'envoi (Entrée ou clic).
    elements.searchInput.addEventListener('input', () => utils.autoResize(elements.searchInput));
    elements.searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const q = elements.searchInput.value.trim();
            if (q) ui.showChat(q);
        }
    });
    elements.sendButton.addEventListener('click', () => {
        const q = elements.searchInput.value.trim();
        if (q) ui.showChat(q);
    });

    // Chat : saisie dans la fenêtre ouverte.
    elements.chatInput.addEventListener('input', () => utils.autoResize(elements.chatInput));
    elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const message = elements.chatInput.value.trim();
            if (message) {
                elements.chatInput.value = '';
                elements.chatInput.style.height = 'auto';
                ui.handleUserQuery(message);
            }
        }
    });

    elements.chatMessages.addEventListener('scroll', utils.checkScrollPosition);
    elements.goToBottomBtn.addEventListener('click', utils.scrollToBottom);

    elements.controls.window.addEventListener('click', () => {
        elements.chatWindow.className = 'chat-window window-mode visible';
    });
    elements.controls.half.addEventListener('click', () => {
        elements.chatWindow.className = 'chat-window half-screen visible';
    });
    elements.controls.full.addEventListener('click', () => {
        elements.chatWindow.className = 'chat-window full-screen visible';
    });
    elements.controls.close.addEventListener('click', ui.hideChat);

    elements.chatHeader.addEventListener('mousedown', dragSystem.start);
    window.addEventListener('resize', utils.updateBlobVisibility);
};

// === INITIALIZATION ===
const init = () => {
    setupEvents();
    setTimeout(() => {
        elements.speechBubble.classList.add('visible');
        setTimeout(() => elements.speechBubble.classList.remove('visible'), 3000);
    }, 2000);
};

init();
