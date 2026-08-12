/* Sahteïn — widget de chat recettes (L'Orient-Le Jour)
 * ---------------------------------------------------------------------------
 * Remplacement direct de /assets/js/sahtein.js. Ce fichier charge l'interface
 * Sahteïn actuelle depuis notre serveur : plus aucun échange de fichier pour les
 * évolutions suivantes (design ou logique), tout arrive automatiquement.
 *
 * AUCUNE modification de la page n'est nécessaire : le balisage existant
 * (#blob, #chatWindow, #searchBar) est simplement masqué, jamais supprimé.
 *
 * SÉCURITÉ DE PRODUCTION — si notre serveur est injoignable, ce fichier
 * REPLIE sur l'ancien widget intégré ci-dessous (identique à la version en
 * ligne, avec ses correctifs). La page n'est donc jamais sans assistant.
 */
(function () {
  "use strict";
  if (window.__sahtenBooted) return;
  window.__sahtenBooted = true;

  var ORIGIN = "https://sahtenbotapi.lorientlejour.com";
  var LEGACY_IDS = ["blob", "speechBubble", "searchBar", "chatWindow"];

  // ── Interface actuelle : bouton flottant + panneau (iframe sur notre origine,
  //    donc styles totalement isolés de ceux du site) ────────────────────────
  function bootModern() {
    var Z = 2147483000;
    var frame = null;

    var btn = document.createElement("button");
    btn.type = "button";
    btn.id = "sahten-launcher";
    btn.setAttribute("aria-label", "Ouvrir Sahteïn — assistant recettes");
    btn.title = "Une idée recette ? Ouvrez Sahteïn.";
    btn.style.cssText =
      "position:fixed;bottom:20px;right:20px;width:60px;height:60px;border-radius:50%;" +
      "border:0;padding:0;cursor:pointer;z-index:" + Z + ";background:#0b5c3f;" +
      "box-shadow:0 6px 20px rgba(0,0,0,.28);display:flex;align-items:center;" +
      "justify-content:center;transition:transform .15s ease";
    var logo = document.createElement("img");
    logo.src = ORIGIN + "/assets/v7_logo_sahten.svg";
    logo.alt = "";
    logo.style.cssText = "width:38px;height:38px;pointer-events:none";
    btn.appendChild(logo);
    btn.addEventListener("mouseenter", function () { btn.style.transform = "scale(1.06)"; });
    btn.addEventListener("mouseleave", function () { btn.style.transform = "scale(1)"; });

    function isMobile() { return window.matchMedia("(max-width: 600px)").matches; }
    function sizeFrame() {
      if (!frame) return;
      var base =
        "position:fixed;border:0;z-index:" + (Z + 1) + ";background:transparent;" +
        "box-shadow:0 12px 48px rgba(0,0,0,.28);max-width:100vw;max-height:100vh";
      frame.style.cssText = isMobile()
        ? base + ";inset:0;width:100%;height:100%;border-radius:0"
        : base + ";bottom:20px;right:20px;width:412px;height:min(680px,calc(100vh - 40px));border-radius:16px";
    }

    function open() {
      if (!frame) {
        frame = document.createElement("iframe");
        frame.src = ORIGIN + "/widget?embed=1";
        frame.title = "Sahteïn — assistant recettes";
        frame.setAttribute("allow", "clipboard-write");
        sizeFrame();
        document.body.appendChild(frame);
        window.addEventListener("resize", sizeFrame);
      } else {
        frame.style.display = "block";
        sizeFrame();
        try { frame.contentWindow.postMessage({ type: "sahten:open" }, ORIGIN); } catch (e) {}
      }
      btn.style.display = "none";
    }
    function close() {
      if (frame) frame.style.display = "none";
      btn.style.display = "flex";
    }

    btn.addEventListener("click", open);
    window.addEventListener("message", function (e) {
      if (e.origin !== ORIGIN) return;               // n'écoute que notre origine
      if ((e.data || {}).type === "sahten:close") close();
    });

    // Masque l'ancien widget (sans toucher au HTML de la page).
    var css = document.createElement("style");
    css.textContent =
      "#" + LEGACY_IDS.join(",#") + "{display:none !important}";
    document.head.appendChild(css);

    document.body.appendChild(btn);
  }

  // ── Repli : ancien widget, si notre serveur ne répond pas ─────────────────
  function bootLegacy() {
    var el = {};
    LEGACY_IDS.concat(["searchInput", "sendButton", "chatMessages", "chatInput", "goToBottomBtn"])
      .forEach(function (id) { el[id] = document.getElementById(id); });
    if (!el.blob || !el.searchInput || !el.chatMessages) return;   // page sans widget

    var sending = false;
    function sid() {
      try {
        var s = localStorage.getItem("sahten_session_id");
        if (!s) {
          s = "ses_" + Math.random().toString(36).slice(2, 14);
          localStorage.setItem("sahten_session_id", s);
        }
        return s;
      } catch (e) { return "ses_" + Math.random().toString(36).slice(2, 14); }
    }
    function clean(html) {
      if (window.DOMPurify) return window.DOMPurify.sanitize(html, { ADD_ATTR: ["target"] });
      var doc = new DOMParser().parseFromString(html, "text/html");
      doc.querySelectorAll("script,iframe,object,embed,link,style").forEach(function (n) { n.remove(); });
      doc.querySelectorAll("*").forEach(function (e2) {
        [].slice.call(e2.attributes).forEach(function (a) {
          var v = (a.value || "").toLowerCase().trim();
          if (a.name.toLowerCase().indexOf("on") === 0 || v.indexOf("javascript:") === 0 ||
              v.indexOf("data:") === 0 || v.indexOf("vbscript:") === 0) e2.removeAttribute(a.name);
        });
      });
      return doc.body.innerHTML;
    }
    function add(type, content) {
      var d = document.createElement("div");
      d.className = "message " + type;
      var c = document.createElement("div");
      c.className = "message-content";
      if (type === "loading") c.innerHTML = "<span></span><span></span><span></span>";
      else if (type.indexOf("assistant") === 0) {
        if (String(content).indexOf("<") >= 0) c.innerHTML = clean(content);
        else c.textContent = content;
        d.classList.add("assistant");
        if (type === "assistant-error") d.classList.add("error");
      } else c.textContent = content;
      d.appendChild(c);
      el.chatMessages.appendChild(d);
      el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
      return d;
    }
    async function ask(q) {
      if (!q || sending) return;
      sending = true;
      add("user", q);
      var loading = add("loading");
      var ctrl = new AbortController();
      var timer = setTimeout(function () { ctrl.abort(); }, 45000);
      try {
        var r = await fetch(ORIGIN + "/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, session_id: sid() }),
          signal: ctrl.signal
        });
        var data = await r.json();
        loading.remove();
        if (data && data.html) add("assistant-html", data.html);
        else if (data && data.answer_sentences)
          add("assistant", data.answer_sentences.map(function (s) { return s.text || ""; }).join(" "));
        else add("assistant", "Aucune réponse trouvée.");
      } catch (err) {
        loading.remove();
        add("assistant-error",
          (err && err.name === "AbortError")
            ? "La réponse tarde. Pourriez-vous réessayer ?"
            : "Oups. Sahteïn est momentanément indisponible. Réessayez dans un instant.");
      } finally {
        clearTimeout(timer);
        sending = false;
      }
    }
    function openChat(q) {
      el.chatWindow.classList.add("visible");
      el.searchBar.classList.remove("visible");
      el.searchInput.value = "";
      if (q) ask(q);
    }
    el.blob.addEventListener("click", function () { el.searchBar.classList.toggle("visible"); });
    el.searchInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); openChat(el.searchInput.value.trim()); }
    });
    if (el.sendButton)
      el.sendButton.addEventListener("click", function () { openChat(el.searchInput.value.trim()); });
    if (el.chatInput)
      el.chatInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          var m = el.chatInput.value.trim();
          el.chatInput.value = "";
          ask(m);
        }
      });
    var closeBtn = document.getElementById("closeBtn");
    if (closeBtn) closeBtn.addEventListener("click", function () { el.chatWindow.classList.remove("visible"); });
  }

  // ── Démarrage : on ne bascule QUE si notre serveur répond ────────────────
  function boot() {
    var done = false;
    var giveUp = setTimeout(function () {
      if (!done) { done = true; bootLegacy(); }
    }, 4000);

    fetch(ORIGIN + "/healthz", { method: "GET", cache: "no-store" })
      .then(function (r) {
        if (done) return;
        done = true;
        clearTimeout(giveUp);
        if (r && r.ok) bootModern();
        else bootLegacy();
      })
      .catch(function () {
        if (done) return;
        done = true;
        clearTimeout(giveUp);
        bootLegacy();
      });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
