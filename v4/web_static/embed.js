/* Sahteïn — chargeur embarquable pour lorientlejour.com
 * ---------------------------------------------------------------------------
 * Intégration DÉFINITIVE : une seule balise sur la page OLJ
 *
 *     <script src="https://sahtenbotapi.lorientlejour.com/embed.js" defer></script>
 *
 * À partir de là, plus RIEN à changer côté site. Le widget (interface + logique)
 * est servi par notre serveur, qui suit GitHub : chaque `git push` met à jour le
 * widget en ligne tout seul. WhiteBeard ne remplace plus jamais de fichier.
 *
 * Modèle : bouton flottant injecté ici + panneau iframe (chargé depuis notre
 * origine, donc isolation CSS totale vis-à-vis du site OLJ). Standard des widgets
 * de chat (Intercom, Crisp, Zendesk).
 */
(function () {
  "use strict";
  if (window.__sahtenEmbedLoaded) return;      // idempotent (double inclusion)
  window.__sahtenEmbedLoaded = true;

  // Origine du script : robuste même si le domaine change un jour.
  var ORIGIN = (function () {
    try {
      var cur = document.currentScript && document.currentScript.src;
      if (cur) return new URL(cur).origin;
    } catch (e) {}
    return "https://sahtenbotapi.lorientlejour.com";
  })();

  var Z = 2147483000;                           // au-dessus de tout
  var LOGO = ORIGIN + "/assets/v7_logo_sahten.svg";

  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  ready(function () {
    // --- Bouton flottant (trigger) ---------------------------------------
    // Lanceur : une bulle « coucou » à gauche + le logo, exactement comme sur
    // la page autonome. Le disque vert saturé de la version précédente ne
    // ressemblait à rien du reste du site, et sans texte le bouton n'invitait
    // pas à cliquer.
    var wrap = document.createElement("div");
    wrap.style.cssText = [
      "position:fixed", "bottom:20px", "right:20px", "z-index:" + Z,
      "display:flex", "flex-direction:row", "align-items:center",
      "gap:8px", "max-width:min(calc(100vw - 32px),300px)", "cursor:pointer",
      "opacity:0", "transform:translateY(8px)",
      "transition:opacity .35s cubic-bezier(.23,1,.32,1),transform .35s cubic-bezier(.23,1,.32,1)"
    ].join(";");

    var btn = document.createElement("button");
    btn.type = "button";
    btn.setAttribute("aria-label", "Ouvrir Sahteïn — assistant recettes L'Orient-Le Jour");
    btn.title = "Une idée recette ? Ouvrez Sahteïn.";
    // Logo seul, sans fond : identique à la page autonome. Une ombre portée
    // légère suffit à le détacher d'une photo d'article.
    btn.style.cssText = [
      "flex-shrink:0", "width:auto", "height:auto", "padding:0", "border:0",
      "background:transparent", "cursor:pointer", "display:flex",
      "align-items:center", "justify-content:center",
      "-webkit-tap-highlight-color:transparent",
      "transition:transform .18s cubic-bezier(.23,1,.32,1)"
    ].join(";");
    var logo = document.createElement("img");
    logo.src = LOGO;
    logo.alt = "";
    logo.width = 52; logo.height = 52;
    logo.style.cssText =
      "display:block;width:52px;height:52px;object-fit:contain;pointer-events:none;" +
      "filter:drop-shadow(0 1px 5px rgba(0,0,0,.18));";
    btn.appendChild(logo);

    var bubble = document.createElement("div");
    bubble.textContent = "Une idée recette ?";
    bubble.style.cssText = [
      "max-width:min(176px,56vw)", "padding:6px 12px", "background:#ffffff",
      "border:1px solid rgba(0,0,0,.07)", "border-radius:999px",
      "box-shadow:0 1px 8px rgba(0,0,0,.05)",
      "font:600 11px/1.25 'Aktiv Grotesk Trial','Aktiv Grotesk','DM Sans',-apple-system,sans-serif",
      "letter-spacing:-.01em", "color:#000", "text-align:center",
      "user-select:none", "white-space:nowrap",
      // Elle se montre au chargement puis s'efface ; revient au survol.
      "opacity:0", "transform:translateX(6px)", "pointer-events:none",
      "transition:opacity .22s cubic-bezier(.23,1,.32,1),transform .22s cubic-bezier(.23,1,.32,1)"
    ].join(";");
    var showBubble = function (on) {
      bubble.style.opacity = on ? "1" : "0";
      bubble.style.transform = on ? "translateX(0)" : "translateX(6px)";
      bubble.style.pointerEvents = on ? "auto" : "none";
    };

    wrap.appendChild(bubble);
    wrap.appendChild(btn);
    wrap.addEventListener("mouseenter", function () {
      btn.style.transform = "scale(1.06)";
      showBubble(true);
    });
    wrap.addEventListener("mouseleave", function () {
      btn.style.transform = "scale(1)";
      showBubble(false);
    });
    // Entrée douce une fois la page posée : une apparition sèche au chargement
    // se remarque plus qu'elle n'invite.
    setTimeout(function () {
      wrap.style.opacity = "1";
      wrap.style.transform = "translateY(0)";
      // Le « coucou » : la bulle se montre, puis s'efface d'elle-même.
      showBubble(true);
      setTimeout(function () { showBubble(false); }, 4200);
    }, 600);

    // --- Panneau iframe (créé au premier clic) ---------------------------
    var frame = null;

    function isMobile() { return window.matchMedia("(max-width: 600px)").matches; }

    // Agrandi (bouton ⤢ dans le widget) : l'iframe couvre tout l'écran en
    // TRANSPARENT — le panneau centré + le voile sont dessinés dedans.
    var expanded = false;

    function sizeFrame() {
      if (!frame) return;
      if (expanded || isMobile()) {
        frame.style.cssText = baseFrameCss() +
          ";inset:0;width:100%;height:100%;border-radius:0" +
          (expanded && !isMobile()
            ? ";background:transparent;box-shadow:none"
            : "");
      } else {
        frame.style.cssText = baseFrameCss() +
          ";bottom:20px;right:20px;width:412px;height:min(680px, calc(100vh - 40px));border-radius:16px;";
      }
    }
    function baseFrameCss() {
      return [
        "position:fixed", "border:0", "z-index:" + (Z + 1),
        "background:#ffffff", "box-shadow:0 12px 48px rgba(0,0,0,.28)",
        "max-width:100vw", "max-height:100vh", "color-scheme:normal"
      ].join(";");
    }

    function openWidget() {
      if (!frame) {
        frame = document.createElement("iframe");
        // `host=desktop` dit au widget de NE PAS basculer en rendu téléphone :
        // son viewport interne (412px) est sous le seuil mobile alors que le
        // visiteur est sur un écran de bureau.
        frame.src = ORIGIN + "/widget?embed=1" +
          (isMobile() ? "" : "&host=desktop") +
          "&v=" + Math.floor(Date.now()/600000);
        frame.title = "Sahteïn — assistant recettes";
        frame.setAttribute("allow", "clipboard-write");
        frame.setAttribute("allowtransparency", "true");
        sizeFrame();
        document.body.appendChild(frame);
        window.addEventListener("resize", sizeFrame);
      } else {
        frame.style.display = "block";
        sizeFrame();
        // L'iframe existe déjà mais le chat s'était replié à la fermeture :
        // on lui redemande de s'ouvrir (conversation préservée).
        try { frame.contentWindow.postMessage({ type: "sahten:open" }, ORIGIN); } catch (e) {}
      }
      wrap.style.display = "none";
    }

    function closeWidget() {
      if (frame) frame.style.display = "none";
      wrap.style.display = "flex";
    }

    // La bulle est une cible d'ouverture, pas une étiquette : le wrap entier
    // est cliquable.
    wrap.addEventListener("click", openWidget);

    // Le widget (dans l'iframe) demande la fermeture via postMessage.
    window.addEventListener("message", function (e) {
      if (e.origin !== ORIGIN) return;          // sécurité : n'écoute que notre origine
      var d = e.data || {};
      if (d && d.type === "sahten:close") { expanded = false; closeWidget(); }
      if (d && d.type === "sahten:size") { expanded = !!d.expanded; sizeFrame(); }
    });

    document.body.appendChild(wrap);
  });
})();
