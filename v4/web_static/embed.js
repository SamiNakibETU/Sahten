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
    var btn = document.createElement("button");
    btn.type = "button";
    btn.setAttribute("aria-label", "Ouvrir Sahteïn — assistant recettes L'Orient-Le Jour");
    btn.title = "Une idée recette ? Ouvrez Sahteïn.";
    btn.style.cssText = [
      "position:fixed", "bottom:20px", "right:20px",
      "width:60px", "height:60px", "border-radius:50%", "border:0",
      "padding:0", "cursor:pointer", "z-index:" + Z,
      "background:#0b5c3f", "box-shadow:0 6px 20px rgba(0,0,0,.28)",
      "display:flex", "align-items:center", "justify-content:center",
      "transition:transform .15s ease, box-shadow .15s ease"
    ].join(";");
    var logo = document.createElement("img");
    logo.src = LOGO;
    logo.alt = "";
    logo.width = 38; logo.height = 38;
    logo.style.cssText = "width:38px;height:38px;pointer-events:none;";
    btn.appendChild(logo);
    btn.addEventListener("mouseenter", function () { btn.style.transform = "scale(1.06)"; });
    btn.addEventListener("mouseleave", function () { btn.style.transform = "scale(1)"; });

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
        frame.src = ORIGIN + "/widget?embed=1&v=" + Math.floor(Date.now()/600000);
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
      btn.style.display = "none";
    }

    function closeWidget() {
      if (frame) frame.style.display = "none";
      btn.style.display = "flex";
    }

    btn.addEventListener("click", openWidget);

    // Le widget (dans l'iframe) demande la fermeture via postMessage.
    window.addEventListener("message", function (e) {
      if (e.origin !== ORIGIN) return;          // sécurité : n'écoute que notre origine
      var d = e.data || {};
      if (d && d.type === "sahten:close") { expanded = false; closeWidget(); }
      if (d && d.type === "sahten:size") { expanded = !!d.expanded; sizeFrame(); }
    });

    document.body.appendChild(btn);
  });
})();
