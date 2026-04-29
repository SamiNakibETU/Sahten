/**
 * Appels API protégés par SAHTEN_ADMIN_API_TOKEN (header X-Sahten-Admin-Token).
 */
(function () {
  "use strict";

  var STORAGE_KEY = "sahten_admin_token";
  var prompting = false;

  function getToken() {
    return localStorage.getItem(STORAGE_KEY) || "";
  }

  function setToken(t) {
    if (t) {
      localStorage.setItem(STORAGE_KEY, t);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }

  function buildHeaders(base, token) {
    var h = base ? new Headers(base) : new Headers();
    if (token) {
      h.set("X-Sahten-Admin-Token", token);
    }
    return h;
  }

  async function adminFetch(url, options) {
    options = options || {};
    var token = getToken();
    var headers = buildHeaders(options.headers, token);
    var r = await fetch(
      url,
      Object.assign({}, options, { headers: headers }),
    );

    if (r.status === 401 && !prompting) {
      prompting = true;
      try {
        var msg =
          "Cette console nécessite le jeton d’administration (SAHTEN_ADMIN_API_TOKEN).";
        var entered = window.prompt(msg, "");
        if (entered && String(entered).trim()) {
          var next = String(entered).trim();
          setToken(next);
          headers = buildHeaders(options.headers, next);
          r = await fetch(
            url,
            Object.assign({}, options, { headers: headers }),
          );
        }
      } finally {
        prompting = false;
      }
    }

    return r;
  }

  window.sahtenAdmin = {
    getToken: getToken,
    setToken: setToken,
    fetch: adminFetch,
  };
})();
