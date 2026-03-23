# Sahten Widget - Integration Guide for L'Orient-Le Jour

## Overview

This guide explains how to integrate the **Sahten** conversational widget into the L'Orient-Le Jour website. The widget provides an AI-powered recipe search experience using a RAG system connected to OLJ's recipe database.

## Repository

**GitHub:** https://github.com/SamiNakibETU/Sahten  
**Branch:** `main`  
**Latest Version:** v2.1 (Frontend fully polished, mobile-ready)

---

## What's New in v2.1

- ✅ **Mobile-first bottom sheet** with swipe gestures (drag handle)
- ✅ **Responsive design** for all screen sizes (window, mid, full modes)
- ✅ **XSS protection** via DOMPurify sanitization
- ✅ **Centered SVG icons** and refined UI polish
- ✅ **Touch interactions** optimized for mobile (tap-to-cycle size)
- ✅ **OLJ demo page** (`demo-olj.html`) simulating "A Table" section integration
- ✅ **Feedback collection** (thumbs up/down) for quality monitoring
- ✅ **Model selection** support (nano/mini/auto) for A/B testing

---

## Files to Use

All files are in the `frontend/` directory of the GitHub repository:

| File | Purpose |
|------|---------|
| `css/sahten.css` | Complete widget styles (desktop + mobile) |
| `js/sahten.js` | Widget logic (ES module) |
| `widget.html` | Clean production widget reference (no test controls) |
| `index.html` | Test page with model selector for QA |
| `demo-olj.html` + `css/demo-olj.css` | Full OLJ "A Table" integration demo |

---

## Integration Steps

### 1. Add CSS

Include the Sahten stylesheet in your page `<head>`:

```html
<link rel="stylesheet" href="/path/to/sahten.css" />
```

### 2. Add XSS Protection (DOMPurify)

Include DOMPurify for secure HTML rendering:

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.8/purify.min.js" 
        crossorigin="anonymous" 
        referrerpolicy="no-referrer"></script>
```

### 3. Add Widget HTML

Copy the widget HTML markup from `widget.html` (lines 95-148) into your page body:

```html
<!-- SAHTEN WIDGET -->
<div class="sahten-backdrop"></div>
<div class="sahten-widget-container" data-size="window">
  <!-- Drag Handle (Mobile) -->
  <div class="sahten-drag-handle"></div>

  <!-- Header -->
  <div class="sahten-header">
    <div class="sahten-title">Sahten</div>
    <div class="sahten-controls">
      <button class="sahten-size-btn" data-action="window" aria-label="Petit">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="4.5" y="3.5" width="7" height="9" rx="1.5"/>
        </svg>
      </button>
      <button class="sahten-size-btn" data-action="mid" aria-label="Moyen">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="2.5" y="2.5" width="11" height="11" rx="1.5"/>
        </svg>
      </button>
      <button class="sahten-size-btn" data-action="full" aria-label="Grand">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <polyline points="10 2 14 2 14 6"/><polyline points="6 14 2 14 2 10"/>
          <line x1="14" y1="2" x2="9.5" y2="6.5"/><line x1="2" y1="14" x2="6.5" y2="9.5"/>
        </svg>
      </button>
      <button class="sahten-close-btn" aria-label="Fermer">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
          <line x1="4" y1="4" x2="12" y2="12"/><line x1="12" y1="4" x2="4" y2="12"/>
        </svg>
      </button>
    </div>
  </div>

  <!-- Chat Body -->
  <div class="sahten-body">
    <!-- Messages will be injected here via JS -->
  </div>

  <!-- Footer / Input -->
  <form id="sahten-form" class="sahten-footer">
    <div class="input-wrapper">
      <input type="text" id="sahten-input" class="sahten-input" 
             placeholder="Je cherche une recette..." autocomplete="off" />
      <button type="submit" class="send-btn" aria-label="Envoyer">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" 
             stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="3" y1="8" x2="13" y2="8"/><polyline points="9 4 13 8 9 12"/>
        </svg>
      </button>
    </div>
  </form>
</div>

<!-- Trigger Button -->
<button class="sahten-trigger" aria-label="Ouvrir Sahten">
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" 
       stroke="currentColor" stroke-width="1.5">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
  </svg>
</button>
```

### 4. Configure API Endpoint

Set the API base URL **before** loading the widget script:

```html
<script>
  // Production API (Railway)
  window.SAHTEN_API_BASE = "https://web-production-73152.up.railway.app/api";
</script>
```

### 5. Initialize Widget

Import and initialize the widget (ES module):

```html
<script type="module">
  import { SahtenChat } from "./js/sahten.js";
  
  // Simple initialization (uses default model)
  const chat = new SahtenChat();
  
  // OR with model selector for A/B testing:
  // const chat = new SahtenChat({
  //   modelSelector: document.getElementById('model-select')
  // });
</script>
```

---

## Testing Locally

### Option 1: View Demo Page

Open `demo-olj.html` directly in your browser (simulates full OLJ "A Table" section).

### Option 2: Test Widget Only

Open `widget.html` for a clean widget view without OLJ page simulation.

### Option 3: Test with Model Selection

Open `index.html` to access test mode with model selector dropdown (nano/mini/auto).

**Note:** Local testing requires the backend API to be running and CORS configured for your local origin.

---

## Mobile Behavior

### Desktop (> 480px)
- **Window mode**: Bottom-right fixed position (400x650px)
- **Mid mode**: Centered, 60vw x 70vh
- **Full mode**: Full screen overlay

### Mobile (≤ 480px)
- **Bottom sheet design** with rounded top corners
- **Drag handle** at top for intuitive swipe gestures:
  - Swipe **down**: shrink size or close
  - Swipe **up**: expand size
  - **Tap handle**: cycle through sizes (window → mid → full)
- **Window mode**: 50vh height
- **Mid mode**: 75vh height
- **Full mode**: calc(100vh - 24px) height
- Size buttons hidden (mobile uses drag handle instead)

---

## API Configuration

### Production API (Railway)
```
https://web-production-73152.up.railway.app/api
```

**Available Endpoints:**
- `POST /api/chat` - Send user message, get AI response with recipe cards
- `POST /api/events` - Track impressions and clicks (analytics)
- `POST /api/feedback` - Submit user feedback (👍/👎)
- `GET /api/models` - Get available LLM models

### CORS Configuration

The backend is configured to accept requests from:
- `https://lorientlejour.com`
- `https://*.lorientlejour.com`
- `http://localhost:*` (for local development)
- `http://127.0.0.1:*` (for local development)

If you need additional origins whitelisted, please provide the domains.

---

## Customization

### Colors (CSS Variables)

Edit `sahten.css` to match OLJ branding:

```css
:root {
  --color-bg: #F8F6F6;           /* Widget background */
  --color-text-primary: #1B1A1A; /* Main text */
  --color-text-secondary: #595959; /* Secondary text */
  --color-liban: #94C2AE;        /* Accent color (green) */
  --color-accent: #DD3B31;       /* Error/highlight (red) */
  --color-border: #E0DEDD;       /* Borders */
}
```

### Widget Positioning

Adjust trigger button position in `sahten.css`:

```css
.sahten-trigger {
  bottom: 32px;  /* Distance from bottom */
  right: 32px;   /* Distance from right */
}
```

---

## Security Features

### XSS Protection
All HTML responses from the API are sanitized using **DOMPurify** before rendering. Only whitelisted tags and attributes are allowed.

### Allowed HTML Tags
```javascript
['div', 'p', 'span', 'strong', 'em', 'u', 'br', 'a', 'article', 
 'h3', 'ul', 'li', 'ol', 'blockquote']
```

### Content Security
- No `<script>`, `<iframe>`, `<object>`, or `<embed>` tags allowed
- All `on*` event handlers stripped
- External links open in new tab with `target="_blank"`

---

## Analytics & Feedback

### Event Tracking
The widget automatically tracks:
- **Impressions**: When recipe cards are displayed
- **Clicks**: When users click recipe links
- **Feedback**: When users rate responses (👍/👎)

All events include:
- `session_id`: Unique user session
- `request_id`: Specific query identifier
- `model_used`: LLM model that generated response
- `intent`: Detected user intent (recipe_search, ingredient_based, etc.)

### Feedback Collection
Each bot response includes feedback buttons:
- ✅ **Positive**: Submits immediately
- ❌ **Negative**: Prompts for optional reason text

Data is sent to `/api/feedback` and `/api/events` endpoints.

---

## Troubleshooting

### Widget Not Appearing
1. Check browser console for errors
2. Verify `sahten.css` is loaded
3. Verify `sahten.js` is loaded as ES module (`type="module"`)
4. Check that DOMPurify is loaded before widget initialization

### API Errors
1. Verify `window.SAHTEN_API_BASE` is set correctly
2. Check network tab for CORS errors
3. Verify backend is running: `curl https://web-production-73152.up.railway.app/api/models`

### Mobile Issues
1. Clear browser cache
2. Test in Chrome DevTools mobile emulation
3. Verify viewport meta tag: `<meta name="viewport" content="width=device-width, initial-scale=1.0">`

---

## Support & Contact

For technical questions or issues during integration:
- **GitHub Issues**: https://github.com/SamiNakibETU/Sahten/issues
- **Backend Status**: Check Railway deployment logs
- **Demo**: `demo-olj.html` for visual reference

---

## Deployment Checklist

- [ ] Copy `css/sahten.css` to your static assets
- [ ] Copy `js/sahten.js` to your static assets
- [ ] Include DOMPurify script in page `<head>`
- [ ] Add widget HTML markup to page body
- [ ] Set `window.SAHTEN_API_BASE` to production API
- [ ] Initialize widget with ES module import
- [ ] Test on desktop (Chrome, Firefox, Safari)
- [ ] Test on mobile (iOS Safari, Android Chrome)
- [ ] Verify analytics events are firing
- [ ] Verify feedback collection works
- [ ] Monitor backend logs for errors

---

**Version:** v2.1  
**Last Updated:** February 10, 2026  
**Maintainer:** Sahten Team
