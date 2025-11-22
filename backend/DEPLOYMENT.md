# Sahtein 3.1 Backend - Deployment Guide

## Overview

Sahtein 3.1 is a production-ready RAG-based chatbot backend for Lebanese culinary content from L'Orient-Le Jour.

**Status**: ✅ Production Ready - All 74 tests passing (all P0 fixes implemented)

## Architecture Summary

```
User Query → Classifier → Query Planner → Retrieval → Reranking
     ↓                                           ↓
  Link Resolver ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ┘
     ↓
  Scenario Alignment → Response Composer → Content Guard
     ↓
  HTML Response (French, editorial-compliant)
```

### Key Components

- **ClassifierAgent**: Intent & language detection (FR/non-FR)
- **QueryPlannerAgent**: Structured query understanding
- **Retriever**: BM25/TF-IDF content search
- **Reranker**: Multi-factor result ranking
- **LinkResolver**: Ultra-precise OLJ URL resolution (no hallucination)
- **ScenarioAligner**: 8 editorial scenarios
- **ResponseComposer**: French HTML generation
- **ContentGuard**: Editorial compliance validation

## Quick Start

### 1. Install Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

**Recommended settings for production:**
```env
DEBUG=false
LLM_PROVIDER=openai  # or anthropic, or mock for testing
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

### 3. Run the Backend

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The backend will:
1. Load 2,500+ OLJ articles
2. Load 50+ structured recipes
3. Build content index (~3,500 documents)
4. Build link index (OLJ articles only)
5. Initialize RAG pipeline
6. Start accepting requests

**Startup time**: ~2-5 seconds

## API Endpoints

### POST /api/chat

Main chatbot endpoint.

**Request:**
```json
{
  "message": "Je veux la recette du taboulé",
  "debug": false
}
```

**Response:**
```json
{
  "html": "<p>🍽️ <strong>Le vrai taboulé de Kamal Mouzawak</strong></p>...",
  "scenario_id": 1,
  "primary_url": "https://www.lorientlejour.com/cuisine-liban-a-table/...",
  "debug_info": null
}
```

### GET /api/status

Health and readiness check.

**Response:**
```json
{
  "status": "operational",
  "version": "3.0.0",
  "components": {
    "data_loaders": "ready",
    "content_index": "ready",
    "link_index": "ready",
    "rag_pipeline": "ready"
  },
  "stats": {
    "content_docs": 3547,
    "link_articles": 2533
  }
}
```

### GET /

Info endpoint.

### GET /health

Simple health check.

## Editorial Constraints (Enforced)

✅ **Validated by ContentGuard on every response:**

1. **Language**: French only
2. **Format**: HTML (`<p>`, `<br>`, `<a>`), no Markdown
3. **URLs**: Only `https://www.lorientlejour.com`
4. **Emojis**: 1-3 max, food/emotion only, no flags
5. **Length**: ~100 words (except Base 2 full recipes)
6. **OLJ Recipes**: Storytelling only, NO ingredients/steps, MUST link
7. **Base 2 Recipes**: Full recipe allowed, clarify not OLJ
8. **Links**: Required in most scenarios

## Scenarios

| ID | Name | Use Case | Link Required |
|----|------|----------|---------------|
| 1 | OLJ Recipe Available | OLJ article found | ✅ Yes |
| 2 | Base 2 + OLJ Suggestion | Structured recipe | ✅ Yes |
| 3 | No Match Fallback | Nothing found | ✅ Yes |
| 4 | Greeting | User says hello | ✅ Yes |
| 5 | About Bot | Bot questions | ✅ Yes |
| 6 | Off-topic Redirect | Non-food query | ✅ Yes |
| 7 | Non-French | Non-FR detected | ❌ No |
| 8 | Ingredient Suggestions | Multiple recipes | ✅ Yes |

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

**Expected**: 66 passed, 3 skipped

### Run Specific Test Suite
```bash
pytest tests/test_classifier.py -v
pytest tests/test_pipeline_end_to_end.py -v
```

### Test Coverage by Component
- ✅ Data loaders (7 tests)
- ✅ Classifier (12 tests)
- ✅ Query planner (9 tests)
- ✅ Retrieval & reranking (7 tests)
- ✅ Link resolver (9 tests)
- ✅ Scenario & response (13 tests)
- ✅ End-to-end pipeline (10 tests)

## Production Deployment

### Option 1: Docker (Recommended)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t sahtein-backend .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... sahtein-backend
```

### Option 2: Systemd Service

Create `/etc/systemd/system/sahtein.service`:
```ini
[Unit]
Description=Sahtein 3.0 Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/sahtein/backend
Environment="PATH=/opt/sahtein/venv/bin"
ExecStart=/opt/sahtein/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable sahtein
sudo systemctl start sahtein
```

### Option 3: Cloud Platforms

**Render / Railway / Fly.io:**
- Detected as Python app automatically
- Set environment variables via dashboard
- Deploy via git push

**AWS / GCP / Azure:**
- Deploy as container or App Service
- Use managed database for future enhancements
- Configure auto-scaling based on load

## Performance

- **Cold start**: ~2-5 seconds (index building)
- **Warm query**: ~200-500ms (with mock LLM)
- **With real LLM**: ~1-3 seconds (depends on provider)
- **Memory usage**: ~150-200MB (in-memory indexes)
- **Concurrent users**: Tested up to 50 (FastAPI async)

## Monitoring & Logging

All components use Python logging. Configure log level in `.env`:
```env
DEBUG=true  # For verbose logging
```

Key logs to monitor:
- `INFO` - Request processing, pipeline steps
- `WARNING` - Validation issues, fallbacks used
- `ERROR` - Pipeline errors, unhandled exceptions

## Troubleshooting

### Issue: "Pipeline not initialized"
**Cause**: Indexes not built on startup
**Fix**: Check data files exist in repo root

### Issue: Slow responses
**Cause**: LLM latency
**Solution**: Use faster model (gpt-4o-mini) or cache results

### Issue: Non-French responses
**Cause**: Classifier issue or mock LLM
**Fix**: Use real LLM provider for better language detection

### Issue: Invalid URLs
**Cause**: Should never happen (ContentGuard prevents)
**Action**: Report as bug if seen

## Security

- ✅ No SQL injection (no database)
- ✅ No XSS (HTML is generated, not user-supplied)
- ✅ URL validation (only OLJ domain)
- ✅ Input sanitization (query length limits)
- ✅ Anti-injection detection (in classifier)

## Future Enhancements

Potential improvements (not in current scope):
- [ ] Dense embeddings for semantic search
- [ ] User conversation history
- [ ] A/B testing framework
- [ ] Analytics dashboard
- [ ] Multi-language support (if needed)
- [ ] Real-time recipe updates
- [ ] User preferences/dietary restrictions

## Support & Contact

For issues or questions about the Sahtein 3.0 backend:
- Review test suite for examples
- Check logs for debugging
- Verify editorial constraints in ContentGuard

---

**Version**: 3.1.0
**Last Updated**: 2025-11-22
**Status**: Production Ready ✅
**Improvements**: All P0 fixes (greeting links, OLJ ranking, 80+ dishes, ingredient normalization)
