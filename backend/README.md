# Sahtein 3.0 Backend

Lebanese culinary chatbot backend for L'Orient-Le Jour, powered by RAG (Retrieval-Augmented Generation).

## Architecture

```
backend/
├── app/
│   ├── api/          # FastAPI routes and dependencies
│   ├── data/         # Data loading and indexing
│   ├── models/       # Pydantic schemas and configuration
│   └── rag/          # RAG pipeline components
├── tests/            # Test suite
├── main.py           # FastAPI application entry point
└── requirements.txt  # Python dependencies
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

## Running the Backend

Development mode (with auto-reload):
```bash
cd backend
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Production mode:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /api/chat` - Main chat endpoint
  - Request: `{"message": "Je veux la recette du taboulé"}`
  - Response: `{"html": "...", "scenario_id": 1, "primary_url": "..."}`

- `GET /` - Root/info endpoint
- `GET /health` - Health check
- `GET /api/status` - API status and component readiness

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_classifier.py -v
```

## Development Phases

- [x] Phase 1: Backend structure and FastAPI setup
- [ ] Phase 2: Data loaders and models
- [ ] Phase 3: Classifier and Query Planner agents
- [ ] Phase 4: RAG retrieval and reranking
- [ ] Phase 5: Link resolver
- [ ] Phase 6: Scenario alignment and response composer
- [ ] Phase 7: Complete RAG pipeline integration
- [ ] Phase 8: Testing and documentation

## Editorial Constraints

- **Language**: French only in responses
- **Cuisine**: Lebanese and Mediterranean focus
- **OLJ Recipes**: Storytelling only, no full recipes, must link to article
- **Base 2 Recipes**: Can provide full recipe, but clarify it's not OLJ content
- **Links**: Only `https://www.lorientlejour.com` domain
- **Format**: HTML output (`<p>`, `<br>`, `<a>`), no Markdown
- **Emojis**: 1-3 max, food/emotion related, no flags
- **Length**: ~100 words (except full Base 2 recipes)

## License

Proprietary - L'Orient-Le Jour
