# Sahtein 3.1 Backend + Frontend

Lebanese culinary chatbot for L'Orient-Le Jour, powered by RAG (Retrieval-Augmented Generation).

**Status**: ✅ Production Ready - All 74 tests passing

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
cd backend
pip install -r requirements.txt

# 3. Run the application
python main.py

# 4. Open in browser
# Frontend UI: http://localhost:8000/
# API docs: http://localhost:8000/docs
# Status: http://localhost:8000/api/status
```

## Architecture

```
v2/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes and dependencies
│   │   ├── data/         # Data loading, indexing, culinary graph
│   │   ├── models/       # Pydantic schemas and configuration
│   │   └── rag/          # RAG pipeline components
│   ├── tests/            # Test suite (74 tests)
│   ├── main.py           # FastAPI application entry point
│   └── requirements.txt  # Python dependencies
├── frontend/
│   └── index.html        # Production-ready chat UI
└── [data files]          # olj_recette_liban_a_table.json, etc.
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

## Running the Application

### Backend + Frontend (Recommended)

The easiest way to run both backend and frontend together:

```bash
cd backend
python main.py
```

Then open your browser to **http://localhost:8000/** to access the chat UI.

### Development Mode

With auto-reload for development:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Frontend Features

The production-ready chat UI (`frontend/index.html`) includes:

- **💬 Intelligent Chat Interface** - Clean, responsive design
- **🔍 Debug Mode Toggle** - Shows scenario IDs, debug info, and backend details
- **🏷️ Metadata Badges** - Displays scenario name and OLJ article links
- **✨ Quick Suggestions** - Pre-built queries for easy demos
- **🗑️ Conversation Reset** - Clear chat history
- **🪟 Flexible Layouts** - Window, half-screen, and full-screen modes
- **⌨️ Keyboard Shortcuts** - Ctrl+K to toggle chat, Esc to close
- **📱 Mobile Responsive** - Works on all devices

## API Endpoints

### Chat Endpoint

**POST** `/api/chat`

Request:
```json
{
  "message": "Je veux la recette du taboulé",
  "debug": false
}
```

Response:
```json
{
  "html": "<p>Voici une délicieuse recette...</p>",
  "scenario_id": 1,
  "primary_url": "https://www.lorientlejour.com/...",
  "debug_info": { ... }  // Only if debug=true
}
```

### Other Endpoints

- **GET** `/` - Serves the frontend UI (or JSON status if frontend not found)
- **GET** `/health` - Health check endpoint
- **GET** `/api/status` - Detailed API status and component readiness
- **GET** `/docs` - Interactive API documentation (Swagger UI)
- **GET** `/redoc` - Alternative API documentation (ReDoc)

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_classifier.py -v
```

## Development Status

### ✅ Completed (v3.1)

- [x] **Phase 1-3**: Backend structure, data loaders, classifier & query planner
- [x] **Phase 4-5**: RAG retrieval, reranking, link resolver, audit & P0 fixes
- [x] **Phase 6-7**: Scenario alignment, response composer, full pipeline integration
- [x] **Phase 8**: Complete testing suite (74 tests) and documentation
- [x] **Frontend**: Production-ready chat UI with debug mode
- [x] **Integration**: Backend serves frontend, full end-to-end flow

### 🎯 Production Ready Features

- ✅ 79+ dishes in culinary graph (expanded from 40)
- ✅ 60+ ingredient equivalence groups (French/English)
- ✅ Strict OLJ vs Base 2 ranking rules
- ✅ Greeting scenario fallback article support
- ✅ All golden examples passing
- ✅ Zero hallucination link resolver
- ✅ Full editorial compliance validation

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
