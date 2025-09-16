# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend (FastAPI)
```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8501

# Train XGBoost model
python train.py

# Setup product embeddings (requires OpenAI API key)
cd app
python setup_product_embeddings.py
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev    # Development server on port 3001
npm run build  # Production build
npm run lint   # ESLint check
```

### Docker Environment
```bash
# Create network and start all services
docker network create shopping-network
docker-compose up -d

# Database initialization
docker exec -i trnAi_postgres psql -U TRN_AI -d TRNAI_DB < init_database.sql

# Check service status
docker-compose ps
```

## Architecture Overview

This is an AI-powered home shopping broadcast recommendation system with these key components:

### Core Services
- **FastAPI Backend** (`backend/app/main.py`): Main API server on port 8501
- **Next.js Frontend** (`frontend/`): Web interface on port 3001
- **PostgreSQL**: Relational data (products, sales, trends)
- **Qdrant Vector DB**: Product embeddings for semantic search on port 6333
- **n8n**: Batch workflow automation on port 5678

### Key Technologies
- **LangChain**: AI workflow orchestration and prompt management
- **XGBoost**: Sales prediction models (`backend/train.py`)
- **OpenAI Embeddings**: Semantic product matching
- **Async Processing**: `asyncio.gather` for parallel Track A/B execution

### Data Flow
1. **n8n** collects trends every 30 minutes from external APIs
2. **FastAPI** receives broadcast time requests from frontend
3. **BroadcastWorkflow** (`backend/app/broadcast_workflow.py`) executes:
   - Track A: Category-based search → XGBoost prediction
   - Track B: Direct product search via embeddings
4. Results merged and ranked using weighted scoring formula
5. **LangChain** generates AI reasoning for each recommendation

## Core Architecture Components

### Main API Endpoints
- `POST /api/v1/broadcast/recommendations`: Core AI recommendation API
- `GET /api/v1/health`: Service health check
- `POST /api/v1/trends/collect`: Batch trend collection (n8n triggered)

### Key Models & Classes
- `BroadcastWorkflow`: Main orchestrator for recommendation logic
- `ProductEmbedder`: Qdrant integration for semantic search
- `TrendProcessor`: Trend analysis and product matching
- `XGBoost Model`: Sales prediction (trained via `train.py`)

### Database Schema
- `TAIGOODS`: Product catalog with categories and keywords
- `TAIPGMTAPE`: Broadcast tape inventory (filters available products)
- `TAIBROADCASTS`: Historical sales data for model training
- `trends_data`: Real-time trend keywords from external APIs

### Environment Configuration
Required environment variables in `backend/.env`:
```
DB_URI=postgresql://TRN_AI:TRN_AI@localhost:5432/TRNAI_DB
OPENAI_API_KEY=your_openai_api_key_here
```

### Docker Service Dependencies
- Backend depends on: PostgreSQL, Qdrant
- Frontend communicates with: Backend (port 8501)
- All services use `shopping-network` Docker network

## Testing & Validation

### API Testing
```bash
# Test main recommendation endpoint
curl -X POST http://localhost:8501/api/v1/broadcast/recommendations \
  -H "Content-Type: application/json" \
  -d '{"broadcastTime": "2025-09-15T22:40:00+09:00", "recommendationCount": 5}'

# Health check
curl http://localhost:8501/api/v1/health
```

### Model Validation
- XGBoost model file: `backend/app/xgb_broadcast_sales.joblib`
- Training data from `TAIBROADCASTS` table with features: broadcast_timestamp, category, weather, competition
- Target variable: `actual_sales_amount`

## Development Notes

### Parallel Processing Pattern
The system uses `asyncio.gather` for parallel execution of Track A (category search) and Track B (product search) in `broadcast_workflow.py`. This pattern optimizes I/O-bound operations.

### Error Handling Strategy
- OpenAI API failures: Return 503 with retry message
- Empty recommendations: Return 503 with service unavailable
- Invalid requests: Return 400 with validation details
- Database failures: Return 500 with generic error

### Performance Considerations
- Response time target: 2-3 seconds average
- Qdrant vector search parameters: k=50 for categories, k=30 for products
- Result caching: Weather/holiday data cached for 1 hour
- Connection pooling: Vector DB connections reused via application state