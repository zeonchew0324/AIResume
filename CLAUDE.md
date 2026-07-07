# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Frontend (run from `frontend/`)

```bash
npm run dev      # Dev server on http://localhost:5173
npm run build    # TypeScript + Vite production build
npm run lint     # ESLint
npm run preview  # Preview production build
```

### Backend (run from `backend/`)

```bash
python -m uvicorn app.main:app --reload  # Dev server on http://localhost:8000
pip install -r requirements.txt          # Install dependencies
pytest                                   # Run test suite (app/tests/)
```

### Environment

Backend `.env` (in `backend/`):
- `OPENAI_API_KEY` — OpenAI key for the LangChain graphs
- `SUPABASE_URL` — Supabase project URL; used to fetch the JWKS for JWT verification
- `DB_URL` — Postgres connection string (asyncpg; TLS verified against certifi or `DB_SSL_ROOT_CERT`)

Frontend `.env` (in `frontend/`, see `.env.example`):
- `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`

### Database migrations

Supabase CLI project lives in `backend/supabase/`. Schema changes go in
`backend/supabase/migrations/` as new timestamped files (never edit applied
migrations); apply with `supabase db push` from `backend/`. `resumes.user_id`
references `auth.users(id)` — user IDs come from the Supabase JWT `sub` claim.

## Architecture

### Data Flow

1. User signs in via Supabase (`/login`, `AuthContext`); all app routes are wrapped in `ProtectedRoute`
2. User uploads PDF resumes on `/my-resume` → `POST /api/resumes` extracts text (PyPDF2) and stores it in Postgres, scoped to the authed user
3. Feature pages (`/analyze`, `/improve`, `/cover-letter`) select a saved resume via `ResumeSelect` dropdown and POST `FormData` (`resume_id` + job fields) with a `Bearer` token
4. Backend verifies the JWT against the Supabase JWKS (`app/auth.py`), loads the resume text from the DB, cleans/caps inputs, and runs the matching LangGraph chain with a timeout
5. Structured Pydantic response is rendered in a tabbed UI (pie-chart score, feedback, suggestions, keywords; DOCX download for improve/cover-letter)

### API Endpoints (all under rate limit 5/min except resumes CRUD)

- `POST /api/analyze` — ATS match score, feedback, suggestions, missing keywords
- `POST /api/improve` — rewritten resume, list of changes, keywords
- `POST /api/coverletter` — generated cover letter
- `POST /api/resumes`, `GET /api/resumes`, `DELETE /api/resumes/{id}` — saved-resume CRUD
- `POST /health` — health check (no auth)

All `/api/*` endpoints require `Authorization: Bearer <supabase-jwt>`.

### Backend (`backend/app/`)

- **`main.py`** — FastAPI app; CORS for `localhost:5173`, slowapi rate limiter, mounts routers
- **`auth.py`** — Supabase JWT verification via cached JWKS client; yields `user_id` from `sub`
- **`routes/`** — HTTP layer (`ats.py`: analyze/improve/coverletter; `resumes.py`: CRUD); delegates to `services/`
- **`services/`** — Business logic; loads resume text, calls LangChain graphs, resume persistence
- **`graphs/`** — LangGraph workflows per feature; ChatOpenAI with structured output
- **`prompts/`** — System/user prompt templates per feature
- **`models/`** — Pydantic schemas (`schemas.py`) and SQLAlchemy `Resume` model (`db.py`)
- **`db/`** — Async SQLAlchemy engine/session with TLS-verified Postgres connection
- **`utils/`** — PDF parsing (PyPDF2), input cleaning with length caps
- **`tests/`** — pytest suite covering routes, auth, and utils

Model used: `gpt-4o-mini` via LangChain-OpenAI (`app/config.py`).

### Frontend (`frontend/src/`)

- **`App.tsx`** — React Router: `/login` public; `/analyze`, `/improve`, `/cover-letter`, `/my-resume` behind `ProtectedRoute` inside `AppLayout`
- **`context/AuthContext.tsx`** — Supabase session state; `authHeaders()` helper for API calls
- **`pages/`** — one page per feature; each owns its form + results flow
- **`components/`** — `AppLayout` (sidebar nav), `ProtectedRoute`, `ResumeSelect` (saved-resume dropdown), `ScoreChart` (Recharts PieChart), shadcn UI primitives in `ui/`
- **`hooks/use-resumes.ts`** — fetches the user's saved resumes
- Path alias `@/` maps to `src/`

### Key Tech

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS 4, shadcn/ui, Recharts, Supabase JS
- **Backend**: FastAPI, LangChain + LangGraph, OpenAI, SQLAlchemy (async) + asyncpg, PyJWT, PyPDF2, slowapi, Pydantic

DISTILLED_AESTHETICS_PROMPT = """
<frontend_aesthetics>
You tend to converge toward generic, "on distribution" outputs. In frontend design, this creates what users call the "AI slop" aesthetic. Avoid this: make creative, distinctive frontends that surprise and delight. Focus on:

Typography: Choose fonts that are beautiful, unique, and interesting. Avoid generic fonts like Arial and Inter; opt instead for distinctive choices that elevate the frontend's aesthetics.

Color & Theme: Commit to a cohesive aesthetic. Use CSS variables for consistency. Dominant colors with sharp accents outperform timid, evenly-distributed palettes. Draw from IDE themes and cultural aesthetics for inspiration.

Motion: Use animations for effects and micro-interactions. Prioritize CSS-only solutions for HTML. Use Motion library for React when available. Focus on high-impact moments: one well-orchestrated page load with staggered reveals (animation-delay) creates more delight than scattered micro-interactions.

Backgrounds: Create atmosphere and depth rather than defaulting to solid colors. Layer CSS gradients, use geometric patterns, or add contextual effects that match the overall aesthetic.

Avoid generic AI-generated aesthetics:

- Overused font families (Inter, Roboto, Arial, system fonts)
- Clichéd color schemes (particularly purple gradients on white backgrounds)
- Predictable layouts and component patterns
- Cookie-cutter design that lacks context-specific character

Interpret creatively and make unexpected choices that feel genuinely designed for the context. Vary between light and dark themes, different fonts, different aesthetics. You still tend to converge on common choices (Space Grotesk, for example) across generations. Avoid this: it is critical that you think outside the box!
</frontend_aesthetics>
"""
