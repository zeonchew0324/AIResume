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
```

Backend requires a `.env` file in `backend/` with `OPENAI_API_KEY`.

## Architecture

### Data Flow

1. User uploads PDF resume + job title/description on the frontend
2. Frontend POSTs `FormData` to `/api/analyze`
3. Vite dev proxy forwards `/api/*` → `http://localhost:8000`
4. Backend route → service (PDF extraction) → LangChain graph (AI scoring)
5. Returns `ResumeAnalysisResponse`: `match_score`, `feedback`, `suggestions`, `missing_keywords`
6. Frontend renders results in tabbed UI (pie chart score, feedback, suggestions, keywords)

### Backend (`backend/app/`)

- **`main.py`** — FastAPI app with CORS for `localhost:5173`, mounts routers
- **`routes/`** — HTTP layer; delegates to `services/`
- **`services/`** — Business logic; extracts PDF text, calls LangChain graph
- **`graphs/`** — LangGraph workflows; constructs the ChatOpenAI chain with structured output
- **`prompts/`** — System/user prompt templates for ATS evaluation
- **`models/`** — Pydantic request/response schemas
- **`utils/`** — PDF parsing (PyPDF2)

Model used: `gpt-4o-mini` via LangChain-OpenAI.

### Frontend (`frontend/src/`)

- **`App.tsx`** — React Router with two routes: `/analyze` (functional), `/improve` (placeholder)
- **`pages/`** — `AnalyzeResume` page owns the full upload + results flow
- **`components/`** — `AppLayout` (sidebar nav), `ScoreChart` (Recharts PieChart), shadcn UI primitives
- Path alias `@/` maps to `src/`

### Key Tech

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS 4, shadcn/ui, Recharts
- **Backend**: FastAPI, LangChain + LangGraph, OpenAI, PyPDF2, Pydantic

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
