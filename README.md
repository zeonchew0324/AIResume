# AIResume

AI-powered resume toolkit. Upload your resume once, then analyze it against job
descriptions, get an ATS-style match score, generate improved rewrites, and
draft tailored cover letters — all scoped to your account.

## Features

- **Analyze** — ATS match score (pie chart), feedback, suggestions, and missing
  keywords for a given job title + description
- **Improve** — AI-rewritten resume tailored to the job, with a list of changes
  and keywords; downloadable as DOCX
- **Cover letter** — tailored cover letter generation with DOCX download
- **My resumes** — upload PDF resumes once; text is extracted and saved so every
  feature works from a dropdown instead of re-uploading
- **Auth** — Supabase email sign-in/sign-up; every API call is JWT-verified and
  data is scoped per user

## Tech Stack

| Layer    | Tech                                                                    |
|----------|-------------------------------------------------------------------------|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS 4, shadcn/ui, Recharts         |
| Backend  | FastAPI, LangChain + LangGraph (`gpt-4o-mini`), SQLAlchemy + asyncpg    |
| Platform | Supabase (auth + Postgres), PyJWT/JWKS verification, slowapi rate limit |

## Getting Started

### Prerequisites

- Node.js 18+, Python 3.11+
- A [Supabase](https://supabase.com) project (auth + Postgres)
- An OpenAI API key
- Supabase CLI (for database migrations)

### 1. Database

```bash
cd backend
supabase link --project-ref <your-project-ref>
supabase db push
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # then fill in the values below
python -m uvicorn app.main:app --reload   # http://localhost:8000
```

`backend/.env`:

| Variable           | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `OPENAI_API_KEY`   | OpenAI key used by the LangChain graphs                      |
| `SUPABASE_URL`     | Supabase project URL (used to fetch JWKS for JWT validation) |
| `DB_URL`           | Postgres connection string (asyncpg)                         |
| `DB_SSL_ROOT_CERT` | Optional: custom CA bundle for DB TLS (defaults to certifi)  |

### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env   # fill in VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY
npm run dev            # http://localhost:5173
```

The Vite dev server proxies `/api/*` to `http://localhost:8000`.

## API

All endpoints (except `/health`) require `Authorization: Bearer <supabase-jwt>`
and the AI endpoints are rate-limited to 5 requests/minute.

| Method   | Endpoint                  | Description                                  |
|----------|---------------------------|----------------------------------------------|
| `POST`   | `/api/analyze`            | Match score, feedback, suggestions, keywords |
| `POST`   | `/api/improve`            | Rewritten resume + changes + keywords        |
| `POST`   | `/api/coverletter`        | Generated cover letter                       |
| `POST`   | `/api/resumes`            | Upload a PDF resume (text extracted, saved)  |
| `GET`    | `/api/resumes`            | List the user's saved resumes                |
| `DELETE` | `/api/resumes/{id}`       | Delete a saved resume                        |

## Testing

```bash
cd backend
pytest        # routes, auth, PDF parsing, input cleaning
```

## Project Structure

```
frontend/src/
  pages/        # AnalyzeResume, ImproveResume, CoverLetter, MyResume, AuthPage
  components/   # AppLayout, ProtectedRoute, ResumeSelect, ScoreChart, ui/
  context/      # AuthContext (Supabase session + auth headers)
  hooks/        # use-resumes
backend/app/
  routes/       # HTTP layer (ats.py, resumes.py)
  services/     # business logic
  graphs/       # LangGraph AI workflows
  prompts/      # prompt templates
  models/       # Pydantic schemas + SQLAlchemy models
  db/           # async engine/session
  auth.py       # Supabase JWT verification
backend/supabase/
  migrations/   # SQL migrations (apply with `supabase db push`)
```
