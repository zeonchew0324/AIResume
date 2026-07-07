import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import * as Sentry from '@sentry/react'
import './index.css'
import App from './App.tsx'
import { TooltipProvider } from '@/components/ui/tooltip'

// No-op when VITE_SENTRY_DSN is unset (local dev, CI builds).
// Session Replay is deliberately not enabled: users type resume content
// into this app, and screen recordings would capture it.
if (import.meta.env.VITE_SENTRY_DSN) {
  Sentry.init({
    dsn: import.meta.env.VITE_SENTRY_DSN,
    environment: import.meta.env.MODE,
    sendDefaultPii: false,
    tracesSampleRate: 0.1,
  })
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Sentry.ErrorBoundary
      fallback={
        <div style={{ padding: '4rem 1.5rem', textAlign: 'center' }}>
          <h1 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>
            Something went wrong
          </h1>
          <p>Please refresh the page. If it keeps happening, we're on it.</p>
        </div>
      }
    >
      <TooltipProvider>
        <App />
      </TooltipProvider>
    </Sentry.ErrorBoundary>
  </StrictMode>,
)
