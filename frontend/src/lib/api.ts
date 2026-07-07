import { supabase } from "@/lib/supabase";

/**
 * Returns the Authorization header carrying the current Supabase access token.
 * Attach this to every request to a protected backend endpoint.
 */
export async function authHeaders(): Promise<HeadersInit> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

const FRIENDLY_MESSAGES: Record<string, string> = {
  "Resume is empty or unreadable":
    "We couldn't read your PDF. Make sure it contains selectable text, not a scanned image.",
};

/**
 * POST a FormData payload to a protected backend endpoint and return the
 * parsed JSON response. Failures throw an Error whose message is safe to
 * show the user: backend `detail` strings pass through (they are written
 * for end users), rate-limit and timeout cases get dedicated messages.
 *
 * The default timeout must exceed the backend's worst case — analyze runs
 * two sequential LLM calls of up to 30s each before its own 504.
 */
export async function postForm<T>(
  path: string,
  form: FormData,
  {
    timeoutMs = 65_000,
    fallbackError = "Something went wrong. Please try again.",
  }: { timeoutMs?: number; fallbackError?: string } = {},
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: await authHeaders(),
      body: form,
      signal: controller.signal,
    });
    if (!res.ok) {
      const body: unknown = await res.json().catch(() => ({}));
      const raw = typeof body === "object" && body !== null
        ? (body as Record<string, unknown>)
        : {};
      // FastAPI errors carry a string `detail`; slowapi 429s carry `error`.
      // 422 validation errors put an object list in `detail` — not showable.
      const detail = [raw.detail, raw.error].find(
        (v): v is string => typeof v === "string",
      );
      throw new Error((detail && (FRIENDLY_MESSAGES[detail] ?? detail)) || fallbackError);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("The request timed out. Please try again.");
    }
    if (err instanceof TypeError) {
      throw new Error("Couldn't reach the server. Check your connection and try again.");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/** Turn any thrown value into a user-facing message. */
export function errorMessage(err: unknown): string {
  return err instanceof Error && err.message
    ? err.message
    : "Something went wrong. Please try again.";
}
