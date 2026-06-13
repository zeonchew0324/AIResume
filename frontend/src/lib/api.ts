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
