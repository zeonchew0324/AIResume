import { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import { authHeaders } from "@/lib/api";

export type SavedResume = {
  id: string;
  name: string;
  created_at: string;
};

/**
 * Loads the current user's saved resumes (id + name) for selection in the
 * service pages. Refetches whenever the signed-in user changes.
 */
export function useSavedResumes() {
  const { user, session } = useAuth();
  const [resumes, setResumes] = useState<SavedResume[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    if (!user) {
      setResumes([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    (async () => {
      try {
        const res = await fetch("/api/resumes", { headers: await authHeaders() });
        if (!res.ok) throw new Error(`Failed to load resumes (${res.status})`);
        const data = await res.json();
        if (active) setResumes(data.resumes ?? []);
      } catch (err) {
        console.error("Failed to load resumes:", err);
        if (active) setResumes([]);
      } finally {
        if (active) setLoading(false);
      }
    })();

    return () => {
      active = false;
    };
    // Depend on the access token too, so an early 401 (token still refreshing)
    // is retried once a valid token is available.
  }, [user?.id, session?.access_token]);

  return { resumes, loading };
}
