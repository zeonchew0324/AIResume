import { Link } from "react-router-dom";
import { useSavedResumes } from "@/hooks/use-resumes";

type Props = {
  value: string;
  onChange: (resumeId: string) => void;
};

/**
 * Dropdown for picking one of the user's saved resumes, replacing per-request
 * PDF uploads on the service pages. Handles loading and empty states.
 */
export function ResumeSelect({ value, onChange }: Props) {
  const { resumes, loading } = useSavedResumes();

  return (
    <div>
      <label className="block text-sm font-medium mb-1.5">Resume</label>

      {loading ? (
        <div className="h-9 rounded-md border border-input bg-background px-3 flex items-center text-sm text-muted-foreground">
          Loading your resumes…
        </div>
      ) : resumes.length === 0 ? (
        <div className="rounded-md border border-dashed border-border p-3 text-sm text-muted-foreground">
          No saved resumes yet.{" "}
          <Link
            to="/my-resume"
            className="text-primary underline underline-offset-2"
          >
            Add one
          </Link>{" "}
          to get started.
        </div>
      ) : (
        <select
          className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="" disabled>
            Select a saved resume
          </option>
          {resumes.map((r) => (
            <option key={r.id} value={r.id}>
              {r.name}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
