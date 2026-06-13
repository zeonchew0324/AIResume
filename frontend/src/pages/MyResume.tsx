import { useState, useRef, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import { authHeaders } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
  SheetFooter,
  SheetClose,
} from "@/components/ui/sheet";
import {
  Upload,
  FileText,
  Trash2,
  Loader2,
  BookMarked,
  AlertCircle,
  Plus,
} from "lucide-react";

type UploadState = "idle" | "uploading" | "error";
type ListState = "loading" | "loaded";

type SavedResume = {
  id: string;
  name: string;
  created_at: string;
};

export default function MyResume() {
  const { user, session } = useAuth();
  const [listState, setListState] = useState<ListState>("loading");
  const [resumes, setResumes] = useState<SavedResume[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const [sheetOpen, setSheetOpen] = useState(false);
  const [uploadState, setUploadState] = useState<UploadState>("idle");
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [resumeName, setResumeName] = useState("");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canUpload = resumeFile && resumeName.trim();

  useEffect(() => {
    fetchResumes();
    // Depend on the access token, not just the user id: an early request can
    // 401 while the token is still being refreshed, and that must be retried
    // once a valid token arrives — otherwise the list stays empty until the
    // next manual action.
  }, [user?.id, session?.access_token]);

  const fetchResumes = async () => {
    if (!user) return;
    setListState("loading");
    setLoadError(null);
    try {
      const res = await fetch("/api/resumes", { headers: await authHeaders() });
      if (!res.ok) throw new Error(`Failed to load resumes (${res.status})`);
      const data = await res.json();
      setResumes(data.resumes ?? []);
    } catch (err) {
      console.error("Failed to load resumes:", err);
      setLoadError("Couldn't load your resumes. Please try again.");
    } finally {
      setListState("loaded");
    }
  };

  const resetSheet = () => {
    setResumeFile(null);
    setResumeName("");
    setUploadState("idle");
    setUploadError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleUpload = async () => {
    if (!canUpload) return;
    setUploadState("uploading");
    setUploadError(null);

    const formData = new FormData();
    formData.append("resume", resumeFile!);
    formData.append("name", resumeName.trim());

    try {
      const res = await fetch("/api/resumes", {
        method: "POST",
        headers: await authHeaders(),
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? "Upload failed. Please try again.");
      }

      setSheetOpen(false);
      resetSheet();
      fetchResumes();
    } catch (err) {
      setUploadError(
        err instanceof Error ? err.message : "Upload failed. Please try again.",
      );
      setUploadState("error");
    }
  };

  const handleDelete = async (id: string) => {
    setDeletingId(id);
    try {
      const res = await fetch(`/api/resumes/${id}`, {
        method: "DELETE",
        headers: await authHeaders(),
      });
      if (!res.ok) throw new Error();
      setResumes((prev) => prev.filter((r) => r.id !== id));
    } finally {
      setDeletingId(null);
    }
  };

  const formatDate = (iso: string) =>
    new Date(iso).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });

  return (
    <div className="min-h-screen text-left">
      <div className="max-w-2xl mx-auto px-4 py-10">
        {/* Header */}
        <div className="flex items-start justify-between mb-8">
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-foreground">
              My Resumes
            </h1>
            <p className="text-muted-foreground mt-1 text-sm">
              Save versions of your resume to reuse across tools.
            </p>
          </div>
          <Button
            size="sm"
            className="gap-1.5 shrink-0"
            onClick={() => {
              resetSheet();
              setSheetOpen(true);
            }}
          >
            <Plus className="size-3.5" />
            Add Resume
          </Button>
        </div>

        {/* List */}
        {listState === "loading" && (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-16 w-full rounded-lg" />
            ))}
          </div>
        )}

        {listState === "loaded" && resumes.length === 0 && loadError && (
          <div className="flex flex-col items-center gap-3 py-20 text-muted-foreground">
            <AlertCircle className="size-10 opacity-20" />
            <div className="text-center">
              <p className="text-sm font-medium">{loadError}</p>
              <button
                onClick={fetchResumes}
                className="text-xs mt-1 text-primary underline underline-offset-2"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {listState === "loaded" && resumes.length === 0 && !loadError && (
          <div className="flex flex-col items-center gap-3 py-20 text-muted-foreground">
            <BookMarked className="size-10 opacity-20" />
            <div className="text-center">
              <p className="text-sm font-medium">No resumes saved yet</p>
              <p className="text-xs mt-1 opacity-60">
                Click "Add Resume" to save your first one.
              </p>
            </div>
          </div>
        )}

        {listState === "loaded" && resumes.length > 0 && (
          <div className="space-y-2 animate-in fade-in duration-300">
            <div className="flex items-center justify-between mb-4">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Saved
              </p>
              <Badge variant="outline" className="text-xs">
                {resumes.length}
              </Badge>
            </div>
            {resumes.map((r) => (
              <div
                key={r.id}
                className="flex items-center justify-between gap-4 px-4 py-3.5 rounded-lg bg-white/5 border border-white/10 group hover:bg-white/[0.07] transition-colors"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="size-8 rounded-md bg-white/8 flex items-center justify-center shrink-0">
                    <FileText className="size-4 text-muted-foreground" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-medium truncate">{r.name}</p>
                    <p className="text-xs text-muted-foreground truncate">
                      {formatDate(r.created_at)}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100 transition-all shrink-0"
                  disabled={deletingId === r.id}
                  onClick={() => handleDelete(r.id)}
                >
                  {deletingId === r.id ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Trash2 className="size-4" />
                  )}
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add Resume Sheet */}
      <Sheet
        open={sheetOpen}
        onOpenChange={(open) => {
          setSheetOpen(open);
          if (!open) resetSheet();
        }}
      >
        <SheetContent
          side="bottom"
          className="rounded-t-2xl max-w-lg mx-auto left-0 right-0 px-6 pb-8"
        >
          <SheetHeader className="px-0 pb-2">
            <SheetTitle>Add a resume</SheetTitle>
            <SheetDescription>
              Upload a PDF and give it a name to save it.
            </SheetDescription>
          </SheetHeader>

          <div className="space-y-4 py-2">
            {/* Drop zone */}
            <div
              className="border-2 border-dashed border-border rounded-xl p-8 text-center cursor-pointer hover:border-primary/40 hover:bg-white/[0.03] transition-all"
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={(e) => {
                  setResumeFile(e.target.files?.[0] ?? null);
                  setUploadState("idle");
                  setUploadError(null);
                }}
              />
              {resumeFile ? (
                <div className="flex items-center justify-center gap-2 text-sm text-foreground">
                  <FileText className="size-4 text-primary" />
                  <span className="font-medium">{resumeFile.name}</span>
                  <span className="text-muted-foreground text-xs">
                    ({(resumeFile.size / 1024).toFixed(0)} KB)
                  </span>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2 text-muted-foreground">
                  <div className="size-10 rounded-full bg-white/5 flex items-center justify-center">
                    <Upload className="size-4" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">Click to upload PDF</p>
                    <p className="text-xs mt-0.5 opacity-60">PDF up to 10MB</p>
                  </div>
                </div>
              )}
            </div>

            {/* Name input */}
            <div>
              <label className="block text-sm font-medium mb-1.5">
                Resume name
              </label>
              <input
                className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                placeholder="e.g. Software Engineer – v2, Google Application"
                value={resumeName}
                onChange={(e) => setResumeName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleUpload()}
              />
            </div>

            {uploadError && (
              <div className="flex items-start gap-2 text-sm text-destructive">
                <AlertCircle className="size-4 mt-0.5 shrink-0" />
                {uploadError}
              </div>
            )}
          </div>

          <SheetFooter className="px-0 pt-2 flex-row gap-2">
            <SheetClose asChild>
              <Button variant="outline" className="flex-1">
                Cancel
              </Button>
            </SheetClose>
            <Button
              className="flex-1"
              disabled={!canUpload || uploadState === "uploading"}
              onClick={handleUpload}
            >
              {uploadState === "uploading" ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  Saving...
                </>
              ) : (
                "Save Resume"
              )}
            </Button>
          </SheetFooter>
        </SheetContent>
      </Sheet>
    </div>
  );
}
