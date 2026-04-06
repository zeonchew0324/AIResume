import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, Upload, FileText, RotateCcw } from "lucide-react";

type AppState = "input" | "loading" | "results";

export default function ImproveResume() {
  const [state, setState] = useState<AppState>("input");
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [jobTitle, setJobTitle] = useState("");

  const [error, setError] = useState<string | null>(null);
  const [improvedResume, setImprovedResume] = useState<string>("");
  const [changes, setChanges] = useState<
    Array<{ section: string; change: string }>
  >([]);
  const [keywordsAdded, setKeywordsAdded] = useState<string[]>([]);
  const [extraInfo, setExtraInfo] = useState("");
  const [copied, setCopied] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const canSubmit = resumeFile && jobDescription.trim() && jobTitle.trim();

  const handleImprove = async () => {
    if (!canSubmit) return;
    setState("loading");
    setError(null);

    const formData = new FormData();
    formData.append("resume", resumeFile!);
    formData.append("job_title", jobTitle);
    formData.append("job_description", jobDescription);
    formData.append("extra_info", extraInfo);

    try {
      const res = await fetch("http://localhost:8000/api/improve", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const errorDetail = body.detail ?? "";
        if (errorDetail === "Resume is empty or unreadable") {
          throw new Error(
            "We couldn't read your PDF. Make sure it contains selectable text, not a scanned image.",
          );
        }
        throw new Error("Action failed. Please try again.");
      }

      const data = await res.json();
      console.log("API Response:", data);
      setImprovedResume(data.improved_resume);
      setChanges(data.changes);
      setKeywordsAdded(data.keywords_added);
      setState("results");
    } catch (err) {
      console.error(err);
      const message =
        err instanceof Error
          ? err.message
          : "Something went wrong. Please try again.";
      setError(message);
      setState("input");
    }
  };

  const reset = () => {
    setState("input");
    setResumeFile(null);
    setJobDescription("");
    setJobTitle("");
    setImprovedResume("");
    setChanges([]);
    setKeywordsAdded([]);
    setExtraInfo("");
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="min-h-screen text-left">
      <div className="max-w-2xl mx-auto px-4 py-10">
        <header className="mb-8">
          <h1 className="text-xl font-semibold tracking-tight text-foreground">
            Improve Resume
          </h1>
          <p className="text-muted-foreground mt-1 text-sm">
            Upload your resume and get AI-powered improvements.
          </p>
        </header>

        {state === "results" ? (
          // Resume Analysis Results
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-600">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Improvement Results</h2>
              <Button variant="outline" size="sm" onClick={reset}>
                <RotateCcw className="size-4" />
                Improve Again
              </Button>
            </div>

            <Card className="bg-white/5 backdrop-blur-md border-white/10">
              <CardContent>
                <Tabs defaultValue="improved-resume">
                  <TabsList className="flex-wrap h-auto gap-1 mb-4">
                    <TabsTrigger value="improved-resume">
                      Improved Resume
                    </TabsTrigger>
                    <TabsTrigger value="changes-made">Changes Made</TabsTrigger>
                  </TabsList>

                  <TabsContent value="improved-resume">
                    <div className="flex justify-end mb-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          navigator.clipboard.writeText(improvedResume);
                          setCopied(true);
                          setTimeout(() => setCopied(false), 2000);
                        }}
                      >
                        {copied ? "Copied!" : "Copy"}
                      </Button>
                    </div>
                    <div className="whitespace-pre-wrap text-sm text-foreground max-h-[500px] overflow-y-auto">
                      {improvedResume}
                    </div>
                  </TabsContent>

                  <TabsContent value="changes-made">
                    <div className="space-y-3">
                      {changes.map((change, idx) => (
                        <div key={idx} className="p-3 bg-muted rounded-md text-sm">
                          <span className="font-medium">{change.section}</span>
                          <p className="text-muted-foreground mt-0.5">{change.change}</p>
                        </div>
                      ))}
                    </div>
                    {keywordsAdded.length > 0 && (
                      <div className="mt-4">
                        <p className="text-sm font-medium mb-2">Keywords Added</p>
                        <div className="flex flex-wrap gap-2">
                          {keywordsAdded.map((kw, i) => (
                            <span key={i} className="px-3 py-1 bg-primary/10 text-primary text-sm rounded-full">
                              {kw}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>
        ) : (
          // Input Form
          <Card className="bg-white/5 backdrop-blur-md border-white/10 animate-in fade-in duration-200">
            <CardContent>
              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    Resume (PDF)
                  </label>
                  <div
                    className="border-2 border-dashed border-border rounded-lg p-6 text-center cursor-pointer hover:border-primary/40 transition-colors"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf"
                      className="hidden"
                      onChange={(e) =>
                        setResumeFile(e.target.files?.[0] ?? null)
                      }
                    />
                    {resumeFile ? (
                      <div className="flex items-center justify-center gap-2 text-sm text-foreground">
                        <FileText className="size-4" />
                        {resumeFile.name}
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-1 text-muted-foreground">
                        <Upload className="size-5" />
                        <span className="text-sm">Click to upload PDF</span>
                      </div>
                    )}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    Job Title
                  </label>
                  <input
                    className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                    placeholder="e.g. Backend Engineer, Data Science"
                    value={jobTitle}
                    onChange={(e) => setJobTitle(e.target.value)}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    Job Description
                  </label>
                  <Textarea
                    placeholder="Paste the full job description here..."
                    rows={6}
                    value={jobDescription}
                    onChange={(e) => setJobDescription(e.target.value)}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    Additional Information <span className="text-muted-foreground">(optional)</span>
                  </label>
                  <Textarea
                    placeholder="e.g. hackathons, side projects, certifications, open source contributions..."
                    rows={3}
                    value={extraInfo}
                    onChange={(e) => setExtraInfo(e.target.value)}
                  />
                </div>

                <Button
                  className="w-full"
                  size="lg"
                  disabled={!canSubmit || state === "loading"}
                  onClick={handleImprove}
                >
                  {state === "loading" ? (
                    <>
                      <Loader2 className="size-4 animate-spin" />
                      Improving...
                    </>
                  ) : (
                    "Improve Resume"
                  )}
                </Button>
                {error && (
                  <p className="text-sm text-destructive text-center">
                    {error}
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
