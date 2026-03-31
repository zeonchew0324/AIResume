import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, Upload, FileText, RotateCcw } from "lucide-react";

type AppState = "input" | "loading" | "results";

interface Section {
  title: string;
  content: string;
}

function parseSections(text: string): Section[] {
  const parts = text.split(/\n(?=##\s)/);
  return parts
    .filter((p) => p.trim().startsWith("##"))
    .map((part) => {
      const firstNewline = part.indexOf("\n");
      const title = part.slice(2, firstNewline).trim();
      const content = part.slice(firstNewline + 1).trim();
      return { title, content };
    });
}

export default function AnalyzeResume() {
  const [state, setState] = useState<AppState>("input");
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [sections, setSections] = useState<Section[]>([]);
  const [rawOutput, setRawOutput] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canSubmit = resumeFile && jobDescription.trim() && jobTitle.trim();

  const handleAnalyze = async () => {
    if (!canSubmit) return;
    setState("loading");

    const formData = new FormData();
    formData.append("resume", resumeFile!);
    formData.append("job_title", jobTitle);
    formData.append("job_description", jobDescription);

    try {
      const res = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      const output = data.output as string;
      setRawOutput(output);
      setSections(parseSections(output));
      setState("results");
    } catch (err) {
      console.error(err);
      setState("input");
    }
  };

  const reset = () => {
    setState("input");
    setResumeFile(null);
    setJobDescription("");
    setJobTitle("");
    setSections([]);
    setRawOutput("");
  };

  return (
    <div className="min-h-screen bg-background text-left">
      <div className="max-w-2xl mx-auto px-4 py-10">
        <header className="mb-8">
          <h1 className="text-xl font-semibold tracking-tight text-foreground">
            Analyze Resume
          </h1>
          <p className="text-muted-foreground mt-1 text-sm">
            Upload your resume and paste a job description to get AI-powered
            feedback.
          </p>
        </header>

        <Card>
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
                    onChange={(e) => setResumeFile(e.target.files?.[0] ?? null)}
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

              <Button
                className="w-full"
                size="lg"
                disabled={!canSubmit || state === "loading"}
                onClick={handleAnalyze}
              >
                {state === "loading" ? (
                  <>
                    <Loader2 className="size-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  "Analyze Resume"
                )}
              </Button>
            </div>

            {state === "results" && sections.length > 0 && (
              <div className="mt-10">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">Feedback</h2>
                  <Button variant="outline" size="sm" onClick={reset}>
                    <RotateCcw className="size-4" />
                    Analyze Again
                  </Button>
                </div>
                <Tabs defaultValue={sections[0].title}>
                  <TabsList className="flex-wrap h-auto gap-1 mb-4">
                    {sections.map((s) => (
                      <TabsTrigger key={s.title} value={s.title}>
                        {s.title}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                  {sections.map((s) => (
                    <TabsContent key={s.title} value={s.title}>
                      <div className="whitespace-pre-wrap text-sm text-foreground leading-relaxed">
                        {s.content}
                      </div>
                    </TabsContent>
                  ))}
                </Tabs>
              </div>
            )}

            {state === "results" && sections.length === 0 && rawOutput && (
              <div className="mt-10">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">Feedback</h2>
                  <Button variant="outline" size="sm" onClick={reset}>
                    <RotateCcw className="size-4" />
                    Analyze Again
                  </Button>
                </div>
                <div className="whitespace-pre-wrap text-sm text-foreground leading-relaxed">
                  {rawOutput}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
