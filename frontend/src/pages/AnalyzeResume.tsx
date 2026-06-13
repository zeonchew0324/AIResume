import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, Upload, FileText, RotateCcw } from "lucide-react";
import { ScoreChart } from "@/components/ScoreChart";
import { Progress } from "@/components/ui/progress";
import { authHeaders } from "@/lib/api";

type AppState = "input" | "loading" | "results";
type ScoreBreakdown = {
  category: string;
  score: number;
  reason: string;
};

export default function AnalyzeResume() {
  const [state, setState] = useState<AppState>("input");
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [jobTitle, setJobTitle] = useState("");

  const [error, setError] = useState<string | null>(null);
  const [matchScore, setMatchScore] = useState<number>(0);
  const [feedback, setFeedback] = useState<string>("");
  const [suggestions, setSuggestions] = useState<
    Array<{ focus_area: string; advice: string }>
  >([]);
  const [missingKeywords, setMissingKeywords] = useState<string[]>([]);
  const [scoreBreakdown, setScoreBreakdown] = useState<ScoreBreakdown[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const canSubmit = resumeFile && jobDescription.trim() && jobTitle.trim();

  const handleAnalyze = async () => {
    if (!canSubmit) return;
    setState("loading");
    setError(null);

    const formData = new FormData();
    formData.append("resume", resumeFile!);
    formData.append("job_title", jobTitle);
    formData.append("job_description", jobDescription);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000); // 30 seconds timeout

    try {
      const res = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: await authHeaders(),
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const errorDetail = body.detail ?? "";
        if (errorDetail === "Resume is empty or unreadable") {
          throw new Error(
            "We couldn't read your PDF. Make sure it contains selectable text, not a scanned image.",
          );
        }
        throw new Error("Analysis failed. Please try again.");
      }

      const data = await res.json();
      console.log("API Response:", data);
      setMatchScore(data.match_score);
      setFeedback(data.feedback);
      setSuggestions(data.suggestions);
      setMissingKeywords(data.missing_keywords);
      setScoreBreakdown(data.score_breakdown);
      setState("results");
    } catch (err) {
      // Handle Request Timeout Error
      console.error(err);
      if (err instanceof Error && err.name === "AbortError") {
        setError("Request Timed Out. Please Try Again.");
        return;
      }

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
    setMatchScore(0);
    setFeedback("");
    setSuggestions([]);
    setMissingKeywords([]);
    setScoreBreakdown([]);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="min-h-screen text-left">
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

        {state === "results" ? (
          // Resume Analysis Results
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-600">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Analysis Results</h2>
              <Button variant="outline" size="sm" onClick={reset}>
                <RotateCcw className="size-4" />
                Analyze Again
              </Button>
            </div>

            <Card className="bg-white/5 backdrop-blur-md border-white/10">
              <CardContent>
                <Tabs defaultValue="score">
                  <TabsList className="flex-wrap h-auto gap-1 mb-4">
                    <TabsTrigger value="score">Score</TabsTrigger>
                    <TabsTrigger value="feedback">Feedback</TabsTrigger>
                    <TabsTrigger value="suggestions">Suggestions</TabsTrigger>
                    <TabsTrigger value="keywords">Missing Keywords</TabsTrigger>
                  </TabsList>

                  <TabsContent value="score" className="min-h-[300px]">
                    <div className="flex flex-col items-center py-4 gap-2">
                      <ScoreChart score={matchScore} />
                      <p className="text-muted-foreground text-sm">
                        ATS Match Score
                      </p>
                    </div>
                    {scoreBreakdown.length > 0 && (
                      <div className="mt-4 space-y-4">
                        {scoreBreakdown.map((item) => (
                          <div key={item.category} className="space-y-1">
                            <div className="flex justify-between text-sm">
                              <span className="font-medium">
                                {item.category}
                              </span>
                              <span className="text-muted-foreground">
                                {item.score}/100
                              </span>
                            </div>
                            <Progress value={item.score} />
                            <p className="text-xs text-muted-foreground">
                              {item.reason}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="feedback" className="min-h-[300px]">
                    <div className="text-sm leading-relaxed">{feedback}</div>
                  </TabsContent>

                  <TabsContent value="suggestions" className="min-h-[300px]">
                    <div className="space-y-3">
                      {suggestions.map((item, i) => (
                        <div
                          key={i}
                          className="text-sm leading-relaxed p-3 bg-muted rounded-md"
                        >
                          <div className="font-medium mb-1">
                            {item.focus_area}
                          </div>
                          <div className="text-muted-foreground">
                            {item.advice}
                          </div>
                        </div>
                      ))}
                    </div>
                  </TabsContent>

                  <TabsContent value="keywords" className="min-h-[300px]">
                    <div className="flex flex-wrap gap-2">
                      {missingKeywords.map((keyword, i) => (
                        <span
                          key={i}
                          className="px-3 py-1 bg-destructive/10 text-destructive text-sm rounded-full"
                        >
                          {keyword}
                        </span>
                      ))}
                    </div>
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
