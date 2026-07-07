import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, RotateCcw, Download, Copy, Check } from "lucide-react";
import { Document, Packer, Paragraph, TextRun } from "docx";
import { saveAs } from "file-saver";
import { ResumeSelect } from "@/components/ResumeSelect";
import { authHeaders } from "@/lib/api";

type AppState = "input" | "loading" | "results";

export default function CoverLetter() {
  const [state, setState] = useState<AppState>("input");
  const [selectedResumeId, setSelectedResumeId] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [companyName, setCompanyName] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [extraInfo, setExtraInfo] = useState("");
  const [coverLetter, setCoverLetter] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const canSubmit =
    selectedResumeId &&
    jobTitle.trim() &&
    companyName.trim() &&
    jobDescription.trim();

  const wordCount = coverLetter
    ? coverLetter.trim().split(/\s+/).filter(Boolean).length
    : 0;

  const handleGenerate = async () => {
    if (!canSubmit) return;
    setState("loading");
    setError(null);

    const formData = new FormData();
    formData.append("resume_id", selectedResumeId);
    formData.append("job_title", jobTitle);
    formData.append("company_name", companyName);
    formData.append("job_description", jobDescription);
    formData.append("extra_info", extraInfo);

    try {
      const res = await fetch("/api/coverletter", {
        method: "POST",
        headers: await authHeaders(),
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
        throw new Error("Generation failed. Please try again.");
      }

      const data = await res.json();
      setCoverLetter(data.cover_letter);
      setState("results");
    } catch (err) {
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
    setSelectedResumeId("");
    setJobTitle("");
    setCompanyName("");
    setJobDescription("");
    setExtraInfo("");
    setCoverLetter("");
    setError(null);
  };

  const downloadDocx = async () => {
    const doc = new Document({
      sections: [
        {
          children: coverLetter
            .split("\n")
            .map((line) => new Paragraph({ children: [new TextRun(line)] })),
        },
      ],
    });
    const blob = await Packer.toBlob(doc);
    saveAs(blob, "cover_letter.docx");
  };

  return (
    <div className="min-h-screen text-left">
      <div className="max-w-3xl mx-auto px-4 py-10">
        <header className="mb-8">
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-xl font-semibold tracking-tight text-foreground">
              Cover Letter
            </h1>
          </div>
          <p className="text-muted-foreground text-sm">
            Generate a tailored, ATS-optimized cover letter for any role.
          </p>
        </header>

        {state === "loading" ? (
          <Card className="bg-white/5 backdrop-blur-md border-white/10">
            <CardContent className="py-16 flex flex-col items-center gap-4">
              <div className="size-12 rounded-full border border-white/10 flex items-center justify-center">
                <Loader2 className="size-5 text-muted-foreground animate-spin" />
              </div>
              <div className="text-center space-y-1">
                <p className="text-sm font-medium">
                  Composing your cover letter...
                </p>
                <p className="text-xs text-muted-foreground">
                  This may take up to 30 seconds
                </p>
              </div>
            </CardContent>
          </Card>
        ) : state === "results" ? (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold">Cover letter ready</h2>
              </div>
              <Button variant="outline" size="sm" onClick={reset}>
                <RotateCcw className="size-3.5" />
                Generate Another
              </Button>
            </div>

            <Card className="bg-white/5 backdrop-blur-md border-white/10">
              <CardContent>
                {/* Metadata strip */}
                <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/10">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                    {jobTitle} · {companyName}
                  </p>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        navigator.clipboard.writeText(coverLetter);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 2000);
                      }}
                    >
                      {copied ? (
                        <Check className="size-3.5" />
                      ) : (
                        <Copy className="size-3.5" />
                      )}
                      {copied ? "Copied" : "Copy"}
                    </Button>
                    <Button variant="outline" size="sm" onClick={downloadDocx}>
                      <Download className="size-3.5" />
                      Download DOCX
                    </Button>
                  </div>
                </div>

                {/* Letter body */}
                <div className="max-h-[600px] overflow-y-auto">
                  <p className="whitespace-pre-wrap text-sm text-foreground/90 leading-relaxed">
                    {coverLetter}
                  </p>
                </div>

                {/* Word count */}
                <p className="text-xs text-muted-foreground mt-4 pt-3 border-t border-white/10 text-right">
                  {wordCount} words
                </p>
              </CardContent>
            </Card>
          </div>
        ) : (
          <Card className="bg-white/5 backdrop-blur-md border-white/10 animate-in fade-in duration-200">
            <CardContent>
              <div className="space-y-5">
                {/* Resume selection */}
                <ResumeSelect
                  value={selectedResumeId}
                  onChange={setSelectedResumeId}
                />

                {/* Job Title + Company Name side by side */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1.5">
                      Job Title
                    </label>
                    <input
                      className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                      placeholder="e.g. Backend Engineer"
                      value={jobTitle}
                      onChange={(e) => setJobTitle(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1.5">
                      Company Name
                    </label>
                    <input
                      className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                      placeholder="e.g. Stripe, Google"
                      value={companyName}
                      onChange={(e) => setCompanyName(e.target.value)}
                    />
                  </div>
                </div>

                {/* Job Description */}
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

                {/* Extra info */}
                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    Additional Information{" "}
                    <span className="text-muted-foreground font-normal text-xs">
                      (optional)
                    </span>
                  </label>
                  <Textarea
                    placeholder="Certifications, side projects, open source contributions..."
                    rows={3}
                    value={extraInfo}
                    onChange={(e) => setExtraInfo(e.target.value)}
                  />
                </div>

                <Button
                  className="w-full"
                  size="lg"
                  disabled={!canSubmit}
                  onClick={handleGenerate}
                >
                  Generate Cover Letter
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
