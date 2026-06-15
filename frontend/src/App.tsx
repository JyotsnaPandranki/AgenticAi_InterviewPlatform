import { useMemo, useState } from "react";
import { Navbar } from "./components/Navbar";
import { Sidebar } from "./components/Sidebar";
import { UploadPage } from "./pages/UploadPage";
import { RecommendedRolesPage } from "./pages/RecommendedRolesPage";
import { LiveInterviewPage } from "./pages/LiveInterviewPage";
import { InterviewSummaryPage } from "./pages/InterviewSummaryPage";
import { useUploadFlow } from "./hooks/useUploadFlow";
import { useLiveInterviewState } from "./hooks/useLiveInterviewState";
import { selectSession, startInterview, submitAnswer, uploadResumeAndRecommend } from "./api/interviewApi";
import type { InterviewSummary, RecommendedRole } from "./types/interview";

const SIDEBAR_STEPS = [
  { key: "upload", label: "Upload Resume" },
  { key: "roles", label: "Recommended Roles" },
  { key: "live", label: "Live Interview" },
  { key: "summary", label: "Interview Summary" },
];

export default function App() {
  const { state: uploadState, setFile, startUpload, updateProgress, startAnalyzing, fail } = useUploadFlow();
  const live = useLiveInterviewState(90);

  const [screen, setScreen] = useState<"upload" | "roles" | "live" | "summary">("upload");
  const [roles, setRoles] = useState<RecommendedRole[]>([]);
  const [selectedRole, setSelectedRole] = useState<RecommendedRole | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<string>("Tell me about a project where you handled model overfitting.");
  const [summary, setSummary] = useState<InterviewSummary | null>(null);
  const [resumeId, setResumeId] = useState<number | null>(null);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [interviewId, setInterviewId] = useState<string | null>(null);

  const sidebarItems = useMemo(
    () => SIDEBAR_STEPS.map((s) => ({ ...s, active: s.key === screen })),
    [screen],
  );

  async function handleAnalyzeResume() {
    if (!uploadState.file) return;

    try {
      startUpload();
      updateProgress(45);

      const rec = await uploadResumeAndRecommend(uploadState.file, 5);

      startAnalyzing();
      setRoles(rec.recommended_jobs || []);
      setResumeId((rec as any).resume_id ?? null);
      setScreen("roles");
    } catch (e) {
      fail(e instanceof Error ? e.message : "Failed to analyze resume.");
    }
  }

  async function handleStartInterview() {
    if (!selectedRole || resumeId == null) return;
    const s = await selectSession({ resume_id: resumeId, mode: "new" });
    setSessionId(s.session.session_id);

    const started = await startInterview({
      session_id: s.session.session_id,
      selected_role: selectedRole.title,
      questions: 5,
      coaching_mode: true,
    });
    setInterviewId(started.interview_id);
    setCurrentQuestion(started.question);
    setScreen("live");
  }

  async function handleSubmitTranscript() {
    if (!interviewId) return;
    const res = await submitAnswer({
      interview_id: interviewId,
      answerText: live.state.transcript || "",
      transcript: live.state.transcript || "",
    });
    if (res.done && res.summary) {
      setSummary(res.summary);
      setScreen("summary");
      return;
    }
    if (!res.done && res.question) {
      setCurrentQuestion(res.question);
      live.reset();
    }
  }

  return (
    <div className="app-shell">
      <Navbar rightSlot={<button className="primary-btn small">Start Interview</button>} />

      <div className="content-shell">
        <Sidebar items={sidebarItems} />

        <main className="main-panel">
          {screen === "upload" && (
            <UploadPage state={uploadState} onPickFile={setFile} onAnalyze={handleAnalyzeResume} />
          )}

          {screen === "roles" && (
            <RecommendedRolesPage
              roles={roles}
              selectedRole={selectedRole}
              onSelectRole={setSelectedRole}
              onStartInterview={handleStartInterview}
            />
          )}

          {screen === "live" && (
            <LiveInterviewPage
              question={currentQuestion}
              interviewState={live.state}
              timerText={live.mmss}
              onStartListening={live.start}
              onStopListening={live.stop}
              onSubmitTranscript={handleSubmitTranscript}
            />
          )}

          {screen === "summary" && <InterviewSummaryPage summary={summary} />}
        </main>
      </div>
    </div>
  );
}
