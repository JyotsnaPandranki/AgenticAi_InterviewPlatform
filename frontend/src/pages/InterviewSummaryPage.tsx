import { useNavigate } from 'react-router-dom';
import { useAppState } from '../context/AppContext';

export default function InterviewSummaryPage() {
  const nav = useNavigate();
  const { state, setInterviewSummary, setProcessingJob, setProcessingLogs } = useAppState();
  const s = state.interviewSummary;
  if (!s) return <div className="glass-card p-8">No summary yet.</div>;

  return (
    <section className="mx-auto max-w-[1450px] py-8">
      <div className="grid gap-8 lg:grid-cols-12">
        <aside className="rounded-2xl border border-white/10 bg-[#090d15]/70 p-6 lg:col-span-2">
          <div className="rounded-xl bg-primary/90 px-4 py-3 font-medium text-black">+ New Assessment</div>
          <div className="mt-4 rounded-xl border border-white/10 bg-white/5 px-4 py-3">Interviews</div>
        </aside>

        <main className="space-y-6 lg:col-span-10">
          <div className="rounded-2xl border border-white/10 bg-[#0b111b]/70 p-10">
            <div className="text-xs uppercase tracking-[0.2em] text-tertiary">Candidate Analysis</div>
            <div className="mt-3 flex items-start justify-between gap-5">
              <div>
                <h2 className="text-6xl font-semibold tracking-[-0.03em]">Interview Summary</h2>
                <p className="mt-4 max-w-4xl text-xl text-white/60">
                  Role: {s.selected_role} · Completed {s.completed_questions}/{s.total_questions}
                </p>
              </div>
              <div className="grid h-40 w-40 place-items-center rounded-full border-4 border-primary/70 text-center">
                <div>
                  <div className="text-5xl font-semibold">{Math.round(Number(s.average_score || 0) * 10)}</div>
                  <div className="text-xs uppercase tracking-[0.2em] text-white/55">Overall</div>
                </div>
              </div>
            </div>
            <div className="mt-7 grid gap-4 md:grid-cols-4">
              <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">Avg Score: {s.average_score}</div>
              <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">Final Difficulty: {s.final_difficulty}</div>
              <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">Selected Role: {s.selected_role}</div>
              <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">Categories: {s.used_categories.length}</div>
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-[#0b111b]/70 p-8">
            <h3 className="text-4xl font-semibold tracking-[-0.02em]">Analytical Breakdown</h3>
            <div className="mt-6 space-y-4">
              {s.interview_history.map((r) => (
                <details key={r.round} className="rounded-xl border border-white/10 p-5" open={r.round === 1}>
                  <summary className="cursor-pointer list-none">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-lg font-semibold">Round {r.round}: {r.category || 'Interview Question'}</div>
                        <div className="text-sm text-white/55">
                          {r.difficulty} · Score {r.evaluation.score}/10 · Tech {r.evaluation.technical_score} · Comm {r.evaluation.communication_score}
                        </div>
                      </div>
                      <div className="text-white/50">⌄</div>
                    </div>
                  </summary>
                  <div className="mt-4 space-y-3 text-white/75">
                    <div><strong>Question:</strong> {r.asked_question}</div>
                    <div><strong>Answer:</strong> {r.candidate_answer}</div>
                    <div><strong>Feedback (combined):</strong> {r.evaluation.feedback}</div>
                    <div className="rounded-lg border border-white/10 bg-white/[0.02] p-3">
                      <div className="text-xs uppercase tracking-[0.16em] text-primary">Coaching Breakdown</div>
                      <div className="mt-2 text-sm">
                        <strong>Technical:</strong>{' '}
                        {(r.evaluation as any)?.coaching_feedback?.technical_feedback || 'N/A'}
                      </div>
                      <div className="mt-1 text-sm">
                        <strong>Communication:</strong>{' '}
                        {(r.evaluation as any)?.coaching_feedback?.communication_feedback || 'N/A'}
                      </div>
                      <div className="mt-1 text-sm">
                        <strong>Behavioral:</strong>{' '}
                        {(r.evaluation as any)?.coaching_feedback?.behavioral_feedback || 'N/A'}
                      </div>
                    </div>
                  </div>
                </details>
              ))}
            </div>
          </div>

          <div className="flex flex-wrap gap-4 pb-8">
            <button className="btn-gradient px-10 py-3" onClick={() => nav('/')}>Return Home</button>
            <button
              className="rounded-full border border-white/20 px-10 py-3 text-white/90"
              onClick={() => {
                setInterviewSummary(null as any);
                setProcessingLogs([]);
                setProcessingJob('interview_init');
                nav('/processing');
              }}
            >
              Repeat Interview
            </button>
          </div>
        </main>
      </div>
      <div className="glass-card p-8">
        <h3 className="text-xl font-semibold mb-4">Round-by-Round</h3>
        <div className="space-y-4">
          {s.interview_history.map((r) => (
            <div key={r.round} className="rounded-xl border border-white/10 p-4">
              <div className="text-sm text-on-surface-variant">Round {r.round} · {r.category}</div>
              <div className="mt-2"><strong>Q:</strong> {r.asked_question}</div>
              <div className="mt-2"><strong>A:</strong> {r.candidate_answer}</div>
              <div className="mt-2 text-sm">Score: {r.evaluation.score} · Tech: {r.evaluation.technical_score} · Comm: {r.evaluation.communication_score}</div>
              <div className="mt-2 text-sm">
                <strong>Technical feedback:</strong> {(r.evaluation as any)?.coaching_feedback?.technical_feedback || 'N/A'}
              </div>
              <div className="mt-1 text-sm">
                <strong>Communication feedback:</strong> {(r.evaluation as any)?.coaching_feedback?.communication_feedback || 'N/A'}
              </div>
              <div className="mt-1 text-sm">
                <strong>Behavioral feedback:</strong> {(r.evaluation as any)?.coaching_feedback?.behavioral_feedback || 'N/A'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
