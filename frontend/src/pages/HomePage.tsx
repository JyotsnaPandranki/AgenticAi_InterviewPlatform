import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { deleteSession, interviewRecordSummary, resumeList, sessionInterviews, sessions } from '../services/api';
import { useAppState } from '../context/AppContext';

export default function HomePage() {
  const nav = useNavigate();
  const { setInterviewSummary } = useAppState();
  const [resumes, setResumes] = useState<any[]>([]);
  const [sessionMap, setSessionMap] = useState<Record<number, any[]>>({});
  const [interviewMap, setInterviewMap] = useState<Record<number, any[]>>({});
  const [deletingSessionId, setDeletingSessionId] = useState<number | null>(null);
  const [reviewingInterviewId, setReviewingInterviewId] = useState<number | null>(null);

  async function loadDashboardData() {
    try {
      const res = await resumeList();
      const rows = res.resumes || [];
      setResumes(rows);
      const map: Record<number, any[]> = {};
      await Promise.all(
        rows.slice(0, 6).map(async (r: any) => {
          try {
            const s = await sessions(Number(r.resume_id));
            map[Number(r.resume_id)] = s.sessions || [];
          } catch {
            map[Number(r.resume_id)] = [];
          }
        }),
      );
      setSessionMap(map);

      const iMap: Record<number, any[]> = {};
      const allSessions = Object.values(map).flat();
      await Promise.all(
        allSessions.map(async (s: any) => {
          try {
            const resp = await sessionInterviews(Number(s.session_id));
            iMap[Number(s.session_id)] = resp.interviews || [];
          } catch {
            iMap[Number(s.session_id)] = [];
          }
        }),
      );
      setInterviewMap(iMap);
    } catch {
      setResumes([]);
    }
  }

  useEffect(() => {
    void loadDashboardData();
  }, []);

  async function onDeleteSession(sessionId: number) {
    const ok = window.confirm(`Delete Session #${sessionId}? This will also delete all interviews in that session.`);
    if (!ok) return;
    try {
      setDeletingSessionId(sessionId);
      await deleteSession(sessionId);
      await loadDashboardData();
    } finally {
      setDeletingSessionId(null);
    }
  }

  async function onReviewInterview(interviewId: number) {
    try {
      setReviewingInterviewId(interviewId);
      const res = await interviewRecordSummary(interviewId);
      setInterviewSummary(res.summary as any);
      nav('/summary');
    } finally {
      setReviewingInterviewId(null);
    }
  }

  return (
    <div className="min-h-screen bg-[#06080c] text-white">
      <div className="mx-auto max-w-[1400px] px-7">
        <section className="pt-28 pb-24 text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.02] px-4 py-1.5 text-[11px] uppercase tracking-[0.28em] text-white/70">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            System Online: v2.4.0
          </div>
          <h1 className="mx-auto mt-10 max-w-5xl text-5xl font-semibold leading-[1.08] tracking-[-0.03em] md:text-8xl">
            AI-Powered Adaptive
            <br />
            Interview Platform
          </h1>
          <p className="mx-auto mt-8 max-w-3xl text-xl leading-relaxed text-white/55">
            Upload your resume and experience personalized AI-driven mock interviews
            designed to simulate elite tech environments and maximize your career
            potential.
          </p>
          <Link to="/upload" className="btn-gradient mt-11 inline-block px-14 py-4 text-2xl">
            Start Interview
          </Link>
          <div className="mt-9 text-base text-white/65">Your Interview training For free of cost</div>
        </section>

        <section className="grid grid-cols-1 gap-6 pb-24 lg:grid-cols-12">
          <article className="relative overflow-hidden rounded-3xl border border-white/10 bg-[#0b1018]/70 p-10 lg:col-span-8 min-h-[430px]">
            <div className="absolute inset-0 opacity-35" style={{ background: 'radial-gradient(circle at 40% 45%, rgba(108,162,255,0.28), transparent 46%), radial-gradient(circle at 60% 65%, rgba(91,225,230,0.16), transparent 44%)' }} />
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(8,14,24,0)_0%,rgba(8,14,24,0.88)_73%)]" />
            <div className="relative z-10 mt-52">
              <div className="text-xs uppercase tracking-[0.22em] text-primary">Core Engine</div>
              <h3 className="mt-5 text-6xl font-medium tracking-[-0.02em]">Neural Sentiment Analysis</h3>
              <p className="mt-5 max-w-2xl text-2xl leading-relaxed text-white/60">
                Our AI tracks your micro-expressions and tone in real-time to
                provide detailed feedback on confidence and clarity.
              </p>
            </div>
          </article>

          <div className="grid gap-6 lg:col-span-4">
            <article className="rounded-3xl border border-white/10 bg-[#0b1018]/70 p-8">
              <div className="text-primary text-3xl">▣</div>
              <h4 className="mt-6 text-4xl font-semibold">Resume Parsing</h4>
              <p className="mt-4 text-xl leading-relaxed text-white/55">
                Instant analysis of your experience to tailor specific technical questions.
              </p>
            </article>

            <article className="rounded-3xl border border-white/10 bg-[#0b1018]/70 p-8">
              <div className="text-tertiary text-3xl">✦</div>
              <h4 className="mt-6 text-4xl font-semibold">Performance Metrics</h4>
              <p className="mt-4 text-xl leading-relaxed text-white/55">
                Benchmark your performance against industry standards for elite tech firms.
              </p>
            </article>
          </div>
        </section>

        <section className="pb-24">
          <div className="rounded-3xl border border-white/10 bg-[#0b1018]/70 p-8">
            <div className="flex items-center justify-between">
              <h3 className="text-3xl font-semibold">Recent Resume Sessions</h3>
              <Link to="/upload" className="text-sm text-primary hover:text-primary/80">Start New Interview</Link>
            </div>
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              {resumes.slice(0, 6).map((r: any) => (
                <div key={r.resume_id} className="rounded-2xl border border-white/10 bg-white/[0.02] p-4">
                  <div className="text-lg font-medium">{r.candidate_name || r.original_resume_name}</div>
                  <div className="mt-1 text-xs text-white/50">Resume ID: {r.resume_id}</div>
                  <div className="mt-1 text-xs text-white/50">File: {r.original_resume_name}</div>
                  <div className="mt-3 text-xs uppercase tracking-[0.18em] text-white/55">Sessions</div>
                  <div className="mt-2 space-y-1 text-sm text-white/70">
                    {(sessionMap[Number(r.resume_id)] || []).length === 0 ? (
                      <div>No sessions yet</div>
                    ) : (
                      (sessionMap[Number(r.resume_id)] || []).map((s: any) => (
                        <div key={s.session_id} className="rounded-lg border border-white/10 bg-white/[0.02] px-3 py-2">
                          <div className="flex items-center justify-between gap-3">
                            <div>Session #{s.session_id} · {s.total_interviews}/3 interviews</div>
                            <button
                              className="rounded-md border border-red-400/40 bg-red-500/10 px-2 py-1 text-[11px] uppercase tracking-[0.12em] text-red-200 hover:bg-red-500/20"
                              disabled={deletingSessionId === Number(s.session_id)}
                              onClick={() => onDeleteSession(Number(s.session_id))}
                            >
                              {deletingSessionId === Number(s.session_id) ? 'Deleting...' : 'Delete'}
                            </button>
                          </div>
                          {(interviewMap[Number(s.session_id)] || []).length > 0 ? (
                            <div className="mt-2 space-y-2">
                              {(interviewMap[Number(s.session_id)] || []).map((it: any) => (
                                <div key={it.interview_id} className="flex items-center justify-between rounded-md border border-white/10 bg-black/20 px-2 py-1 text-xs">
                                  <div>
                                    Interview #{it.interview_number} · {it.target_role || 'role'} · Score {Number(it.overall_score || 0).toFixed(1)}
                                  </div>
                                  <button
                                    className="rounded-md border border-cyan-300/40 bg-cyan-300/10 px-2 py-1 uppercase tracking-[0.12em] text-cyan-200 hover:bg-cyan-300/20"
                                    disabled={reviewingInterviewId === Number(it.interview_id)}
                                    onClick={() => onReviewInterview(Number(it.interview_id))}
                                  >
                                    {reviewingInterviewId === Number(it.interview_id) ? 'Loading...' : 'Review'}
                                  </button>
                                </div>
                              ))}
                            </div>
                          ) : null}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="about" className="pb-24">
          <div className="rounded-3xl border border-white/10 bg-[#0b1018]/70 p-8">
            <h3 className="text-3xl font-semibold">About The Project</h3>
            <p className="mt-4 text-lg leading-relaxed text-white/70">
              In 4 days, we built an AI-powered adaptive mock interview platform that takes a resume, generates role-aligned interview tracks,
              runs a real-time voice interview loop with automatic speech detection and silence-based answer completion, and delivers multimodal
              round-wise feedback. The system combines resume parsing, retrieval-guided and generated question orchestration, live transcription,
              communication metrics, CV-based engagement signals, and session memory so each interview can adapt over time instead of repeating
              static question patterns.
            </p>
            <div className="mt-6 text-xs uppercase tracking-[0.2em] text-primary">Roles & Responsibilities</div>
            <div className="mt-3 space-y-2 text-base text-white/75">
              <div>Jyotsna: Complete Frontend(UI/UX) and the involved in the interview questions orchestration</div>
              <div>Harshith: complete Computer Vision pipeline</div>
              <div>Suhas: Complete Audio transcription pipeline</div>
              <div>Pranav: Resume parsing/analyer and interview question generation logic and the feedback and metrics.</div>
            </div>
          </div>
        </section>

        <section className="grid grid-cols-2 gap-y-10 pb-28 text-center md:grid-cols-4">
          <div>
            <div className="text-6xl font-semibold">94%</div>
            <div className="mt-4 text-xs uppercase tracking-[0.25em] text-white/50">Success Rate</div>
          </div>
          <div>
            <div className="text-6xl font-semibold">250k</div>
            <div className="mt-4 text-xs uppercase tracking-[0.25em] text-white/50">Mock Interviews</div>
          </div>
          <div>
            <div className="text-6xl font-semibold">15ms</div>
            <div className="mt-4 text-xs uppercase tracking-[0.25em] text-white/50">Inference Latency</div>
          </div>
          <div>
            <div className="text-6xl font-semibold">180+</div>
            <div className="mt-4 text-xs uppercase tracking-[0.25em] text-white/50">Global Skillsets</div>
          </div>
        </section>
      </div>

      <footer className="border-t border-white/10">
        <div className="mx-auto flex max-w-[1400px] flex-col gap-8 px-7 py-9 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="text-[26px] font-semibold tracking-[-0.02em]">HIA (Humanists Interview Agent)</div>
            <div className="mt-2 text-xs tracking-[0.08em] text-white/55">
              Humanists beleive in AI is only a tool of Human Intelligence.
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-7 text-sm uppercase tracking-[0.18em] text-white/50">
            <span>Terms Of Service</span>
            <span>Privacy Policy</span>
            <span>Status</span>
            <span>Contact</span>
          </div>
        </div>
        <div className="mx-auto max-w-[1400px] px-7 pb-8">
          <div className="inline-block rounded-full border border-cyan-300/40 bg-cyan-300/10 px-4 py-1 text-sm font-semibold tracking-[0.14em] text-cyan-300 shadow-[0_0_16px_rgba(82,255,255,0.35)]">
            IPHIPI HACAKTHON-2026
          </div>
        </div>
      </footer>
    </div>
  );
}
