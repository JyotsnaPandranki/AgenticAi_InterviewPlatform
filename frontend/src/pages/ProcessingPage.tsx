import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppState } from '../context/AppContext';
import { startInterview, startSession, uploadResume } from '../services/api';

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

export default function ProcessingPage() {
  const nav = useNavigate();
  const startedRef = useRef(false);
  const {
    state,
    setResumeId,
    setRoles,
    setSessions,
    setIsNewResume,
    setSelectedSessionId,
    setInterviewId,
    setCurrentQuestion,
    setCurrentRound,
    setTotalQuestions,
    setProcessingLogs,
    appendProcessingLog,
    setProcessingJob,
  } = useAppState();

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    (async () => {
      if (state.processingJob === 'resume_upload') {
        if (!state.resumeFile) {
          nav('/upload');
          return;
        }

        setProcessingLogs([]);
        appendProcessingLog('Loading job index and embedding model...');
        await wait(250);
        appendProcessingLog('Extracting resume text and understanding candidate profile...');
        await wait(250);
        appendProcessingLog('Generating top role recommendations...');
        const data = await uploadResume(state.resumeFile, 5);
        setResumeId(data.resume_id);
        setIsNewResume(Boolean(data.is_new_resume));
        setRoles(data.recommended_jobs || []);
        setSessions(data.sessions || []);
        appendProcessingLog('Resume analysis complete.');
        setProcessingJob(null);
        if (!Boolean(data.is_new_resume) && (data.sessions || []).length > 0) {
          appendProcessingLog('Matching resume sessions found. Redirecting to session selection...');
          nav('/session-select');
        } else {
          nav('/roles');
        }
        return;
      }

      if (state.processingJob === 'interview_init') {
        if (!state.resumeId || !state.selectedRole) {
          nav('/roles');
          return;
        }

        setProcessingLogs([]);
        appendProcessingLog(`Starting adaptive interview for role: ${state.selectedRole.title}`);
        appendProcessingLog('Initializing retriever, orchestrator, and adaptive memory...');

        const existingAvailable = (state.sessions || []).filter((s: any) => Number(s.total_interviews || 0) < 3);
        let selectedSessionId = state.selectedSessionId || null;

        if (state.sessionMode === 'continue' && selectedSessionId) {
          appendProcessingLog(`Continuing session #${selectedSessionId}...`);
        } else {
          if (!state.isNewResume && existingAvailable.length > 0) {
            appendProcessingLog('Resume already exists. Starting a new session...');
          } else if (!state.isNewResume && existingAvailable.length === 0) {
            appendProcessingLog('All existing sessions reached max interviews (3). Creating a new session...');
          } else {
            appendProcessingLog('Creating interview session...');
          }
          const sess = await startSession({
            resume_id: state.resumeId,
            mode: 'new',
          });
          selectedSessionId = sess.session.session_id;
          setSelectedSessionId(selectedSessionId);
        }

        const started = await startInterview({
          session_id: Number(selectedSessionId),
          selected_role: state.selectedRole.title,
          questions: Number(state.questionCount || state.totalQuestions || 5),
          coaching_mode: Boolean(state.coachingMode),
        });

        setInterviewId(started.interview_id);
        setTotalQuestions(Number(started.total_questions || (Number(state.questionCount || state.totalQuestions || 5) + 1)));
        setCurrentQuestion(started.question || '');
        setCurrentRound(Number(started.round || 1));
        appendProcessingLog('Interview ready.');
        setProcessingJob(null);
        nav('/live');
        return;
      }

      nav('/upload');
    })().catch((e: any) => {
      const detail = e?.response?.data?.detail || e?.message || 'Unknown error';
      appendProcessingLog(`Unexpected processing error: ${String(detail)}`);
      setTimeout(() => nav('/upload'), 1200);
    });
  }, []);

  return (
    <section className="mx-auto max-w-6xl py-14 text-center">
      <div className="mx-auto h-36 w-36 animate-spin rounded-full border border-primary/25 border-t-primary" />
      <h2 className="mt-10 text-5xl font-semibold tracking-[-0.02em]">Synchronizing Intelligent Assessment</h2>
      <p className="mx-auto mt-3 max-w-3xl text-xl text-white/60">
        Refining evaluation metrics and preparing the digital environment for a human-centric interview experience.
      </p>

      <div className="mx-auto mt-12 max-w-4xl rounded-2xl border border-primary/30 bg-[#0b121f]/50 p-5 text-left">
        <div className="text-xs uppercase tracking-[0.2em] text-primary">Runtime Logs</div>
        <div className="mt-4 space-y-2 text-sm text-white/75">
          {state.processingLogs.map((line: string, idx: number) => (
            <div key={`${line}-${idx}`}>• {line}</div>
          ))}
        </div>
      </div>
    </section>
  );
}
