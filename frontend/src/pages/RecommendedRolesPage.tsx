import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppState } from '../context/AppContext';

export default function RecommendedRolesPage() {
  const nav = useNavigate();
  const {
    state,
    setSelectedRole,
    setSessionMode,
    setSelectedSessionId,
    setQuestionCount,
    setCoachingMode,
    setProcessingJob,
    setProcessingLogs,
  } = useAppState();

  const availableSessions = useMemo(
    () => (state.sessions || []).filter((s: any) => Number(s.total_interviews || 0) < 3),
    [state.sessions],
  );

  async function onStart(role: any) {
    setSelectedRole(role);
    if (!state.isNewResume && state.sessionMode === 'continue' && state.selectedSessionId) {
      setSessionMode('continue');
      setSelectedSessionId(Number(state.selectedSessionId));
    } else {
      setSessionMode('new');
      setSelectedSessionId(null as any);
    }
    setProcessingLogs([]);
    setProcessingJob('interview_init');
    nav('/processing');
  }

  return (
    <section className="mx-auto max-w-[1400px] px-2 py-8">
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        <aside className="rounded-2xl border border-white/10 bg-[#090d15]/70 p-6 lg:col-span-2">
          <div className="text-xs uppercase tracking-[0.2em] text-white/50">Main Menu</div>
          <div className="mt-5 space-y-3">
            <div className="rounded-xl bg-primary/90 px-4 py-3 font-medium text-black">+ New Assessment</div>
            <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-3">Interviews</div>
          </div>
          <div className="mt-8">
            <div className="text-xs uppercase tracking-[0.2em] text-white/50">Session Strategy</div>
            <p className="mt-2 text-sm text-white/60">
              {state.isNewResume
                ? 'New resume detected. A fresh session will be created.'
                : state.sessionMode === 'continue' && state.selectedSessionId
                  ? `Resume exists. Continuing session #${state.selectedSessionId}.`
                  : availableSessions.length > 0
                    ? 'Resume exists. You can continue a previous session or start a new one.'
                    : 'All prior sessions reached 3 interviews. A new session will be created.'}
            </p>
            <div className="mt-5 space-y-3">
              <label className="block text-xs text-white/70">Questions (number only)</label>
              <input
                type="number"
                min={1}
                max={20}
                value={state.questionCount}
                onChange={(e) => setQuestionCount(Math.max(1, Number(e.target.value || 1)))}
                className="w-full rounded-lg border border-white/10 bg-[#0b111b] px-3 py-2"
              />
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={Boolean(state.coachingMode)}
                  onChange={(e) => setCoachingMode(e.target.checked)}
                />
                Enable Coaching Mode
              </label>
            </div>
          </div>
        </aside>

        <main className="lg:col-span-10">
          <h2 className="text-6xl font-semibold tracking-[-0.03em]">
            Recommended <span className="text-primary">Roles</span>
          </h2>
          <p className="mt-4 max-w-4xl text-2xl text-white/60">
            Our AI has analyzed your skills, experience, and performance patterns to identify elite opportunities matching your profile.
          </p>

          <div className="mt-8 grid gap-6 xl:grid-cols-3">
            {state.roles.map((r: any, i: number) => (
              <article key={`${r.job_id}-${i}`} className="rounded-3xl border border-white/10 bg-[#0b111b]/60 p-7">
                <div className="flex items-start justify-between">
                  <div className="text-xs uppercase tracking-[0.2em] text-primary">Top Match</div>
                  <div className="rounded-full border border-primary/40 px-3 py-1 text-sm text-primary">
                    {Number(r.match_score || 0).toFixed(2)}%
                  </div>
                </div>
                <h3 className="mt-4 text-4xl font-semibold leading-tight">{r.title}</h3>
                <p className="mt-4 text-sm text-white/65">
                  {(r.matched_skills || []).slice(0, 4).join(' • ') || 'Skill alignment verified'}
                </p>
                <p className="mt-5 min-h-24 text-sm text-white/50">{r.why_fit || 'Strong semantic and profile alignment with your resume trajectory.'}</p>
                <button className="btn-gradient mt-8 w-full px-8 py-3 text-sm uppercase tracking-[0.18em]" onClick={() => onStart(r)}>
                  Start Interview
                </button>
              </article>
            ))}
          </div>
        </main>
      </div>
    </section>
  );
}
