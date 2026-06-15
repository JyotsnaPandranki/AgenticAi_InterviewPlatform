import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppState } from '../context/AppContext';

export default function SessionSelectPage() {
  const nav = useNavigate();
  const { state, setSessionMode, setSelectedSessionId } = useAppState();

  const availableSessions = useMemo(
    () => (state.sessions || []).filter((s: any) => Number(s.total_interviews || 0) < 3),
    [state.sessions],
  );

  return (
    <section className="mx-auto max-w-4xl py-10">
      <div className="glass-card p-8">
        <h2 className="text-3xl font-semibold">Select Interview Session</h2>
        <p className="mt-2 text-on-surface-variant">
          Matching sessions for this uploaded resume are shown below.
        </p>

        {availableSessions.length === 0 ? (
          <div className="mt-6 rounded-xl border border-white/10 bg-white/[0.03] p-4 text-white/70">
            No available session for this resume. A new session will be created automatically.
          </div>
        ) : (
          <div className="mt-6 space-y-3">
            {availableSessions.map((s: any) => (
              <button
                key={s.session_id}
                className="w-full rounded-xl border border-white/10 bg-white/[0.03] p-4 text-left hover:border-primary/40"
                onClick={() => {
                  setSessionMode('continue');
                  setSelectedSessionId(Number(s.session_id));
                  nav('/roles');
                }}
              >
                <div className="font-medium">Session #{s.session_id}</div>
                <div className="mt-1 text-sm text-white/60">
                  {s.total_interviews}/3 interviews · Latest role: {s.latest_role || 'N/A'}
                </div>
              </button>
            ))}
          </div>
        )}

        <div className="mt-6 flex gap-3">
          <button
            className="rounded-lg border border-white/10 bg-white/5 px-4 py-2"
            onClick={() => {
              setSessionMode('new');
              setSelectedSessionId(null as any);
              nav('/roles');
            }}
          >
            Start New Session
          </button>
          <button className="rounded-lg border border-white/10 px-4 py-2" onClick={() => nav('/upload')}>
            Back
          </button>
        </div>
      </div>
    </section>
  );
}

