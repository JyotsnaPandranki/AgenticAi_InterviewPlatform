import { createContext, useContext, useMemo, useState } from 'react';
import type { InterviewSummary, RecommendedRole, SessionItem } from '../types/models';

type ProcessingJob = 'resume_upload' | 'interview_init' | null;
type InterviewUiState =
  | 'idle'
  | 'dictating_question'
  | 'listening'
  | 'speech_detected'
  | 'processing_answer'
  | 'moving_next_question'
  | 'interview_complete';

interface AppState {
  resumeFile: File | null;
  resumeId: number | null;
  isNewResume: boolean;
  roles: RecommendedRole[];
  selectedRole: RecommendedRole | null;
  sessions: SessionItem[];
  selectedSessionId: number | null;
  sessionMode: 'new' | 'continue';
  coachingMode: boolean;
  questionCount: number;
  interviewId: string | null;
  currentRound: number;
  totalQuestions: number;
  currentQuestion: string;
  transcript: string;
  audioAnalysis: Record<string, number>;
  cvAnalysis: Record<string, number>;
  transcriptionConfidence: number;
  processingJob: ProcessingJob;
  processingLogs: string[];
  interviewUiState: InterviewUiState;
  interviewSummary: InterviewSummary | null;
  lastEvaluation: Record<string, any> | null;
  coachingTip: string;
}

const initialState: AppState = {
  resumeFile: null,
  resumeId: null,
  isNewResume: true,
  roles: [],
  selectedRole: null,
  sessions: [],
  selectedSessionId: null,
  sessionMode: 'new',
  coachingMode: false,
  questionCount: 5,
  interviewId: null,
  currentRound: 1,
  totalQuestions: 5,
  currentQuestion: '',
  transcript: '',
  audioAnalysis: {},
  cvAnalysis: {},
  transcriptionConfidence: 0,
  processingJob: null,
  processingLogs: [],
  interviewUiState: 'idle',
  interviewSummary: null,
  lastEvaluation: null,
  coachingTip: '',
};

const Ctx = createContext<any>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AppState>(initialState);

  const value = useMemo(() => ({
    state,
    setResumeFile: (f: File | null) => setState((s) => ({ ...s, resumeFile: f })),
    setResumeId: (id: number) => setState((s) => ({ ...s, resumeId: id })),
    setIsNewResume: (isNewResume: boolean) => setState((s) => ({ ...s, isNewResume })),
    setRoles: (roles: RecommendedRole[]) => setState((s) => ({ ...s, roles })),
    setSelectedRole: (role: RecommendedRole | null) => setState((s) => ({ ...s, selectedRole: role })),
    setSessions: (sessions: SessionItem[]) => setState((s) => ({ ...s, sessions })),
    setSelectedSessionId: (id: number) => setState((s) => ({ ...s, selectedSessionId: id })),
    setSessionMode: (mode: 'new' | 'continue') => setState((s) => ({ ...s, sessionMode: mode })),
    setCoachingMode: (coachingMode: boolean) => setState((s) => ({ ...s, coachingMode })),
    setQuestionCount: (questionCount: number) => setState((s) => ({ ...s, questionCount })),
    setInterviewId: (id: string) => setState((s) => ({ ...s, interviewId: id })),
    setCurrentRound: (n: number) => setState((s) => ({ ...s, currentRound: n })),
    setTotalQuestions: (n: number) => setState((s) => ({ ...s, totalQuestions: n })),
    setCurrentQuestion: (q: string) => setState((s) => ({ ...s, currentQuestion: q })),
    setTranscript: (t: string) => setState((s) => ({ ...s, transcript: t })),
    setAudioAnalysis: (audioAnalysis: Record<string, number>) => setState((s) => ({ ...s, audioAnalysis })),
    setCvAnalysis: (cvAnalysis: Record<string, number>) => setState((s) => ({ ...s, cvAnalysis })),
    setTranscriptionConfidence: (v: number) => setState((s) => ({ ...s, transcriptionConfidence: v })),
    setProcessingJob: (processingJob: ProcessingJob) => setState((s) => ({ ...s, processingJob })),
    setProcessingLogs: (processingLogs: string[]) => setState((s) => ({ ...s, processingLogs })),
    appendProcessingLog: (line: string) => setState((s) => ({ ...s, processingLogs: [...s.processingLogs, line] })),
    setInterviewUiState: (interviewUiState: InterviewUiState) => setState((s) => ({ ...s, interviewUiState })),
    setInterviewSummary: (summary: InterviewSummary) => setState((s) => ({ ...s, interviewSummary: summary })),
    setLastEvaluation: (lastEvaluation: Record<string, any> | null) => setState((s) => ({ ...s, lastEvaluation })),
    setCoachingTip: (coachingTip: string) => setState((s) => ({ ...s, coachingTip })),
    resetInterviewRuntime: () =>
      setState((s) => ({
        ...s,
        interviewId: null,
        currentRound: 1,
        currentQuestion: '',
        transcript: '',
        audioAnalysis: {},
        cvAnalysis: {},
        transcriptionConfidence: 0,
        interviewUiState: 'idle',
        lastEvaluation: null,
        coachingTip: '',
      })),
  }), [state]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useAppState = () => {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useAppState must be used within AppProvider');
  return ctx;
};
