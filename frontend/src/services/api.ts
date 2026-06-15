import axios from 'axios';
import type { AudioTranscriptionResult, CvFrameResult, InterviewSummary, RecommendedRole, SessionItem } from '../types/models';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 180000,
});

export async function uploadResume(file: File, topK = 5): Promise<{ resume_id: number; sessions: SessionItem[]; recommended_jobs: RecommendedRole[]; structured_profile: any }> {
  const form = new FormData();
  form.append('resume', file);
  form.append('top_k', String(topK));
  const { data } = await api.post('/upload_resume', form, { headers: { 'Content-Type': 'multipart/form-data' } });
  return data;
}

export async function analyzeResume(resumeId: number): Promise<any> {
  const { data } = await api.post('/analyze_resume', { resume_id: resumeId });
  return data;
}

export async function getRecommendedRoles(resumeId: number): Promise<{ recommended_jobs: RecommendedRole[] }> {
  const { data } = await api.get('/recommended_roles', { params: { resume_id: resumeId } });
  return data;
}

export async function startSession(payload: { resume_id: number; mode: 'new' | 'continue'; session_id?: number; session_name?: string }) {
  const { data } = await api.post('/start_session', payload);
  return data;
}

export async function startInterview(payload: { session_id: number; selected_role: string; questions: number; coaching_mode: boolean }) {
  const { data } = await api.post('/start_interview', payload);
  return data;
}

export async function nextQuestion(payload: { interview_id: string }) {
  const { data } = await api.post('/next_question', payload);
  return data;
}

export async function transcribeAudio(audio: Blob): Promise<AudioTranscriptionResult> {
  const form = new FormData();
  form.append('audio', audio, 'answer.webm');
  const { data } = await api.post('/transcribe_audio', form, { headers: { 'Content-Type': 'multipart/form-data' } });
  return data;
}

export async function analyzeVideoFrame(image: string, timestamp: number): Promise<CvFrameResult> {
  const { data } = await api.post('/analyze_video_frame', { image, timestamp });
  return data;
}

export async function submitAnswer(payload: {
  interview_id: string;
  answer_text: string;
  transcript?: string;
  audio_analysis?: Record<string, number>;
  cv_analysis?: Record<string, number>;
}) {
  const { data } = await api.post('/submit_answer', payload);
  return data;
}

export async function endInterview(payload: { interview_id: string; reason?: string }) {
  const { data } = await api.post('/end_interview', payload);
  return data;
}

export async function interviewSummary(interviewId: string): Promise<InterviewSummary> {
  const { data } = await api.get('/interview_summary', { params: { interview_id: interviewId } });
  return data;
}

export async function sessions(resumeId: number): Promise<{ sessions: SessionItem[] }> {
  const { data } = await api.get('/sessions', { params: { resume_id: resumeId } });
  return data;
}

export async function sessionInterviews(sessionId: number): Promise<{ interviews: Array<{
  interview_id: number;
  session_id: number;
  interview_number: number;
  target_role: string;
  interview_timestamp: string;
  overall_score: number;
  communication_score: number;
  technical_score: number;
  confidence_score: number;
}> }> {
  const { data } = await api.get('/session_interviews', { params: { session_id: sessionId } });
  return data;
}

export async function interviewRecordSummary(interviewId: number): Promise<{ summary: InterviewSummary }> {
  const { data } = await api.get(`/interview_record/${interviewId}`);
  return data;
}

export async function deleteSession(sessionId: number): Promise<{ deleted: boolean; session_id: number; deleted_interviews: number; deleted_memory_rows: number }> {
  const { data } = await api.delete(`/session/${sessionId}`);
  return data;
}

export async function resumeList() {
  const { data } = await api.get('/resume_list');
  return data;
}

export async function resumeById(id: number) {
  const { data } = await api.get(`/resume/${id}`);
  return data;
}
