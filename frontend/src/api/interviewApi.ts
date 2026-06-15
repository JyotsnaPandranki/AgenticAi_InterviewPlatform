import { api } from "./client";
import type { InterviewSummary, RecommendationResponse, RecommendedRole } from "../types/interview";

export async function uploadResumeAndRecommend(file: File, topK = 5): Promise<RecommendationResponse> {
  const form = new FormData();
  form.append("resume", file);
  form.append("top_k", String(topK));

  // Hook to Python/Node backend endpoint: POST /resume/upload
  const { data } = await api.post("/resume/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function selectSession(payload: {
  resume_id: number;
  mode: "new" | "continue";
  session_id?: number;
  session_name?: string;
}): Promise<{ session: { session_id: number } }> {
  const { data } = await api.post("/session/select", payload);
  return data;
}

export async function startInterview(payload: {
  session_id: number;
  selected_role: string;
  questions: number;
  coaching_mode: boolean;
}): Promise<{ interview_id: string; session_id: number; round: number; question: string }> {
  // Hook endpoint: POST /interview/start
  const { data } = await api.post("/interview/start", payload);
  return data;
}

export async function submitAnswer(payload: {
  interview_id: string;
  answerText: string;
  transcript?: string;
}): Promise<{ done: boolean; question?: string; summary?: InterviewSummary; evaluation?: unknown }> {
  // Hook endpoint: POST /interview/answer
  const { data } = await api.post("/interview/answer", {
    interview_id: payload.interview_id,
    answer_text: payload.answerText,
    transcript: payload.transcript || payload.answerText,
  });
  return data;
}

export async function fetchInterviewSummary(interviewId: string): Promise<InterviewSummary> {
  // Hook endpoint: GET /interview/summary/:interview_id
  const { data } = await api.get<InterviewSummary>(`/interview/summary/${interviewId}`);
  return data;
}
