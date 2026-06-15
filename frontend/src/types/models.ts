export interface RecommendedRole {
  job_id: string;
  title: string;
  match_score: number;
  semantic_score: number;
  skill_overlap_score: number;
  matched_skills: string[];
  missing_skills: string[];
  why_fit: string;
}

export interface SessionItem {
  session_id: number;
  session_name: string;
  total_interviews: number;
  latest_role?: string;
  latest_score?: number;
  session_status: string;
}

export interface InterviewSummary {
  selected_role: string;
  total_questions: number;
  completed_questions: number;
  average_score: number;
  final_difficulty: string;
  used_categories: string[];
  interview_history: Array<{
    round: number;
    asked_question: string;
    category: string;
    difficulty: string;
    candidate_answer: string;
    evaluation: {
      score: number;
      communication_score: number;
      technical_score: number;
      confidence_estimate: number;
      feedback: string;
      warnings?: Array<{ source: string; metric: string; level: string; message: string }>;
    };
  }>;
}

export interface AudioTranscriptionResult {
  transcript: string;
  answer_text: string;
  transcription_confidence: number;
  no_speech_prob: number;
  communication_score: number;
  audio_analysis: Record<string, number>;
}

export interface CvFrameResult {
  available: boolean;
  face_detected: boolean;
  cv_analysis: {
    engagement_score?: number;
    eye_contact_score?: number;
    attention_score?: number;
    blink_rate?: number;
    perclos?: number;
    distraction_score?: number;
    face_detected?: boolean;
  };
}
