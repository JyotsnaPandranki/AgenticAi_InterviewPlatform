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

export interface InterviewRoundEvaluation {
  score: number;
  communication_score: number;
  technical_score: number;
  confidence_estimate: number;
  feedback: string;
  recommended_next_difficulty: "Easy" | "Medium" | "Hard";
  warnings?: Array<{
    source: "voice" | "cv";
    metric: string;
    level: "low" | "medium" | "high";
    message: string;
  }>;
}

export interface InterviewRound {
  round: number;
  asked_question: string;
  category: string;
  difficulty: string;
  candidate_answer: string;
  evaluation: InterviewRoundEvaluation;
}

export interface InterviewSummary {
  selected_role: string;
  total_questions: number;
  completed_questions: number;
  average_score: number;
  final_difficulty: string;
  used_categories: string[];
  interview_history: InterviewRound[];
}

export interface RecommendationResponse {
  candidate_name?: string;
  recommended_jobs: RecommendedRole[];
}
