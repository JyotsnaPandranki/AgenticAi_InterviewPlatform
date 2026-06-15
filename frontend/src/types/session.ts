export interface ResumeSession {
  session_id: number;
  session_name: string;
  total_interviews: number;
  latest_role?: string;
  latest_score?: number;
  session_status: "ACTIVE" | "COMPLETED";
}

export interface UploadFlowState {
  file: File | null;
  fileName: string;
  uploadProgress: number;
  isUploading: boolean;
  isAnalyzing: boolean;
  error?: string;
}

export interface LiveInterviewState {
  isRunning: boolean;
  secondsLeft: number;
  isListening: boolean;
  transcript: string;
  confidence?: number;
}
