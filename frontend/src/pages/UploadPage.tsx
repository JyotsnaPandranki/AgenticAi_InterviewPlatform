import { Card } from "../components/Card";
import type { UploadFlowState } from "../types/session";

interface UploadPageProps {
  state: UploadFlowState;
  onPickFile: (file: File | null) => void;
  onAnalyze: () => void;
}

export function UploadPage({ state, onPickFile, onAnalyze }: UploadPageProps) {
  return (
    <div className="page-grid one-col">
      <Card title="Upload Resume PDF" subtitle="Clean, simple, and secure processing.">
        <label className="upload-zone">
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => onPickFile(e.target.files?.[0] || null)}
            hidden
          />
          <p>Drag and drop or browse files</p>
          <small>Supports PDF only (Max 10MB)</small>
        </label>

        {state.fileName && (
          <div className="file-chip">
            <span>{state.fileName}</span>
            <span>{state.uploadProgress}%</span>
          </div>
        )}

        <button className="primary-btn" disabled={!state.file || state.isAnalyzing} onClick={onAnalyze}>
          {state.isUploading ? "Uploading..." : state.isAnalyzing ? "Analyzing..." : "Analyze Resume"}
        </button>

        {state.error && <p className="error-text">{state.error}</p>}
      </Card>
    </div>
  );
}
