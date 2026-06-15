import { useState } from "react";
import type { UploadFlowState } from "../types/session";

const initialState: UploadFlowState = {
  file: null,
  fileName: "",
  uploadProgress: 0,
  isUploading: false,
  isAnalyzing: false,
};

export function useUploadFlow() {
  const [state, setState] = useState<UploadFlowState>(initialState);

  const setFile = (file: File | null) => {
    setState((s) => ({ ...s, file, fileName: file?.name || "", error: undefined }));
  };

  const startUpload = () => setState((s) => ({ ...s, isUploading: true, uploadProgress: 10 }));
  const updateProgress = (value: number) => setState((s) => ({ ...s, uploadProgress: value }));
  const startAnalyzing = () => setState((s) => ({ ...s, isUploading: false, isAnalyzing: true, uploadProgress: 100 }));
  const fail = (error: string) => setState((s) => ({ ...s, isUploading: false, isAnalyzing: false, error }));
  const reset = () => setState(initialState);

  return { state, setFile, startUpload, updateProgress, startAnalyzing, fail, reset };
}
