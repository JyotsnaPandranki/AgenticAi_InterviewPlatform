import { useNavigate } from 'react-router-dom';
import { GradientButton } from '../components/GradientButton';
import { UploadZone } from '../components/UploadZone';
import { useAppState } from '../context/AppContext';

export default function UploadResumePage() {
  const nav = useNavigate();
  const { state, setResumeFile, setProcessingJob, setProcessingLogs } = useAppState();

  return (
    <section className="mx-auto max-w-4xl py-10">
      <div className="glass-card p-10">
        <h2 className="text-2xl font-semibold">Upload Resume PDF</h2>
        <p className="mt-2 text-on-surface-variant">Clean, simple, and secure processing.</p>
        <div className="mt-8"><UploadZone onFile={setResumeFile} /></div>
        <div className="mt-5 text-sm text-on-surface-variant">{state.resumeFile?.name || 'No file selected'}</div>
        <div className="mt-8">
          <GradientButton
            onClick={() => {
              if (!state.resumeFile) return;
              setProcessingLogs([]);
              setProcessingJob('resume_upload');
              nav('/processing');
            }}
            disabled={!state.resumeFile}
          >
            Analyze Resume
          </GradientButton>
        </div>
      </div>
    </section>
  );
}
