export function UploadZone({ onFile }: { onFile: (f: File | null) => void }) {
  return (
    <label className="glass-card block cursor-pointer border-dashed p-10 text-center">
      <input type="file" accept="application/pdf" hidden onChange={(e) => onFile(e.target.files?.[0] || null)} />
      <div className="text-4xl mb-4">⬆</div>
      <p>Drag and drop or browse files</p>
      <small className="text-on-surface-variant">PDF only (Max 10MB)</small>
    </label>
  );
}
