export function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="glass-card p-5 text-center">
      <div className="text-2xl font-semibold text-primary">{value}</div>
      <div className="text-xs uppercase tracking-wider text-on-surface-variant mt-1">{label}</div>
    </div>
  );
}
