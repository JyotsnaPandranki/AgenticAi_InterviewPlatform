export function GradientButton({ children, ...props }: any) {
  return (
    <button className="btn-gradient px-6 py-3" {...props}>
      {children}
    </button>
  );
}
