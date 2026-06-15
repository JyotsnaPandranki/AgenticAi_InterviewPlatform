import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function InterviewInitPage() {
  const nav = useNavigate();
  useEffect(() => {
    nav('/processing');
  }, []);

  return (
    <section className="mx-auto max-w-4xl py-16 text-center text-white/70">Preparing interview...</section>
  );
}
