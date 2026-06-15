import { useLocation, useNavigate } from 'react-router-dom';
import { useEffect } from 'react';

export function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();
  const isHome = location.pathname === '/';
  const isAbout = isHome && location.hash === '#about';

  useEffect(() => {
    if (!isHome || location.hash !== '#about') return;
    const el = document.getElementById('about');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [isHome, location.hash]);

  function goDashboard() {
    if (!isHome || location.hash) {
      navigate('/');
      return;
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function goAbout() {
    if (isHome) {
      const el = document.getElementById('about');
      if (el) {
        navigate('/#about');
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } else {
        navigate('/#about');
      }
      return;
    }
    navigate('/#about');
  }

  return (
    <header className="fixed top-0 z-50 w-full border-b border-white/10 bg-[#090b10]/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-[1400px] items-center justify-between px-7">
        <div className="text-[20px] md:text-[22px] leading-none font-semibold tracking-[-0.01em]">
          HIA <span className="hidden md:inline text-white/70">(Humanists Interview Agent)</span>
        </div>
        <nav className="hidden items-center gap-10 md:flex text-[27px] text-white/65">
          <button
            type="button"
            onClick={goDashboard}
            className={`transition-colors hover:text-white ${isHome && !isAbout ? 'border-b border-primary pb-1 text-white' : ''}`}
          >
            Dashboard
          </button>
          <button
            type="button"
            onClick={goAbout}
            className={`transition-colors hover:text-white ${isAbout ? 'border-b border-primary pb-1 text-white' : ''}`}
          >
            About
          </button>
        </nav>
        <div className="flex items-center gap-3">
          <button className="hidden md:grid h-9 w-9 place-items-center rounded-full border border-white/10 text-white/70">
            <span className="text-sm">◔</span>
          </button>
          <button className="hidden md:grid h-9 w-9 place-items-center rounded-full border border-white/10 text-white/70">
            <span className="text-sm">?</span>
          </button>
        </div>
      </div>
    </header>
  );
}
