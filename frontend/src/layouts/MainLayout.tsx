import { Outlet } from 'react-router-dom';
import { useLocation } from 'react-router-dom';
import { Footer } from '../components/Footer';
import { Navbar } from '../components/Navbar';

export function MainLayout() {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <div className="min-h-screen">
      <div className="grain-overlay" />
      <Navbar />
      <main className={`relative z-10 ${isHome ? 'pt-16' : 'pt-16 p-6'}`}>
        <Outlet />
        {!isHome ? <Footer /> : null}
      </main>
    </div>
  );
}
