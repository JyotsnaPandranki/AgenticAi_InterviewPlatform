import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { MainLayout } from './layouts/MainLayout';
import HomePage from './pages/HomePage';
import InterviewInitPage from './pages/InterviewInitPage';
import InterviewSummaryPage from './pages/InterviewSummaryPage';
import LiveInterviewPage from './pages/LiveInterviewPage';
import ProcessingPage from './pages/ProcessingPage';
import RecommendedRolesPage from './pages/RecommendedRolesPage';
import SessionSelectPage from './pages/SessionSelectPage';
import UploadResumePage from './pages/UploadResumePage';
import './styles/index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <AppProvider>
        <Routes>
          <Route element={<MainLayout />}>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadResumePage />} />
            <Route path="/processing" element={<ProcessingPage />} />
            <Route path="/session-select" element={<SessionSelectPage />} />
            <Route path="/roles" element={<RecommendedRolesPage />} />
            <Route path="/init" element={<InterviewInitPage />} />
            <Route path="/live" element={<LiveInterviewPage />} />
            <Route path="/summary" element={<InterviewSummaryPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </AppProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
