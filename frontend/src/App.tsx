import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "@/context/AuthContext";
import ProtectedRoute from "@/components/ProtectedRoute";
import AppLayout from "./components/AppLayout";
import AuthPage from "./pages/AuthPage";
import AnalyzeResume from "./pages/AnalyzeResume";
import ImproveResume from "./pages/ImproveResume";
import CoverLetter from "./pages/CoverLetter";
import MyResume from "./pages/MyResume";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<AuthPage />} />
          <Route
            element={
              <ProtectedRoute>
                <AppLayout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Navigate to="/analyze" replace />} />
            <Route path="/analyze" element={<AnalyzeResume />} />
            <Route path="/improve" element={<ImproveResume />} />
            <Route path="/cover-letter" element={<CoverLetter />} />
            <Route path="/my-resume" element={<MyResume />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
