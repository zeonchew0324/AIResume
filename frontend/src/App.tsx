import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import AppLayout from "./components/AppLayout";
import AnalyzeResume from "./pages/AnalyzeResume";
import ImproveResume from "./pages/ImproveResume";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<Navigate to="/analyze" replace />} />
          <Route path="/analyze" element={<AnalyzeResume />} />
          <Route path="/improve" element={<ImproveResume />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
