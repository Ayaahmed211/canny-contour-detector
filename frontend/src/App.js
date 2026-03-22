import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./HomePage";
import CannyApp from "./CannyApp";
import ActiveContourPage from "./ActiveContourPage";
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/canny" element={<CannyApp />} />
        <Route path="/snake" element={<ActiveContourPage />} />
      </Routes>
    </BrowserRouter>
  );
}