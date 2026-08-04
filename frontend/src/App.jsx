import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import Search from "./pages/Search.jsx";
import Admin from "./pages/Admin.jsx";
import Metrics from "./pages/Metrics.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Navbar />
        <Routes>
          <Route path="/" element={<Search />} />
          <Route path="/admin" element={<Admin />} />
          <Route path="/metrics" element={<Metrics />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
