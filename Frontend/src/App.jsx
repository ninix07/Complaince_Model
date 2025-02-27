/** @format */

import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import HomePage from "./pages/HomePage.jsx";
import ComplianceCategoryPage from "./pages/ComplianceCategoryPage.jsx";
import FlowChartPage from "./pages/FlowChartPage.jsx";
import ComplianceChartsPage from "./pages/ComplianceChartsPage.jsx";
function App() {
  return (
    <Router>
      <div className="w-screen min-h-screen">
        <Navbar />
        <main className="max-w-7xl mx-auto pt-24 px-4">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route
              path="/compliance-category"
              element={<ComplianceCategoryPage />}
            />
            <Route path="/flow-chart" element={<FlowChartPage />} />
            <Route
              path="/compliance-charts"
              element={<ComplianceChartsPage />}
            />

          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
