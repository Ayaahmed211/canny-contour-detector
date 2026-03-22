import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./ActiveContour.css"; // Ensure this path is correct

// --- Interactive Chain Code Visualizer Component ---
const ChainCodeVisualizer = ({ chainCode }) => {
  const [step, setStep] = useState(chainCode.length);

  useEffect(() => {
    setStep(chainCode.length);
  }, [chainCode]);

  if (!chainCode || chainCode.length === 0) return null;

  const moves = {
    0: [1, 0],   // Right
    1: [1, 1],   // Down-Right
    2: [0, 1],   // Down
    3: [-1, 1],  // Down-Left
    4: [-1, 0],  // Left
    5: [-1, -1], // Up-Left
    6: [0, -1],  // Up
    7: [1, -1]   // Up-Right
  };

  let x = 0, y = 0;
  let minX = 0, maxX = 0, minY = 0, maxY = 0;
  const allPoints = [[0, 0]];

  chainCode.forEach((code) => {
    const [dx, dy] = moves[code] || [0, 0];
    x += dx;
    y += dy;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    allPoints.push([x, y]);
  });

  const width = Math.max(maxX - minX, 1);
  const height = Math.max(maxY - minY, 1);
  const pad = Math.max(width, height) * 0.15;
  const viewBox = `${minX - pad} ${minY - pad} ${width + pad * 2} ${height + pad * 2}`;
  const strokeW = Math.max(width, height) * 0.015 || 0.5;

  const visiblePoints = allPoints.slice(0, step + 1);
  const fullPathString = allPoints.map((p) => `${p[0]},${p[1]}`).join(" ");
  const activePathString = visiblePoints.map((p) => `${p[0]},${p[1]}`).join(" ");
  const currentPoint = visiblePoints[visiblePoints.length - 1];

  return (
    <div className="chain-viz-container">
      <div className="chain-viz-card">
        <svg viewBox={viewBox} className="chain-viz-svg">
          <polyline points={fullPathString} fill="none" stroke="var(--dimmed)" strokeWidth={strokeW} strokeLinejoin="round" />
          <polyline points={activePathString} fill="none" stroke="var(--green)" strokeWidth={strokeW} strokeLinejoin="round" strokeLinecap="round" />
          <circle cx={allPoints[0][0]} cy={allPoints[0][1]} r={strokeW * 2.5} fill="var(--orange)" />
          <circle cx={currentPoint[0]} cy={currentPoint[1]} r={strokeW * 2.5} fill="var(--cyan)" />
        </svg>

        <div className="chain-slider-row">
          <span className="chain-step-text">
            {step}/{chainCode.length}
          </span>
          <input
            type="range"
            min="0"
            max={chainCode.length}
            value={step}
            onChange={(e) => setStep(Number(e.target.value))}
            className="chain-viz-slider"
          />
        </div>
      </div>

      <div className="chain-legend">
        <div>0=→ 1=↘ 2=↓ 3=↙ 4=← 5=↖ 6=↑ 7=↗</div>
      </div>

      <div className="chain-code-box">
        <span className="chain-code-active">{chainCode.slice(0, step).join("")}</span>
        <span className="chain-code-inactive">{chainCode.slice(step).join("")}</span>
      </div>
    </div>
  );
};


// --- Main Page Component ---
export default function ActiveContourPage() {
  const navigate = useNavigate();
  const imageRef = useRef(null);

  const [imageSrc, setImageSrc] = useState(null);
  const [points, setPoints] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  // Extended Algorithm Parameters
  const [params, setParams] = useState({
    alpha: 0.3,
    beta: 0.5,
    gamma: 1.5,
    max_iterations: 200,
    convergence_threshold: 0.5,
  });

  const paramConfig = {
    alpha: { min: 0, max: 2, step: 0.1, label: "Alpha (Continuity)" },
    beta: { min: 0, max: 2, step: 0.1, label: "Beta (Curvature)" },
    gamma: { min: 0, max: 5, step: 0.1, label: "Gamma (Edge Pull)" },
    max_iterations: { min: 10, max: 500, step: 10, label: "Max Iterations" },
    convergence_threshold: { min: 0.1, max: 2.0, step: 0.1, label: "Convergence Thresh" },
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImageSrc(event.target.result);
        setPoints([]); 
        setResults(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleImageClick = (e) => {
    if (!imageRef.current || results) return; 
    
    const rect = imageRef.current.getBoundingClientRect();
    const scaleX = imageRef.current.naturalWidth / rect.width;
    const scaleY = imageRef.current.naturalHeight / rect.height;
    
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    
    setPoints([...points, [x, y]]);
  };

  const resetContour = () => {
    setPoints([]);
    setResults(null);
  };

  const handleRunSnake = async () => {
    if (points.length < 3) {
      alert("Please select at least 3 points to form a closed polygon.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/snake/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imageSrc, 
          initial_contour: points, 
          alpha: params.alpha,
          beta: params.beta,
          gamma: params.gamma,
          max_iterations: params.max_iterations,
          convergence_threshold: params.convergence_threshold
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Server error occurred");
      }

      setResults({
        visualization: data.visualization,
        perimeter: data.perimeter,
        area: data.area,
        chainCode: data.chainCode 
      });
    } catch (error) {
      console.error("Error running snake:", error);
      alert("Failed to process image: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Helper to map image coordinates to screen coordinates for the SVG
  const getMappedPointsString = () => {
    return points.map(p => {
      const rect = imageRef.current?.getBoundingClientRect();
      const scaleX = rect ? rect.width / imageRef.current.naturalWidth : 1;
      const scaleY = rect ? rect.height / imageRef.current.naturalHeight : 1;
      return `${p[0] * scaleX},${p[1] * scaleY}`;
    }).join(" ");
  };

  return (
    <div className="page-container">
      {/* Topbar */}
      <div className="topbar">
        <div className="topbar-left">
          <button onClick={() => navigate("/")} className="back-button">
            ←
          </button>
          <div>
            <div className="page-title">Active Contour (Snake)</div>
            <div className="page-subtitle">Greedy Algorithm</div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="main-content">
        
        {/* Left Column: Scrollable Sidebar */}
        <div className="sidebar">
          
          {/* Upload Panel */}
          <div className="panel">
            <div className="panel-title">1. Source Image</div>
            <label className="upload-label">
              <input type="file" accept="image/*" onChange={handleImageUpload} style={{ display: "none" }} />
              Upload Image
            </label>

            {imageSrc && (
              <div className="button-row">
                <button onClick={resetContour} className="secondary-button">
                  {results ? "Redraw New Contour" : "Clear Points"}
                </button>
              </div>
            )}
          </div>

          {/* Parameters Panel */}
          <div className="panel">
            <div className="panel-title">2. Parameters</div>
            {Object.entries(params).map(([key, value]) => {
              const config = paramConfig[key];
              return (
                <div key={key} className="parameter-row">
                  <div className="parameter-labels">
                    <span>{config.label}</span>
                    <span>{value}</span>
                  </div>
                  <input 
                    type="range" 
                    min={config.min} 
                    max={config.max} 
                    step={config.step} 
                    value={value}
                    onChange={(e) => setParams({...params, [key]: parseFloat(e.target.value)})}
                    className="range-slider"
                  />
                </div>
              )
            })}

            <button
              onClick={handleRunSnake}
              disabled={!imageSrc || points.length < 3 || loading}
              className={`run-button ${loading ? "loading" : ""}`}
            >
              {loading ? "Computing..." : results ? "Recompute with New Params" : "Run Active Contour"}
            </button>
          </div>

          {/* Results Panel */}
          {results && (
            <div className="panel results-panel">
              <div className="results-title">Results</div>
              
              <div className="metric-row">
                <span className="metric-label">Perimeter</span>
                <span className="metric-value">{results.perimeter.toFixed(2)} px</span>
              </div>
              
              <div className="metric-row">
                <span className="metric-label">Area</span>
                <span className="metric-value">{results.area.toFixed(2)} px²</span>
              </div>

              <div>
                <div className="chain-code-label">Chain Code Visualizer</div>
                <ChainCodeVisualizer chainCode={results.chainCode} />
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Interactive Workspace */}
        <div className="workspace">
          {!imageSrc ? (
            <div className="workspace-placeholder">
              Upload an image, then click to add polygon vertices.
            </div>
          ) : results ? (
            // Result View
            <div className="result-view">
              <img src={results.visualization} alt="Snake Result" className="result-image" />
            </div>
          ) : (
            // Interactive Point-and-Click View
            <div className="interactive-view">
              <img 
                ref={imageRef}
                src={imageSrc} 
                alt="Workspace" 
                onClick={handleImageClick}
                className="workspace-image"
              />
              
              <svg className="workspace-svg-overlay">
                {points.length > 2 ? (
                  <polygon
                    points={getMappedPointsString()}
                    fill="var(--orange-alpha-20)"
                    stroke="var(--orange)"
                    strokeWidth="2"
                    strokeDasharray="4 4"
                  />
                ) : points.length === 2 ? (
                  <polyline
                    points={getMappedPointsString()}
                    fill="none"
                    stroke="var(--orange)"
                    strokeWidth="2"
                    strokeDasharray="4 4"
                  />
                ) : null}

                {points.map((p, i) => {
                  const rect = imageRef.current?.getBoundingClientRect();
                  const scaleX = rect ? rect.width / imageRef.current.naturalWidth : 1;
                  const scaleY = rect ? rect.height / imageRef.current.naturalHeight : 1;
                  return (
                    <circle 
                      key={i} 
                      cx={p[0] * scaleX} 
                      cy={p[1] * scaleY} 
                      r="4" 
                      fill={i === 0 ? "var(--green)" : "var(--orange)"} 
                    />
                  );
                })}
              </svg>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}