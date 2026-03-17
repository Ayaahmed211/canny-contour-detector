import { useState, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";

const API_URL = "http://localhost:8000/api/upload/";

/* ─── Design tokens ─────────────────────────────────────────────────────── */
const C = {
  bg:        "#07090f",
  surface:   "#0d1120",
  border:    "#1a2236",
  borderHov: "#2a3a56",
  text:      "#d4dff2",
  muted:     "#4a5a78",
  dimmed:    "#2a3550",
  cyan:      "#38bdf8",
  orange:    "#ff6b35",
  green:     "#00e5a0",
  violet:    "#c850ff",
  red:       "#ff4820",
  white:     "#ffffff",
};

const TABS = [
  { key: "original",     label: "Original",     icon: "⬜", color: C.cyan   },
  { key: "edges",        label: "Canny Edges",   icon: "〰",  color: C.white  },
  { key: "lines_img",    label: "Lines",         icon: "╱",  color: C.red    },
  { key: "circles_img",  label: "Circles",       icon: "◯",  color: C.green  },
  { key: "ellipses_img", label: "Ellipses",      icon: "⬭",  color: C.violet },
  { key: "all_img",      label: "All Shapes",    icon: "✦",  color: C.orange },
];

/* ─── Reusable components ───────────────────────────────────────────────── */

function StatCard({ label, value, color, icon }) {
  return (
    <div style={{
      flex: 1, minWidth: 90,
      background: C.surface,
      border: `1px solid ${color}30`,
      borderRadius: 14,
      padding: "16px 18px",
      display: "flex", flexDirection: "column", gap: 6,
      boxShadow: `0 0 20px ${color}10`,
      transition: "box-shadow .3s",
    }}
      onMouseEnter={e => e.currentTarget.style.boxShadow = `0 0 28px ${color}30`}
      onMouseLeave={e => e.currentTarget.style.boxShadow = `0 0 20px ${color}10`}
    >
      <span style={{ fontSize: 22 }}>{icon}</span>
      <span style={{ fontSize: 28, fontWeight: 900, color, fontFamily: "monospace", lineHeight: 1 }}>
        {value ?? "—"}
      </span>
      <span style={{ fontSize: 10, color: C.muted, letterSpacing: 2.5, textTransform: "uppercase" }}>
        {label}
      </span>
    </div>
  );
}

function ParamSlider({ label, name, min, max, step, value, onChange, color = C.cyan }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 10, color: C.muted, letterSpacing: 1.8, textTransform: "uppercase" }}>
          {label}
        </span>
        <span style={{
          fontSize: 11, color, fontFamily: "monospace",
          background: `${color}15`, padding: "2px 7px", borderRadius: 5,
        }}>
          {value}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(name, parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: color, cursor: "pointer", height: 4 }}
      />
    </div>
  );
}

function ImageViewer({ src, label, color, badge }) {
  const [zoom, setZoom] = useState(false);

  return (
    <>
      <div
        onClick={() => src && setZoom(true)}
        style={{
          width: "100%",
          aspectRatio: "4/3",
          borderRadius: 16,
          overflow: "hidden",
          border: `1px solid ${src ? color + "40" : C.border}`,
          background: C.surface,
          display: "flex", alignItems: "center", justifyContent: "center",
          cursor: src ? "zoom-in" : "default",
          position: "relative",
          boxShadow: src ? `0 0 32px ${color}18` : "none",
          transition: "box-shadow .3s",
        }}
        onMouseEnter={e => { if (src) e.currentTarget.style.boxShadow = `0 0 48px ${color}30`; }}
        onMouseLeave={e => e.currentTarget.style.boxShadow = src ? `0 0 32px ${color}18` : "none"}
      >
        {src
          ? <>
              <img src={src} alt={label}
                style={{ width: "100%", height: "100%", objectFit: "contain" }} />
              {badge && (
                <div style={{
                  position: "absolute", top: 10, right: 10,
                  background: `${color}22`, border: `1px solid ${color}60`,
                  backdropFilter: "blur(8px)",
                  borderRadius: 8, padding: "3px 10px",
                  fontSize: 11, color, fontFamily: "monospace", fontWeight: 700,
                }}>
                  {badge}
                </div>
              )}
              <div style={{
                position: "absolute", bottom: 10, right: 10,
                color: C.muted, fontSize: 11,
              }}>
                🔍 click to enlarge
              </div>
            </>
          : <div style={{ color: C.dimmed, fontSize: 48, userSelect: "none" }}>◌</div>
        }
      </div>

      {/* Lightbox */}
      {zoom && (
        <div
          onClick={() => setZoom(false)}
          style={{
            position: "fixed", inset: 0,
            background: "rgba(0,0,0,0.88)",
            backdropFilter: "blur(6px)",
            zIndex: 999,
            display: "flex", alignItems: "center", justifyContent: "center",
            cursor: "zoom-out",
          }}
        >
          <img src={src} alt={label}
            style={{ maxWidth: "92vw", maxHeight: "92vh", borderRadius: 12,
              border: `2px solid ${color}50`,
              boxShadow: `0 0 80px ${color}40` }} />
        </div>
      )}
    </>
  );
}

/* ─── Main Canny App ──────────────────────────────────────────────────────── */
export default function CannyApp() {
  const navigate = useNavigate();
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [dragging, setDragging] = useState(false);
  const [activeTab, setActiveTab] = useState("original");
  const inputRef = useRef();

  const [params, setParams] = useState({
    canny_sigma:  1.4,
    canny_low:    0.05,
    canny_high:   0.15,
    lines_thresh: 80,
    circles_p2:   30,
  });
  const setParam = useCallback((k, v) => setParams(p => ({ ...p, [k]: v })), []);

  const handleFile = (f) => {
    if (!f) return;
    setFile(f); setPreview(URL.createObjectURL(f));
    setResult(null); setError(null); setActiveTab("original");
  };

  const onDrop = (e) => {
    e.preventDefault(); setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const detect = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    const fd = new FormData();
    fd.append("image", file);
    Object.entries(params).forEach(([k, v]) => fd.append(k, v));
    try {
      const res  = await fetch(API_URL, { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Server error");
      setResult(data);
      setActiveTab("all_img");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const activeTabDef = TABS.find(t => t.key === activeTab);
  const badgeMap = {
    lines_img:    result ? `${result.lines} lines`    : null,
    circles_img:  result ? `${result.circles} circles` : null,
    ellipses_img: result ? `${result.ellipses} ellipses` : null,
  };

  return (
    <div style={{
      minHeight: "100vh", background: C.bg,
      fontFamily: "'Segoe UI', 'Helvetica Neue', sans-serif",
      color: C.text,
    }}>

      {/* ── Topbar ── */}
      <div style={{
        borderBottom: `1px solid ${C.border}`,
        background: `linear-gradient(180deg, #0d1525 0%, ${C.bg} 100%)`,
        padding: "0 40px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        height: 64,
      }}>
        <button
          onClick={() => navigate("/")}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 14,
            padding: 0,
          }}
        >
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: `linear-gradient(135deg, ${C.cyan}, ${C.violet})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18,
          }}>◉</div>
          <div style={{ textAlign: "left" }}>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: -0.3, color: C.text }}>
              Edge &amp; Shape Detector
            </div>
            <div style={{ fontSize: 10, color: C.muted, letterSpacing: 2, textTransform: "uppercase" }}>
              Canny · Lines · Circles · Ellipses
            </div>
          </div>
        </button>
        <div style={{
          fontSize: 10, color: C.dimmed, letterSpacing: 2, textTransform: "uppercase",
          fontFamily: "monospace",
        }}>
          Computer Vision / Assignment 2
        </div>
      </div>

      {/* ── Body ── */}
      <div style={{ display: "flex", height: "calc(100vh - 64px)" }}>

        {/* ── Left panel ── */}
        <div style={{
          width: 300, flexShrink: 0,
          borderRight: `1px solid ${C.border}`,
          background: C.surface,
          padding: "24px 20px",
          display: "flex", flexDirection: "column", gap: 20,
          overflowY: "auto",
        }}>

          {/* Upload zone */}
          <div
            onClick={() => inputRef.current.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            style={{
              borderRadius: 14,
              border: `2px dashed ${dragging ? C.cyan : C.border}`,
              background: dragging ? `${C.cyan}08` : C.bg,
              padding: preview ? 0 : "28px 16px",
              cursor: "pointer", textAlign: "center",
              transition: "all .2s", overflow: "hidden",
            }}
          >
            <input ref={inputRef} type="file" accept="image/*"
              style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />
            {preview
              ? <img src={preview} alt="preview"
                  style={{ width: "100%", display: "block", maxHeight: 200, objectFit: "contain",
                    padding: 8 }} />
              : <>
                  <div style={{ fontSize: 32, marginBottom: 8 }}>🖼</div>
                  <div style={{ fontSize: 12, color: C.muted }}>
                    Drop image or click to browse
                  </div>
                </>
            }
          </div>

          {file && (
            <div style={{
              fontSize: 11, color: C.muted, textAlign: "center",
              background: C.bg, borderRadius: 8, padding: "6px 10px",
              border: `1px solid ${C.border}`,
            }}>
              📎 {file.name}
            </div>
          )}

          {/* Params */}
          <div style={{
            background: C.bg, borderRadius: 14, border: `1px solid ${C.border}`,
            padding: "16px 14px", display: "flex", flexDirection: "column", gap: 14,
          }}>
            <div style={{ fontSize: 10, color: C.muted, letterSpacing: 3, textTransform: "uppercase" }}>
              Parameters
            </div>

            <div style={{ paddingBottom: 12, borderBottom: `1px solid ${C.border}`, display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ fontSize: 10, color: C.cyan, letterSpacing: 2, textTransform: "uppercase" }}>
                ◈ Canny
              </div>
              <ParamSlider label="Sigma"      name="canny_sigma" min={0.5}  max={3}    step={0.1}  value={params.canny_sigma}  onChange={setParam} color={C.cyan} />
              <ParamSlider label="Low ratio"  name="canny_low"   min={0.01} max={0.2}  step={0.01} value={params.canny_low}   onChange={setParam} color={C.cyan} />
              <ParamSlider label="High ratio" name="canny_high"  min={0.05} max={0.5}  step={0.01} value={params.canny_high}  onChange={setParam} color={C.cyan} />
            </div>

            <div style={{ paddingBottom: 12, borderBottom: `1px solid ${C.border}`, display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ fontSize: 10, color: C.red, letterSpacing: 2, textTransform: "uppercase" }}>
                ╱ Lines
              </div>
              <ParamSlider label="Threshold" name="lines_thresh" min={20} max={200} step={5} value={params.lines_thresh} onChange={setParam} color={C.red} />
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ fontSize: 10, color: C.green, letterSpacing: 2, textTransform: "uppercase" }}>
                ◯ Circles
              </div>
              <ParamSlider label="Acc. Thresh" name="circles_p2" min={10} max={100} step={5} value={params.circles_p2} onChange={setParam} color={C.green} />
            </div>
          </div>

          {/* Run */}
          <button
            onClick={detect} disabled={!file || loading}
            style={{
              background: file && !loading
                ? `linear-gradient(135deg, #0284c7, ${C.cyan})`
                : C.border,
              color: file && !loading ? "#fff" : C.muted,
              border: "none", borderRadius: 12,
              padding: "13px 0", fontSize: 13, fontWeight: 700,
              cursor: file && !loading ? "pointer" : "not-allowed",
              letterSpacing: 0.8, transition: "all .2s",
              boxShadow: file && !loading ? `0 4px 24px ${C.cyan}30` : "none",
            }}
          >
            {loading ? "⚙  Processing…" : "▶  Run Detection"}
          </button>

          {error && (
            <div style={{
              background: "#ff000015", border: `1px solid #ff000035`,
              borderRadius: 10, padding: "10px 13px",
              fontSize: 12, color: "#f87171",
            }}>
              ⚠ {error}
            </div>
          )}
        </div>

        {/* ── Right panel ── */}
        <div style={{
          flex: 1, display: "flex", flexDirection: "column",
          overflow: "hidden",
        }}>

          {/* Stats bar */}
          <div style={{
            borderBottom: `1px solid ${C.border}`,
            padding: "16px 32px",
            display: "flex", gap: 12,
            background: `${C.surface}88`,
          }}>
            <StatCard label="Lines"    value={result?.lines}    color={C.red}    icon="╱" />
            <StatCard label="Circles"  value={result?.circles}  color={C.green}  icon="◯" />
            <StatCard label="Ellipses" value={result?.ellipses} color={C.violet} icon="⬭" />
          </div>

          {/* Tab bar */}
          <div style={{
            display: "flex", gap: 0,
            borderBottom: `1px solid ${C.border}`,
            padding: "0 24px",
            background: C.surface,
            overflowX: "auto",
          }}>
            {TABS.map(tab => {
              const active = activeTab === tab.key;
              return (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  style={{
                    background: "none", border: "none",
                    borderBottom: active ? `2px solid ${tab.color}` : "2px solid transparent",
                    color: active ? tab.color : C.muted,
                    padding: "14px 18px",
                    cursor: "pointer", fontSize: 12, fontWeight: active ? 700 : 400,
                    letterSpacing: 0.5,
                    transition: "all .15s",
                    whiteSpace: "nowrap",
                    display: "flex", alignItems: "center", gap: 7,
                  }}
                  onMouseEnter={e => { if (!active) e.currentTarget.style.color = C.text; }}
                  onMouseLeave={e => { if (!active) e.currentTarget.style.color = C.muted; }}
                >
                  <span style={{ fontSize: 16 }}>{tab.icon}</span>
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Image area */}
          <div style={{
            flex: 1, overflow: "auto",
            padding: "28px 32px",
            display: "flex", flexDirection: "column", gap: 20,
          }}>

            {/* Active big view */}
            <div style={{ maxWidth: 700 }}>
              <div style={{
                fontSize: 10, color: activeTabDef?.color,
                letterSpacing: 3, textTransform: "uppercase",
                marginBottom: 10, fontFamily: "monospace",
              }}>
                {activeTabDef?.icon}  {activeTabDef?.label}
              </div>
              <ImageViewer
                src={result?.[activeTab] ?? (activeTab === "original" ? preview : null)}
                label={activeTabDef?.label}
                color={activeTabDef?.color}
                badge={badgeMap[activeTab]}
              />
            </div>

            {/* Thumbnail strip — only when results available */}
            {result && (
              <div>
                <div style={{
                  fontSize: 10, color: C.muted, letterSpacing: 3,
                  textTransform: "uppercase", marginBottom: 12,
                }}>
                  All Views
                </div>
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                  {TABS.map(tab => {
                    const src = result[tab.key] ?? (tab.key === "original" ? preview : null);
                    const active = activeTab === tab.key;
                    return (
                      <div
                        key={tab.key}
                        onClick={() => setActiveTab(tab.key)}
                        style={{
                          width: 110, cursor: "pointer",
                          borderRadius: 10, overflow: "hidden",
                          border: `2px solid ${active ? tab.color : C.border}`,
                          boxShadow: active ? `0 0 16px ${tab.color}40` : "none",
                          transition: "all .2s",
                          background: C.surface,
                        }}
                        onMouseEnter={e => { if (!active) e.currentTarget.style.borderColor = tab.color + "60"; }}
                        onMouseLeave={e => { if (!active) e.currentTarget.style.borderColor = C.border; }}
                      >
                        <div style={{
                          aspectRatio: "1/1",
                          display: "flex", alignItems: "center", justifyContent: "center",
                          background: C.bg,
                        }}>
                          {src
                            ? <img src={src} alt={tab.label}
                                style={{ width: "100%", height: "100%", objectFit: "contain" }} />
                            : <span style={{ color: C.dimmed, fontSize: 20 }}>◌</span>
                          }
                        </div>
                        <div style={{
                          padding: "5px 8px",
                          fontSize: 9, color: active ? tab.color : C.muted,
                          letterSpacing: 1.5, textTransform: "uppercase",
                          textAlign: "center",
                        }}>
                          {tab.icon} {tab.label}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Legend */}
            {result && (
              <div style={{
                display: "flex", gap: 20, flexWrap: "wrap",
                background: C.surface, borderRadius: 12,
                border: `1px solid ${C.border}`,
                padding: "12px 20px", alignItems: "center",
              }}>
                <span style={{ fontSize: 10, color: C.muted, letterSpacing: 2, textTransform: "uppercase" }}>
                  Legend
                </span>
                {[
                  { color: C.red,    label: "Lines" },
                  { color: C.green,  label: "Circles" },
                  { color: C.violet, label: "Ellipses" },
                ].map(({ color, label }) => (
                  <div key={label} style={{ display: "flex", alignItems: "center", gap: 7 }}>
                    <div style={{
                      width: 28, height: 3, background: color, borderRadius: 2,
                      boxShadow: `0 0 6px ${color}`,
                    }} />
                    <span style={{ fontSize: 12, color: C.muted }}>{label}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Empty state */}
            {!result && !loading && (
              <div style={{
                flex: 1, display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center",
                color: C.dimmed, gap: 14, paddingTop: 60,
                textAlign: "center",
              }}>
                <div style={{ fontSize: 64, opacity: 0.4 }}>◉</div>
                <div style={{ fontSize: 13, letterSpacing: 2.5, textTransform: "uppercase", color: C.muted }}>
                  Upload an image &amp; run detection
                </div>
                <div style={{ fontSize: 11, color: C.dimmed, maxWidth: 300 }}>
                  Results will appear here with separate views for lines, circles, and ellipses
                </div>
              </div>
            )}

            {/* Spinner */}
            {loading && (
              <div style={{
                flex: 1, display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center",
                color: C.cyan, gap: 16, paddingTop: 60,
              }}>
                <div style={{
                  width: 52, height: 52, borderRadius: "50%",
                  border: `3px solid ${C.border}`,
                  borderTopColor: C.cyan,
                  animation: "spin .75s linear infinite",
                }} />
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                <div style={{ fontSize: 12, letterSpacing: 2.5, textTransform: "uppercase" }}>
                  Running pipeline…
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
