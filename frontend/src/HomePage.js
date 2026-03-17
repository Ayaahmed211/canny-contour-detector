import { useNavigate } from "react-router-dom";

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

export default function HomePage() {
  const navigate = useNavigate();

  const MenuButton = ({ icon, title, description, color, onClick }) => (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        minWidth: 300,
        padding: "48px 32px",
        background: `linear-gradient(135deg, ${C.surface}dd, ${C.surface}aa)`,
        border: `2px solid ${color}40`,
        borderRadius: 16,
        cursor: "pointer",
        transition: "all .3s ease",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 16,
        boxShadow: `0 0 32px ${color}15`,
      }}
      onMouseEnter={e => {
        e.currentTarget.style.background = `linear-gradient(135deg, ${C.surface}, ${C.surface}dd)`;
        e.currentTarget.style.boxShadow = `0 0 48px ${color}35`;
        e.currentTarget.style.borderColor = `${color}80`;
        e.currentTarget.style.transform = "translateY(-4px)";
      }}
      onMouseLeave={e => {
        e.currentTarget.style.background = `linear-gradient(135deg, ${C.surface}dd, ${C.surface}aa)`;
        e.currentTarget.style.boxShadow = `0 0 32px ${color}15`;
        e.currentTarget.style.borderColor = `${color}40`;
        e.currentTarget.style.transform = "translateY(0)";
      }}
    >
      <div style={{ fontSize: 56 }}>{icon}</div>
      <div style={{
        fontSize: 28,
        fontWeight: 700,
        color: color,
        letterSpacing: -0.5,
      }}>
        {title}
      </div>
      <div style={{
        fontSize: 13,
        color: C.muted,
        textAlign: "center",
        lineHeight: 1.5,
      }}>
        {description}
      </div>
    </button>
  );

  return (
    <div style={{
      minHeight: "100vh",
      background: C.bg,
      fontFamily: "'Segoe UI', 'Helvetica Neue', sans-serif",
      color: C.text,
      display: "flex",
      flexDirection: "column",
    }}>
      {/* Topbar */}
      <div style={{
        borderBottom: `1px solid ${C.border}`,
        background: `linear-gradient(180deg, #0d1525 0%, ${C.bg} 100%)`,
        padding: "0 40px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        height: 64,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 36,
            height: 36,
            borderRadius: 10,
            background: `linear-gradient(135deg, ${C.cyan}, ${C.violet})`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 18,
          }}>
            ◉
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: -0.3 }}>
              Edge &amp; Shape Detector
            </div>
            <div style={{ fontSize: 10, color: C.muted, letterSpacing: 2, textTransform: "uppercase" }}>
              Computer Vision Tools
            </div>
          </div>
        </div>
        <div style={{
          fontSize: 10,
          color: C.dimmed,
          letterSpacing: 2,
          textTransform: "uppercase",
          fontFamily: "monospace",
        }}>
          Choose Algorithm
        </div>
      </div>

      {/* Main content */}
      <div style={{
        flex: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "60px 40px",
      }}>
        <div style={{
          display: "flex",
          gap: 32,
          flexWrap: "wrap",
          justifyContent: "center",
          maxWidth: 1000,
        }}>
          <MenuButton
            icon="〰"
            title="Canny Edges"
            description="Detect edges and edges, lines, circles, and ellipses"
            color={C.cyan}
            onClick={() => navigate("/canny")}
          />
          <MenuButton
            icon="◯"
            title="Active Contour"
            description="Coming soon..."
            color={C.orange}
            onClick={() => alert("Active Contour mode coming soon!")}
          />
        </div>
      </div>
    </div>
  );
}
