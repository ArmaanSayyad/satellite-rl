export default function Legend() {
  return (
    <div className="legend">
      <span className="legend-item">
        <span className="legend-swatch" style={{ background: "#4fd1ff" }} />
        Your satellite
      </span>
      <span className="legend-item">
        <span className="legend-swatch" style={{ background: "#ff8a3d" }} />
        Other object (real conjunction partner)
      </span>
      <span className="legend-item">
        <span className="legend-swatch legend-line" style={{ borderColor: "#ff5c5c" }} />
        Live separation
      </span>
      <span className="legend-item">
        <span className="legend-swatch legend-ring" style={{ borderColor: "#ff4d4d" }} />
        Collision threshold (combined hard-body radius)
      </span>
    </div>
  );
}
