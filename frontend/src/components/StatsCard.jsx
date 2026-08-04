export default function StatsCard({ label, value, accent }) {
  return (
    <div className={`stat-card${accent ? " stat-card--accent" : ""}`}>
      <div className="stat-card__value">{value}</div>
      <div className="stat-card__label">{label}</div>
    </div>
  );
}
