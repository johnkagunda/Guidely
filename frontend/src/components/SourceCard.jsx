export default function SourceCard({ source }) {
  return (
    <div className="source-card">
      <div className="source-card__top">
        <span className="source-card__doc">{source.document}</span>
        <span className="source-card__score">match {(source.score * 100).toFixed(0)}%</span>
      </div>
      {source.section && (
        <div className="source-card__section">{source.section}</div>
      )}
      <p className="source-card__snippet">"{source.snippet}"</p>
    </div>
  );
}
