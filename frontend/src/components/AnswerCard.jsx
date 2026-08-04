import SourceCard from "./SourceCard.jsx";

export default function AnswerCard({ result }) {
  const { answer, sources, retrieved_chunks, latency_ms } = result;

  return (
    <>
      <div className="answer-card">
        <div className="answer-card__label">Answer</div>
        <div className="answer-card__text">{answer}</div>
        <div className="answer-card__meta">
          {retrieved_chunks} chunk{retrieved_chunks === 1 ? "" : "s"} retrieved ·{" "}
          {latency_ms ? `${Math.round(latency_ms)} ms` : "cached"}
        </div>
      </div>

      {sources.length > 0 && (
        <div className="sources">
          <div className="sources__label">Sources</div>
          {sources.map((s, i) => (
            <SourceCard key={i} source={s} />
          ))}
        </div>
      )}
    </>
  );
}
