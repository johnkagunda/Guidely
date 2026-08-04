import { useEffect, useState } from "react";
import StatsCard from "../components/StatsCard.jsx";
import LoadingState from "../components/LoadingState.jsx";
import ErrorMessage from "../components/ErrorMessage.jsx";
import { getMetrics } from "../services/api.js";

export default function Metrics() {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getMetrics()
      .then(setData)
      .catch((err) => setError(err.message || "Could not load metrics."))
      .finally(() => setIsLoading(false));
  }, []);

  if (isLoading) return (
    <main className="page">
      <LoadingState message="Loading metrics…" />
    </main>
  );

  if (error) return (
    <main className="page">
      <ErrorMessage message={error} />
    </main>
  );

  const errorEntries = Object.entries(data.errors || {}).filter(([, v]) => v > 0);

  return (
    <main className="page page--wide">
      <p className="page__eyebrow">Operations</p>
      <h1 className="page__title">Metrics</h1>
      <p className="page__subtitle">
        Live counters recorded automatically from queries and indexing runs.
      </p>

      <section>
        <div className="stats-grid">
          <StatsCard label="Documents" value={data.documents} />
          <StatsCard label="Chunks" value={data.chunks} />
          <StatsCard label="Queries served" value={data.queries_served} />
          <StatsCard label="Median latency" value={`${data.latency.median_ms} ms`} accent />
          <StatsCard label="P95 latency" value={`${data.latency.p95_ms} ms`} accent />
          <StatsCard
            label="Cache hit rate"
            value={`${Math.round(data.embedding_cache.hit_rate * 100)}%`}
          />
        </div>
      </section>

      <section>
        <p className="sources__label">Indexing</p>
        <div className="stats-grid">
          <StatsCard label="Documents indexed" value={data.indexing.documents_indexed} />
          <StatsCard label="Chunks created" value={data.indexing.chunks_created} />
          <StatsCard label="Skipped (unchanged)" value={data.indexing.skipped_unchanged} />
        </div>
      </section>

      <section>
        <p className="sources__label">Errors</p>
        {errorEntries.length === 0 ? (
          <p className="empty-state">No errors recorded.</p>
        ) : (
          <div className="stats-grid">
            {errorEntries.map(([key, value]) => (
              <StatsCard key={key} label={key.replace(/_/g, " ")} value={value} />
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
