function formatBytes(bytes) {
  if (!bytes) return "0 KB";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

function formatDate(iso) {
  if (!iso) return "never";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function DocumentCard({ doc, onDelete, onReindex, busy }) {
  return (
    <div className="document-card">
      <div>
        <div className="document-card__name">{doc.filename}</div>
        <div className="document-card__details">
          <span className={`status-pill status-pill--${doc.status}`}>{doc.status}</span>
          {"  ·  "}
          {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"}
          {"  ·  "}
          {formatBytes(doc.size_bytes)}
          {"  ·  "}
          last indexed {formatDate(doc.last_indexed_at)}
        </div>
      </div>
      <div className="document-card__actions">
        <button className="btn" disabled={busy} onClick={() => onReindex(doc.id)}>
          Re-index
        </button>
        <button className="btn btn--danger" disabled={busy} onClick={() => onDelete(doc.id)}>
          Delete
        </button>
      </div>
    </div>
  );
}
