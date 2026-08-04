import { useEffect, useState, useCallback } from "react";
import DocumentCard from "../components/DocumentCard.jsx";
import LoadingState from "../components/LoadingState.jsx";
import ErrorMessage from "../components/ErrorMessage.jsx";
import {
  listDocuments,
  uploadDocument,
  deleteDocument,
  reindexDocument,
} from "../services/api.js";

export default function Admin() {
  const [documents, setDocuments] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [busyId, setBusyId] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await listDocuments();
      setDocuments(data.documents);
    } catch (err) {
      setError(err.message || "Could not load documents.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      await uploadDocument(file);
      await refresh();
    } catch (err) {
      setUploadError(err.message || "Upload failed.");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function handleDelete(id) {
    setBusyId(id);
    try {
      await deleteDocument(id);
      await refresh();
    } catch (err) {
      setError(err.message || "Delete failed.");
    } finally {
      setBusyId(null);
    }
  }

  async function handleReindex(id) {
    setBusyId(id);
    try {
      await reindexDocument(id);
      await refresh();
    } catch (err) {
      setError(err.message || "Re-index failed.");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <main className="page">
      <p className="page__eyebrow">Collection management</p>
      <h1 className="page__title">Documents</h1>
      <p className="page__subtitle">
        Upload the source material Guidely should learn from. Unchanged
        documents are skipped automatically on re-upload.
      </p>

      <div className="upload-drop">
        Drop a .txt, .md, .pdf, or .docx file, or choose one below.
        <br />
        <input type="file" accept=".txt,.md,.pdf,.docx" onChange={handleUpload} disabled={uploading} />
        {uploading && <div style={{ marginTop: 8 }}>Uploading and indexing…</div>}
      </div>

      {uploadError && <ErrorMessage message={uploadError} />}

      {isLoading && <LoadingState message="Loading documents…" />}
      {!isLoading && error && <ErrorMessage message={error} />}

      {!isLoading && !error && documents.length === 0 && (
        <p className="empty-state">
          No documents yet. Upload your first document to build the index.
        </p>
      )}

      {!isLoading &&
        documents.map((doc) => (
          <DocumentCard
            key={doc.id}
            doc={doc}
            busy={busyId === doc.id}
            onDelete={handleDelete}
            onReindex={handleReindex}
          />
        ))}
    </main>
  );
}
