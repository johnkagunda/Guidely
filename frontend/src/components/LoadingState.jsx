export default function LoadingState({ message = "Searching the archive…" }) {
  return (
    <div className="loading-state" role="status">
      <span className="loading-state__dot" />
      <span>{message}</span>
    </div>
  );
}
