export default function ErrorMessage({ message }) {
  if (!message) return null;
  return (
    <div className="error-message" role="alert">
      <div className="error-message__label">Something went wrong</div>
      {message}
    </div>
  );
}
