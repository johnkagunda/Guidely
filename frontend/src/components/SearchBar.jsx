import { useState } from "react";

export default function SearchBar({ onSearch, isLoading }) {
  const [value, setValue] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || isLoading) return;
    onSearch(trimmed);
  }

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="How many annual leave days do employees receive?"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        aria-label="Ask a question"
      />
      <button type="submit" disabled={isLoading || !value.trim()}>
        {isLoading ? "Asking…" : "Ask"}
      </button>
    </form>
  );
}
