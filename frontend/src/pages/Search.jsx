import { useState } from "react";
import SearchBar from "../components/SearchBar.jsx";
import AnswerCard from "../components/AnswerCard.jsx";
import LoadingState from "../components/LoadingState.jsx";
import ErrorMessage from "../components/ErrorMessage.jsx";
import { search } from "../services/api.js";

export default function Search() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSearch(query) {
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await search(query);
      setResult(data);
    } catch (err) {
      setError(err.message || "The search request failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page">
      <p className="page__eyebrow">Reference desk</p>
      <h1 className="page__title">Ask Guidely</h1>
      <p className="page__subtitle">
        Ask a question in plain language. Answers are drawn only from
        documents your team has uploaded, with sources cited below.
      </p>

      <SearchBar onSearch={handleSearch} isLoading={isLoading} />

      {isLoading && <LoadingState />}
      {!isLoading && error && <ErrorMessage message={error} />}
      {!isLoading && !error && result && <AnswerCard result={result} />}
    </main>
  );
}
