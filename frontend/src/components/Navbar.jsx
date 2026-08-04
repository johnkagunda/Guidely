import { NavLink } from "react-router-dom";

export default function Navbar() {
  return (
    <header className="navbar">
      <NavLink to="/" className="navbar__brand">
        <span className="navbar__mark">Guidely</span>
        <span className="navbar__tagline">Internal Knowledge Assistant</span>
      </NavLink>
      <nav className="navbar__tabs">
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            "navbar__tab" + (isActive ? " navbar__tab--active" : "")
          }
        >
          Search
        </NavLink>
        <NavLink
          to="/admin"
          className={({ isActive }) =>
            "navbar__tab" + (isActive ? " navbar__tab--active" : "")
          }
        >
          Documents
        </NavLink>
        <NavLink
          to="/metrics"
          className={({ isActive }) =>
            "navbar__tab" + (isActive ? " navbar__tab--active" : "")
          }
        >
          Metrics
        </NavLink>
      </nav>
    </header>
  );
}
