# Frontend Changes

---

## Feature 1 — Sidebar Toggle Button

### Summary
Added a sidebar toggle button that collapses and expands the left sidebar panel with a smooth animated transition.

### Files Changed

#### `frontend/index.html`
- Added `<button type="button" class="sidebar-toggle" id="sidebarToggle">` with three `<span>` children (hamburger lines) above `<aside class="sidebar">`.
- Added `id="sidebar"` to `<aside>` so JS can target it.
- Bumped cache-busting version from `v=9` to `v=10`.

#### `frontend/style.css`
- **`.sidebar-toggle`** — fixed-position button (top-left) with border-radius, hover/focus states, and flex column layout for the three hamburger lines.
- **`.sidebar-toggle span`** — each line transitions via `transform` and `opacity`.
- **`.sidebar-toggle[aria-expanded="true"]`** — animates the three lines into an × (cross) shape when sidebar is open.
- **`.sidebar`** — added `transition` on `width`, `padding`, and `opacity`; `padding-top: 4rem` to clear the toggle button.
- **`.sidebar.collapsed`** — collapses to `width: 0`, `padding: 0`, `opacity: 0`.
- **Mobile** — toggle repositions to bottom-left; `.sidebar.collapsed` uses `max-height: 0`.

#### `frontend/script.js`
- Added `sidebarToggle` and `sidebar` DOM element references.
- Added `click` listener that toggles `.collapsed` on sidebar and flips `aria-expanded`.

### Behaviour
- Default: sidebar open, button shows × icon. Click to collapse → hamburger icon.
- All transitions are CSS-driven (300 ms ease).
- `aria-expanded` kept in sync for accessibility.

---

## Feature 2 — Light Theme Variant

### Summary
Added a full light theme with accessible color contrast, a fixed theme-toggle button (top-right), and localStorage persistence so the user's preference survives page reloads.

### Files Changed

#### `frontend/index.html`
- Added `<button type="button" class="theme-toggle" id="themeToggle">` containing two inline SVGs — a moon icon (visible in dark mode) and a sun icon (visible in light mode).
- Bumped cache-busting version to `v=11`.

#### `frontend/style.css`
- **`[data-theme="light"]` block** — overrides all CSS custom properties:
  - `--background: #f8fafc` / `--surface: #ffffff` / `--surface-hover: #f1f5f9`
  - `--text-primary: #0f172a` / `--text-secondary: #475569` (≥4.5:1 contrast ratio on light backgrounds)
  - `--border-color: #e2e8f0`
  - `--primary-color: #1d4ed8` / `--primary-hover: #1e40af` (deeper blue for AA contrast on white)
  - `--user-message: #1d4ed8` / `--assistant-message: #f1f5f9`
  - `--shadow`, `--focus-ring`, `--welcome-bg`, `--welcome-border` adjusted accordingly.
- **`body`** — added `transition: background-color 0.25s ease, color 0.25s ease` for smooth theme switch.
- **`.theme-toggle`** — fixed-position button (top-right); hover rotates the SVG icon 20°.
- **`.icon-moon` / `.icon-sun`** — visibility swapped via `display` depending on `[data-theme="light"]`.

#### `frontend/script.js`
- Added `themeToggle` DOM reference.
- On `DOMContentLoaded`: reads `localStorage.getItem('theme')` and applies `data-theme` to `<html>` before first paint (avoids flash of wrong theme).
- `click` listener: toggles `data-theme` attribute between `"light"` and `"dark"` on `document.documentElement` and writes the choice to `localStorage`.

### Behaviour
- Default: dark theme (no `data-theme` attribute → `:root` dark variables).
- Clicking the moon/sun button switches theme instantly with a 250 ms CSS cross-fade.
- Chosen theme persists across page reloads via `localStorage`.
- Accessible: primary blue (`#1d4ed8`) on white background meets WCAG AA (4.58:1); dark text `#0f172a` on `#f8fafc` is ≥15:1.
