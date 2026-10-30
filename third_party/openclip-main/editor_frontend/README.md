# OpenClip editor frontend

React + TypeScript + Vite frontend for the OpenClip post-generation clip editor.

## Scripts

- `npm run dev` — local Vite dev server
- `npm run build` — type-check + production build
- `npm run lint` — ESLint
- `npm run test` — Vitest component tests

## Current scope

The frontend now targets the real manifest-backed editor API:

- loads projects from `/api/projects/:project_id`
- edits one active clip at a time
- sends bounds, subtitle, and cover-title updates to the Python service
- queues targeted rerender jobs and polls job status
- falls back to a clearly labeled demo shell only when the editor service is unavailable

The remaining gaps are UX polish and deeper end-to-end media verification, not the basic frontend/service contract.
