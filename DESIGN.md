# Archetype AI Design System

## Build a front-end

→ **[`skills/atai-design-system/SKILL.md`](skills/atai-design-system/SKILL.md)** — when and how to use it.
Components, tokens, fonts, fallback behavior, and styling are owned by the packages and the agent config
the CLI installs; the skill only gets you to the scaffold.

Quick start:

```bash
npx @archetypeai/ds-cli create my-app --codeagent claude   # new SvelteKit app, fully wired
cd existing-app && npx @archetypeai/ds-cli init --codeagent claude   # add to an existing app
```

`--codeagent claude` installs the design system's own agent configuration into your
project — `ds-manifest.json` (the machine-readable component catalog) plus its skills and
rules — which is the source of truth for component usage once scaffolded.

## The packages (npm, `@archetypeai/`, MIT)

| Package | Role |
|---------|------|
| `@archetypeai/ds-lib-tokens` | Tailwind v4 theme — OKLCH semantic + brand tokens, scales, dark variant, HTML base layer (pure CSS; the only tier a non-Svelte demo can use) |
| `@archetypeai/ds-ui-svelte-console` | Stable Svelte 5 primitives (button, card, badge, dialog, table, …) |
| `@archetypeai/ds-ui-svelte-labs` | Experimental Svelte 5 primitives composing console (logo, menubar, sensor/scatter/area charts, video player, …) |
| `@archetypeai/ds-cli` (`ds`) | Scaffold (`create` / `init` / `add`) + ship the agent config |
