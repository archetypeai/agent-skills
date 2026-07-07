---
name: atai-design-system
description: >
  Build the front-end for a Newton demo with the Archetype AI Design
  System — the published, versioned packages (`@archetypeai/ds-lib-tokens`,
  `@archetypeai/ds-ui-svelte-console`, `@archetypeai/ds-ui-svelte-labs`)
  and the `ds` scaffolding CLI (`@archetypeai/ds-cli`) — instead of
  hand-rolling tokens, components, fonts, or brand styling. Use this skill
  when the user wants to scaffold a new dashboard/UI, add the design system
  to an existing SvelteKit app, or pull in branded primitives (menubar,
  logo, sensor/scatter charts, video player, card, badge, table, dialog…).
  The CLI installs the design system's OWN agent config — `ds-manifest.json`
  plus skills and rules into `.claude/` — which is the source of truth for
  component usage, fonts, fallback behavior, and styling once scaffolded;
  this skill's only job is to get you there. The UI is always built on the
  Svelte stack — Svelte 5 + SvelteKit + Tailwind v4 + the DS packages.
  Do NOT use for Newton API / backend work (see the `atai-newton-*` skills).
---

# Archetype AI Design System — Scaffold, Don't Reinvent

There is a real, published, MIT-licensed design system on npm. Front-ends are **assembled by installing it**, not by transcribing tokens or copy-pasting component markup. The `ds` CLI scaffolds a SvelteKit + Tailwind v4 project, installs the packages, wires the order-critical CSS, and drops the design system's own agent configuration (`ds-manifest.json` + skills + rules) into your repo. After that, **defer to that shipped config** — it owns components, fonts, fallback behavior, and styling. This file only gets you to the scaffold.

**Every UI built with this skill targets the Svelte stack** — Svelte 5 + SvelteKit + Tailwind v4 + the DS packages, exactly what `ds create` scaffolds. Build Newton demo front-ends there; don't reach for React, Vue, or server-rendered templates.

## When to Apply

- User wants to start a new Newton demo front-end / dashboard / monitoring UI
- User wants to add Archetype AI branding + tokens to an existing SvelteKit app
- User wants branded components: menubar, logo, cards, badges, tables, dialogs, sensor/scatter/area charts, a video player, a code block, toasts

**Do not use this skill when:**
- The task is Newton API / model work — text·image·video reasoning, embeddings, KNN, data prep. Use the `atai-newton-fusion-model`, `atai-newton-omega-model`, or `atai-newton-omega-model-data-prep` skills ([source](https://github.com/archetypeai/agent-skills/tree/main/skills)).

## The System (v0.8.x, MIT)

Four packages under the `@archetypeai/` scope:

| Package | Role | What you get |
|---------|------|--------------|
| `@archetypeai/ds-lib-tokens` | Tailwind v4 theme | OKLCH **semantic** tokens (`background`, `foreground`, `card`, `primary`, `muted`, `destructive`, `border`, `ring`, `chart-1..5`, `sidebar`, `atai-{neutral,good,warning,critical}`) + **8 brand palettes** (`atai-{tangerine,sunshine-yellow,lime,screen-green,fire-red,energy-pink,cool-purple,baby-blue}`, shades 50–950) + spacing/radius/icon-stroke scales + `.dark` variant + font-family stacks with system fallbacks + a **semantic HTML base layer** that styles bare `h1`–`h6`/`p`/`code` on-brand. Pure CSS — no JS. |
| `@archetypeai/ds-ui-svelte-console` | Stable primitives | ~24 Svelte 5 components: `alert`, `badge`, `button`, `card`, `checkbox`, `codeblock`, `collapsible`, `dialog`, `dropdown-menu`, `dropzone`, `empty-state`, `input`, `input-group`, `item`, `label`, `progress`, `select`, `separator`, `sonner`, `spinner`, `table`, `tabs`, `textarea`, `tooltip` — plus a `theme` helper and `utils` (`cn`). |
| `@archetypeai/ds-ui-svelte-labs` | Experimental primitives (composes console) | `aspect-ratio`, `chart`, `kbd`, **`logo`**, **`menubar`**, **`scatter-chart`**, **`sensor-chart`**, `slider`, `switch`, `toggle`, **`video-player`**. Charts are built on `layerchart` (pinned to an exact prerelease). |
| `@archetypeai/ds-cli` (`ds`) | Scaffold + agent config | `create` / `init` / `add`. Installs the packages + peers, writes the order-critical `layout.css`, sets up `shadcn-svelte`, and installs the design system's **own** agent config. |

Import subpaths are per-component, e.g. `import { Button } from "@archetypeai/ds-ui-svelte-console/primitives/button"`, `import { Menubar } from "@archetypeai/ds-ui-svelte-labs/primitives/menubar"`. The authoritative catalog of every component, its import path, variant axes, and usage rules is `ds-manifest.json` (installed by the CLI) — read that, don't guess.

## Assembly Path — the `ds` CLI

This is how a front-end is assembled. Prefer it over manual installation.

**Always invoke the CLI with the `@latest` tag** (`npx @archetypeai/ds-cli@latest …`). Without it, `npx` silently reuses whatever `ds-cli` version it has cached, which can scaffold against stale packages, an outdated `ds-manifest.json`, or a superseded `layerchart` pin. `@latest` forces npx to resolve the newest published CLI on every run.

**Confirm the scaffold location before running `create`.** `ds create <name>` creates a **new folder named `<name>` in the current working directory** — it does not prompt for placement. Before running it, check the cwd and confirm the parent directory and project name with the user, so the app doesn't land somewhere unexpected (e.g. inside a skills directory or another repo). For adding the system to an existing project, `cd` into that project first and use `init`, not `create`.

```bash
# New project: SvelteKit (TS) + Tailwind v4 + tokens + console + labs + a demo page
npx @archetypeai/ds-cli@latest create my-app --codeagent claude

# Existing SvelteKit project: add the system in place (no demo page)
cd my-existing-app && npx @archetypeai/ds-cli@latest init --codeagent claude

# Granular adds
npx @archetypeai/ds-cli@latest add ds-ui-svelte          # editable component SOURCE (modify workflow)
npx @archetypeai/ds-cli@latest add ds-lib-tokens         # tokens + CSS imports only
npx @archetypeai/ds-cli@latest add ds-config-codeagent --claude   # (re)install agent config only
```

`create` does the full job: scaffolds the SvelteKit app, patches `svelte.config.js` so Svelte 5 runes work inside third-party packages, installs all three DS packages + peers (pinning `layerchart` to the exact prerelease the labs charts need), initializes `components.json` + `src/lib/utils.ts` (one `cn` in the tree), writes the order-critical `src/routes/layout.css`, installs the agent config, and writes a demo `+page.svelte`.

The CLI is **agent-aware**: with no TTY or `--defaults` it lists the options it still needs (`--pm`, `--fonts`/`--no-fonts`, `--codeagent`) and exits, so you ask the user before proceeding rather than guessing. Fold the target directory / project name into that same confirmation.

### Two install models

- **Compose from npm packages (default).** `create`/`init` import components from the installed packages; upgrades come via `npm update`. This is what you want unless the user needs to edit component internals.
- **Source install / "modify" workflow.** `ds add ds-ui-svelte` vendors **editable** component source into the project via `shadcn-svelte`, pulled from the two registries (labs: `https://design-system-labs.archetypeai.workers.dev`, console: `https://design-system-console.archetypeai.workers.dev`). Individual components: `npx shadcn-svelte@latest add <registry>/r/<name>.json`.

## After Scaffolding, Defer to the Shipped Config

`--codeagent claude` writes, to the project root and `.claude/`:

- **`ds-manifest.json`** — machine-readable catalog of every component in both tiers (import subpaths, variant axes, registry source URLs, usage rules). **This is the component source of truth.** When building UI, read it instead of inventing prop/variant names.
- **Rules** — `accessibility`, `charts`, `components`, `design-principles`, `frontend-architecture`, `linting`, `state`, `styling`. Hierarchy, mono-on-technical-data, status-color semantics, radius, layout, fonts, and fallback behavior are encoded here, versioned with the packages.
- **Skills** — `apply-ds`, `build-component`, `create-dashboard`, `setup-chart`, `deploy-worker`, `fix-accessibility`, `fix-metadata`. Use `create-dashboard` for full-viewport monitoring layouts, `build-component` for composing new pieces from primitives, `setup-chart` for the chart primitives.

Composite demo pieces that are **not** standalone primitives — a replay/scrubber bar, a per-subsystem stage panel, a log-line item, a status-badge grid — are built on top of the primitives (`slider` + `button` + `card` + `badge`) via `build-component` / `create-dashboard`. Don't expect them as imports; do reuse the primitive layer rather than raw markup.

## The Stack

Every UI is **Svelte 5 + SvelteKit + Tailwind v4 + bits-ui** with the DS packages on top — the stack `ds create` scaffolds. Build Newton demo front-ends here; don't target another framework, and use the published primitives rather than hand-building against the raw tokens or copying assets into the repo. Everything the demos need (logo, charts, menubar, …) ships as a primitive.

## Common Pitfalls

- **Don't transcribe tokens, fonts, or component styles by hand.** They live in the packages and `ds-manifest.json`. A hand-copied second source of truth silently drifts. Read the manifest; install the package.
- **Prefer the CLI to manual installs.** It applies the `svelte.config.js` runes patch, pins `layerchart` to the exact prerelease, and keeps a single `cn` in the tree — replicate all three if you install by hand or the package components won't compile / charts break.
- **Always the Svelte stack.** Build the UI with the `ds-ui-svelte-*` primitives on SvelteKit; don't target another framework or hand-roll markup against the raw tokens.
- **Always pin `@latest` on the CLI.** Run `npx @archetypeai/ds-cli@latest …`, never bare `npx @archetypeai/ds-cli`. Bare npx reuses a cached CLI and can scaffold against stale packages, manifest, or the wrong `layerchart` pin.
- **Confirm where you scaffold.** `ds create <name>` drops a new `<name>/` folder into the current working directory without prompting. Check the cwd and confirm the parent dir + project name with the user first; use `init` (after `cd`-ing in) to add the system to an existing project.

## File Layout

```
skills/atai-design-system/
└── SKILL.md   ← this file (delegates to the published packages + the CLI-installed ds-manifest.json/skills/rules)
```

There are no reference scripts here by design: the design system is the published npm packages and the `ds` CLI, and component usage, fonts, and styling are owned by the `ds-manifest.json` + skills + rules the CLI installs into your project. This skill points at those rather than duplicating them.
