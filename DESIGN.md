# Design System Inspiration of Archetype AI

## 1. Visual Theme & Atmosphere

Archetype AI's design system is built for **Physical AI interfaces** — dashboards, monitoring views, and sensor-driven applications where humans reason about the physical world alongside AI. The aesthetic is dark-first, data-dense, and technically precise — closer to mission control than consumer software.

The foundation is a near-black background (`oklch(0.141 0.005 285.823)`) with light foreground text (`oklch(0.985 0 0)`), creating a high-contrast canvas where data visualizations, status indicators, and sensor readouts command attention. Cards float on a slightly lighter surface (`oklch(0.21 0.006 285.885)`) with `10%` white borders that create subtle depth without distraction.

Typography uses **Geist** — Vercel's open-source geometric sans-serif — paired with **Geist Mono** for technical data. The mono font appears on badges, card headers, numeric readouts, and status labels — never on body text. Headings are set in normal weight (400) with tight tracking, giving them an understated, engineering-grade quality rather than marketing boldness.

The brand's color palette is deliberately scientific: **Baby Blue** for neutral states, **Screen Green** for healthy/good, **Sunshine Yellow** for warnings, and **Fire Red** for critical alerts. A secondary palette of **Cool Purple**, **Energy Pink**, **Tangerine**, and **Lime** provides chart series colors. These are OKLCH-native, perceptually uniform, and designed for data legibility across light and dark modes.

**Key Characteristics:**
- Dark-first design with near-black backgrounds and high-contrast text
- Geist (sans) + Geist Mono (mono) type pairing
- OKLCH color system — perceptually uniform across all palettes
- Semantic status colors: green (good), yellow (warning), red (critical), blue (neutral)
- Data-dense layouts: full-viewport dashboards, no scrolling, 2×2/3×2 panel grids
- Mono font for all technical data: badges, headers, timestamps, scores
- Minimal border-radius (`0.125rem` / 2px) — sharp, technical aesthetic
- Card-based composition with `BackgroundCard` as the default container
- Physical context always visible: sensor IDs, locations, time ranges, camera sources

## 2. Color Palette & Roles

### Semantic Surface Tokens (Dark Mode — Primary)
- **Background**: `oklch(0.141 0.005 285.823)` — near-black page background
- **Card**: `oklch(0.21 0.006 285.885)` — elevated card surfaces
- **Muted**: `oklch(0.274 0.006 286.033)` — secondary surfaces, inactive areas
- **Border**: `oklch(1 0 0 / 10%)` — subtle white borders
- **Input**: `oklch(1 0 0 / 15%)` — input field backgrounds

### Semantic Surface Tokens (Light Mode)
- **Background**: `oklch(1 0 0)` — pure white
- **Card**: `oklch(0.99 0 0)` — near-white cards
- **Muted**: `oklch(0.967 0.001 286.375)` — light gray surfaces
- **Border**: `oklch(0.92 0.004 286.32)` — light gray borders

### Text Scale
- **Foreground (dark)**: `oklch(0.985 0 0)` — near-white primary text
- **Muted Foreground (dark)**: `oklch(0.705 0.015 286.067)` — secondary text, descriptions
- **Foreground (light)**: `oklch(0.141 0.005 285.823)` — near-black primary text
- **Muted Foreground (light)**: `oklch(0.552 0.016 285.938)` — gray secondary text

### Status System (ATAI Brand)
- **Good / Healthy**: Screen Green 200 `oklch(0.822 0.208 146.907)` — normal operations, passing checks
- **Warning**: Sunshine Yellow 50–100 `oklch(0.95 0.18 110.904)` — attention needed, degraded performance
- **Critical**: Fire Red 500 `oklch(0.641 0.214 25.595)` — errors, failures, incidents
- **Neutral / Info**: Baby Blue 300 `oklch(0.794 0.091 250.497)` — informational, no severity

### Chart Series
- **Chart 1**: Cool Purple 500 `oklch(0.66 0.177 299.333)` — primary data series
- **Chart 2**: Fire Red 500 `oklch(0.641 0.214 25.595)` — secondary series
- **Chart 3**: Sunshine Yellow 500 `oklch(0.63 0.12 110.904)` — tertiary series
- **Chart 4**: Baby Blue 800 `oklch(0.534 0.129 250.497)` — quaternary series
- **Chart 5**: Energy Pink 600 `oklch(0.603 0.2 5.943)` — fifth series

### Brand Palette (Full Scale, 50–950)
- **Tangerine**: Hue 85.144 — warm amber accent
- **Sunshine Yellow**: Hue 110.904 — warning states, chart color
- **Lime**: Hue 123.573 — secondary green
- **Screen Green**: Hue 146.907 — healthy/good status
- **Fire Red**: Hue 25.595 — critical/destructive states
- **Energy Pink**: Hue 5.943 — chart accent
- **Cool Purple**: Hue 299.333 — primary chart color
- **Baby Blue**: Hue 250.497 — neutral info, sidebar accents

## 3. Typography Rules

### CSS Import Order

```css
@import "@archetypeai/ds-lib-tokens/theme.css";
@import "@fontsource-variable/geist";
@import "@fontsource-variable/geist-mono";
@import "tailwindcss";
@import "tw-animate-css";
```

Order matters — tokens and fonts must come before Tailwind.

> Note for Agent Skills demos: do **not** import `@archetypeai/ds-lib-tokens/fonts.css`. That file expects PP Neue Montreal font files to be copy-pasted into the app — a commercial license we deliberately avoid in public demos. Pull Geist directly from `@fontsource-variable/geist*` instead until the tokens package is updated to ship Geist itself.

### Font Families
- **Sans**: `Geist Variable`, fallbacks: `system-ui, -apple-system, sans-serif`
- **Mono**: `Geist Mono Variable`, fallbacks: `ui-monospace, SFMono-Regular, Menlo, monospace`

Use **Geist Font from Vercel** — official source: [vercel.com/font](https://vercel.com/font#get). Geist is open-source under the SIL Open Font License, so it's safe to bundle in public repos. Install via `npm i @fontsource-variable/geist @fontsource-variable/geist-mono` for framework-agnostic projects (this is the path Vercel themselves recommend outside Next.js); Next.js apps can use Vercel's official `geist` package directly. If the font fails to load, the fallback stack (`system-ui` for sans, `ui-monospace` for mono) renders the UI without licensing risk and without breaking the visual hierarchy.

### Font Weights Available
Geist supports the full 100–900 weight axis (variable). Named cuts:
- **Sans**: 100 (Thin), 200 (UltraLight), 300 (Light), 400 (Regular), 500 (Medium), 600 (SemiBold), 700 (Bold), 800 (Black), 900 (UltraBlack)
- **Mono**: 100 (Thin), 200 (UltraLight), 300 (Light), 400 (Regular), 500 (Medium), 600 (SemiBold), 700 (Bold), 800 (Black), 900 (UltraBlack)

### Heading Hierarchy

| Element | Font | Size | Weight | Line Height | Tracking | Transform | Notes |
|---------|------|------|--------|-------------|----------|-----------|-------|
| `h1` | Geist | 2.25rem (36px) | 400 (normal) | normal | tight | capitalize | Primary page title |
| `h2` | Geist | 1.875rem (30px) | 400 (normal) | tight | tight | — | Section heading |
| `h3` | Geist | 1.5rem (24px) | 400 (normal) | normal | tight | — | Subsection heading |
| `h4` | Geist | 1.25rem (20px) | 400 (normal) | tight | tight | — | Muted foreground color |
| `h5` | Geist | 1.125rem (18px) | 400 (normal) | tight | tight | uppercase | Muted, uppercase label |
| `h6` | Geist | 1rem (16px) | 400 (normal) | tight | tight | — | Muted smallest heading |
| `p` | Geist | 0.875rem (14px) | 400 | tight | normal | — | Body text |
| `small` | Geist | 0.75rem (12px) | 400 | tight | normal | — | Captions, fine print |
| `code` | Geist Mono | 0.875rem (14px) | 400 | relaxed | normal | — | Inline code |

### Mono Font Usage (Critical Convention)
The mono font (`font-mono`) is used deliberately on specific UI elements — never as body text:
- **Badge labels**: status text, category tags
- **Card headers**: BackgroundCard titles (uppercase, tracking-wider)
- **Button labels**: all button text
- **Numeric values**: scores, percentages, counts, timestamps
- **Status indicators**: health scores, sensor readings
- **Table headers**: data table column labels

### Principles
- **Normal weight headings**: All headings use weight 400 (normal) — no bold headings. The hierarchy comes from size and color, not weight.
- **Tight tracking on headings**: `tracking-tight` creates a technical, precise feel.
- **Muted lower headings**: `h4`, `h5`, `h6` use `text-muted-foreground` — headings de-escalate in visual prominence.
- **Uppercase sparingly**: Only `h5` and BackgroundCard titles use uppercase. Combined with mono and tracking-wider for a label/category effect.

## 4. Component Stylings

### Buttons

**Default (Primary)**
- Background: `bg-primary` (near-black in light, light gray in dark)
- Text: `text-primary-foreground`
- Radius: `rounded-xs` (2px)
- Height: `h-9` (36px), padding: `px-4`
- Hover: `bg-primary/90`
- Shadow: `shadow-xs`

**Outline**
- Background: `bg-background`, border: `border-border`
- Dark: `bg-input/30`, `border-input`, hover `bg-input/50`
- Radius: `rounded-xs` (2px)

**Ghost**
- Transparent background
- Hover: `bg-accent` with `text-accent-foreground`

**Destructive**
- Background: `bg-destructive` (Fire Red 500)
- Text: white
- Dark: `bg-destructive/60`

**Sizes**: default (`h-9`), sm (`h-8`), lg (`h-10`), icon (`size-9`), icon-sm (`size-8`), icon-lg (`size-10`)

### Cards & Containers

**BackgroundCard** (default container for single-purpose panels)
- Padding: `p-4`
- Background: `bg-card` semantic token
- Border: 1px `border-border`
- Radius: `rounded-xs` (2px)
- Header: mono font, uppercase, `tracking-wider`, with optional Lucide icon
- Content: `flex flex-col gap-6`

**Card**
- Background: `bg-card text-card-foreground`
- Radius: `rounded-xs`
- Border: 1px
- Shadow: `shadow-sm`
- Layout: `flex flex-col gap-6 py-6`

### Badges

**Default Badge**
- Radius: `rounded-md` (6px) — the one element with rounded corners
- Font: `font-mono text-xs`
- Padding: `px-2.5 py-0.5`
- Variants: default, secondary, outline, destructive

**StatusBadge Pattern**
- Grid layout: avatar + label + percentage
- Avatar with status-colored background (green/yellow/red)
- Label: mono, uppercase, truncated
- Score: mono, right-aligned

### FlatLogItem Pattern
- Left color stripe (2px) — `bg-atai-good`, `bg-atai-warning`, `bg-atai-critical`, or `bg-muted-foreground`
- Status badge with icon: CircleCheck (good), TriangleAlert (warning), CircleX (critical), Info (neutral)
- Badge text: mono, uppercase, colored background matching status
- Body text: `text-muted-foreground`, `whitespace-pre-wrap`
- Detail (timestamp): `font-mono text-sm`, right-aligned

### Inputs
- Background: transparent
- Border: `border-input`
- Radius: `rounded-xs`
- Focus: `ring-ring/50`, 3px ring
- Invalid: `aria-invalid:border-destructive`, `aria-invalid:ring-destructive/20`

### PlaybackBar (replay / time-scrub control)

Used in any demo that replays a time-indexed dataset — `archetypeai-swat-demo` (10× SWaT replay), `archetypeai-wifi-demo` (per-window walkthrough), `archetypeai-obd2-demo` (1×–20× session playback). A horizontal strip immediately under the Menubar (`border-b px-4 py-2`), with these elements left-to-right:

- **Play / pause icon button** — `Button variant="outline" size="icon"`, swaps a `Play` and `Pause` Lucide icon
- **Reset icon button** — same `outline icon`, `RotateCcw` Lucide icon
- **Time readout** — mono, tabular nums, e.g. `52:40 / 52:40`. `text-muted-foreground` with the separator slash at `opacity-50`.
- **Scrubber** — `<input type="range">` (`flex-1`) styled with a `border`-colored track and a `bg-foreground` circular thumb ringed by 2px `bg-background`
- **Speed selector** — segmented group of buttons inside a 1px `border` container, mono uppercase labels (`1× 2× 3× 5× 10× 20×`), active state uses `bg-accent` + `text-foreground`, rest uses `text-muted-foreground`

Separating the bar from the panel grid (own row, own border) keeps temporal control out of the data area and makes it obvious it acts globally on every panel.

### Icons (Lucide)
- Default stroke width: 1 (`--atai-icon-stroke-default`)
- Interactive: 1.25 (`--atai-icon-stroke-interactive`)
- Status indicators: 1.5 (`--atai-icon-stroke-status`)
- Emphasis/action: 2 (`--atai-icon-stroke-emphasis`)
- Sizing: `size-4` (16px) in buttons, `size-6` (24px) in card headers

### Schematic Equipment Icons (custom SVG)

For domain hardware that doesn't exist in Lucide — tanks, ultrafiltration units, reverse-osmosis stages, drilling rigs, pumps, dosing skids — use hand-drawn schematic SVGs rather than reaching for a stock library. The convention is consistent across `archetypeai-swat-demo`, `archetypeai-drilling-demo`, and the other Newton demos that visualize physical equipment:

```svelte
<svg
  viewBox="0 0 120 40"
  class="text-muted-foreground h-10 w-full"
  preserveAspectRatio="xMidYMid meet"
  fill="none"
  stroke="currentColor"
  stroke-width="1.25"
  stroke-linecap="round"
  stroke-linejoin="round"
>
  <!-- equipment paths, e.g. raw-water tank with outflow pipe: -->
  <rect x="18" y="8" width="36" height="26" rx="1" />
  <path d="M 22 18 Q 27 15 32 18 T 42 18 T 50 18" stroke-width="0.75" />
  <line x1="54" y1="26" x2="110" y2="26" />
  <path d="M 106 22 L 110 26 L 106 30" />
</svg>
```

Conventions:
- **viewBox**: `0 0 120 40` (3:1) for a stage / process unit; `0 0 24 24` for inline icons.
- **Stroke**: `currentColor` with `stroke-width="1.25"` matching the interactive icon weight from Lucide. Thinner accents (water surface, dotted flow) drop to `0.75`.
- **Caps & joins**: always `round`.
- **Fill**: `none` — equipment is line-art, not filled shapes. The dark canvas does the visual work.
- **Color**: `text-muted-foreground` by default; switch to `text-atai-critical/70` (or another semantic status) when the stage is anomalous so the schematic reads as a status surface, not just decoration.
- **Aspect**: `preserveAspectRatio="xMidYMid meet"` so the schematic centers cleanly when the container width changes.

If you have a vector source from the customer (e.g. a P&ID excerpt) you can adapt that into this style by stripping fills, normalizing strokes to 1.25, and snapping the geometry to a 120×40 grid.

## 5. Layout Principles

### Spacing System
- **xs**: 0.25rem (4px)
- **sm**: 0.5rem (8px)
- **md**: 0.75rem (12px)
- **lg**: 1rem (16px)
- **xl**: 1.5rem (24px)
- Standard Tailwind spacing scale also available

### Dashboard Layout (Primary Pattern)
- Full viewport: `h-screen w-screen overflow-hidden` — no page scrolling
- Grid: `grid-rows-[auto_1fr]` — fixed menubar + flexible content area
- Content: `grid grid-cols-2 grid-rows-2 gap-4 p-4` — 2×2 panel grid
- Each panel: `max-h-full` with internal scroll via `ScrollArea`

### Split-pane Layout (Sensor / Analysis Pattern)

When the demo has a clear "raw input | model interpretation" duality — timeline of sensor values on one side, model output on the other — use a horizontal split instead of the 2×2 grid:

- Outer grid: `grid-rows-[auto_auto_1fr]` — Menubar / PlaybackBar / content
- Content grid: `grid-cols-[240px_1fr_1fr] gap-3 p-3` — sidebar (session list) + left pane (raw signals) + right pane (analysis)
- Each pane handles its own internal scroll; the page never scrolls
- Both data panes are visually weighted equally (`1fr` each) — neither side is "subordinate"

`archetypeai-obd2-demo` uses this for raw OBD-II playback on the left and the Omega embedding trajectory on the right, time-synced through a shared scrubber. The pattern generalizes to any "compare what the sensors saw against what the model concluded" case.

### Menubar

Identical across `archetypeai-swat-demo`, `archetypeai-drilling-demo`, `archetypeai-wifi-demo`, `archetypeai-grid-demo`, `archetypeai-wildfire-demo` — treat this as the canonical pattern, copy verbatim rather than re-inventing:

```svelte
<header class="border-border flex items-center justify-between border-b px-4 py-2">
  <div class="flex items-center gap-3">
    <Logo class="h-6" />
    <SeparatorIcon class="text-muted-foreground size-6" strokeWidth={1} />
    {@render partnerLogo()}    <!-- or a Badge variant="outline" placeholder -->
  </div>
  <div class="flex items-center gap-2">
    {@render children()}        <!-- action buttons, status pills, etc. -->
    <Button variant="outline" size="icon" onclick={toggleDark}>
      {#if darkMode}<SunIcon />{:else}<MoonIcon />{/if}
    </Button>
  </div>
</header>
```

Key details:
- Full-bleed (`border-b`, no max-width container) — the menubar always spans the viewport.
- **Wordmark asset:** [`assets/archetype-wordmark.svg`](../../assets/archetype-wordmark.svg) in this repo. Single 190×35 `viewBox`, one `fill="currentColor"` path (~14 KB). The path is achromatic — it renders in whatever `color` the *surrounding* element resolves to, so the same file works on both backgrounds (in dark mode, give it `text-foreground` ≈ near-white; in light mode, `text-foreground` ≈ near-black; the wordmark flips automatically with the theme). **Inline** the SVG (drop the `<svg>` element directly into your JSX/Svelte) so the `currentColor` cascade works — `<img src="…">` and `background-image: url(…)` ignore the parent's `color`, so the wordmark renders as nothing visible. The seven Newton demos in this catalogue (swat, drilling, wifi, grid, wildfire, traffic, earthquake) each hand-inlined this same SVG in their own `src/lib/components/ui/patterns/logo/logo.svelte` — those copies are byte-identical, but the version in `assets/` is now the single source of truth, so future demos should reference *that* file rather than copy from a sibling demo. `@archetypeai/ds-lib-tokens` ships theme/token CSS but does **not** include the wordmark asset.
- Separator is a Lucide `Minus` icon at `strokeWidth={1}` and `size-6` — *not* a CSS divider or pipe character.
- Partner branding goes to the right of the separator. When you don't have one, a `<Badge variant="outline">Partner Logo</Badge>` is the standard placeholder so the slot stays visible during development.
- The dark-mode button is a `Button variant="outline" size="icon"` with Sun/Moon swap, wrapped in `document.startViewTransition` so the canvas crossfades instead of flicker-swapping.
- Right side accepts arbitrary action content via a slot/snippet — a "Start analysis" pill, a status badge ("API READY"), a "Logout" button, etc., all to the *left* of the dark-mode toggle.

### Stage / Pipeline Panel Pattern

When the demo shows a multi-stage physical process (water-treatment stages, drilling-rig subsystems, manufacturing line cells), each stage gets a dedicated `BackgroundCard` laid out as a column of: **stage code + status dot + status badge → schematic SVG → mono stage name → tabular readouts → optional history strip**. The pattern is from `archetypeai-swat-demo`'s `stage-card.svelte` (P1–P6 of the water-treatment plant) and transfers wholesale to any per-subsystem dashboard.

```svelte
<BackgroundCard class="flex flex-col gap-3 p-4">
  <header class="flex items-center justify-between">
    <div class="flex items-center gap-2">
      <span class="text-muted-foreground font-mono text-sm">{stageId}</span>
      <span class="size-2 rounded-full {tokens.dot}"></span>      <!-- status dot -->
    </div>
    <Badge variant="outline" class="font-mono text-xs {tokens.pill}">{tokens.label}</Badge>
  </header>

  <StageSchematic stageId={stageId} class={status === 'attack' && 'text-atai-critical/70'} />

  <p class="font-mono text-sm leading-tight">{stageName}</p>

  <dl class="flex flex-col gap-1 text-xs">
    {#each columns as col}
      <div class="flex items-baseline justify-between gap-2">
        <dt class="text-muted-foreground font-mono">{col}</dt>
        <dd class="font-mono">{fmt(liveRow?.[col])}</dd>
      </div>
    {/each}
  </dl>

  <!-- optional: recent-classifications dot strip mt-auto -->
</BackgroundCard>
```

Composition notes:
- The card padding is the standard `p-4`, but the *internal* gap shrinks to `gap-3` (vs `gap-6` in a generic BackgroundCard) — stage panels are intentionally denser.
- **Stage code + dot** sit on the left; **status badge** on the right. The dot is `size-2` rounded-full filled with a semantic status token (`bg-atai-good`, `bg-atai-critical`, `bg-atai-warning`, `bg-muted`).
- **Stage name uses mono** (`font-mono text-sm`) — this is an exception to the "mono only on technical data" rule, justified because stage names ("Raw water intake", "UV dechlorination") read more like equipment IDs than prose.
- A dedicated `STATUS_TOKEN` map keeps dot color + pill color + label in lockstep so adding a new status (`warmup`, `pending`, `standby`, `unmonitored`, `idle`) is one record, not three.

### Readout Tables (Tabular Sensor Values)

The `<dl>` block inside a stage panel is the canonical "label-value" readout — used wherever you want a tight column of sensor IDs and their current values without a true table's chrome:

```svelte
<dl class="flex flex-col gap-1 text-xs">
  <div class="flex items-baseline justify-between gap-2">
    <dt class="text-muted-foreground font-mono">FIT101</dt>
    <dd class="font-mono">2.42</dd>
  </div>
  …
</dl>
```

Conventions:
- Both `dt` and `dd` are `font-mono`. The label is `text-muted-foreground`; the value is full-strength `text-foreground`.
- `text-xs` (12px) is the default density. Bump to `text-sm` only when the panel is the entire dashboard.
- Rows are `flex items-baseline justify-between` so the decimal points roughly stack visually without a true tabular-num font feature.
- Value formatting (from `stage-card.svelte`):

  ```js
  function fmt(v) {
    if (v === undefined || v === null || v === '') return '—';
    const n = parseFloat(v);
    if (isNaN(n)) return String(v);
    if (Math.abs(n) >= 1000) return n.toFixed(0);  // 1450
    if (Math.abs(n) >= 10)   return n.toFixed(1);  // 521.3
    return n.toFixed(2);                            //   2.42
  }
  ```

  The three-tier rounding keeps the column visually balanced at any scale and gives unknowns a single em-dash glyph instead of "N/A" or "null".

### Whitespace Philosophy
- **Data density over decoration**: Panels use every pixel — content areas flex to fill, no decorative whitespace.
- **Gap-based spacing**: Consistent `gap-4` between panels, `gap-6` inside cards. No margin-based spacing.
- **Overflow management**: Each panel handles its own scroll. The page never scrolls.
- **Padding consistency**: `p-4` on cards and main content area.

### Border Radius Scale
- **Interactive (xs)**: 0.125rem (2px) — buttons, cards, inputs, all interactive elements
- **Badge (md)**: 0.375rem (6px) — badges only
- **Full**: 9999px — avatars, circular indicators

## 6. Depth & Elevation

| Level | Treatment | Use |
|-------|-----------|-----|
| Flat (Level 0) | No shadow, no border | Page background |
| Card (Level 1) | `shadow-sm` + 1px border | Cards, panels, popovers |
| Elevated (Level 2) | `shadow-lg` | Dialogs, sheets, dropdowns |
| Focus Ring | `ring-[3px] ring-ring/50` | Keyboard focus indicators |

**Depth Philosophy**: Archetype AI uses minimal elevation. In dark mode, depth comes primarily from **surface color stepping** (background → card → muted) and **border opacity** (`oklch(1 0 0 / 10%)`) rather than shadows. Shadows are barely perceptible in dark mode — borders do the heavy lifting. This creates a flat, technical aesthetic appropriate for data-dense dashboards.

## 7. Do's and Don'ts

### Do
- Use the dark mode as the primary design target — it's the default for monitoring interfaces
- Apply `font-mono` to badges, card headers, numeric values, timestamps, and buttons — technical UI elements
- Use semantic status tokens (`atai-good`, `atai-warning`, `atai-critical`) only for meaningful states — never decoratively
- Keep all headings at weight 400 (normal) — size and color create hierarchy, not boldness
- Use `rounded-xs` (2px) for all interactive elements — the sharp aesthetic is intentional
- Compose layouts with BackgroundCard as the default panel container
- Show physical context: sensor IDs, camera locations, time ranges, data provenance
- Use `cn()` from `$lib/utils.js` for all class merging — never string concatenation
- Design for full-viewport dashboards with no page scroll
- Frame AI output as observation, not conclusion — the human decides

### Don't
- Don't use mono font for body text, descriptions, or paragraphs — only for technical data elements
- Don't use bold (700) headings — the system uses normal weight throughout
- Don't apply status colors for decoration — green/yellow/red carry severity meaning
- Don't use large border-radius (8px+) on cards or buttons — `rounded-xs` (2px) is the standard
- Don't add shadows for depth in dark mode — use surface color stepping and border opacity instead
- Don't create scrolling pages — use fixed-viewport layouts with internal panel scroll
- Don't use `text-primary` for secondary content — use `text-muted-foreground` for descriptions and metadata
- Don't hardcode colors — always use semantic tokens that adapt to light/dark mode
- Don't add decorative whitespace — every pixel should serve data display or readability

## 8. Responsive Behavior

### Design Target
Archetype AI interfaces are primarily designed for **desktop monitoring environments** — wide screens, high resolution, landscape orientation. The full-viewport dashboard pattern assumes 1280px+ widths.

### Breakpoints (Standard Tailwind)
| Name | Width | Key Changes |
|------|-------|-------------|
| sm | 640px | Stack panels vertically |
| md | 768px | 2-column layouts possible |
| lg | 1024px | Full 2×2 dashboard grid |
| xl | 1280px | Comfortable dashboard density |
| 2xl | 1536px | Extra breathing room |

### Collapsing Strategy
- Dashboard grid: 2×2 → 2×1 → 1×1 stacked columns on smaller screens
- Menubar: logo + controls stay; partner branding may collapse
- BackgroundCard: panels become full-width and individually scrollable
- Charts: maintain aspect ratio, reduce detail density

### Touch Considerations
- Icon buttons: `size-9` (36px) minimum touch target
- Input fields: `h-9` (36px) standard height
- Buttons: `h-8` to `h-10` range for comfortable tapping

## 9. Agent Prompt Guide

### Quick Color Reference (Dark Mode)
- Background: `oklch(0.141 0.005 285.823)` — near-black
- Card surface: `oklch(0.21 0.006 285.885)` — dark gray
- Text: `oklch(0.985 0 0)` — near-white
- Secondary text: `oklch(0.705 0.015 286.067)` — muted gray
- Border: `oklch(1 0 0 / 10%)` — 10% white
- Good/healthy: `oklch(0.822 0.208 146.907)` — Screen Green
- Warning: `oklch(0.95 0.18 110.904)` — Sunshine Yellow
- Critical: `oklch(0.641 0.214 25.595)` — Fire Red
- Neutral: `oklch(0.794 0.091 250.497)` — Baby Blue

### Example Component Prompts
- "Create a status dashboard card: dark background `oklch(0.21 0.006 285.885)`, 1px border at `oklch(1 0 0 / 10%)`, 2px border-radius. Header in Geist Mono, uppercase, tracking-wider, with a Lucide icon at stroke-width 1.25. Content area with `gap-6`."
- "Design a status badge: rounded-md (6px radius), Geist Mono at 12px, uppercase. Use Screen Green `oklch(0.822 0.208 146.907)` background for healthy state with dark text."
- "Build a log item: 2px left color stripe (green/yellow/red by severity). Status badge with icon, mono uppercase label. Body text in muted gray `oklch(0.705 0.015 286.067)`, timestamp right-aligned in mono."
- "Create a monitoring dashboard: full viewport, no scroll. Grid with fixed menubar top row. 2×2 panel grid with 16px gap. Each panel is a BackgroundCard with mono uppercase title and Lucide icon header."
- "Design a line chart: use Cool Purple `oklch(0.66 0.177 299.333)` for primary series, Fire Red for secondary. Natural curve interpolation, 1.5px stroke. Dark background card container."

### Implementation Tips (from production demos)
- **Markdown rendering**: Newton returns markdown (bold, lists, numbered items). Use `marked` library to render responses — raw `whitespace-pre-wrap` shows literal `**asterisks**`.
- **SVG charts**: Use `ResizeObserver` to dynamically size SVG viewBox to match the container's pixel dimensions. Avoid `preserveAspectRatio="none"` which distorts circles into ovals.
- **ScrollArea in cards**: BackgroundCard's `CardContent` needs `min-h-0 flex-1` for `ScrollArea` to properly constrain and scroll inside flex containers.
- **Status inference**: When classifying Newton's text responses (e.g., traffic normal vs congestion), check for negation patterns ("no visible incidents") before keyword matching to avoid false positives.
- **Camera watermarks**: ALERTCalifornia cameras have a "UC San Diego" watermark. Newton reads it and assumes location — explicitly tell Newton to ignore watermarks in the instruction prompt.
- **Plotly traces**: set `paper_bgcolor: "transparent"` and `plot_bgcolor: "transparent"` so the surrounding `bg-card` is visible — never hardcode a chart background color. Axis font: `family: "Geist Mono, ui-monospace, monospace"`. Grid lines in dark mode: `gridcolor: "rgba(255,255,255,0.08)"` (matches the 10% white border convention). For a single primary line series, use Cool Purple `oklch(0.66 0.177 299.333)`; for time-revealed scatters (UMAP/t-SNE/PCA over a session timeline), `colorscale: "Viridis"` with the colorbar labeled in mono.
- **Plotly `scatter3d` doesn't honor per-marker opacity arrays** (works in 2D `scatter`, silently ignored in 3D). To reveal points progressively in 3D, slice the `x`/`y`/`z` arrays each frame (`coords.slice(0, activeIdx + 1)`) instead of toggling opacity.
- **Plotly with React + Vite**: skip `react-plotly.js@2.x` — its CommonJS factory breaks under Vite + React 19 with `createPlotlyComponent is not a function`. Call `Plotly.react(ref.current, traces, layout)` directly inside a `useEffect` and use a `ResizeObserver` for autosizing.

### Iteration Guide
1. Start with the dark canvas — `oklch(0.141 0.005 285.823)` background
2. Cards step up to `oklch(0.21 0.006 285.885)` with `oklch(1 0 0 / 10%)` borders
3. Geist for everything — Geist Mono for technical data, Geist Sans for prose
4. All headings weight 400 — hierarchy from size and color, not boldness
5. Status colors carry meaning: green = good, yellow = warning, red = critical, blue = info
6. 2px border-radius on everything except badges (6px) and avatars (full)
7. Full-viewport dashboards, panel-based composition, no page scroll
8. Mono + uppercase + tracking-wider for card headers and labels
9. Data provenance is always visible — sensors, locations, timestamps

## 10. Reference Implementations

| Demo | Data Type | Newton API | Repo |
|------|-----------|------------|------|
| **Traffic Monitor** | HLS video stream (Caltrans CCTV) | Lens session + `model.query` (vision) | [archetypeai/archetypeai-traffic-demo](https://github.com/archetypeai/archetypeai-traffic-demo) |
| **Wildfire Watch** | JPEG snapshots (ALERTCalifornia 1,200+ cameras) | Lens session + `model.query` (vision) | [archetypeai/archetypeai-wildfire-demo](https://github.com/archetypeai/archetypeai-wildfire-demo) |
| **Earthquake Monitor** | USGS earthquake catalog (structured text) | Direct query `/v0.5/query` (reasoning) | [archetypeai/archetypeai-earthquake-demo](https://github.com/archetypeai/archetypeai-earthquake-demo) |
| **Grid Monitor** | CAISO supply/demand CSVs (5-min intervals) | Direct query `/v0.5/query` (reasoning) | [archetypeai/archetypeai-grid-demo](https://github.com/archetypeai/archetypeai-grid-demo) |
| **Drilling Monitor** | Volve oil field sensor data (14 wells, North Sea) | Machine State Lens (SSE streaming) | [archetypeai/archetypeai-drilling-demo](https://github.com/archetypeai/archetypeai-drilling-demo) |
| **WiFi Occupancy** | GHOST-IoT smart-home capture (10 days, 9 WiFi clients, anonymized) | Direct query `/v0.5/query` (reasoning) | [archetypeai/archetypeai-wifi-demo](https://github.com/archetypeai/archetypeai-wifi-demo) |
| **Water Treatment Plant Monitor** | SWaT dataset (iTrust/SUTD) — 6-stage plant, 7 days normal + 4 days of 36 cyber-physical attacks | Machine State Lens (6 parallel SSE sessions) + `/query` (operator suggestions) | [archetypeai/archetypeai-swat-demo](https://github.com/archetypeai/archetypeai-swat-demo) |
| **OBD-II Embedding Viewer** | OBD-II logs from a 2020 Lexus RX 450hL (2 sessions × ~50 min, 25 sensors) | Omega 1.3 encoder (local checkpoint, precomputed embeddings) | [archetypeai/archetypeai-obd2-demo](https://github.com/archetypeai/archetypeai-obd2-demo) |
