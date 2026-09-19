/**
 * Fixed HUD frame: neon corner brackets + faint scanline texture + edge
 * vignette. Purely decorative, pointer-events disabled.
 */
export function HudFrame() {
  return (
    <div className="pointer-events-none fixed inset-0 z-40" aria-hidden>
      {/* Corner brackets */}
      <span className="hud-corner left-3 top-3 border-l-2 border-t-2" />
      <span className="hud-corner right-3 top-3 border-r-2 border-t-2" />
      <span className="hud-corner bottom-3 left-3 border-b-2 border-l-2" />
      <span className="hud-corner bottom-3 right-3 border-b-2 border-r-2" />

      {/* Scanlines */}
      <div className="absolute inset-0 opacity-[0.35] hud-scanlines" />

      {/* Subtle edge vignette */}
      <div className="absolute inset-0 shadow-[inset_0_0_140px_rgba(0,0,0,0.55)]" />
    </div>
  )
}
