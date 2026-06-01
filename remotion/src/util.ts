/**
 * Resolve a media reference to a loadable URL.
 *
 * Absolute URLs (http/https/data) pass through untouched; everything else is
 * treated as a path relative to the output root the render service serves
 * statically, so ``video_inputs/x.png`` becomes ``<staticBase>/video_inputs/x.png``.
 */
export const resolveSrc = (src: string, staticBase: string): string => {
  if (!src) return src;
  if (/^(https?:|data:|blob:)/.test(src)) return src;
  const base = (staticBase || "").replace(/\/$/, "");
  const path = src.replace(/^\//, "");
  return base ? `${base}/${path}` : `/${path}`;
};
