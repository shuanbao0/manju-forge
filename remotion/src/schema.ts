/**
 * VideoSpec — zod schema. Node-side half of the contract mirrored by
 * ``src/models/remotion_spec.py`` on the Python side. Field names are
 * snake_case to match the pydantic payload verbatim (no alias translation),
 * except KenBurnsImage's ``from`` which is a JS reserved-ish key but legal as
 * an object property.
 */
import { z } from "zod";

const focus = z.object({
  scale: z.number().default(1),
  x: z.number().default(0),
  y: z.number().default(0),
});

const subtitleCue = z.object({
  text: z.string(),
  from_sec: z.number().default(0),
  to_sec: z.number(),
});

const transitionIn = z.object({
  type: z.enum(["fade", "slide", "wipe", "none"]).default("fade"),
  duration_sec: z.number().default(0.5),
  direction: z
    .enum(["from-left", "from-right", "from-top", "from-bottom"])
    .nullish(),
});

const audioTrack = z.object({
  src: z.string(),
  from_sec: z.number().default(0),
  volume: z.number().default(1),
  loop: z.boolean().default(false),
});

// ── Layers (discriminated union on `type`) ──────────────────────────────
const kenBurnsImage = z.object({
  type: z.literal("kenburns_image"),
  src: z.string(),
  from: focus.default({}),
  to: focus.default({ scale: 1.12 }),
  easing: z
    .enum(["linear", "ease", "ease-in", "ease-out", "ease-in-out"])
    .default("ease-in-out"),
  fit: z.enum(["cover", "contain"]).default("cover"),
});

const titleCard = z.object({
  type: z.literal("title_card"),
  text: z.string(),
  subtitle: z.string().nullish(),
  align: z.enum(["center", "left", "right"]).default("center"),
  color: z.string().default("#ffffff"),
  font_size: z.number().default(80),
  enter: z.enum(["fade", "rise", "none"]).default("rise"),
});

const subtitle = z.object({
  type: z.literal("subtitle"),
  cues: z.array(subtitleCue).default([]),
  position: z.enum(["bottom", "top", "center"]).default("bottom"),
  color: z.string().default("#ffffff"),
  font_size: z.number().default(44),
  box: z.boolean().default(true),
});

const bulletList = z.object({
  type: z.literal("bullet_list"),
  title: z.string().nullish(),
  items: z.array(z.string()).default([]),
  color: z.string().default("#ffffff"),
  stagger_sec: z.number().default(0.4),
});

const background = z.object({
  type: z.literal("background"),
  color: z.string().nullish(),
  gradient: z.array(z.string()).nullish(),
});

export const layer = z.discriminatedUnion("type", [
  kenBurnsImage,
  titleCard,
  subtitle,
  bulletList,
  background,
]);

export const clip = z.object({
  duration_sec: z.number().positive().default(5),
  layers: z.array(layer).default([]),
  transition_in: transitionIn.nullish(),
  audio_src: z.string().nullish(),
});

export const videoSpec = z.object({
  fps: z.number().default(30),
  width: z.number().default(1080),
  height: z.number().default(1920),
  clips: z.array(clip).default([]),
  audio: z.array(audioTrack).default([]),
});

// The composition receives the spec plus a `staticBase` URL the render
// service injects so relative media refs resolve to the served output dir.
export const compositionProps = videoSpec.extend({
  staticBase: z.string().default(""),
});

export type VideoSpec = z.infer<typeof videoSpec>;
export type CompositionProps = z.infer<typeof compositionProps>;
export type Clip = z.infer<typeof clip>;
export type Layer = z.infer<typeof layer>;
