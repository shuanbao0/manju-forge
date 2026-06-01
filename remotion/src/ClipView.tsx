import {
  AbsoluteFill,
  Audio,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import type { Clip, Layer } from "./schema";
import { resolveSrc } from "./util";
import { KenBurnsImage } from "./components/KenBurnsImage";
import { TitleCard } from "./components/TitleCard";
import { Subtitle } from "./components/Subtitle";
import { BulletList } from "./components/BulletList";
import { Background } from "./components/Background";

const LayerView: React.FC<{ layer: Layer; staticBase: string }> = ({
  layer,
  staticBase,
}) => {
  switch (layer.type) {
    case "kenburns_image":
      return <KenBurnsImage layer={layer} staticBase={staticBase} />;
    case "title_card":
      return <TitleCard layer={layer} />;
    case "subtitle":
      return <Subtitle layer={layer} />;
    case "bullet_list":
      return <BulletList layer={layer} />;
    case "background":
      return <Background layer={layer} />;
    default:
      return null;
  }
};

/**
 * Applies ``transition_in`` as an entrance effect over the first frames of the
 * clip. Kept overlap-free (no duration borrowing) so total video duration is
 * exactly the sum of clip durations — simplifies the composition's
 * ``calculateMetadata``.
 */
const TransitionWrap: React.FC<{
  transition?: Clip["transition_in"];
  children: React.ReactNode;
}> = ({ transition, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  if (!transition || transition.type === "none") {
    return <AbsoluteFill>{children}</AbsoluteFill>;
  }

  const dur = Math.max(1, Math.round(transition.duration_sec * fps));
  const p = interpolate(frame, [0, dur], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  let transform = "none";
  if (transition.type === "slide") {
    const offset = interpolate(p, [0, 1], [100, 0]);
    switch (transition.direction) {
      case "from-left":
        transform = `translateX(${-offset}%)`;
        break;
      case "from-top":
        transform = `translateY(${-offset}%)`;
        break;
      case "from-bottom":
        transform = `translateY(${offset}%)`;
        break;
      default:
        transform = `translateX(${offset}%)`;
    }
  }

  return (
    <AbsoluteFill style={{ opacity: transition.type === "fade" ? p : 1, transform }}>
      {children}
    </AbsoluteFill>
  );
};

export const ClipView: React.FC<{ clip: Clip; staticBase: string }> = ({
  clip,
  staticBase,
}) => {
  return (
    <TransitionWrap transition={clip.transition_in}>
      {clip.layers.map((layer, i) => (
        <LayerView key={i} layer={layer} staticBase={staticBase} />
      ))}
      {clip.audio_src ? <Audio src={resolveSrc(clip.audio_src, staticBase)} /> : null}
    </TransitionWrap>
  );
};
