import { AbsoluteFill, Audio, Sequence, useVideoConfig } from "remotion";
import type { CompositionProps } from "./schema";
import { ClipView } from "./ClipView";
import { resolveSrc } from "./util";

/**
 * Interprets a VideoSpec: lays clips back-to-back as Sequences (so each layer's
 * ``useCurrentFrame`` is clip-relative) and overlays global audio tracks.
 */
export const VideoSpecVideo: React.FC<CompositionProps> = ({
  clips,
  audio,
  staticBase,
}) => {
  const { fps } = useVideoConfig();

  let offset = 0;
  const sequences = clips.map((clip, i) => {
    const dur = Math.max(1, Math.round(clip.duration_sec * fps));
    const from = offset;
    offset += dur;
    return (
      <Sequence key={i} from={from} durationInFrames={dur}>
        <ClipView clip={clip} staticBase={staticBase} />
      </Sequence>
    );
  });

  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      {sequences}
      {(audio || []).map((track, i) => (
        <Audio
          key={`audio-${i}`}
          src={resolveSrc(track.src, staticBase)}
          volume={track.volume}
          loop={track.loop}
          startFrom={Math.max(0, Math.round((track.from_sec || 0) * fps))}
        />
      ))}
    </AbsoluteFill>
  );
};
