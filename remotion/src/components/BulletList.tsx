import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export const BulletList: React.FC<{
  layer: {
    title?: string | null;
    items: string[];
    color: string;
    stagger_sec: number;
  };
}> = ({ layer }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  // Guard against partial specs: Remotion does not apply zod `.default()` to
  // render-time input props, so a missing stagger_sec would yield NaN ranges.
  const staggerFrames = Math.max(1, Math.round((layer.stagger_sec ?? 0.4) * fps));
  const items = layer.items ?? [];

  return (
    <AbsoluteFill
      style={{
        justifyContent: "center",
        padding: "10%",
        color: layer.color,
        fontFamily: "sans-serif",
      }}
    >
      {layer.title ? (
        <div style={{ fontSize: 64, fontWeight: 800, marginBottom: "0.6em" }}>
          {layer.title}
        </div>
      ) : null}
      {items.map((item, i) => {
        const appearAt = (i + 1) * staggerFrames;
        const opacity = interpolate(frame, [appearAt, appearAt + fps * 0.4], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        const x = interpolate(frame, [appearAt, appearAt + fps * 0.4], [30, 0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        return (
          <div
            key={i}
            style={{
              fontSize: 44,
              fontWeight: 500,
              marginBottom: "0.5em",
              opacity,
              transform: `translateX(${x}px)`,
            }}
          >
            • {item}
          </div>
        );
      })}
    </AbsoluteFill>
  );
};
