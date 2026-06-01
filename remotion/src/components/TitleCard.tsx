import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export const TitleCard: React.FC<{
  layer: {
    text: string;
    subtitle?: string | null;
    align: "center" | "left" | "right";
    color: string;
    font_size: number;
    enter: "fade" | "rise" | "none";
  };
}> = ({ layer }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress =
    layer.enter === "none"
      ? 1
      : spring({ frame, fps, config: { damping: 200 }, durationInFrames: fps });
  const opacity = layer.enter === "none" ? 1 : interpolate(progress, [0, 1], [0, 1]);
  const translateY = layer.enter === "rise" ? interpolate(progress, [0, 1], [40, 0]) : 0;

  return (
    <AbsoluteFill
      style={{
        justifyContent: "center",
        alignItems:
          layer.align === "center"
            ? "center"
            : layer.align === "left"
              ? "flex-start"
              : "flex-end",
        padding: "8%",
        opacity,
        transform: `translateY(${translateY}px)`,
      }}
    >
      <div
        style={{
          color: layer.color,
          fontFamily: "sans-serif",
          textAlign: layer.align,
          textShadow: "0 4px 24px rgba(0,0,0,0.6)",
        }}
      >
        <div style={{ fontSize: layer.font_size, fontWeight: 800, lineHeight: 1.15 }}>
          {layer.text}
        </div>
        {layer.subtitle ? (
          <div
            style={{
              fontSize: Math.round(layer.font_size * 0.45),
              fontWeight: 500,
              marginTop: "0.5em",
              opacity: 0.85,
            }}
          >
            {layer.subtitle}
          </div>
        ) : null}
      </div>
    </AbsoluteFill>
  );
};
