import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";

type Cue = { text: string; from_sec: number; to_sec: number };

const POSITION: Record<string, React.CSSProperties> = {
  bottom: { justifyContent: "flex-end", paddingBottom: "8%" },
  top: { justifyContent: "flex-start", paddingTop: "8%" },
  center: { justifyContent: "center" },
};

export const Subtitle: React.FC<{
  layer: {
    cues: Cue[];
    position: "bottom" | "top" | "center";
    color: string;
    font_size: number;
    box: boolean;
  };
}> = ({ layer }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const sec = frame / fps;
  const cue = layer.cues.find((c) => sec >= c.from_sec && sec <= c.to_sec);
  if (!cue) return null;

  return (
    <AbsoluteFill
      style={{
        ...POSITION[layer.position],
        alignItems: "center",
        display: "flex",
      }}
    >
      <span
        style={{
          color: layer.color,
          fontSize: layer.font_size,
          fontWeight: 700,
          fontFamily: "sans-serif",
          textAlign: "center",
          maxWidth: "86%",
          lineHeight: 1.3,
          padding: layer.box ? "0.25em 0.6em" : 0,
          borderRadius: 12,
          background: layer.box ? "rgba(0,0,0,0.55)" : "transparent",
          textShadow: layer.box ? "none" : "0 2px 8px rgba(0,0,0,0.9)",
        }}
      >
        {cue.text}
      </span>
    </AbsoluteFill>
  );
};
