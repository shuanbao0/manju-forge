import { AbsoluteFill } from "remotion";

export const Background: React.FC<{
  layer: { color?: string | null; gradient?: string[] | null };
}> = ({ layer }) => {
  const background =
    layer.gradient && layer.gradient.length >= 2
      ? `linear-gradient(135deg, ${layer.gradient.join(", ")})`
      : layer.color || "#000000";
  return <AbsoluteFill style={{ background }} />;
};
