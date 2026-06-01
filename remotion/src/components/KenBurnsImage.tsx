import {
  AbsoluteFill,
  Img,
  interpolate,
  Easing,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { resolveSrc } from "../util";

const EASING: Record<string, (n: number) => number> = {
  linear: Easing.linear,
  ease: Easing.ease,
  "ease-in": Easing.in(Easing.ease),
  "ease-out": Easing.out(Easing.ease),
  "ease-in-out": Easing.inOut(Easing.ease),
};

type Focus = { scale: number; x: number; y: number };

export const KenBurnsImage: React.FC<{
  layer: {
    src: string;
    from: Focus;
    to: Focus;
    easing: string;
    fit: "cover" | "contain";
  };
  staticBase: string;
}> = ({ layer, staticBase }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const easing = EASING[layer.easing] ?? EASING["ease-in-out"];
  const end = Math.max(1, durationInFrames - 1);

  const lerp = (a: number, b: number) =>
    interpolate(frame, [0, end], [a, b], {
      easing,
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });

  const scale = lerp(layer.from.scale, layer.to.scale);
  const x = lerp(layer.from.x, layer.to.x);
  const y = lerp(layer.from.y, layer.to.y);

  return (
    <AbsoluteFill style={{ overflow: "hidden" }}>
      <Img
        src={resolveSrc(layer.src, staticBase)}
        style={{
          width: "100%",
          height: "100%",
          objectFit: layer.fit,
          transform: `scale(${scale}) translate(${x * 100}%, ${y * 100}%)`,
        }}
      />
    </AbsoluteFill>
  );
};
