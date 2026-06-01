import { Composition } from "remotion";
import { compositionProps } from "./schema";
import { VideoSpecVideo } from "./VideoSpecVideo";

/**
 * Single composition driven entirely by input props. Duration / fps / size are
 * derived from the spec in ``calculateMetadata`` so one composition renders any
 * VideoSpec — a single-clip motion shot (flow A) or a multi-clip MG video
 * (flow B). The static defaults are only placeholders for Remotion Studio.
 */
export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="VideoSpec"
      component={VideoSpecVideo}
      schema={compositionProps}
      durationInFrames={150}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={{
        fps: 30,
        width: 1080,
        height: 1920,
        clips: [],
        audio: [],
        staticBase: "",
      }}
      calculateMetadata={({ props }) => {
        const fps = props.fps || 30;
        const totalSec =
          (props.clips || []).reduce((s, c) => s + (c.duration_sec || 0), 0) || 1;
        return {
          durationInFrames: Math.max(1, Math.round(totalSec * fps)),
          fps,
          width: props.width || 1080,
          height: props.height || 1920,
          props,
        };
      }}
    />
  );
};
