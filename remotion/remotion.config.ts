import { Config } from "@remotion/cli/config";

// Only affects `remotion studio` / `remotion render` CLI usage. The render
// microservice (server.mjs) sets its own renderMedia options programmatically.
Config.setVideoImageFormat("jpeg");
Config.setOverwriteOutput(true);
