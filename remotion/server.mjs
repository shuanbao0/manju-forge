/**
 * Remotion render microservice.
 *
 * Long-lived Node process the Python backend (RemotionRenderClient) talks to.
 *
 *   GET  /health            -> 200 {ok:true}
 *   GET  /static/<relpath>  -> serves files from OUTPUT_ROOT (media for layers)
 *   POST /render {spec, outputRel} -> renders to OUTPUT_ROOT/outputRel, {ok, seconds}
 *
 * The bundle is built once at boot (warm Chrome each render avoids re-bundling).
 * Renders are serialized through a small queue to bound memory.
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { bundle } from "@remotion/bundler";
import { selectComposition, renderMedia } from "@remotion/renderer";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PORT = Number(process.env.REMOTION_RENDER_PORT || 3001);
const OUTPUT_ROOT = path.resolve(process.env.REMOTION_OUTPUT_ROOT || "output");
const PUBLIC_URL = (
  process.env.REMOTION_PUBLIC_URL || `http://localhost:${PORT}`
).replace(/\/$/, "");
const STATIC_BASE = `${PUBLIC_URL}/static`;
const COMPOSITION_ID = "VideoSpec";

const CONTENT_TYPES = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".m4a": "audio/mp4",
  ".mp4": "video/mp4",
};

let serveUrlPromise = null;
const getServeUrl = () => {
  if (!serveUrlPromise) {
    console.log("[remotion] bundling composition…");
    serveUrlPromise = bundle({
      entryPoint: path.join(__dirname, "src", "index.ts"),
      onProgress: () => {},
    }).then((url) => {
      console.log("[remotion] bundle ready");
      return url;
    });
  }
  return serveUrlPromise;
};

// ── serialize renders ─────────────────────────────────────────────────────
let renderChain = Promise.resolve();
const enqueue = (job) => {
  const run = renderChain.then(job, job);
  // keep the chain alive regardless of individual job outcome
  renderChain = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
};

// ── safe path resolution under OUTPUT_ROOT ────────────────────────────────
const resolveUnderRoot = (relPath) => {
  const clean = decodeURIComponent(relPath).replace(/^\/+/, "");
  const abs = path.resolve(OUTPUT_ROOT, clean);
  if (abs !== OUTPUT_ROOT && !abs.startsWith(OUTPUT_ROOT + path.sep)) {
    throw new Error(`path escapes output root: ${relPath}`);
  }
  return abs;
};

const sendJson = (res, status, body) => {
  const payload = JSON.stringify(body);
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(payload);
};

const readBody = (req) =>
  new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });

const serveStatic = (res, relPath) => {
  let abs;
  try {
    abs = resolveUnderRoot(relPath);
  } catch {
    res.writeHead(403);
    return res.end("forbidden");
  }
  fs.stat(abs, (err, stat) => {
    if (err || !stat.isFile()) {
      res.writeHead(404);
      return res.end("not found");
    }
    const type = CONTENT_TYPES[path.extname(abs).toLowerCase()] || "application/octet-stream";
    res.writeHead(200, { "Content-Type": type, "Content-Length": stat.size });
    fs.createReadStream(abs).pipe(res);
  });
};

const handleRender = async (res, body) => {
  let parsed;
  try {
    parsed = JSON.parse(body);
  } catch (e) {
    return sendJson(res, 400, { ok: false, error: `invalid JSON: ${e.message}` });
  }
  const { spec, outputRel } = parsed || {};
  if (!spec || !outputRel) {
    return sendJson(res, 400, { ok: false, error: "spec and outputRel are required" });
  }

  let outputPath;
  try {
    outputPath = resolveUnderRoot(outputRel);
  } catch (e) {
    return sendJson(res, 400, { ok: false, error: e.message });
  }

  try {
    const seconds = await enqueue(async () => {
      const serveUrl = await getServeUrl();
      const inputProps = { ...spec, staticBase: STATIC_BASE };
      const composition = await selectComposition({
        serveUrl,
        id: COMPOSITION_ID,
        inputProps,
      });
      fs.mkdirSync(path.dirname(outputPath), { recursive: true });
      const started = process.hrtime.bigint();
      await renderMedia({
        composition,
        serveUrl,
        codec: "h264",
        outputLocation: outputPath,
        inputProps,
        overwrite: true,
      });
      return Number(process.hrtime.bigint() - started) / 1e9;
    });
    return sendJson(res, 200, { ok: true, seconds, output: outputRel });
  } catch (e) {
    console.error("[remotion] render failed:", e);
    return sendJson(res, 500, { ok: false, error: String(e && e.message ? e.message : e) });
  }
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, PUBLIC_URL);

  if (req.method === "GET" && url.pathname === "/health") {
    return sendJson(res, 200, { ok: true });
  }
  if (req.method === "GET" && url.pathname.startsWith("/static/")) {
    return serveStatic(res, url.pathname.slice("/static/".length));
  }
  if (req.method === "POST" && url.pathname === "/render") {
    const body = await readBody(req);
    return handleRender(res, body);
  }
  res.writeHead(404);
  res.end("not found");
});

server.listen(PORT, () => {
  console.log(`[remotion] render service on ${PUBLIC_URL}`);
  console.log(`[remotion] output root: ${OUTPUT_ROOT}`);
  // Warm the bundle so the first real render is fast.
  getServeUrl().catch((e) => console.error("[remotion] bundle failed:", e));
});
