const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const remotionDir = path.join(root, 'remotion');

// Pin the renderer's output root to the SAME absolute dir the backend serves
// from (backend default = abspath("output") relative to repo root). If these
// two ever diverge, a render "succeeds" but the MP4 lands where the backend
// can't serve it -> broken <video>. Keeping them equal here removes that
// whole class of local-dev bug.
const outputRoot = path.join(root, 'output');

if (!fs.existsSync(path.join(remotionDir, 'node_modules'))) {
  console.warn(
    '[remotion] node_modules missing — skipping render service.\n' +
    '[remotion] Run `cd remotion && npm install` to enable Remotion video rendering.'
  );
  // Exit 0 so concurrently --kill-others-on-fail does not tear down the
  // backend/frontend just because the optional renderer is not installed.
  process.exit(0);
}

const child = spawn('node', ['server.mjs'], {
  cwd: remotionDir,
  stdio: 'inherit',
  env: { ...process.env, REMOTION_OUTPUT_ROOT: outputRoot },
});

child.on('exit', (code) => process.exit(code || 0));
