# ManjuForge Remotion 渲染微服务

把一份 `VideoSpec` JSON 用 [Remotion](https://www.remotion.dev/) 确定性渲染成 MP4。
ManjuForge 后端两条新视频流程共用这一个渲染器：

- **A**（`RemotionVideoAdapter`）：单 clip 的 spec → 替代 i2v 的"伪运镜"片段，接入 `merge_videos`。
- **B**（`RemotionMGGenerator`）：多 clip 的 spec → 整条图文/解说视频。

> ⚠️ Remotion 是自定义非开源 License，营利组织 >3 名员工需购买 Company License。商用前请确认合规。

## 本地运行

```bash
cd remotion
npm install
REMOTION_OUTPUT_ROOT=../output npm run server   # 默认监听 :3001
```

后端通过 `REMOTION_RENDER_URL`（默认 `http://localhost:3001`）调用它。

预览/调试 composition：`npm run studio`。

## 接口

- `GET /health` → `{ok:true}`
- `GET /static/<相对路径>` → 从 `REMOTION_OUTPUT_ROOT` 提供媒体文件（供 `kenburns_image` 等图层按相对路径加载）
- `POST /render` body `{spec, outputRel}` → 渲染到 `OUTPUT_ROOT/outputRel`，返回 `{ok, seconds}`

## 契约

`VideoSpec` 的两端定义：

- Node（source of truth，含默认值）：`src/schema.ts`
- Python（DTO 镜像 + 构造）：`src/models/remotion_spec.py`

媒体引用（`src`/`audio_src`）一律用 **相对 output 根** 的路径；`http(s)://` / `data:` 原样透传。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `REMOTION_RENDER_PORT` | `3001` | 服务端口 |
| `REMOTION_OUTPUT_ROOT` | `output` | 共享输出根（与后端同一目录/卷） |
| `REMOTION_PUBLIC_URL` | `http://localhost:<port>` | headless Chrome 访问静态资源的基址 |

## 依赖版本

所有 `@remotion/*` 包必须同版本。升级时用 `npx remotion versions` 校验对齐。
