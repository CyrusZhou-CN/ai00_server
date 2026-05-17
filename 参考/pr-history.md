# PR 历史记录

## PR #168: update web-rwkv to 0.10.20 (已合并)
- 日期: 2026-05-17
- 分支: `update-webrwkv`
- 内容: web-rwkv 版本 0.10.18 → 0.10.20
- 关键变更:
  - `enumerate_adapters()` 改为异步方法，调用处需加 `.await`
  - 修复多处 clippy 警告 (useless_conversion, unused import, unnecessary_min_or_max)

## PR #169: fix batch completion (已合并)
- 日期: 2026-05-17
- 分支: `fix-batch-completion`
- 内容: 修复 /v1/completions 批量推理只返回一组数据的 BUG
- 根因: `Vec::from(prompt).join("")` 把所有 prompt 拼接成一个
- 修复:
  - `into_generate_request(self)` → `to_generate_request(&self, prompt: String)` (避免 move)
  - `respond_one`: 用 `JoinSet` 并发处理每个 prompt，按 index 排序返回多个 choices
  - `respond_stream`: 共享 channel 传递 `(index, Token)` 对，SSE 事件带正确 index
  - `Vec::from(request.prompt)` → `Vec::from(request.prompt.clone())` (避免 partial move)