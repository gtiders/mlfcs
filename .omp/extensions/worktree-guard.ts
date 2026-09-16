/**
 * 重构期间的写入边界守卫。
 *
 * 只允许对 worktree 内的文件产生写入（write / edit / ast_edit 等），
 * 其余任何路径一律 block，工具不会执行。策略文本见 .omp/rules/worktree-only.md。
 *
 * 边界目录不存在时不启用（fail-open），避免在别处的 clone 里把写操作全挡掉。
 */
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { existsSync } from "node:fs";
import * as nodePath from "node:path";

const REPO_ROOT = nodePath.resolve(import.meta.dir, "..", "..");
const WORKTREE_ROOT = nodePath.resolve(REPO_ROOT, ".worktrees", "reflector");

const WRITE_TOOLS: Record<string, true> = {
  write: true,
  edit: true,
  ast_edit: true,
  apply_patch: true,
  notebook_edit: true,
};

const PATH_KEYS: Record<string, true> = {
  path: true,
  paths: true,
  file: true,
  files: true,
  target: true,
  targets: true,
};

function collectPaths(value: unknown, key: string, out: string[]): string[] {
  if (typeof value === "string") {
    if (PATH_KEYS[key] === true) out.push(value);
    return out;
  }
  if (Array.isArray(value)) {
    for (const item of value) collectPaths(item, key, out);
    return out;
  }
  if (value && typeof value === "object") {
    for (const [childKey, child] of Object.entries(value)) collectPaths(child, childKey, out);
  }
  return out;
}

export default function worktreeGuard(pi: ExtensionAPI) {
  if (!existsSync(WORKTREE_ROOT)) return;

  pi.on("tool_call", async (event, ctx) => {
    if (WRITE_TOOLS[event.toolName] !== true) return;

    for (const raw of collectPaths(event.input, "", [])) {
      const abs = nodePath.resolve(ctx.cwd, raw);
      if (abs === WORKTREE_ROOT || abs.startsWith(WORKTREE_ROOT + nodePath.sep)) continue;
      return {
        block: true,
        reason:
          `写入被 worktree 边界拦住：${raw} → ${abs}\n` +
          `重构期间只允许写 ${WORKTREE_ROOT} 内的文件。` +
          `把这次改动落到 worktree 内的对应路径上重试。策略见 .omp/rules/worktree-only.md。`,
      };
    }
  });
}
