#!/usr/bin/env python3
"""Validate modular-assemblies graph/reward invariants in browser."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any

from playwright.async_api import async_playwright


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "dynamics_validation"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def collect_invariant_report(
    url: str,
    loop_count: int,
    sample_every: int,
    expected_reward_mode: str,
    min_abs_mean_reward: float,
) -> dict[str, Any]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await page.wait_for_timeout(2_500)

        raw = await page.evaluate(
            """
            ({ loopCount, sampleEvery, expectedRewardMode, minAbsMeanReward }) => {
              const d = window.__assembliesDemo || window.__mujocoDemo;
              if (!d || !d.data || !d.model || typeof d.render !== "function") {
                return { ok: false, reason: "No MuJoCo demo object found on window" };
              }
              if (!d.graph || !Array.isArray(d.graph.parent) || typeof d.graph.connectedComponents !== "function") {
                return { ok: false, reason: "Demo does not expose modular graph API" };
              }
              if (d.renderer && typeof d.renderer.setAnimationLoop === "function") {
                d.renderer.setAnimationLoop(null);
              }
              if (d.mujoco && typeof d.mujoco.mj_forward === "function") {
                d.mujoco.mj_forward(d.model, d.data);
              }

              const limbs = d.graph.parent.length;
              const eps = 1e-6;
              const samples = [];
              const violations = [];
              const rewardMode = d.rewardMode || "standing";

              const simTime = () => (typeof d.simTimeSec === "number") ? d.simTimeSec : d.data.time;

              const sortedEdgesFromParent = () => {
                const out = [];
                for (let child = 0; child < d.graph.parent.length; child++) {
                  const parent = d.graph.parent[child];
                  if (parent >= 0) out.push(`${parent}->${child}`);
                }
                out.sort();
                return out;
              };

              const sortedEqEdges = () => {
                const out = [];
                const map = d.directedConstraintIndex || {};
                for (const [edge, eqId] of Object.entries(map)) {
                  if ((d.data.eq_active?.[eqId] || 0) > 0) out.push(edge);
                }
                out.sort();
                return out;
              };

              const checkNoCycle = () => {
                const parent = d.graph.parent;
                const n = parent.length;
                for (let i = 0; i < n; i++) {
                  let cur = i;
                  const seen = new Set([i]);
                  for (let hop = 0; hop < n + 1; hop++) {
                    cur = parent[cur];
                    if (cur < 0) break;
                    if (seen.has(cur)) return false;
                    seen.add(cur);
                  }
                }
                return true;
              };

              const checkRewardSharing = () => {
                const rewards = d.limbRewards || [];
                const components = d.graph.connectedComponents();
                for (const component of components) {
                  if (component.length <= 1) continue;
                  const first = rewards[component[0]];
                  for (const node of component) {
                    if (Math.abs((rewards[node] ?? 0) - first) > eps) {
                      return false;
                    }
                  }
                }
                return true;
              };

              const checkObservationDims = () => {
                if (typeof d.getObservationBatch !== "function") return false;
                const batch = d.getObservationBatch();
                const expected = d.observationDim || 16;
                if (!Array.isArray(batch) || batch.length !== limbs) return false;
                return batch.every((row) => row && row.obsDim === expected);
              };

              const snapshot = (tag) => {
                const edges = sortedEdgesFromParent();
                const eqEdges = sortedEqEdges();
                const noCycle = checkNoCycle();
                const rewardShared = checkRewardSharing();
                const obsDimsOk = checkObservationDims();
                const edgeParity = JSON.stringify(edges) === JSON.stringify(eqEdges);
                const parentCountOk = d.graph.parent.every((p) => Number.isInteger(p) && p < limbs);
                const ok = noCycle && rewardShared && obsDimsOk && edgeParity && parentCountOk;
                const meanReward = (d.limbRewards || []).reduce((acc, x) => acc + x, 0.0) / Math.max(1, limbs);
                if (!ok) {
                  violations.push({
                    tag,
                    simTime: simTime(),
                    controlTick: d.controlTick,
                    noCycle,
                    rewardShared,
                    obsDimsOk,
                    edgeParity,
                    parentCountOk,
                    edges,
                    eqEdges,
                    rewards: Array.from(d.limbRewards || []),
                    meanReward,
                  });
                }
                return {
                  tag,
                  simTime: simTime(),
                  controlTick: d.controlTick,
                  rewardMode,
                  meanReward,
                  edgeCount: edges.length,
                  componentCount: d.graph.connectedComponents().length,
                  noCycle,
                  rewardShared,
                  obsDimsOk,
                  edgeParity,
                  parentCountOk,
                  ok,
                };
              };

              const start = snapshot("start");
              for (let k = 1; k <= loopCount; k++) {
                if (typeof d.stepPhysics === "function") {
                  d.stepPhysics(1);
                } else if (typeof d.stepPhysicsFrame === "function") {
                  d.stepPhysicsFrame();
                } else if (d.mujoco && typeof d.mujoco.mj_step === "function") {
                  d.mujoco.mj_step(d.model, d.data);
                } else {
                  d.render(k * 16.0);
                }
                if (k % sampleEvery === 0) {
                  samples.push(snapshot(`k=${k}`));
                }
              }
              const end = snapshot("end");
              const rewardModeOk = rewardMode === expectedRewardMode;
              const rewardSignalOk =
                rewardMode !== "locomotion" ||
                samples.some((s) => Math.abs(s.meanReward) >= minAbsMeanReward) ||
                Math.abs(end.meanReward) >= minAbsMeanReward;
              if (!rewardModeOk) {
                violations.push({
                  tag: "reward_mode_mismatch",
                  rewardMode,
                  expectedRewardMode,
                });
              }
              if (!rewardSignalOk) {
                violations.push({
                  tag: "reward_signal_too_small",
                  rewardMode,
                  minAbsMeanReward,
                  endMeanReward: end.meanReward,
                });
              }

              return {
                ok:
                  violations.length === 0 &&
                  end.controlTick > start.controlTick &&
                  end.simTime > start.simTime &&
                  rewardModeOk &&
                  rewardSignalOk,
                demo_kind: d.constructor?.name || "UnknownDemo",
                limb_count: limbs,
                observation_dim: d.observationDim || null,
                reward_mode: rewardMode,
                start,
                end,
                samples,
                violations,
              };
            }
            """,
            {
                "loopCount": loop_count,
                "sampleEvery": sample_every,
                "expectedRewardMode": expected_reward_mode,
                "minAbsMeanReward": min_abs_mean_reward,
            },
        )
        await browser.close()
    return raw


def with_query_param(url: str, key: str, value: str) -> str:
    split = urlsplit(url)
    pairs = dict(parse_qsl(split.query, keep_blank_values=True))
    pairs[key] = value
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(pairs), split.fragment))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8877/modular_assemblies.html")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--loop-count", type=int, default=600)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--reward-mode", choices=("standing", "locomotion"), default="standing")
    parser.add_argument("--min-abs-mean-reward", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.loop_count <= 0:
        raise SystemExit("--loop-count must be positive")
    if args.sample_every <= 0:
        raise SystemExit("--sample-every must be positive")
    if args.min_abs_mean_reward < 0:
        raise SystemExit("--min-abs-mean-reward must be non-negative")

    url = with_query_param(args.url, "reward", args.reward_mode)

    report = asyncio.run(
        collect_invariant_report(
            url=url,
            loop_count=args.loop_count,
            sample_every=args.sample_every,
            expected_reward_mode=args.reward_mode,
            min_abs_mean_reward=args.min_abs_mean_reward,
        )
    )
    if not report.get("ok") and "reason" in report:
        raise SystemExit(f"Invariant probe failed: {report}")

    output = {
        "created_utc": utc_now_iso(),
        "url": url,
        "loop_count": args.loop_count,
        "sample_every": args.sample_every,
        "expected_reward_mode": args.reward_mode,
        "min_abs_mean_reward": args.min_abs_mean_reward,
        **report,
    }

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"assemblies_invariant_report_{args.reward_mode}_{run_id}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(out_path)
    print(
        json.dumps(
            {
                "ok": output["ok"],
                "demo_kind": output.get("demo_kind"),
                "limb_count": output.get("limb_count"),
                "observation_dim": output.get("observation_dim"),
                "reward_mode": output.get("reward_mode"),
                "start_control_tick": output.get("start", {}).get("controlTick"),
                "end_control_tick": output.get("end", {}).get("controlTick"),
                "start_sim_time": output.get("start", {}).get("simTime"),
                "end_sim_time": output.get("end", {}).get("simTime"),
                "sample_count": len(output.get("samples", [])),
                "violation_count": len(output.get("violations", [])),
            },
            indent=2,
        )
    )

    if not output["ok"]:
        raise SystemExit(
            "Invariant validation failed: "
            f"violations={len(output.get('violations', []))}, "
            f"start={output.get('start')}, end={output.get('end')}"
        )


if __name__ == "__main__":
    main()
