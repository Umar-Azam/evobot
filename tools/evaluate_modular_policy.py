#!/usr/bin/env python3
"""Evaluate modular-assemblies policy weights with a rollout."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "policy_evals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8877/modular_assemblies.html?headless=1&reward=locomotion",
    )
    parser.add_argument("--weights", required=True, help="Path to policy weights JSON")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--disable-messages", action="store_true")
    parser.add_argument("--random-torque-scale", type=float, default=0.03)
    return parser.parse_args()


async def evaluate_once(
    url: str,
    weights: dict[str, Any],
    steps: int,
    use_messages: bool,
    random_torque_scale: float,
) -> dict[str, Any]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await page.wait_for_timeout(2_000)
        raw = await page.evaluate(
            """
            ({ weights, stepCount, useMessages, randomTorqueScale }) => {
              const d = window.__assembliesDemo || window.__mujocoDemo;
              if (!d || typeof d.collectRollout !== "function" || typeof d.loadPolicyWeights !== "function") {
                return { ok: false, reason: "required runtime APIs are not available" };
              }
              if (typeof d.resetEpisode === "function") {
                d.resetEpisode();
              }
              if (typeof d.setMessagePassingEnabled === "function") {
                d.setMessagePassingEnabled(useMessages);
              } else {
                d.useMessages = !!useMessages;
              }
              if (Number.isFinite(randomTorqueScale)) {
                d.randomTorqueScale = randomTorqueScale;
              }
              if (d.renderer && typeof d.renderer.setAnimationLoop === "function") {
                d.renderer.setAnimationLoop(null);
              }
              d.loadPolicyWeights(weights);
              const rollout = d.collectRollout(stepCount);
              if (!Array.isArray(rollout) || rollout.length < 2) {
                return { ok: false, reason: "rollout too short" };
              }

              const edgeCounts = rollout.map((s) => (Array.isArray(s.edges) ? s.edges.length : 0));
              const edgeMean = edgeCounts.reduce((acc, x) => acc + x, 0.0) / edgeCounts.length;
              const edgeNonzeroFrac = edgeCounts.filter((x) => x > 0).length / edgeCounts.length;
              const maxEdges = edgeCounts.reduce((acc, x) => Math.max(acc, x), 0);

              const meanReward =
                rollout.reduce((acc, s) => {
                  const rs = Array.isArray(s.rewards) ? s.rewards : [];
                  const m = rs.length ? rs.reduce((racc, x) => racc + x, 0.0) / rs.length : 0.0;
                  return acc + m;
                }, 0.0) / rollout.length;

              const centroid = (step) => {
                const obs = step.observations || [];
                if (!obs.length) {
                  return [0.0, 0.0, 0.0];
                }
                let cx = 0.0;
                let cy = 0.0;
                let cz = 0.0;
                for (const row of obs) {
                  cx += row[0] || 0.0;
                  cy += row[1] || 0.0;
                  cz += row[2] || 0.0;
                }
                const inv = 1.0 / obs.length;
                return [cx * inv, cy * inv, cz * inv];
              };
              const c0 = centroid(rollout[0]);
              const c1 = centroid(rollout[rollout.length - 1]);
              const dx = c1[0] - c0[0];
              const dy = c1[1] - c0[1];
              const dz = c1[2] - c0[2];
              const displacement = Math.sqrt(dx * dx + dy * dy + dz * dz);

              return {
                ok: true,
                metrics: {
                  edge_mean: edgeMean,
                  edge_nonzero_frac: edgeNonzeroFrac,
                  max_edges: maxEdges,
                  mean_reward: meanReward,
                  x_displacement: dx,
                  centroid_displacement: displacement,
                  start_control_tick: rollout[0].controlTick || 0,
                  end_control_tick: rollout[rollout.length - 1].controlTick || 0,
                  rollout_len: rollout.length,
                },
                first_edges: rollout[0].edges || [],
                last_edges: rollout[rollout.length - 1].edges || [],
                rollout,
              };
            }
            """,
            {
                "weights": weights,
                "stepCount": steps,
                "useMessages": use_messages,
                "randomTorqueScale": random_torque_scale,
            },
        )
        await browser.close()
    return raw


def main() -> None:
    args = parse_args()
    if args.steps <= 1:
        raise SystemExit("--steps must be > 1")

    weights = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    report = asyncio.run(
        evaluate_once(
            url=args.url,
            weights=weights,
            steps=args.steps,
            use_messages=not args.disable_messages,
            random_torque_scale=args.random_torque_scale,
        )
    )
    if not report.get("ok"):
        raise SystemExit(f"evaluation failed: {report}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    summary = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "url": args.url,
        "weights": str(Path(args.weights).resolve()),
        "steps": args.steps,
        "use_messages": not args.disable_messages,
        "random_torque_scale": args.random_torque_scale,
        "metrics": report["metrics"],
        "first_edges": report.get("first_edges", []),
        "last_edges": report.get("last_edges", []),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "rollout.jsonl").open("w", encoding="utf-8") as f:
        for step in report.get("rollout", []):
            f.write(json.dumps(step, separators=(",", ":")) + "\n")

    print(run_dir)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
