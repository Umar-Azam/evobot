#!/usr/bin/env python3
"""Train modular-assemblies policy weights with a small CEM loop."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "training_runs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_train_vector(weights: dict[str, Any]) -> list[float]:
    vec: list[float] = []
    for row in weights["wTorque"]:
        vec.extend(float(v) for v in row)
    vec.extend(float(v) for v in weights["wAttach"])
    vec.extend(float(v) for v in weights["wDetach"])
    return vec


def inject_train_vector(base_weights: dict[str, Any], vec: list[float]) -> dict[str, Any]:
    out = copy.deepcopy(base_weights)
    idx = 0
    for i in range(len(out["wTorque"])):
        for j in range(len(out["wTorque"][i])):
            out["wTorque"][i][j] = float(vec[idx])
            idx += 1
    for i in range(len(out["wAttach"])):
        out["wAttach"][i] = float(vec[idx])
        idx += 1
    for i in range(len(out["wDetach"])):
        out["wDetach"][i] = float(vec[idx])
        idx += 1
    if idx != len(vec):
        raise ValueError(f"vector length mismatch: consumed {idx} but got {len(vec)}")
    return out


def score_metrics(metrics: dict[str, float], args: argparse.Namespace) -> float:
    return (
        args.w_edge_mean * metrics["edge_mean"]
        + args.w_edge_nonzero * metrics["edge_nonzero_frac"]
        + args.w_forward_x * max(0.0, metrics["x_displacement"])
        + args.w_displacement * metrics["centroid_displacement"]
        + args.w_reward * metrics["mean_reward"]
    )


async def evaluate_policy(
    page: Any,
    weights: dict[str, Any],
    steps: int,
    use_messages: bool,
    random_torque_scale: float,
) -> dict[str, Any]:
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
    if not raw.get("ok"):
        raise RuntimeError(f"evaluation failed: {raw}")
    return raw


async def run_training(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
        await page.wait_for_timeout(2_000)

        base_weights = await page.evaluate(
            """
            () => {
              const d = window.__assembliesDemo || window.__mujocoDemo;
              if (!d || typeof d.exportPolicyWeights !== "function") {
                return null;
              }
              return d.exportPolicyWeights();
            }
            """
        )
        if not base_weights:
            raise RuntimeError("failed to read policy weights from runtime")

        mean_vec = extract_train_vector(base_weights)
        dim = len(mean_vec)
        sigma = args.init_sigma

        baseline_raw = await evaluate_policy(
            page=page,
            weights=base_weights,
            steps=args.rollout_steps,
            use_messages=not args.disable_messages,
            random_torque_scale=args.eval_random_torque_scale,
        )
        baseline_metrics = baseline_raw["metrics"]
        baseline_score = score_metrics(baseline_metrics, args)

        best_score = baseline_score
        best_metrics = baseline_metrics
        best_weights = copy.deepcopy(base_weights)
        best_meta = {"source": "baseline", "generation": -1, "candidate": -1}
        generations: list[dict[str, Any]] = []

        for gen in range(args.generations):
            candidates: list[dict[str, Any]] = []
            for cand_idx in range(args.population):
                if gen == 0 and cand_idx == 0:
                    vec = mean_vec[:]
                else:
                    vec = [m + sigma * rng.gauss(0.0, 1.0) for m in mean_vec]
                cand_weights = inject_train_vector(base_weights, vec)
                raw = await evaluate_policy(
                    page=page,
                    weights=cand_weights,
                    steps=args.rollout_steps,
                    use_messages=not args.disable_messages,
                    random_torque_scale=args.eval_random_torque_scale,
                )
                metrics = raw["metrics"]
                score = score_metrics(metrics, args)
                candidates.append(
                    {
                        "candidate": cand_idx,
                        "score": score,
                        "metrics": metrics,
                        "vector": vec,
                    }
                )
                if score > best_score:
                    best_score = score
                    best_metrics = metrics
                    best_weights = cand_weights
                    best_meta = {"source": "candidate", "generation": gen, "candidate": cand_idx}

            candidates.sort(key=lambda x: x["score"], reverse=True)
            elite_count = min(args.elite, len(candidates))
            elites = candidates[:elite_count]
            mean_vec = [
                sum(candidate["vector"][j] for candidate in elites) / elite_count
                for j in range(dim)
            ]
            sigma *= args.sigma_decay

            generations.append(
                {
                    "generation": gen,
                    "sigma": sigma,
                    "best_score": candidates[0]["score"],
                    "mean_score": sum(c["score"] for c in candidates) / len(candidates),
                    "best_metrics": candidates[0]["metrics"],
                }
            )

        # Repeat-evaluate baseline and best to reduce single-rollout noise.
        baseline_repeat: list[dict[str, Any]] = []
        best_repeat: list[dict[str, Any]] = []
        for _ in range(max(1, args.final_eval_repeats)):
            baseline_eval = await evaluate_policy(
                page=page,
                weights=base_weights,
                steps=args.final_eval_steps,
                use_messages=not args.disable_messages,
                random_torque_scale=args.eval_random_torque_scale,
            )
            best_eval = await evaluate_policy(
                page=page,
                weights=best_weights,
                steps=args.final_eval_steps,
                use_messages=not args.disable_messages,
                random_torque_scale=args.eval_random_torque_scale,
            )
            baseline_repeat.append(baseline_eval["metrics"])
            best_repeat.append(best_eval["metrics"])

        def avg_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
            keys = rows[0].keys()
            return {k: sum(float(r[k]) for r in rows) / len(rows) for k in keys}

        baseline_final = avg_metrics(baseline_repeat)
        best_final = avg_metrics(best_repeat)

        output = {
            "created_utc": utc_now_iso(),
            "url": args.url,
            "seed": args.seed,
            "train_dim": dim,
            "objective": {
                "w_edge_mean": args.w_edge_mean,
                "w_edge_nonzero": args.w_edge_nonzero,
                "w_forward_x": args.w_forward_x,
                "w_displacement": args.w_displacement,
                "w_reward": args.w_reward,
            },
            "config": {
                "generations": args.generations,
                "population": args.population,
                "elite": args.elite,
                "init_sigma": args.init_sigma,
                "sigma_decay": args.sigma_decay,
                "rollout_steps": args.rollout_steps,
                "disable_messages": args.disable_messages,
                "eval_random_torque_scale": args.eval_random_torque_scale,
                "final_eval_steps": args.final_eval_steps,
                "final_eval_repeats": args.final_eval_repeats,
            },
            "baseline": {
                "single_rollout": baseline_metrics,
                "single_rollout_score": baseline_score,
                "final_eval_avg": baseline_final,
                "final_eval_avg_score": score_metrics(baseline_final, args),
            },
            "best": {
                "selection_meta": best_meta,
                "single_rollout": best_metrics,
                "single_rollout_score": best_score,
                "final_eval_avg": best_final,
                "final_eval_avg_score": score_metrics(best_final, args),
            },
            "generations": generations,
        }

        (run_dir / "training_summary.json").write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        (run_dir / "best_policy_weights.json").write_text(
            json.dumps(best_weights, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        (run_dir / "base_policy_weights.json").write_text(
            json.dumps(base_weights, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        await browser.close()

    return {"run_dir": str(run_dir), "summary": output}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8877/modular_assemblies.html?headless=1&reward=locomotion",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--init-sigma", type=float, default=0.06)
    parser.add_argument("--sigma-decay", type=float, default=0.92)
    parser.add_argument("--rollout-steps", type=int, default=420)
    parser.add_argument("--disable-messages", action="store_true")
    parser.add_argument("--eval-random-torque-scale", type=float, default=0.03)
    parser.add_argument("--final-eval-steps", type=int, default=700)
    parser.add_argument("--final-eval-repeats", type=int, default=3)
    parser.add_argument("--w-edge-mean", type=float, default=1.2)
    parser.add_argument("--w-edge-nonzero", type=float, default=0.8)
    parser.add_argument("--w-forward-x", type=float, default=5.0)
    parser.add_argument("--w-displacement", type=float, default=1.2)
    parser.add_argument("--w-reward", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generations <= 0:
        raise SystemExit("--generations must be positive")
    if args.population <= 0:
        raise SystemExit("--population must be positive")
    if args.elite <= 0:
        raise SystemExit("--elite must be positive")
    if args.elite > args.population:
        raise SystemExit("--elite cannot be greater than --population")
    if args.rollout_steps <= 1:
        raise SystemExit("--rollout-steps must be > 1")
    if args.final_eval_steps <= 1:
        raise SystemExit("--final-eval-steps must be > 1")
    if args.final_eval_repeats <= 0:
        raise SystemExit("--final-eval-repeats must be positive")
    if args.init_sigma <= 0:
        raise SystemExit("--init-sigma must be positive")
    if not (0.0 < args.sigma_decay <= 1.0):
        raise SystemExit("--sigma-decay must be in (0, 1]")

    result = asyncio.run(run_training(args))
    print(result["run_dir"])
    print(
        json.dumps(
            {
                "baseline_final_eval_score": result["summary"]["baseline"]["final_eval_avg_score"],
                "best_final_eval_score": result["summary"]["best"]["final_eval_avg_score"],
                "improvement": (
                    result["summary"]["best"]["final_eval_avg_score"]
                    - result["summary"]["baseline"]["final_eval_avg_score"]
                ),
                "best_selection": result["summary"]["best"]["selection_meta"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
