#!/usr/bin/env python3
"""Optimize a single-limb policy for sustained upright stability with CEM."""

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
    vec.extend(float(v) for v in weights["b1"])
    return vec


def inject_train_vector(base_weights: dict[str, Any], vec: list[float]) -> dict[str, Any]:
    out = copy.deepcopy(base_weights)
    idx = 0
    for i in range(len(out["wTorque"])):
        for j in range(len(out["wTorque"][i])):
            out["wTorque"][i][j] = float(vec[idx])
            idx += 1
    for i in range(len(out["b1"])):
        out["b1"][i] = float(vec[idx])
        idx += 1
    if idx != len(vec):
        raise ValueError(f"vector length mismatch: consumed {idx} but got {len(vec)}")
    return out


def score_metrics(metrics: dict[str, float], args: argparse.Namespace) -> float:
    return (
        args.w_mean_top * metrics["mean_top_norm"]
        + args.w_upright * metrics["upright_frac"]
        + args.w_tail_upright * metrics["tail_upright_frac"]
        + args.w_tail_top * metrics["tail_mean_top_norm"]
        + args.w_streak * metrics["longest_upright_streak_frac"]
        - args.w_top_std * metrics["std_top_norm"]
        - args.w_upright_std * metrics["std_upright"]
        - args.w_angvel * metrics["mean_ang_speed"]
    )


async def evaluate_policy(
    page: Any,
    weights: dict[str, Any],
    steps: int,
    torque_scale: float,
) -> dict[str, Any]:
    raw = await page.evaluate(
        """
        ({ weights, stepCount, torqueScale }) => {
          const d = window.__assembliesDemo || window.__mujocoDemo;
          if (!d || !d.model || !d.data || typeof d.stepPhysics !== "function") {
            return { ok: false, reason: "runtime APIs unavailable" };
          }
          if (!Array.isArray(d.limbBodyIds) || d.limbBodyIds.length !== 1) {
            return { ok: false, reason: `expected one limb, got ${d.limbBodyIds?.length ?? "unknown"}` };
          }
          if (d.renderer && typeof d.renderer.setAnimationLoop === "function") {
            d.renderer.setAnimationLoop(null);
          }
          if (typeof d.resetEpisode === "function") {
            d.resetEpisode();
          }

          // Disable all assists: optimize pure torque control.
          const zeroFields = [
            "stallPulseLift", "stallPulseForward", "maxDynamicNoiseBoost",
            "stallKickVelZ", "stallKickVelX", "standingLiftGain",
            "standingLateralPulse", "standingKickVelZ", "standingKickVelY",
            "standingInwardGain", "standingTorqueGain",
          ];
          for (const key of zeroFields) {
            if (typeof d[key] === "number") d[key] = 0.0;
          }
          if (typeof d.stallDetachEvery === "number") d.stallDetachEvery = 0;
          if (typeof d.standingDetachEvery === "number") d.standingDetachEvery = 0;
          if (typeof d.randomTorqueScale === "number") d.randomTorqueScale = 0.0;
          if (typeof d.torqueScale === "number") d.torqueScale = torqueScale;

          if (typeof d.loadPolicyWeights !== "function") {
            return { ok: false, reason: "missing loadPolicyWeights" };
          }
          d.loadPolicyWeights(weights);

          const limbIdx = 0;
          const bodyId = d.limbBodyIds[0];
          const sitePos = (siteId) => {
            const b = siteId * 3;
            return [d.data.site_xpos[b + 0], d.data.site_xpos[b + 1], d.data.site_xpos[b + 2]];
          };
          const childSiteId = d.childSiteIds[0];
          const parentSiteId = d.parentSiteIds[0];

          let radius = 0.0;
          for (let g = 0; g < d.model.ngeom; g++) {
            if (d.model.geom_bodyid[g] !== bodyId) continue;
            const s0 = d.model.geom_size[g * 3 + 0];
            if (Number.isFinite(s0) && s0 > radius) {
              radius = s0;
            }
          }
          if (!(Number.isFinite(radius) && radius > 0.0)) {
            return { ok: false, reason: "unable to infer limb radius" };
          }

          const c0 = sitePos(childSiteId);
          const p0 = sitePos(parentSiteId);
          const dx0 = p0[0] - c0[0];
          const dy0 = p0[1] - c0[1];
          const dz0 = p0[2] - c0[2];
          const centerlineLength = Math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
          const tipToTipLength = centerlineLength + 2.0 * radius;
          const invLen = 1.0 / Math.max(1e-8, tipToTipLength);

          const N = Math.max(1, Math.floor(stepCount));
          const tailStart = Math.floor(N * 0.7);

          let sumTop = 0.0;
          let sumTop2 = 0.0;
          let sumUpright = 0.0;
          let sumUpright2 = 0.0;
          let sumAng = 0.0;
          let uprightCount = 0;
          let tailTop = 0.0;
          let tailUprightCount = 0;
          let tailCount = 0;
          let maxTopNorm = -1e9;
          let maxUpright = -1e9;
          let longestStreak = 0;
          let streak = 0;

          for (let k = 0; k < N; k++) {
            d.stepPhysics(1);
            const c = sitePos(childSiteId);
            const p = sitePos(parentSiteId);
            const vx = p[0] - c[0];
            const vy = p[1] - c[1];
            const vz = p[2] - c[2];
            const axisNorm = Math.sqrt(vx * vx + vy * vy + vz * vz);
            const uprightCos = axisNorm > 1e-8 ? Math.abs(vz / axisNorm) : 0.0;
            const topNorm = (Math.max(c[2], p[2]) + radius) * invLen;
            const cvelBase = bodyId * 6;
            const wx = d.data.cvel[cvelBase + 0];
            const wy = d.data.cvel[cvelBase + 1];
            const wz = d.data.cvel[cvelBase + 2];
            const angSpeed = Math.sqrt(wx * wx + wy * wy + wz * wz);

            sumTop += topNorm;
            sumTop2 += topNorm * topNorm;
            sumUpright += uprightCos;
            sumUpright2 += uprightCos * uprightCos;
            sumAng += angSpeed;
            if (topNorm > maxTopNorm) maxTopNorm = topNorm;
            if (uprightCos > maxUpright) maxUpright = uprightCos;

            const uprightNow = uprightCos >= 0.9 && topNorm >= 0.88;
            if (uprightNow) {
              uprightCount += 1;
              streak += 1;
              if (streak > longestStreak) longestStreak = streak;
            } else {
              streak = 0;
            }

            if (k >= tailStart) {
              tailCount += 1;
              tailTop += topNorm;
              if (uprightNow) tailUprightCount += 1;
            }
          }

          const meanTopNorm = sumTop / N;
          const meanUpright = sumUpright / N;
          const meanAngSpeed = sumAng / N;
          const varTop = Math.max(0.0, sumTop2 / N - meanTopNorm * meanTopNorm);
          const varUpright = Math.max(0.0, sumUpright2 / N - meanUpright * meanUpright);
          const stdTopNorm = Math.sqrt(varTop);
          const stdUpright = Math.sqrt(varUpright);

          return {
            ok: true,
            metrics: {
              steps: N,
              tip_to_tip_length: tipToTipLength,
              mean_top_norm: meanTopNorm,
              std_top_norm: stdTopNorm,
              max_top_norm: maxTopNorm,
              mean_upright: meanUpright,
              std_upright: stdUpright,
              max_upright: maxUpright,
              upright_frac: uprightCount / N,
              tail_mean_top_norm: tailCount > 0 ? tailTop / tailCount : 0.0,
              tail_upright_frac: tailCount > 0 ? tailUprightCount / tailCount : 0.0,
              longest_upright_streak_frac: longestStreak / N,
              mean_ang_speed: meanAngSpeed,
            },
          };
        }
        """,
        {
            "weights": weights,
            "stepCount": steps,
            "torqueScale": torque_scale,
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

        runtime_weights = await page.evaluate(
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
        if not runtime_weights:
            raise RuntimeError("failed to read policy weights from runtime")

        if args.init_weights:
            init_path = Path(args.init_weights).resolve()
            base_weights = json.loads(init_path.read_text(encoding="utf-8"))
        else:
            base_weights = runtime_weights

        mean_vec = extract_train_vector(base_weights)
        sigma = args.init_sigma

        baseline_raw = await evaluate_policy(page, base_weights, args.rollout_steps, args.torque_scale)
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
                weights = inject_train_vector(base_weights, vec)
                raw = await evaluate_policy(page, weights, args.rollout_steps, args.torque_scale)
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

            candidates.sort(key=lambda x: x["score"], reverse=True)
            elite = candidates[: args.elite]
            dim = len(mean_vec)
            new_mean = []
            for i in range(dim):
                new_mean.append(sum(entry["vector"][i] for entry in elite) / len(elite))
            mean_vec = new_mean
            sigma *= args.sigma_decay

            gen_best = candidates[0]
            if gen_best["score"] > best_score:
                best_score = gen_best["score"]
                best_metrics = gen_best["metrics"]
                best_weights = inject_train_vector(base_weights, gen_best["vector"])
                best_meta = {
                    "source": "candidate",
                    "generation": gen,
                    "candidate": gen_best["candidate"],
                }

            generations.append(
                {
                    "generation": gen,
                    "sigma": sigma,
                    "best_score": gen_best["score"],
                    "best_metrics": gen_best["metrics"],
                    "mean_score": sum(x["score"] for x in candidates) / len(candidates),
                    "elite_scores": [x["score"] for x in elite],
                }
            )

        baseline_finals: list[dict[str, float]] = []
        best_finals: list[dict[str, float]] = []
        for _ in range(args.final_eval_repeats):
            baseline_eval = await evaluate_policy(
                page,
                base_weights,
                args.final_eval_steps,
                args.torque_scale,
            )
            best_eval = await evaluate_policy(
                page,
                best_weights,
                args.final_eval_steps,
                args.torque_scale,
            )
            baseline_finals.append(baseline_eval["metrics"])
            best_finals.append(best_eval["metrics"])

        def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
            keys = rows[0].keys()
            out: dict[str, float] = {}
            for k in keys:
                out[k] = sum(float(r[k]) for r in rows) / len(rows)
            return out

        baseline_final = mean_metrics(baseline_finals)
        best_final = mean_metrics(best_finals)

        summary = {
            "created_utc": utc_now_iso(),
            "url": args.url,
            "seed": args.seed,
            "torque_scale": args.torque_scale,
            "init_weights": str(Path(args.init_weights).resolve()) if args.init_weights else None,
            "weights": {
                "w_mean_top": args.w_mean_top,
                "w_upright": args.w_upright,
                "w_tail_upright": args.w_tail_upright,
                "w_tail_top": args.w_tail_top,
                "w_streak": args.w_streak,
                "w_top_std": args.w_top_std,
                "w_upright_std": args.w_upright_std,
                "w_angvel": args.w_angvel,
            },
            "config": {
                "generations": args.generations,
                "population": args.population,
                "elite": args.elite,
                "init_sigma": args.init_sigma,
                "sigma_decay": args.sigma_decay,
                "rollout_steps": args.rollout_steps,
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
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "best_policy_weights.json").write_text(
            json.dumps(best_weights, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        (run_dir / "base_policy_weights.json").write_text(
            json.dumps(base_weights, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        await browser.close()

    return {"run_dir": str(run_dir), "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=(
            "http://127.0.0.1:8877/modular_assemblies.html?headless=1&scene=modular_assemblies_1limb.xml"
            "&reward=standing&noise=0&torque_scale=8"
            "&stand_lift_gain=0&stand_kick_vz=0&stand_kick_vy=0&stand_inward=0&stand_torque=0&stand_lat=0"
            "&stall_lift=0&stall_fwd=0&stall_kick_vz=0&stall_kick_vx=0&max_boost=0"
            "&stall_detach_every=0&stand_detach_every=0"
        ),
    )
    parser.add_argument("--init-weights", default="")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--torque-scale", type=float, default=8.0)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--init-sigma", type=float, default=0.06)
    parser.add_argument("--sigma-decay", type=float, default=0.97)
    parser.add_argument("--rollout-steps", type=int, default=1200)
    parser.add_argument("--final-eval-steps", type=int, default=3000)
    parser.add_argument("--final-eval-repeats", type=int, default=6)
    parser.add_argument("--w-mean-top", type=float, default=2.0)
    parser.add_argument("--w-upright", type=float, default=3.2)
    parser.add_argument("--w-tail-upright", type=float, default=5.0)
    parser.add_argument("--w-tail-top", type=float, default=2.4)
    parser.add_argument("--w-streak", type=float, default=4.2)
    parser.add_argument("--w-top-std", type=float, default=1.6)
    parser.add_argument("--w-upright-std", type=float, default=2.0)
    parser.add_argument("--w-angvel", type=float, default=0.12)
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
    if args.torque_scale <= 0:
        raise SystemExit("--torque-scale must be positive")

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
                "best_final_metrics": result["summary"]["best"]["final_eval_avg"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
