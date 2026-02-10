#!/usr/bin/env python3
"""Validate whether a single limb can stand upright without kick-assist terms."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "single_limb_validation"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_probe(url: str, steps: int, disable_control_torque: bool) -> dict[str, Any]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await page.wait_for_timeout(2_200)

        raw = await page.evaluate(
            """
            ({ stepCount, disableControlTorque }) => {
              const d = window.__assembliesDemo || window.__mujocoDemo;
              if (!d || !d.model || !d.data || typeof d.stepPhysics !== "function") {
                return { ok: false, reason: "No modular assemblies demo object found" };
              }
              if (!Array.isArray(d.limbBodyIds) || d.limbBodyIds.length !== 1) {
                return { ok: false, reason: `Expected exactly one limb, got ${d.limbBodyIds?.length ?? "unknown"}` };
              }
              if (d.renderer && typeof d.renderer.setAnimationLoop === "function") {
                d.renderer.setAnimationLoop(null);
              }
              if (typeof d.resetEpisode === "function") {
                d.resetEpisode();
              }

              // Disable all kick/assist terms to answer the "without kicks" question.
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
              if (disableControlTorque && typeof d.torqueScale === "number") d.torqueScale = 0.0;

              const limbBodyId = d.limbBodyIds[0];
              let radius = 0.0;
              for (let g = 0; g < d.model.ngeom; g++) {
                if (d.model.geom_bodyid[g] !== limbBodyId) continue;
                const s0 = d.model.geom_size[g * 3 + 0];
                if (Number.isFinite(s0) && s0 > radius) {
                  radius = s0;
                }
              }
              if (!(Number.isFinite(radius) && radius > 0.0)) {
                return { ok: false, reason: "Unable to infer limb radius" };
              }

              const sitePos = (siteId) => {
                const b = siteId * 3;
                return [d.data.site_xpos[b + 0], d.data.site_xpos[b + 1], d.data.site_xpos[b + 2]];
              };
              const childSiteId = d.childSiteIds[0];
              const parentSiteId = d.parentSiteIds[0];
              const child0 = sitePos(childSiteId);
              const parent0 = sitePos(parentSiteId);
              const dx0 = parent0[0] - child0[0];
              const dy0 = parent0[1] - child0[1];
              const dz0 = parent0[2] - child0[2];
              const centerlineLength = Math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
              const tipToTipLength = centerlineLength + 2.0 * radius;

              let maxTopZ = -1e9;
              let maxUprightCos = -1e9;
              let uprightCount = 0;
              let meanTopZ = 0.0;
              const samples = [];

              const N = Math.max(1, Math.floor(stepCount));
              for (let k = 0; k < N; k++) {
                d.stepPhysics(1);
                const c = sitePos(childSiteId);
                const p = sitePos(parentSiteId);
                const vx = p[0] - c[0];
                const vy = p[1] - c[1];
                const vz = p[2] - c[2];
                const norm = Math.sqrt(vx * vx + vy * vy + vz * vz);
                const uprightCos = norm > 1e-8 ? Math.abs(vz / norm) : 0.0;
                const topZ = Math.max(c[2], p[2]) + radius;
                if (uprightCos > maxUprightCos) maxUprightCos = uprightCos;
                if (topZ > maxTopZ) maxTopZ = topZ;
                if (uprightCos >= 0.9) uprightCount += 1;
                meanTopZ += topZ;
                if (k % Math.max(1, Math.floor(N / 12)) === 0 || k === N - 1) {
                  samples.push({ k, topZ, uprightCos });
                }
              }
              meanTopZ /= N;

              const topToLengthRatio = maxTopZ / Math.max(1e-8, tipToTipLength);
              const uprightFrac = uprightCount / N;
              const standsUpright = topToLengthRatio >= 0.75 && uprightFrac >= 0.05;

              return {
                ok: true,
                disableControlTorque,
                geometry: {
                  centerline_length: centerlineLength,
                  radius,
                  tip_to_tip_length: tipToTipLength,
                },
                stats: {
                  steps: N,
                  max_top_z: maxTopZ,
                  mean_top_z: meanTopZ,
                  max_upright_cos: maxUprightCos,
                  upright_fraction: uprightFrac,
                  top_to_length_ratio: topToLengthRatio,
                  stands_upright: standsUpright,
                },
                samples,
              };
            }
            """,
            {"stepCount": steps, "disableControlTorque": disable_control_torque},
        )
        await browser.close()
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8877/modular_assemblies.html?headless=1&scene=modular_assemblies_1limb.xml&reward=standing&noise=0",
    )
    parser.add_argument("--steps", type=int, default=4500)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    active = asyncio.run(run_probe(args.url, args.steps, disable_control_torque=False))
    if not active.get("ok"):
        raise SystemExit(f"Probe failed (active torque): {active}")
    passive = asyncio.run(run_probe(args.url, args.steps, disable_control_torque=True))
    if not passive.get("ok"):
        raise SystemExit(f"Probe failed (torque off): {passive}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"single_limb_standing_report_{run_id}.json"
    output = {
        "created_utc": utc_now_iso(),
        "url": args.url,
        "steps": args.steps,
        "active_torque": active,
        "passive_no_torque": passive,
    }
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(out_path)
    print(
        json.dumps(
            {
                "tip_to_tip_length": active["geometry"]["tip_to_tip_length"],
                "active_max_top_z": active["stats"]["max_top_z"],
                "active_top_to_length_ratio": active["stats"]["top_to_length_ratio"],
                "active_stands_upright": active["stats"]["stands_upright"],
                "passive_max_top_z": passive["stats"]["max_top_z"],
                "passive_top_to_length_ratio": passive["stats"]["top_to_length_ratio"],
                "passive_stands_upright": passive["stats"]["stands_upright"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
