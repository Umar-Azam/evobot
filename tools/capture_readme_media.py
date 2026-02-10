#!/usr/bin/env python3
"""Capture README screenshots/videos for key scenarios at torque_scale=6.5."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from playwright.async_api import async_playwright


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEDIA_DIR = REPO_ROOT / "docs" / "media"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8877


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_runtime_assets() -> None:
    required = [
        REPO_ROOT / "node_modules" / "three" / "build" / "three.module.js",
        REPO_ROOT / "node_modules" / "three" / "examples" / "jsm" / "controls" / "OrbitControls.js",
        REPO_ROOT / "node_modules" / "mujoco-js" / "dist" / "mujoco_wasm.js",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise SystemExit(
            "Missing required runtime assets for capture. Run `npm install` in repo root first.\n"
            + lines
        )


def force_torque_scale(url: str, value: float) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["torque_scale"] = f"{value:g}"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


@dataclass(frozen=True)
class Scenario:
    slug: str
    title: str
    description: str
    url: str
    warmup_ms: int
    record_ms: int


SCENARIOS = [
    Scenario(
        slug="single_limb_standing",
        title="Single-Limb Upright Stability",
        description="Single sphere-cylinder limb, standing reward, no kick-assist terms.",
        url=(
            "http://127.0.0.1:8877/modular_assemblies.html"
            "?scene=modular_assemblies_1limb.xml"
            "&reward=standing"
            "&policy=assets/policies/single_limb_standing_stability_stage2_best.json"
            "&noise=0"
            "&stand_lift_gain=0"
            "&stand_kick_vz=0"
            "&stand_kick_vy=0"
            "&stand_inward=0"
            "&stand_torque=0"
            "&stand_lat=0"
            "&stall_lift=0"
            "&stall_fwd=0"
            "&stall_kick_vz=0"
            "&stall_kick_vx=0"
            "&max_boost=0"
            "&stall_detach_every=0"
            "&stand_detach_every=0"
        ),
        warmup_ms=2800,
        record_ms=5200,
    ),
    Scenario(
        slug="eight_limb_standing",
        title="Eight-Limb Standing",
        description="8-limb assemblies with geometry-aware standing policy.",
        url=(
            "http://127.0.0.1:8877/modular_assemblies.html"
            "?scene=modular_assemblies_8limb.xml"
            "&reward=standing"
            "&policy=assets/policies/standing_8limb_geomv2_best.json"
            "&noise=0.02"
        ),
        warmup_ms=3200,
        record_ms=5600,
    ),
    Scenario(
        slug="eight_limb_locomotion",
        title="Eight-Limb Locomotion",
        description="8-limb assemblies with geometry-aware locomotion policy.",
        url=(
            "http://127.0.0.1:8877/modular_assemblies.html"
            "?scene=modular_assemblies_8limb.xml"
            "&reward=locomotion"
            "&policy=assets/policies/locomotion_8limb_geomv2_best.json"
            "&noise=0.02"
        ),
        warmup_ms=3200,
        record_ms=5600,
    ),
]


def start_server(host: str, port: int, root: Path) -> tuple[ThreadingHTTPServer, threading.Thread]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


async def capture_scenario(
    browser,
    scenario: Scenario,
    out_dir: Path,
    torque_scale: float,
    viewport_w: int,
    viewport_h: int,
) -> dict:
    scenario_dir = out_dir / scenario.slug
    scenario_dir.mkdir(parents=True, exist_ok=True)
    tmp_video_dir = scenario_dir / "_tmp_video"
    tmp_video_dir.mkdir(parents=True, exist_ok=True)

    context = await browser.new_context(
        viewport={"width": viewport_w, "height": viewport_h},
        record_video_dir=str(tmp_video_dir),
        record_video_size={"width": viewport_w, "height": viewport_h},
    )
    page = await context.new_page()
    url = force_torque_scale(scenario.url, torque_scale)
    await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    await page.wait_for_function(
        """
        () => {
          const d = window.__assembliesDemo || window.__mujocoDemo;
          return !!(d && d.model && d.data && typeof d.stepPhysics === "function");
        }
        """,
        timeout=60_000,
    )
    await page.evaluate(
        """
        () => {
          const d = window.__assembliesDemo || window.__mujocoDemo;
          if (!d) return;
          if (d.renderer && typeof d.renderer.setAnimationLoop === "function") {
            d.renderer.setAnimationLoop(null);
          }
          if (typeof d.resetEpisode === "function") {
            d.resetEpisode();
          }
        }
        """
    )

    warmup_frames = max(1, scenario.warmup_ms // 16)
    await page.evaluate(
        """
        async ({ frames, substeps }) => {
          const d = window.__assembliesDemo || window.__mujocoDemo;
          for (let i = 0; i < frames; i++) {
            d.stepPhysics(substeps);
          }
          if (typeof d.syncVisualTransforms === "function") {
            d.syncVisualTransforms();
          }
          if (d.renderer && d.scene && d.camera) {
            d.renderer.render(d.scene, d.camera);
          }
          return { simTime: d.simTimeSec, controlTick: d.controlTick };
        }
        """,
        {"frames": warmup_frames, "substeps": 3},
    )

    screenshot_path = scenario_dir / f"{scenario.slug}.png"
    await page.screenshot(path=str(screenshot_path), full_page=False)

    record_frames = max(30, scenario.record_ms // 33)
    sim_state = await page.evaluate(
        """
        async ({ frames, substeps, delayMs }) => {
          const d = window.__assembliesDemo || window.__mujocoDemo;
          for (let i = 0; i < frames; i++) {
            d.stepPhysics(substeps);
            if (typeof d.syncVisualTransforms === "function") {
              d.syncVisualTransforms();
            }
            if (d.renderer && d.scene && d.camera) {
              d.renderer.render(d.scene, d.camera);
            }
            await new Promise((resolve) => setTimeout(resolve, delayMs));
          }
          return { simTime: d.simTimeSec, controlTick: d.controlTick };
        }
        """,
        {"frames": record_frames, "substeps": 2, "delayMs": 33},
    )
    video = page.video
    await page.close()
    video_tmp = Path(await video.path())
    await context.close()

    video_path = scenario_dir / f"{scenario.slug}.webm"
    if video_path.exists():
        video_path.unlink()
    shutil.move(str(video_tmp), str(video_path))
    shutil.rmtree(tmp_video_dir, ignore_errors=True)

    return {
        "slug": scenario.slug,
        "title": scenario.title,
        "description": scenario.description,
        "url": url,
        "screenshot": str(screenshot_path.relative_to(REPO_ROOT)),
        "video": str(video_path.relative_to(REPO_ROOT)),
        "warmup_ms": scenario.warmup_ms,
        "record_ms": scenario.record_ms,
        "end_state": sim_state,
    }


async def run_capture(args: argparse.Namespace) -> dict:
    media_dir = Path(args.out_dir).resolve()
    media_dir.mkdir(parents=True, exist_ok=True)
    server, thread = start_server(args.host, args.port, REPO_ROOT)
    del thread
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                executable_path="/usr/bin/google-chrome",
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            rows = []
            for scenario in SCENARIOS:
                rows.append(
                    await capture_scenario(
                        browser=browser,
                        scenario=scenario,
                        out_dir=media_dir,
                        torque_scale=args.torque_scale,
                        viewport_w=args.width,
                        viewport_h=args.height,
                    )
                )
            await browser.close()
    finally:
        server.shutdown()
        server.server_close()

    manifest = {
        "created_utc": utc_now_iso(),
        "torque_scale": args.torque_scale,
        "viewport": {"width": args.width, "height": args.height},
        "scenarios": rows,
    }
    manifest_path = media_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_MEDIA_DIR))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--torque-scale", type=float, default=6.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_runtime_assets()
    if args.port <= 0 or args.port > 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if args.torque_scale <= 0:
        raise SystemExit("--torque-scale must be positive")
    manifest = asyncio.run(run_capture(args))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
