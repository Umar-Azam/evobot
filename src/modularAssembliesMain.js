import * as THREE from "three";
import { OrbitControls } from "../vendor/three/examples/jsm/controls/OrbitControls.js";
import load_mujoco from "../vendor/mujoco-js/mujoco_wasm.js";
import { loadSceneFromURL, getPosition, getQuaternion } from "./mujocoUtils.js";
import { AssemblyGraphManager } from "./assemblies/AssemblyGraphManager.js";
import { SimpleDGNPolicy } from "./assemblies/SimpleDGNPolicy.js";

const mujoco = await load_mujoco();
const bootQuery = typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;
const requestedScene = bootQuery ? (bootQuery.get("scene") || "").trim() : "";
const safeScene = /^[A-Za-z0-9_.-]+$/.test(requestedScene) ? requestedScene : "";
const sceneFile = safeScene || "modular_assemblies_4limb.xml";
mujoco.FS.mkdir("/working");
mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");
mujoco.FS.writeFile(
  `/working/${sceneFile}`,
  await (await fetch(`./assets/scenes/${sceneFile}`)).text()
);

function decodeName(model, addr) {
  const names = new Uint8Array(model.names);
  let end = addr;
  while (end < names.length && names[end] !== 0) {
    end += 1;
  }
  return new TextDecoder("utf-8").decode(names.subarray(addr, end));
}

function getEqName(model, eqId) {
  const addr = model.name_eqadr[eqId];
  if (addr < 0) {
    return "";
  }
  return decodeName(model, addr);
}

function getBodyIdByName(model, name) {
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, name);
}

function getBodyName(model, bodyId) {
  const addr = model.name_bodyadr[bodyId];
  if (addr < 0) {
    return "";
  }
  return decodeName(model, addr);
}

function getSiteIdByName(model, name) {
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, name);
}

function discoverLimbNames(model) {
  const names = [];
  for (let bodyId = 1; bodyId < model.nbody; bodyId++) {
    const name = getBodyName(model, bodyId);
    const match = name.match(/^limb(\d+)$/);
    if (!match) {
      continue;
    }
    names.push({ name, index: Number(match[1]) });
  }
  names.sort((a, b) => a.index - b.index);
  return names.map((x) => x.name);
}

function queryNumber(query, key, fallback) {
  if (!query) {
    return fallback;
  }
  const raw = query.get(key);
  if (raw === null || raw.trim() === "") {
    return fallback;
  }
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

export class ModularAssembliesDemo {
  constructor() {
    const query = typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;
    this.headlessMode = query ? query.get("headless") === "1" : false;
    const rewardQuery = query ? (query.get("reward") || "").toLowerCase() : "";
    this.rewardMode = rewardQuery === "locomotion" ? "locomotion" : "standing";
    this.useMessages = !query || query.get("messages") !== "0";
    this.policyURL = query ? query.get("policy") || "" : "";
    this.forwardAxis = 0; // +x direction
    this.prevComponentForward = {};
    this.mujoco = mujoco;
    this.model = mujoco.MjModel.loadFromXML(`/working/${sceneFile}`);
    this.data = new mujoco.MjData(this.model);
    this.disableReflector = true;
    this.enableShadows = false;
    this.bodies = {};
    this.lights = {};
    this.mujocoRoot = null;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0.10, 0.14, 0.18);
    this.scene.fog = new THREE.Fog(this.scene.background, 8, 18);

    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 100);
    this.camera.position.set(1.8, 1.35, 1.8);
    this.scene.add(this.camera);
    if (!this.headlessMode) {
      const floorTint = new THREE.Mesh(
        new THREE.PlaneGeometry(8, 8),
        new THREE.MeshStandardMaterial({
          color: 0x7e858d,
          roughness: 1.0,
          metalness: 0.0,
          transparent: true,
          opacity: 0.22,
        })
      );
      floorTint.rotation.x = -Math.PI / 2;
      floorTint.position.y = 0.001;
      this.scene.add(floorTint);

      const floorGrid = new THREE.GridHelper(8, 32, 0x4e5c67, 0x44505a);
      floorGrid.position.y = 0.002;
      this.scene.add(floorGrid);
    }

    this.overlay = null;
    if (!this.headlessMode) {
      this.renderer = new THREE.WebGLRenderer({ antialias: true });
      this.renderer.setPixelRatio(1.0);
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.renderer.shadowMap.enabled = this.enableShadows;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      this.renderer.useLegacyLights = true;

      document.body.appendChild(this.renderer.domElement);
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.target.set(0.0, 0.25, 0.0);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.update();

      window.addEventListener("resize", this.onResize.bind(this));
      this.renderer.setAnimationLoop(this.render.bind(this));

      this.overlay = document.createElement("div");
      this.overlay.style.position = "absolute";
      this.overlay.style.top = "12px";
      this.overlay.style.left = "12px";
      this.overlay.style.padding = "10px 12px";
      this.overlay.style.font = "12px monospace";
      this.overlay.style.color = "#e8f5ff";
      this.overlay.style.background = "rgba(10, 18, 24, 0.74)";
      this.overlay.style.border = "1px solid rgba(210, 230, 245, 0.25)";
      this.overlay.style.borderRadius = "8px";
      this.overlay.style.zIndex = "1000";
      this.overlay.style.whiteSpace = "pre";
      document.body.appendChild(this.overlay);
    } else {
      this.renderer = null;
      this.controls = { update() {} };
    }

    this.timeMS = 0.0;
    this.simTimeSec = 0.0;
    this.lastControlTime = -1.0;
    this.controlDt = 0.05;
    this.controlTick = 0;
    this.joinDist = 0.5;
    this.torqueScale = queryNumber(query, "torque_scale", 0.9);
    this.attachThreshold = 0.58;
    this.detachThreshold = 0.92;
    this.randomTorqueScale = 0.08;
    this.attractionForceScale = 0.65;
    this.attractionMaxDist = 0.9;
    // Locomotion shaping and anti-stall controls.
    this.locomotionHeightWeight = queryNumber(query, "loc_h_w", 0.45);
    this.locomotionHeightBaseline = queryNumber(query, "loc_h0", 0.075);
    this.locomotionStallSpeed = queryNumber(query, "loc_v_eps", 0.018);
    this.locomotionStallPenalty = queryNumber(query, "loc_stall_pen", 0.018);
    this.stallTickThreshold = Math.max(0, Math.floor(queryNumber(query, "stall_ticks", 8)));
    this.maxDynamicNoiseBoost = Math.max(0.0, queryNumber(query, "max_boost", 0.2));
    this.stallPulseLift = Math.max(0.0, queryNumber(query, "stall_lift", 0.34));
    this.stallPulseForward = Math.max(0.0, queryNumber(query, "stall_fwd", 0.11));
    this.stallDetachEvery = Math.max(0, Math.floor(queryNumber(query, "stall_detach_every", 18)));
    this.stallKickVelZ = Math.max(0.0, queryNumber(query, "stall_kick_vz", 1.2));
    this.stallKickVelX = Math.max(0.0, queryNumber(query, "stall_kick_vx", 0.6));
    this.stallKickEvery = Math.max(1, Math.floor(queryNumber(query, "stall_kick_every", 6)));
    // Standing-mode anti-flat assistance.
    this.standingTargetHeight = queryNumber(query, "stand_h_target", 0.22);
    this.standingLiftGain = Math.max(0.0, queryNumber(query, "stand_lift_gain", 1.8));
    this.standingDetachEvery = Math.max(0, Math.floor(queryNumber(query, "stand_detach_every", 16)));
    this.standingLateralPulse = Math.max(0.0, queryNumber(query, "stand_lat", 0.09));
    this.standingStallVx = Math.max(0.0, queryNumber(query, "stand_vx_eps", 0.012));
    this.standingInwardGain = Math.max(0.0, queryNumber(query, "stand_inward", 0.85));
    this.standingTorqueGain = Math.max(0.0, queryNumber(query, "stand_torque", 0.55));
    this.standingKickVelZ = Math.max(0.0, queryNumber(query, "stand_kick_vz", 1.6));
    this.standingKickVelY = Math.max(0.0, queryNumber(query, "stand_kick_vy", 0.5));
    this.standingKickEvery = Math.max(1, Math.floor(queryNumber(query, "stand_kick_every", 6)));
    this.maxAbsQvel = 14.0;
    this.maxAbsPos = 6.0;
    this.maxAbsCoord = 8.0;
    this.observationDim = 16;
    this.messageDim = 8;
    const noiseQuery = query ? Number(query.get("noise")) : NaN;
    if (Number.isFinite(noiseQuery) && noiseQuery >= 0.0) {
      this.randomTorqueScale = noiseQuery;
    }
    this.currentTorques = [];
    this.limbRewards = [];
    this.componentDebug = [];
    this.policySource = this.policyURL ? `pending:${this.policyURL}` : "default";
    this.centroidStartX = null;
    this.centroidX = 0.0;
    this.deltaControlX = 0.0;
    this.deltaFromStartX = 0.0;
    this.lastCentroidX = null;
    this.stallTicks = 0;
    this.dynamicNoiseBoost = 0.0;
    this.lastStallDetachTick = -1;
    this.standingMeanHeight = 0.0;
    this.standingLiftAssist = 0.0;
    this.standingStallTicks = 0;
    this.lastStandingDetachTick = -1;
    this.latestActions = [];
    this.telemetryHistory = [];
    this.telemetryHistoryLimit = 512;
    this.lastSnapshot = null;
  }

  async init() {
    [this.model, this.data, this.bodies, this.lights] = await loadSceneFromURL(mujoco, sceneFile, this);
    this.model.opt.disableflags = this.model.opt.disableflags & ~mujoco.mjtDisableBit.mjDSBL_EQUALITY.value;

    this.limbNames = discoverLimbNames(this.model);
    if (this.limbNames.length < 1) {
      throw new Error("Expected at least one limb body named limb0..limbN in scene");
    }
    this.limbBodyIds = this.limbNames.map((name) => getBodyIdByName(this.model, name));
    this.parentSiteIds = this.limbNames.map((name) => getSiteIdByName(this.model, `${name}_parent`));
    this.childSiteIds = this.limbNames.map((name) => getSiteIdByName(this.model, `${name}_child`));
    if (this.parentSiteIds.some((x) => x < 0) || this.childSiteIds.some((x) => x < 0)) {
      throw new Error("Each limb body must define both child and parent sites");
    }
    this.limbJointIds = this.limbBodyIds.map((bodyId) => this.model.body_jntadr[bodyId]);
    this.limbQposAdr = this.limbJointIds.map((jointId) => this.model.jnt_qposadr[jointId]);
    this.limbQvelAdr = this.limbJointIds.map((jointId) => this.model.jnt_dofadr[jointId]);
    this.limbOuterRadius = this.limbBodyIds.map((bodyId) => this.estimateLimbOuterRadius(bodyId));

    this.directedConstraintIndex = {};
    for (let e = 0; e < this.model.neq; e++) {
      const name = getEqName(this.model, e);
      const match = name.match(/^conn_(\d+)_(\d+)$/);
      if (!match) {
        continue;
      }
      const key = `${match[1]}->${match[2]}`;
      this.directedConstraintIndex[key] = e;
      this.data.eq_active[e] = 0;
    }

    this.graph = new AssemblyGraphManager(this.limbBodyIds.length, this.directedConstraintIndex, this.joinDist);
    this.policy = new SimpleDGNPolicy(this.observationDim, this.messageDim, 16);
    this.currentTorques = Array.from({ length: this.limbBodyIds.length }, () => [0, 0, 0]);
    this.limbRewards = new Array(this.limbBodyIds.length).fill(0.0);
    this.latestActions = Array.from({ length: this.limbBodyIds.length }, () => ({ attachProb: 0, detachProb: 0 }));

    this.mujoco.mj_forward(this.model, this.data);
    this.simTimeSec = 0.0;
    this.lastControlTime = -1.0;
    this.initialLimbQpos = this.limbQposAdr.map((adr) => Array.from(this.data.qpos.slice(adr, adr + 7)));
    this.prevComponentForward = {};
    if (this.policyURL) {
      await this.loadPolicyWeightsFromURL(this.policyURL);
    }
    this.lastStallDetachTick = -1;
    this.lastStandingDetachTick = -1;
    this.computeRewards();
    this.updateCentroidMetrics();
    this.updateStallState();
    this.updateStandingState();
    this.pushSnapshot();
    this.updateOverlay();
  }

  onResize() {
    if (!this.renderer) {
      return;
    }
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  sitePos(siteId) {
    const base = siteId * 3;
    return [
      this.data.site_xpos[base + 0],
      this.data.site_xpos[base + 1],
      this.data.site_xpos[base + 2],
    ];
  }

  limbBodyPos(limbIdx) {
    const bodyId = this.limbBodyIds[limbIdx];
    const posSrc = this.data.xipos || this.data.xpos;
    const base = bodyId * 3;
    return [
      posSrc[base + 0],
      posSrc[base + 1],
      posSrc[base + 2],
    ];
  }

  estimateLimbOuterRadius(bodyId) {
    let radius = 0.02;
    for (let geomId = 0; geomId < this.model.ngeom; geomId++) {
      if (this.model.geom_bodyid[geomId] !== bodyId) {
        continue;
      }
      const r = this.model.geom_size[geomId * 3 + 0];
      if (Number.isFinite(r)) {
        radius = Math.max(radius, r);
      }
    }
    return radius;
  }

  limbTopHeight(limbIdx) {
    const child = this.sitePos(this.childSiteIds[limbIdx]);
    const parent = this.sitePos(this.parentSiteIds[limbIdx]);
    const radius = this.limbOuterRadius?.[limbIdx] ?? 0.0;
    return Math.max(child[2], parent[2]) + radius;
  }

  limbMidpointXY(limbIdx) {
    const child = this.sitePos(this.childSiteIds[limbIdx]);
    const parent = this.sitePos(this.parentSiteIds[limbIdx]);
    return [(child[0] + parent[0]) * 0.5, (child[1] + parent[1]) * 0.5];
  }

  computeCentroidX() {
    if (!this.limbBodyIds || this.limbBodyIds.length === 0) {
      return 0.0;
    }
    const posSrc = this.data.xipos || this.data.xpos;
    let out = 0.0;
    for (const bodyId of this.limbBodyIds) {
      out += posSrc[bodyId * 3 + this.forwardAxis];
    }
    return out / this.limbBodyIds.length;
  }

  updateCentroidMetrics() {
    const current = this.computeCentroidX();
    if (this.centroidStartX === null) {
      this.centroidStartX = current;
      this.lastCentroidX = current;
    }
    this.deltaControlX = current - (this.lastCentroidX ?? current);
    this.lastCentroidX = current;
    this.centroidX = current;
    this.deltaFromStartX = current - (this.centroidStartX ?? current);
  }

  updateStallState() {
    if (this.rewardMode !== "locomotion") {
      this.stallTicks = 0;
      this.dynamicNoiseBoost = 0.0;
      return;
    }
    let meanAbsVx = 0.0;
    let vxCount = 0;
    for (const entry of this.componentDebug) {
      if (entry.metricLabel !== "vx" || !Number.isFinite(entry.metricValue)) {
        continue;
      }
      meanAbsVx += Math.abs(entry.metricValue);
      vxCount += 1;
    }
    meanAbsVx = vxCount > 0 ? meanAbsVx / vxCount : 0.0;
    const stalledNow = meanAbsVx < this.locomotionStallSpeed;
    if (stalledNow) {
      this.stallTicks += 1;
    } else {
      this.stallTicks = Math.max(0, this.stallTicks - 2);
    }
    if (this.stallTicks <= this.stallTickThreshold) {
      this.dynamicNoiseBoost = 0.0;
      return;
    }
    const excess = this.stallTicks - this.stallTickThreshold;
    this.dynamicNoiseBoost = Math.min(this.maxDynamicNoiseBoost, excess * 0.01);
  }

  updateStandingState() {
    if (this.rewardMode !== "standing") {
      this.standingMeanHeight = 0.0;
      this.standingLiftAssist = 0.0;
      this.standingStallTicks = 0;
      return;
    }
    let sumH = 0.0;
    for (let i = 0; i < this.limbBodyIds.length; i++) {
      sumH += this.limbTopHeight(i);
    }
    this.standingMeanHeight = sumH / Math.max(1, this.limbBodyIds.length);
    const lowMotion = Math.abs(this.deltaControlX) < this.standingStallVx * this.controlDt;
    const heightDeficit = Math.max(0.0, this.standingTargetHeight - this.standingMeanHeight);
    if (lowMotion && heightDeficit > 0.0) {
      this.standingStallTicks += 1;
    } else {
      this.standingStallTicks = Math.max(0, this.standingStallTicks - 1);
    }
    const stallBoost = Math.min(0.7, this.standingStallTicks * 0.006);
    this.standingLiftAssist = this.standingLiftGain * heightDeficit + stallBoost;
  }

  maybeApplyStandingDetach() {
    if (this.rewardMode !== "standing") {
      return;
    }
    if (this.standingDetachEvery <= 0) {
      return;
    }
    if (this.standingStallTicks < 8) {
      return;
    }
    if (this.controlTick - this.lastStandingDetachTick < this.standingDetachEvery) {
      return;
    }
    for (let child = this.graph.parent.length - 1; child >= 0; child--) {
      if (this.graph.parent[child] < 0) {
        continue;
      }
      this.graph.detach(child, this.data);
      this.lastStandingDetachTick = this.controlTick;
      return;
    }
  }

  applyStandingRecoveryVelocityKick() {
    if (this.rewardMode !== "standing") {
      return;
    }
    if (this.standingStallTicks < 8) {
      return;
    }
    if (this.controlTick % this.standingKickEvery !== 0) {
      return;
    }
    for (let limbIdx = 0; limbIdx < this.limbBodyIds.length; limbIdx++) {
      const qv = this.limbQvelAdr[limbIdx];
      const phase = this.controlTick * 0.31 + limbIdx * 1.11;
      this.data.qvel[qv + 2] += this.standingKickVelZ * (0.5 + 0.5 * Math.sin(phase));
      this.data.qvel[qv + 1] += this.standingKickVelY * Math.cos(phase);
    }
  }

  maybeApplyStallDetach() {
    if (this.rewardMode !== "locomotion") {
      return;
    }
    if (this.stallDetachEvery <= 0) {
      return;
    }
    if (this.stallTicks < this.stallTickThreshold + 10) {
      return;
    }
    if (this.controlTick - this.lastStallDetachTick < this.stallDetachEvery) {
      return;
    }
    for (let child = this.graph.parent.length - 1; child >= 0; child--) {
      if (this.graph.parent[child] < 0) {
        continue;
      }
      this.graph.detach(child, this.data);
      this.lastStallDetachTick = this.controlTick;
      return;
    }
  }

  applyStallRecoveryVelocityKick() {
    if (this.rewardMode !== "locomotion") {
      return;
    }
    if (this.stallTicks < this.stallTickThreshold + 6) {
      return;
    }
    if (this.controlTick % this.stallKickEvery !== 0) {
      return;
    }
    for (let limbIdx = 0; limbIdx < this.limbBodyIds.length; limbIdx++) {
      const qv = this.limbQvelAdr[limbIdx];
      const phase = this.controlTick * 0.35 + limbIdx * 1.33;
      this.data.qvel[qv + 0] += this.stallKickVelX * Math.cos(phase);
      this.data.qvel[qv + 2] += this.stallKickVelZ * (0.5 + 0.5 * Math.sin(phase));
    }
  }

  nearestAttachCandidate(parentIdx) {
    const parentSite = this.sitePos(this.parentSiteIds[parentIdx]);
    let best = -1;
    let bestDist = Infinity;
    for (let childIdx = 0; childIdx < this.limbBodyIds.length; childIdx++) {
      if (childIdx === parentIdx) {
        continue;
      }
      if (this.graph.parent[childIdx] >= 0) {
        continue;
      }
      if (this.graph.isInSameComponent(parentIdx, childIdx)) {
        continue;
      }
      const childSite = this.sitePos(this.childSiteIds[childIdx]);
      const dx = parentSite[0] - childSite[0];
      const dy = parentSite[1] - childSite[1];
      const dz = parentSite[2] - childSite[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < bestDist) {
        bestDist = dist;
        best = childIdx;
      }
    }
    return { child: best, dist: bestDist };
  }

  buildObservation(limbIdx, componentSizes) {
    const bodyId = this.limbBodyIds[limbIdx];
    const posBase = bodyId * 3;
    const cvelBase = bodyId * 6;
    const posSrc = this.data.xipos || this.data.xpos;
    const pos = [
      posSrc[posBase + 0],
      posSrc[posBase + 1],
      posSrc[posBase + 2],
    ];
    const omega = [
      this.data.cvel[cvelBase + 0],
      this.data.cvel[cvelBase + 1],
      this.data.cvel[cvelBase + 2],
    ];
    const linear = [
      this.data.cvel[cvelBase + 3],
      this.data.cvel[cvelBase + 4],
      this.data.cvel[cvelBase + 5],
    ];

    const selfChildSite = this.sitePos(this.childSiteIds[limbIdx]);
    const selfParentSite = this.sitePos(this.parentSiteIds[limbIdx]);
    const axisVec = [
      selfParentSite[0] - selfChildSite[0],
      selfParentSite[1] - selfChildSite[1],
      selfParentSite[2] - selfChildSite[2],
    ];
    const axisNorm = Math.hypot(axisVec[0], axisVec[1], axisVec[2]);
    if (axisNorm > 1e-8) {
      axisVec[0] /= axisNorm;
      axisVec[1] /= axisNorm;
      axisVec[2] /= axisNorm;
    } else {
      axisVec[0] = 0.0;
      axisVec[1] = 0.0;
      axisVec[2] = 1.0;
    }

    const nearest = this.nearestAttachCandidate(limbIdx);

    const hasParent = this.graph.parent[limbIdx] >= 0;
    const childCount = this.graph.children[limbIdx].size;
    const componentSize = componentSizes[limbIdx];
    const nearestInv = nearest.dist < Infinity ? 1.0 / (0.05 + nearest.dist) : 0.0;

    const obsVec = [
      pos[0], pos[1], pos[2],
      linear[0], linear[1], linear[2],
      omega[0], omega[1], omega[2],
      axisVec[0], axisVec[1], axisVec[2],
      hasParent ? 1.0 : 0.0,
      childCount / 3.0,
      componentSize / this.limbBodyIds.length,
      nearestInv,
    ];

    return {
      obsVec,
      meta: {
        nearestChild: nearest.child,
        nearestDist: nearest.dist,
        hasParent,
        childCount,
        height: this.limbTopHeight(limbIdx),
      },
    };
  }

  computeRewards() {
    const rewards = new Array(this.limbBodyIds.length).fill(0.0);
    const components = this.graph.connectedComponents();
    const nextForward = {};
    const componentDebug = [];
    for (const component of components) {
      const sorted = component.slice().sort((a, b) => a - b);
      if (this.rewardMode === "standing") {
        let maxHeight = -Infinity;
        for (const limbIdx of sorted) {
          maxHeight = Math.max(maxHeight, this.limbTopHeight(limbIdx));
        }
        for (const limbIdx of sorted) {
          rewards[limbIdx] = maxHeight;
        }
        componentDebug.push({
          nodes: sorted,
          reward: maxHeight,
          metricLabel: "h",
          metricValue: maxHeight,
        });
        continue;
      }

      let meanForward = 0.0;
      let meanHeight = 0.0;
      const posSrc = this.data.xipos || this.data.xpos;
      for (const limbIdx of sorted) {
        const bodyId = this.limbBodyIds[limbIdx];
        meanForward += posSrc[bodyId * 3 + this.forwardAxis];
        meanHeight += this.limbTopHeight(limbIdx);
      }
      meanForward /= sorted.length;
      meanHeight /= sorted.length;

      const key = sorted.join("-");
      const prev = this.prevComponentForward[key];
      const delta = prev === undefined ? 0.0 : meanForward - prev;
      const forwardVelocity = delta / Math.max(this.controlDt, 1e-6);
      const postureBonus =
        this.locomotionHeightWeight * (meanHeight - this.locomotionHeightBaseline);
      const stallPenalty =
        Math.abs(forwardVelocity) < this.locomotionStallSpeed ? this.locomotionStallPenalty : 0.0;
      const locomotionReward = forwardVelocity + postureBonus - stallPenalty;
      for (const limbIdx of sorted) {
        rewards[limbIdx] = locomotionReward;
      }
      nextForward[key] = meanForward;
      componentDebug.push({
        nodes: sorted,
        reward: locomotionReward,
        metricLabel: "vx",
        metricValue: forwardVelocity,
        meanHeight,
        postureBonus,
        stallPenalty,
      });
    }
    if (this.rewardMode === "locomotion") {
      this.prevComponentForward = nextForward;
    }
    this.limbRewards = rewards;
    this.componentDebug = componentDebug;
  }

  computeComponentSizes(components) {
    const sizes = new Array(this.limbBodyIds.length).fill(1);
    for (const component of components) {
      for (const node of component) {
        sizes[node] = component.length;
      }
    }
    return sizes;
  }

  getObservationBatch() {
    const components = this.graph.connectedComponents();
    const componentSizes = this.computeComponentSizes(components);
    const rows = [];
    for (let limbIdx = 0; limbIdx < this.limbBodyIds.length; limbIdx++) {
      const obs = this.buildObservation(limbIdx, componentSizes);
      rows.push({
        limbIdx,
        obsDim: obs.obsVec.length,
        obsVec: obs.obsVec.slice(),
        meta: { ...obs.meta },
      });
    }
    return rows;
  }

  getActiveEdges() {
    const edges = [];
    for (let child = 0; child < this.graph.parent.length; child++) {
      const parent = this.graph.parent[child];
      if (parent >= 0) {
        edges.push(`${parent}->${child}`);
      }
    }
    return edges;
  }

  getActiveEqEdges() {
    const active = [];
    for (const [edge, eqId] of Object.entries(this.directedConstraintIndex)) {
      if (this.data.eq_active[eqId] > 0) {
        active.push(edge);
      }
    }
    active.sort();
    return active;
  }

  buildSnapshot() {
    const components = this.graph.connectedComponents().map((c) => c.slice().sort((a, b) => a - b));
    const componentByLimb = new Array(this.limbBodyIds.length).fill(-1);
    for (let componentId = 0; componentId < components.length; componentId++) {
      for (const limbIdx of components[componentId]) {
        componentByLimb[limbIdx] = componentId;
      }
    }
    const limbStates = [];
    for (let limbIdx = 0; limbIdx < this.limbBodyIds.length; limbIdx++) {
      limbStates.push({
        limbIdx,
        parent: this.graph.parent[limbIdx],
        childCount: this.graph.children[limbIdx].size,
        componentId: componentByLimb[limbIdx],
        bodyPos: this.limbBodyPos(limbIdx),
        reward: this.limbRewards[limbIdx] ?? 0.0,
        torque: (this.currentTorques[limbIdx] || [0, 0, 0]).slice(),
        attachProb: this.latestActions[limbIdx]?.attachProb ?? 0.0,
        detachProb: this.latestActions[limbIdx]?.detachProb ?? 0.0,
      });
    }
    return {
      simTime: this.simTimeSec,
      controlTick: this.controlTick,
      rewardMode: this.rewardMode,
      centroidX: this.centroidX,
      deltaControlX: this.deltaControlX,
      deltaFromStartX: this.deltaFromStartX,
      edges: this.getActiveEdges(),
      eqActiveEdges: this.getActiveEqEdges(),
      components,
      componentDebug: this.componentDebug.map((row) => ({ ...row, nodes: row.nodes.slice() })),
      limbRewards: this.limbRewards.slice(),
      limbStates,
    };
  }

  pushSnapshot() {
    this.lastSnapshot = this.buildSnapshot();
    this.telemetryHistory.push(this.lastSnapshot);
    if (this.telemetryHistory.length > this.telemetryHistoryLimit) {
      this.telemetryHistory.shift();
    }
  }

  getDebugState() {
    return {
      observationDim: this.observationDim,
      messageDim: this.messageDim,
      useMessages: this.useMessages,
      rewardMode: this.rewardMode,
      latest: this.lastSnapshot ? { ...this.lastSnapshot } : null,
      recent: this.telemetryHistory.slice(-32),
    };
  }

  setMessagePassingEnabled(enabled) {
    this.useMessages = Boolean(enabled);
  }

  async loadPolicyWeightsFromURL(url) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const weights = await response.json();
      this.loadPolicyWeights(weights);
      this.policySource = url;
      return true;
    } catch (err) {
      console.error("Failed to load policy weights:", url, err);
      this.policySource = `load_failed:${url}`;
      return false;
    }
  }

  resetEpisode() {
    this.mujoco.mj_resetData(this.model, this.data);
    for (const eqId of Object.values(this.directedConstraintIndex)) {
      this.data.eq_active[eqId] = 0;
    }
    this.graph = new AssemblyGraphManager(this.limbBodyIds.length, this.directedConstraintIndex, this.joinDist);
    this.currentTorques = Array.from({ length: this.limbBodyIds.length }, () => [0, 0, 0]);
    this.limbRewards = new Array(this.limbBodyIds.length).fill(0.0);
    this.latestActions = Array.from(
      { length: this.limbBodyIds.length },
      () => ({ attachProb: 0.0, detachProb: 0.0 })
    );
    this.telemetryHistory = [];
    this.lastSnapshot = null;

    this.timeMS = 0.0;
    this.simTimeSec = 0.0;
    this.lastControlTime = -1.0;
    this.controlTick = 0;
    this.prevComponentForward = {};
    this.centroidStartX = null;
    this.centroidX = 0.0;
    this.deltaControlX = 0.0;
    this.deltaFromStartX = 0.0;
    this.lastCentroidX = null;
    this.stallTicks = 0;
    this.dynamicNoiseBoost = 0.0;
    this.standingMeanHeight = 0.0;
    this.standingLiftAssist = 0.0;
    this.standingStallTicks = 0;
    this.lastStandingDetachTick = -1;
    this.lastStallDetachTick = -1;

    this.syncEqActiveWithGraph();
    this.mujoco.mj_forward(this.model, this.data);
    this.computeRewards();
    this.updateCentroidMetrics();
    this.updateStallState();
    this.updateStandingState();
    this.pushSnapshot();
    this.updateOverlay();
  }

  exportPolicyWeights() {
    return this.policy.getWeights();
  }

  loadPolicyWeights(weights) {
    this.policy.setWeights(weights);
  }

  collectRollout(stepCount = 600) {
    const steps = Math.max(1, Math.floor(stepCount));
    const rollout = [];
    for (let k = 0; k < steps; k++) {
      const obs = this.getObservationBatch().map((row) => row.obsVec.slice());
      this.stepPhysics(1);
      rollout.push({
        simTime: this.simTimeSec,
        controlTick: this.controlTick,
        rewardMode: this.rewardMode,
        useMessages: this.useMessages,
        edges: this.getActiveEdges(),
        observations: obs,
        rewards: this.limbRewards.slice(),
        torques: this.currentTorques.map((t) => t.slice()),
        attachProbs: this.latestActions.map((a) => a.attachProb),
        detachProbs: this.latestActions.map((a) => a.detachProb),
      });
    }
    return rollout;
  }

  syncEqActiveWithGraph() {
    for (const eqId of Object.values(this.directedConstraintIndex)) {
      this.data.eq_active[eqId] = 0;
    }
    for (let child = 0; child < this.graph.parent.length; child++) {
      const parent = this.graph.parent[child];
      if (parent < 0) {
        continue;
      }
      const key = `${parent}->${child}`;
      const eqId = this.directedConstraintIndex[key];
      if (eqId !== undefined) {
        this.data.eq_active[eqId] = 1;
      }
    }
  }

  controlStep() {
    const components = this.graph.connectedComponents();
    const componentSizes = this.computeComponentSizes(components);

    const order = this.graph.topologicalPostOrder();
    const messages = Array.from(
      { length: this.limbBodyIds.length },
      () => new Array(this.messageDim).fill(0.0)
    );
    const outputs = new Array(this.limbBodyIds.length).fill(null);
    const inStallRecovery =
      this.rewardMode === "locomotion" && this.stallTicks > this.stallTickThreshold;
    const effectiveDetachThreshold = inStallRecovery
      ? Math.max(0.28, this.detachThreshold - 0.35)
      : this.detachThreshold;
    const effectiveAttachThreshold = inStallRecovery
      ? Math.max(0.40, this.attachThreshold - 0.12)
      : this.attachThreshold;

    for (const node of order) {
      const childMsg = new Array(this.messageDim).fill(0.0);
      if (this.useMessages) {
        for (const child of this.graph.children[node]) {
          const m = messages[child];
          for (let k = 0; k < m.length; k++) {
            childMsg[k] += m[k];
          }
        }
      }
      const obs = this.buildObservation(node, componentSizes);
      const out = this.policy.forward(obs.obsVec, childMsg, obs.meta);
      outputs[node] = { ...out, meta: obs.meta };
      messages[node] = out.message;
    }

    // Detach first so attach can form new edges in the same control tick.
    for (let child = 0; child < this.limbBodyIds.length; child++) {
      const out = outputs[child];
      if (!out) {
        continue;
      }
      if (out.detachProb > effectiveDetachThreshold && this.graph.parent[child] >= 0) {
        this.graph.detach(child, this.data);
      }
    }

    const attachCandidates = [];
    for (let parent = 0; parent < this.limbBodyIds.length; parent++) {
      const out = outputs[parent];
      if (!out || out.meta.nearestChild < 0) {
        continue;
      }
      attachCandidates.push({ parent, score: out.attachProb, child: out.meta.nearestChild });
    }
    attachCandidates.sort((a, b) => b.score - a.score);
    for (const candidate of attachCandidates) {
      if (candidate.score <= effectiveAttachThreshold) {
        continue;
      }
      const parentSite = this.sitePos(this.parentSiteIds[candidate.parent]);
      const childSite = this.sitePos(this.childSiteIds[candidate.child]);
      if (!this.graph.canAttach(candidate.parent, candidate.child, parentSite, childSite)) {
        continue;
      }
      this.graph.attach(candidate.parent, candidate.child, this.data);
    }

    for (let i = 0; i < this.limbBodyIds.length; i++) {
      this.currentTorques[i] = outputs[i].torque;
      this.latestActions[i] = {
        attachProb: outputs[i].attachProb,
        detachProb: outputs[i].detachProb,
      };
    }

    this.maybeApplyStallDetach();
    this.maybeApplyStandingDetach();
    this.applyStallRecoveryVelocityKick();
    this.applyStandingRecoveryVelocityKick();
    this.syncEqActiveWithGraph();
    this.mujoco.mj_forward(this.model, this.data);
    this.computeRewards();
    this.updateCentroidMetrics();
    this.updateStallState();
    this.updateStandingState();
    this.controlTick += 1;
    this.pushSnapshot();
    this.updateOverlay();
  }

  applyControlTorques() {
    if (!this.data.xfrc_applied) {
      return;
    }
    for (let i = 0; i < this.data.xfrc_applied.length; i++) {
      this.data.xfrc_applied[i] = 0.0;
    }
    let standCx = 0.0;
    let standCy = 0.0;
    if (this.rewardMode === "standing" && this.limbBodyIds.length > 0) {
      for (let limbIdx = 0; limbIdx < this.limbBodyIds.length; limbIdx++) {
        const [mx, my] = this.limbMidpointXY(limbIdx);
        standCx += mx;
        standCy += my;
      }
      standCx /= this.limbBodyIds.length;
      standCy /= this.limbBodyIds.length;
    }
    for (let limbIdx = 0; limbIdx < this.limbBodyIds.length; limbIdx++) {
      const bodyId = this.limbBodyIds[limbIdx];
      const base = bodyId * 6;
      const torque = this.currentTorques[limbIdx];
      const noise = this.randomTorqueScale + this.dynamicNoiseBoost;
      this.data.xfrc_applied[base + 3] =
        (torque[0] + (Math.random() * 2.0 - 1.0) * noise) * this.torqueScale;
      this.data.xfrc_applied[base + 4] =
        (torque[1] + (Math.random() * 2.0 - 1.0) * noise) * this.torqueScale;
      this.data.xfrc_applied[base + 5] =
        (torque[2] + (Math.random() * 2.0 - 1.0) * noise) * this.torqueScale;

      if (this.rewardMode === "locomotion" && this.stallTicks > this.stallTickThreshold) {
        const phase = this.controlTick * 0.45 + limbIdx * 1.7;
        this.data.xfrc_applied[base + 0] += this.stallPulseForward * Math.cos(phase);
        this.data.xfrc_applied[base + 2] += this.stallPulseLift * (0.5 + 0.5 * Math.sin(phase));
      }

      if (this.rewardMode === "standing" && this.standingLiftAssist > 0.0) {
        const phase = this.controlTick * 0.38 + limbIdx * 1.45;
        const [px, py] = this.limbMidpointXY(limbIdx);
        const rx = px - standCx;
        const ry = py - standCy;
        this.data.xfrc_applied[base + 2] += this.standingLiftAssist * (0.55 + 0.45 * Math.sin(phase));
        this.data.xfrc_applied[base + 0] += -this.standingInwardGain * rx;
        this.data.xfrc_applied[base + 1] += -this.standingInwardGain * ry;
        this.data.xfrc_applied[base + 1] += this.standingLateralPulse * Math.cos(phase);
        this.data.xfrc_applied[base + 3] += -this.standingTorqueGain * ry;
        this.data.xfrc_applied[base + 4] += this.standingTorqueGain * rx;
      }

      if (this.graph.parent[limbIdx] < 0) {
        const nearest = this.nearestAttachCandidate(limbIdx);
        if (nearest.child >= 0 && nearest.dist < this.attractionMaxDist) {
          const parentSite = this.sitePos(this.parentSiteIds[limbIdx]);
          const childSite = this.sitePos(this.childSiteIds[nearest.child]);
          let dx = childSite[0] - parentSite[0];
          let dy = childSite[1] - parentSite[1];
          let dz = childSite[2] - parentSite[2];
          const norm = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (norm > 1e-6) {
            dx /= norm;
            dy /= norm;
            dz /= norm;
            const gain =
              this.attractionForceScale * Math.max(0.0, 1.0 - nearest.dist / this.attractionMaxDist);
            this.data.xfrc_applied[base + 0] += dx * gain;
            this.data.xfrc_applied[base + 1] += dy * gain;
            this.data.xfrc_applied[base + 2] += dz * gain;
          }
        }
      }
    }
  }

  updateOverlay() {
    if (!this.overlay) {
      return;
    }
    const snapshot = this.lastSnapshot || this.buildSnapshot();
    const componentCount = snapshot.components.length;
    const meanReward =
      this.limbRewards.reduce((acc, x) => acc + x, 0.0) / Math.max(1, this.limbRewards.length);
    const stallDetachAge =
      this.lastStallDetachTick >= 0 ? this.controlTick - this.lastStallDetachTick : null;
    const policyLabel =
      this.policySource.length > 44 ? `...${this.policySource.slice(-44)}` : this.policySource;
    const actionRangeLabel = `l0..l${Math.max(0, this.limbBodyIds.length - 1)}`;
    const previewCount = Math.min(8, this.latestActions.length);
    const attachPreview =
      this.latestActions.slice(0, previewCount).map((x) => x.attachProb.toFixed(2)).join(" ") +
      (this.latestActions.length > previewCount ? " ..." : "");
    const detachPreview =
      this.latestActions.slice(0, previewCount).map((x) => x.detachProb.toFixed(2)).join(" ") +
      (this.latestActions.length > previewCount ? " ..." : "");
    const componentSummary = this.componentDebug
      .map((entry) => {
        const metric = Number.isFinite(entry.metricValue) ? entry.metricValue : 0.0;
        const height = Number.isFinite(entry.meanHeight) ? ` h=${entry.meanHeight.toFixed(3)}` : "";
        const bonus = Number.isFinite(entry.postureBonus) ? ` b=${entry.postureBonus.toFixed(3)}` : "";
        const pen = Number.isFinite(entry.stallPenalty) ? ` p=${entry.stallPenalty.toFixed(3)}` : "";
        return (
          `[${entry.nodes.join(",")}]:${entry.metricLabel}=${metric.toFixed(3)}` +
          `${height}${bonus}${pen} r=${entry.reward.toFixed(3)}`
        );
      })
      .join("  ");
    this.overlay.textContent =
      "Modular Assemblies Prototype\n" +
      `sim_time: ${this.simTimeSec.toFixed(3)} s\n` +
      `control_tick: ${this.controlTick}\n` +
      `reward_mode: ${this.rewardMode}\n` +
      `messages: ${this.useMessages ? "on" : "off"}\n` +
      `policy: ${policyLabel}\n` +
      `centroid_x: ${this.centroidX.toFixed(3)}\n` +
      `dx(control): ${this.deltaControlX.toFixed(4)}\n` +
      `dx(start): ${this.deltaFromStartX.toFixed(3)}\n` +
      `stall_ticks: ${this.stallTicks}\n` +
      `noise_eff: ${(this.randomTorqueScale + this.dynamicNoiseBoost).toFixed(3)}\n` +
      `stand_h: ${this.standingMeanHeight.toFixed(3)}\n` +
      `stand_lift: ${this.standingLiftAssist.toFixed(3)}\n` +
      `stand_stall: ${this.standingStallTicks}\n` +
      `stall_detach_age: ${stallDetachAge === null ? "n/a" : stallDetachAge}\n` +
      `active_edges: ${this.graph.edgeCount()} (${snapshot.edges.join(", ") || "none"})\n` +
      `components: ${componentCount}\n` +
      `component_reward: ${componentSummary || "n/a"}\n` +
      `reward(mean): ${meanReward.toFixed(4)}\n` +
      `attach_p(${actionRangeLabel}): ${attachPreview}\n` +
      `detach_p(${actionRangeLabel}): ${detachPreview}`;
  }

  syncVisualTransforms() {
    for (let b = 0; b < this.model.nbody; b++) {
      if (this.bodies[b]) {
        getPosition(this.data.xpos, b, this.bodies[b].position);
        getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        const tmpPos = new THREE.Vector3();
        const tmpDir = new THREE.Vector3();
        getPosition(this.data.light_xpos, l, tmpPos);
        getPosition(this.data.light_xdir, l, tmpDir);
        this.lights[l].position.copy(tmpPos);
        this.lights[l].lookAt(tmpDir.add(tmpPos));
      }
    }
  }

  clearGraphLinksForNode(nodeIdx) {
    for (const child of Array.from(this.graph.children[nodeIdx])) {
      this.graph.detach(child, this.data);
    }
    if (this.graph.parent[nodeIdx] >= 0) {
      this.graph.detach(nodeIdx, this.data);
    }
  }

  enforceStabilityGuards() {
    let changed = false;
    for (let limbIdx = 0; limbIdx < this.limbBodyIds.length; limbIdx++) {
      const qv = this.limbQvelAdr[limbIdx];
      for (let i = 0; i < 6; i++) {
        const value = this.data.qvel[qv + i];
        if (!Number.isFinite(value)) {
          this.data.qvel[qv + i] = 0.0;
          changed = true;
          continue;
        }
        if (value > this.maxAbsQvel) {
          this.data.qvel[qv + i] = this.maxAbsQvel;
          changed = true;
        } else if (value < -this.maxAbsQvel) {
          this.data.qvel[qv + i] = -this.maxAbsQvel;
          changed = true;
        }
      }

      const pos = this.limbBodyPos(limbIdx);
      const runaway =
        !Number.isFinite(pos[0]) ||
        !Number.isFinite(pos[1]) ||
        !Number.isFinite(pos[2]) ||
        Math.abs(pos[0]) > this.maxAbsCoord ||
        Math.abs(pos[1]) > this.maxAbsCoord ||
        Math.abs(pos[2]) > this.maxAbsCoord ||
        Math.sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]) > this.maxAbsPos;

      if (runaway) {
        this.clearGraphLinksForNode(limbIdx);
        const qp = this.limbQposAdr[limbIdx];
        const qvAddr = this.limbQvelAdr[limbIdx];
        const init = this.initialLimbQpos[limbIdx];
        for (let i = 0; i < 7; i++) {
          this.data.qpos[qp + i] = init[i];
        }
        for (let i = 0; i < 6; i++) {
          this.data.qvel[qvAddr + i] = 0.0;
        }
        changed = true;
      }
    }
    if (changed) {
      this.syncEqActiveWithGraph();
      this.mujoco.mj_forward(this.model, this.data);
    }
  }

  stepPhysicsFrame() {
    if (this.lastControlTime < 0.0 || this.simTimeSec - this.lastControlTime >= this.controlDt) {
      this.controlStep();
      this.lastControlTime = this.simTimeSec;
    }
    this.syncEqActiveWithGraph();
    this.enforceStabilityGuards();
    this.applyControlTorques();
    this.mujoco.mj_step(this.model, this.data);
    this.simTimeSec += this.model.opt.timestep;
  }

  stepPhysics(stepCount = 1) {
    const dtMS = this.model.opt.timestep * 1000.0;
    const frames = Math.max(1, Math.floor(stepCount));
    for (let i = 0; i < frames; i++) {
      this.stepPhysicsFrame();
      this.timeMS += dtMS;
    }
  }

  render(timeMS) {
    if (this.controls) {
      this.controls.update();
    }

    const dtMS = this.model.opt.timestep * 1000.0;
    if (timeMS - this.timeMS > 35.0) {
      this.timeMS = timeMS;
    }
    while (this.timeMS < timeMS) {
      this.stepPhysicsFrame();
      this.timeMS += dtMS;
    }

    if (this.renderer) {
      this.syncVisualTransforms();
      this.renderer.render(this.scene, this.camera);
    }
  }
}

const demo = new ModularAssembliesDemo();
await demo.init();
if (typeof window !== "undefined") {
  window.__assembliesDemo = demo;
  window.__mujocoDemo = demo;
}
