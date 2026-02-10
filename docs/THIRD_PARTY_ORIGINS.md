# Third-Party File Origins

This repository is self-contained and includes a minimal subset of code copied from upstream projects.

## Source Repositories

1. `mujoco_wasm` (upstream project by Sergei Surovtsev and Johnathon Selstad)
2. Local project workspace `evorobot` (tools and modular-assemblies implementation developed during this project)

## Copied From `mujoco_wasm`

- `modular_assemblies.html`
- `package.json`
- `package-lock.json`
- `src/modularAssembliesMain.js`
- `src/mujocoUtils.js`
- `src/assemblies/AssemblyGraphManager.js`
- `src/assemblies/SimpleDGNPolicy.js`
- `src/utils/Reflector.js`
- `assets/scenes/modular_assemblies_1limb.xml`
- `assets/scenes/modular_assemblies_4limb.xml`
- `assets/scenes/modular_assemblies_8limb.xml`
- `assets/favicon.png`
- `THIRD_PARTY_LICENSES/mujoco_wasm_LICENSE.txt`

Notes:
- These files were copied from the local `mujoco_wasm` repository present in the parent workspace.
- Some copied files contain local modifications made during this project (geometry updates, standing reward/observation/control updates, policy tuning support).

## Copied From Local `evorobot` Workspace

- `tools/train_modular_policy_cem.py`
- `tools/evaluate_modular_policy.py`
- `tools/validate_modular_assemblies_invariants.py`
- `tools/validate_single_limb_standing.py`
- `tools/train_single_limb_stability_cem.py`
- `assets/policies/standing_8limb_geomv2_best.json`
- `assets/policies/locomotion_8limb_geomv2_best.json`
- `assets/policies/single_limb_standing_torque8_long_best.json`
- `assets/policies/single_limb_standing_stability_stage2_best.json`

## Why This Exists

The goal is traceability: every copied file is explicitly labeled with its origin so this repository remains auditable and reproducible.
