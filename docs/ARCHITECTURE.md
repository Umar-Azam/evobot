# Architecture

## Runtime Stack

1. `modular_assemblies.html`
   - Browser entrypoint.
   - Loads import map and starts `src/modularAssembliesMain.js`.

2. `src/modularAssembliesMain.js`
   - Main simulation loop.
   - Handles scene loading, reward modes (`standing` / `locomotion`), control, graph connectivity, and overlay telemetry.

3. `src/assemblies/AssemblyGraphManager.js`
   - Dynamic parent-child attachment graph and equality-constraint synchronization.

4. `src/assemblies/SimpleDGNPolicy.js`
   - Lightweight shared policy head for per-limb torque + attach/detach action generation.

5. `src/mujocoUtils.js` + `src/utils/Reflector.js`
   - MuJoCo -> Three.js scene bridging utilities.

## Data Assets

- `assets/scenes/*.xml`
  - `modular_assemblies_1limb.xml`
  - `modular_assemblies_4limb.xml`
  - `modular_assemblies_8limb.xml`

- `assets/policies/*.json`
  - Tuned policy checkpoints used directly via URL query param.

## Tooling Layer (`tools/`)

- `train_modular_policy_cem.py`
  - CEM optimization for multi-limb standing/locomotion.
- `evaluate_modular_policy.py`
  - Rollout metrics for a saved policy.
- `validate_modular_assemblies_invariants.py`
  - Invariant checks over rollouts.
- `validate_single_limb_standing.py`
  - Active-vs-passive upright check for 1-limb setup.
- `train_single_limb_stability_cem.py`
  - Stability-focused single-limb optimizer.

## Query-Driven Configuration

The runtime is configured primarily through URL query params:
- `scene`
- `reward`
- `policy`
- `noise`
- `torque_scale`
- standing/locomotion tuning knobs (`stand_*`, `stall_*`, `loc_*`)

This keeps scenarios reproducible without source edits.
