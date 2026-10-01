# navsim-transfuser: Design Spec

> Status: **Planned (draft, details to confirm)** · Owner: Moin Palekar · Resume dates: Oct 2026 to Present · Also: CSE 593 applied project (Spring 2027), CVPR WAD workshop submission target
> Purpose of this doc: everything needed to build the project from zero without re-deriving decisions. Items marked **[DECIDE]** are open and get confirmed before the relevant milestone.

---

## 1. Problem

End-to-end driving models map raw sensors straight to a planned trajectory. Evaluating them is hard: open-loop metrics (L2 distance to the human trajectory) reward imitation and miss safety, while full closed-loop simulation is expensive and suffers from a sim-to-real gap. **NAVSIM** sits in between: it scores a planned trajectory with a simulation-based metric (PDMS in v1, EPDMS in v2) that checks collisions, drivable-area compliance, time-to-collision, progress, and comfort, on real-world data (OpenScene, derived from nuPlan).

**TransFuser** is the standard NAVSIM baseline: a camera + LiDAR fusion model with a transformer backbone and BEV auxiliary heads. The NAVSIM paper reports it matching much larger end-to-end models (TransFuser 84.0 PDMS on navtest vs UniAD and PARA-Drive in the same range) at a fraction of the compute.

This project (1) faithfully reproduces TransFuser on NAVSIM, trained from scratch with distributed multi-GPU training on ASU Sol, then (2) adds one extension aligned with current end-to-end driving research **[DECIDE]**, evaluated with ablations.

**Why this project exists (portfolio):** anchors the AV and MLE resumes (MLE set = NAVSIM + ragserve). Together with mot3d it covers perception, sensor fusion, planning, large-scale training, and benchmark rigor. It doubles as the MS capstone and a workshop paper.

## 2. Goals and non-goals

### Goals
1. **Verified pipeline:** official checkpoints evaluated locally reproduce published scores before any training.
2. **Faithful reproduction:** TransFuser trained from scratch on navtrain matches the published navtest score within a stated tolerance, over multiple seeds.
3. **Distributed training on Sol:** multi-GPU DDP via SLURM, with measured scaling (throughput vs GPU count).
4. **Analysis beyond the headline number:** per-sub-metric breakdown, failure taxonomy with visualizations, navhard two-stage results.
5. **One extension** with clean ablations against the reproduced baseline **[DECIDE]**.
6. **Reproducibility:** pinned devkit commit, configs, seeds, and SLURM scripts committed; anyone with data can rerun.

### Non-goals
- Beating the leaderboard top (world-action models, large pretrained backbones).
- Real-vehicle or full closed-loop deployment.
- Rewriting the devkit; we extend it.

## 3. Background (what must be true before reading code)

| Item | Fact |
|---|---|
| Data | OpenScene (nuPlan-derived, ~120 h of driving at reduced frame rate). `navtrain` ≈ 103k samples, ≈ 450 GB. A `mini` split exists for development. |
| Task | Given ~1.5 s of past sensor frames + ego status + route, output a 4 s future trajectory (8 poses at 2 Hz). |
| Metric v1: PDMS | Multiplicative penalties (no-at-fault collision, drivable-area compliance) × weighted average of ego progress, time-to-collision, comfort. |
| Metric v2: EPDMS | Extends PDMS with additional terms (e.g. driving direction, traffic lights, lane keeping, extended comfort) and two-stage pseudo-simulation (`navhard_two_stage`). |
| TransFuser inputs | Stitched front-left / front / front-right cameras + LiDAR BEV histogram + ego status. |
| TransFuser outputs | Trajectory head + auxiliary heads: BEV semantic segmentation and agent bounding boxes. |
| Published numbers | TransFuser 84.0 PDMS, Latent TransFuser (camera-only) 83.8 PDMS on navtest (v1). Training ≈ 1 GPU-day per the paper. |
| Official assets | Devkit `autonomousvision/navsim`; baseline checkpoints on Hugging Face `autonomousvision/navsim_baselines`. |

Verify the v2 baseline EPDMS numbers from the current devkit docs / leaderboard at M1 and record them in §8.

## 4. Approach

### Phase A: Reproduce
1. Install the devkit at a pinned commit; download `mini`, then `navtrain` / `navtest` to Sol scratch.
2. Build the feature cache and metric cache (devkit caching scripts).
3. Evaluate the official TransFuser and LTF checkpoints on navtest (v1 PDMS) and the v2 splits (EPDMS). Must match published values within ±0.3 before moving on. This isolates environment bugs from training bugs.
4. Train TransFuser from scratch: PyTorch Lightning DDP across N GPUs on Sol, 3 seeds. Report mean ± std.
5. Repeat for LTF (camera-only) to quantify what LiDAR buys.

### Phase B: Analyze
- Sub-metric breakdown (NC, DAC, TTC, EP, comfort, and v2 terms): where does TransFuser lose points?
- Failure taxonomy: sample the lowest-scoring scenes, render BEV + camera + planned vs human trajectory, classify (unprotected turns, dense traffic, lane changes, stationary agents, etc.).
- navhard two-stage: how scores drop under the harder pseudo-simulation.
- Scaling: samples/s and wall-clock vs GPUs (1, 2, 4, 8 if allocation allows); data-loading bottlenecks.

### Phase C: Extend **[DECIDE one]**
| Option | Idea | Fits | Cost |
|---|---|---|---|
| C1. Diffusion / flow planning head | Replace the regression trajectory head with a truncated diffusion or flow-matching head (DiffusionDrive-style); keep the backbone | Current planner trend; multimodality | Medium |
| C2. Latent world-model auxiliary loss | Add a head predicting future BEV features / occupancy from the fused latent; train jointly | World-models research interest; leaderboard direction (world-action models) | Medium to high |
| C3. Robustness and failure study | Systematic navhard two-stage evaluation + perturbations (sensor dropout, camera-only fallback, ego-status ablation) + failure taxonomy | AV test / validation roles; cheapest | Low to medium |

Whichever is chosen: ablate against the reproduced baseline (same seeds, same budget), report EPDMS and sub-metrics, and for C1 also trajectory diversity.

## 5. System and infrastructure

### Sol / SLURM
- `slurm/train.sbatch`: `--gres=gpu:N`, `srun` launches Lightning DDP; checkpoints and logs to scratch, synced to project storage.
- `slurm/cache.sbatch`: CPU-heavy caching job, array-parallel over log shards.
- `slurm/eval.sbatch`: PDMS/EPDMS scoring (CPU-parallel).
- Resume-from-checkpoint on preemption / time limit.
- Data on scratch; record purge policy and re-staging steps in `docs/sol.md`.

### Logging and reproducibility
- Weights & Biases (or TensorBoard) for losses, LR, throughput, eval scores.
- Every run records: devkit commit, repo commit, config hash, seed, GPU count, wall-clock.
- `environment.yml` / lockfile; container option (Apptainer) if Sol modules drift.

### Repo layout (separate repo, devkit as dependency, not a fork)
```
navsim/              # git submodule pinned to a devkit commit
agents/              # our agents (extension), registered via Hydra configs
configs/             # Hydra overrides: experiments, ablations
slurm/               # sbatch scripts
analysis/            # sub-metric breakdowns, failure taxonomy, plots
viz/                 # scene rendering (BEV + cameras + trajectories)
scripts/             # download, cache, eval wrappers
docs/SPEC.md  docs/sol.md  docs/results.md  docs/adr/
```

## 6. Evaluation protocol

| What | Split | Metric | Runs |
|---|---|---|---|
| Pipeline check | navtest (+ v2 splits) | PDMS / EPDMS | official checkpoints |
| Reproduction | navtest (+ v2 splits) | PDMS / EPDMS + sub-metrics | 3 seeds, mean ± std |
| Sensor ablation | same | same | TransFuser vs LTF |
| Hard scenarios | navhard two-stage | EPDMS | best seed of each |
| Extension | same as above | EPDMS + sub-metrics (+ diversity for C1) | 3 seeds vs baseline |
| Scaling | train | samples/s, wall-clock | 1/2/4/8 GPUs |

Results fill the resume placeholders: reproduced XX.X PDMS (published 84.0), YY.Y EPDMS, NN× training speedup on N GPUs, ZZ-point gain from the extension.

## 7. Milestones

| M | Scope | Done when |
|---|---|---|
| **M1: Pipeline verified** | Devkit installed on Sol, mini + full splits staged, caches built, official checkpoints scored | Local scores match published within ±0.3. **Repo goes public, README v1** with setup + verified table |
| **M2: Reproduction** | TransFuser + LTF trained from scratch, multi-GPU DDP, 3 seeds; scaling measurements | Reproduced score within tolerance of published; results in `docs/results.md` |
| **M3: Analysis** *(resume target)* | Sub-metric breakdown, failure taxonomy with renders, navhard two-stage, README with figures | README shows reproduction table, scaling chart, failure gallery |
| **M4: Extension** | Chosen option from §4 Phase C with ablations | Extension vs baseline table over 3 seeds |
| **M5: Write-up** | CSE 593 capstone report + CVPR WAD workshop paper (check deadline and format when the call is out) | Paper submitted; capstone deliverable accepted |

## 8. Results log (fill as you go)

| Model | Source | navtest PDMS | v2 EPDMS | navhard 2-stage EPDMS | Notes |
|---|---|---|---|---|---|
| TransFuser | paper | 84.0 | [fill at M1] | [fill at M1] | |
| LTF | paper | 83.8 | [fill at M1] | [fill at M1] | |
| TransFuser | official ckpt, our eval | | | | |
| TransFuser | ours, from scratch (3 seeds) | | | | |
| Extension | ours | | | | |

## 9. Capstone and paper logistics
- CSE 593 applied project, Spring 2027; faculty advisor self-sourced (Prof. Pavlic first contact; outreach by mid-October).
- Capstone deliverable = M1 to M4 + report; paper = M4 results + analysis.
- Sol allocation: confirm GPU type, hours budget, and max GPUs per job early; it bounds the scaling study and the extension.

## 10. Resume mapping

Paste the AV / MLE resume bullets for this project here. Bullets are written as ongoing; the benchmark result is left to fill.

| Bullet (paste) | Resume | Milestone | Evidence in repo |
|---|---|---|---|
| | AV | | |
| | MLE | | |

Typical claim → evidence pairs:
- camera + LiDAR BEV fusion, transformer backbone, detection/segmentation heads → reproduction + architecture section in README
- distributed multi-GPU training on Sol via SLURM → `slurm/`, scaling chart
- benchmark result → `docs/results.md`, §8 table

## 11. Interview talking points
- Why NAVSIM's metric is better than open-loop L2, and what it still can't capture (non-reactive agents in stage one).
- Verify-before-train: why scoring official checkpoints first separates environment bugs from training bugs.
- What LiDAR buys over camera-only (TransFuser vs LTF) and where.
- Why a modest model matches large end-to-end stacks on this benchmark.
- DDP mechanics: gradient all-reduce, effective batch size and LR scaling, data-loader bottlenecks on HPC.
- The failure taxonomy: which scenario types break the planner and why.

## 12. Open questions
- **[DECIDE]** Extension: C1, C2, or C3.
- Primary headline metric: v1 PDMS (comparable to the paper) vs v2 EPDMS (current leaderboard). Default: report both, lead with EPDMS.
- Leaderboard submission (Hugging Face) for the extension?
- GPU budget on Sol and storage quota for the ≈ 450 GB training split plus caches.
