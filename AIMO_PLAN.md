# AIMO_PLAN.md  
**Legitimacy-First Squiggle Research → AIMO Competition Path**

> Goal:  
> Establish *credible, reproducible evidence* that **squiggle-based geometric events** correlate with learning and generalization in math-reasoning LLMs, using architectures and datasets that scale naturally to AIMO-grade models.

This plan prioritizes **research legitimacy over leaderboard position**. Competitive performance is treated as a downstream validation, not the primary objective.

---

## Roadmap Summary (v0)

1. **Lock the spec and build the harnesses**
2. **Identify the candidate pool (breadth-first)**: ~2k–10k candidate training problems
3. **Probe cycle (Impact Clusters)**: run probes on ~50–200 families; measure DIS distribution; identify top ~10 clusters
4. **Expand winners and generate neighbors**: expand to ~200–500 problems via targeted generation (same mechanism, varied surface)
5. **Confirmation on a larger open model** (e.g., DeepSeek-R1-0): curated corpus vs random; same budget; evaluate on holdout/validation subsets
6. **Write the dataset + write-up for submission**

### Step 1 deliverables: spec + harnesses (v0)

- [ ] **Probe harness implementation** in `squiggle-instrumentation` (micro-finetune probe runner)
- [ ] **Probe output contract** uses `squiggle_core.schemas.probe_summar.ProbeSummary` (schema `probe_summary@2.0`)
- [ ] **DIS computation** follows the DIS separation in Squiggle docs (Magnitude × Coherence × Novelty; operational ranking)
- [ ] **Aggregation output** (per experiment) can be written as:
  - `probe_summaries.parquet`
  - `probe_events_candidates.parquet`
  using the existing writer in `squiggle-analysis` (`squiggle_analysis/io/write_probe_parquet_per_experiment.py`)

**DoD:**
Given a base checkpoint and a set of families, we can run probes and produce a Parquet table of DIS + signatures suitable for clustering.

## 0. Guiding Principles (Non-Negotiable)

- **Not a toy model**: architectures, datasets, and training regimes must resemble real LLM practice.
- **Seed-invariant evidence > single-run anecdotes**
- **Separation of concerns**:
  - Training ≠ Analysis
  - Probes ≠ Interpretation
  - Events ≠ Consensus
- **Every artifact traceable** to:
  - run_id
  - dataset slice
  - seed
  - curriculum
  - config hash

**Definition of Done (Global):**
- Another researcher can reproduce *at least one* squiggle-learning claim using only the repo + data instructions.

---

## 0.1 Confirmed Decisions (Current)

- **Probe harness location:** Implement the probe harness in `squiggle-instrumentation` (per `squiggle-matching/README.md` repo responsibilities).
- **Confirmation model strategy:** Start with LoRA fine-tuning for squiggle detection; escalate to full fine-tuning if needed.
- **Aggregation location (Option A):** Keep experiment-level Parquet + `_manifest.json` export in `squiggle-analysis`.
- **AIMO repo output policy:** Store only exported Parquet + manifests (polished, reviewable artifacts), not raw probe JSONs.
- **Experiment naming:** Use human-readable `EXP_ID` plus a hashed `analysis_id` inside exported tables for strict traceability.

## 1. Baseline Infrastructure Readiness ✅

### 1.1 Repo & pipeline sanity
- [x] Scout training pipeline runs end-to-end
- [x] Capture → geometry → events → report pipeline works
- [x] Artifacts written using `squiggle_core.paths`
- [x] `run + analyze` one-command wrapper exists

**DoD:**  
A Scout run produces a `reports/report.md` with geometry + events sections populated.

---

### 1.2 Artifact contract stabilization
- [ ] Confirm canonical artifact locations in docs:
  - `runs/<run_id>/reports/report.md`
  - `geometry_state/<run_id>.parquet`
  - `events_candidates/<run_id>.parquet` = *single-run candidate events*
- [ ] Add **explicit “v0 semantics”** note to docs:
  - `events_candidates` are *per-run change points*, not seed-consensus

**DoD:**  
Docs and code agree on paths and semantics with no ambiguity.

---

## 2. Research-Grade Instrumentation Model (Core Legitimacy Work)

### 2.1 Model selection (NOT Scout)
**Target:** A *real* LLM-style decoder, small enough to instrument heavily.

- [ ] Choose architecture:
  - Decoder-only
  - RMSNorm
  - RoPE
  - SwiGLU
  - Tied embeddings
- [ ] Choose size tier:
  - ⬜ ~350M (fast iteration, many seeds)
  - ⬜ ~1.3B (stronger legitimacy)
- [ ] Fix context length (e.g., 1k–2k)

**DoD:**  
A model that “looks like” a modern LLM to an external reviewer.

---

### 2.2 Training regime definition
- [ ] From-scratch training (no hidden pretraining)
- [ ] Deterministic seeding
- [ ] Curriculum explicitly defined (ordering, mixing)
- [ ] Logging cadence defined (dense early, sparse later)

**DoD:**  
Two independent runs with identical configs produce comparable loss curves.

---

## 3. Dataset Strategy (Legitimacy-Aligned)

### 3.0 Problem family definition (operational)

A **problem family** is a parameterized generator of many problem instances that share the same internal solution program.

- **Underlying solution strategy invariant**
  - Same core algorithm / proof skeleton
  - Same reasoning moves
- **Surface form varies**
  - Different constants, parameters, objects, or constraints
  - Different wrappers / presentations
- **Controlled difficulty**
  - Easy → hard instances via explicit knobs
- **Fully specified and verifiable**
  - Deterministic answer
  - Preferably deterministic proof trace / outline
- **Programmatically sampleable**
  - A generative object, not a static list

### 3.1 Primary dataset
- [ ] Integrate NVIDIA-released math reasoning dataset (https://huggingface.co/datasets/nvidia/OpenMathReasoning/viewer)
- [ ] Verify licensing + competition compatibility
- [ ] Build **dataset manifest**:
  - source
  - size
  - token count
  - reasoning style (CoT / TIR / short)

Notes:
- Initial candidate pool target: **~2k–10k problems**, preferably drawn from OpenMathReasoning.
- If target problem types are underrepresented (modular arithmetic/invariants, discrete log/order/cyclic groups, bijections/counting, graph/path recurrence counting, functional equations, invariant/monovariant puzzles), generate supplemental families.

**DoD:**  
Dataset slice can be reconstructed from a manifest file alone.

---

### 3.2 Holdout & contamination control
- [ ] Define **strict holdout set**
- [ ] Ensure no overlap with:
  - training data
  - probe A
- [ ] Track dataset hashes per run

**DoD:**  
Holdout evaluation produces non-trivial error even when training loss → 0.

---

## 4. Squiggle Discovery Experiments (Core Research Claim)

### 4.1 Multi-seed geometry capture
- [ ] Run ≥3 seeds with identical configs
- [ ] Capture identical layers/metrics
- [ ] Normalize timelines (time-warp allowed)

**DoD:**  
Geometry state exists for all runs with matching schemas.

---

### 4.2 Event detection (v0)
- [ ] Detect per-run change points
- [ ] Label events with:
  - metric
  - layer
  - step interval
  - magnitude
- [ ] Store as candidate events (single-run)

**DoD:**  
Each run produces a non-empty events table.

---

### 4.3 Seed-invariance analysis
- [ ] Compare event distributions across seeds
- [ ] Identify:
  - repeated event *types*
  - approximate alignment windows
- [ ] Flag candidate “structural” events

**DoD:**  
At least one event family appears in ≥2 seeds within tolerance.

---

## 5. Probe & Capability Correlation

### 5.1 Probe design
- [ ] Probe A: in-distribution (training-like)
- [ ] Probe B: holdout (generalization)
- [ ] Probes fixed, immutable, logged once per run

**DoD:**  
Probe metrics appear in scalars parquet and report.

---

### 5.2 Correlation analysis
- [ ] Align probe changes with event windows
- [ ] Identify:
  - probe jumps
  - loss curvature changes
  - geometry changes

**DoD:**  
At least one event correlates with a probe or loss regime shift.

---

## 6. Reporting & Evidence Packaging

### 6.1 Report upgrades
- [ ] Add:
  - probe curves
  - event summaries
  - geometry metric trends
- [ ] Explicitly label:
  - hypotheses
  - observations
  - non-claims

**DoD:**  
`report.md` reads like a *research log*, not a training log.

---

### 6.2 Shareable evidence bundle
- [ ] Minimal reproduction instructions
- [ ] Fixed seeds + configs
- [ ] Clear statement of limitations

**DoD:**  
Another LLM (or human) could summarize findings without asking clarifying questions.

---

## 7. Transition to AIMO-Targeted Runs

### 7.1 Hypothesis transfer
- [ ] Identify which squiggle events:
  - scale-invariant
  - architecture-invariant
- [ ] Select 1–2 candidate signals to test on larger models

**DoD:**  
Clear hypothesis: *“If event X appears, capability Y improves.”*

---

### 7.2 H100 usage plan
- [ ] Reserve H100 time for:
  - LoRA fine-tunes
  - short validation runs
- [ ] Keep instrumentation lightweight but compatible

**DoD:**  
At least one AIMO-style model run includes squiggle-informed logging.

---

## 8. Explicit Non-Goals (v0)

- ❌ Winning the leaderboard immediately
- ❌ Full consensus events across dozens of seeds
- ❌ Cross-model matching at scale
- ❌ Interpretability claims beyond geometry + correlation

---

## Success Criteria (Legitimacy Bar)

This project is **successful** if we can say:

> “We observed repeatable, seed-robust geometric events in a non-toy model trained on real math reasoning data, and those events correlate with learning behavior in a way that is not reducible to loss alone.”
