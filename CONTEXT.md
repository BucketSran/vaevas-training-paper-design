# vaEVAS Research Context

This context defines the paper-level language used to separate benchmark/evaluator work from model-training work.

## Language

**Benchmark Paper**:
A paper centered on vaBench as a credible behavioral Verilog-A benchmark and EVAS as a Spectre-aligned fast evaluator. Its claims depend on benchmark quality, EVAS/Spectre parity, score-readiness, and same-slice speed evidence.
_Avoid_: Training paper, model paper, SFT paper

**Training Paper**:
A separate paper centered on training a local Verilog-A generation model with SFT followed by GRPO. Its claims depend on contamination-safe training data, held-out evaluation, SFT/GRPO comparisons, reward ablations, and reproducible training logs.
_Avoid_: Benchmark paper, vaBench paper

**Verifier-Grounded Training**:
A training approach where model updates are evaluated against executable behavioral artifacts, simulator execution, and functional checkers. In this project, the training-time verifier should be EVAS-accelerated while its credibility is anchored by Spectre agreement on the supported subset.
_Avoid_: Prompt engineering, benchmark scoring

**Direct Generation**:
A training/evaluation mode where the model receives a specification and must produce Verilog-A artifacts without seeing verifier feedback for that attempt. It measures first-attempt generation quality.
_Avoid_: Feedback repair, debug loop

**Feedback Repair**:
A training/evaluation mode where the model receives a specification, broken Verilog-A artifacts, and verifier feedback, then produces a corrected version. It measures whether the model can use compiler, simulator, checker, or Spectre-audit feedback to repair code.
_Avoid_: Direct generation, prompt-only retry

**Diagnostic Verifier Reward**:
A multi-level reward design that scores generated artifacts by structure, compile/elaboration diagnostics, simulation health, functional checker subclaims, feedback-repair quality, and anti-hacking guards. It treats verifier logs as learning signals, not only pass/fail labels.
_Avoid_: Binary compile reward, single scalar pass/fail

**Clean-Room Training Set**:
A newly constructed training dataset with explicit provenance, contamination audit, and verifier evidence for every example. Historical experiment outputs are excluded by default and may only inform error taxonomies unless re-audited item by item.
_Avoid_: Historical sweep reuse, imported raw results

**LLM Synthetic Expansion**:
A clean-room data construction process that uses LLMs to generate new specifications, Verilog-A artifacts, testbenches, and repair cases from safe seeds and explicit transformation constraints. It is a training-paper method contribution only when paired with provenance, diversity controls, verifier filtering, and contamination checks.
_Avoid_: Prompt paraphrasing, unchecked synthetic data

**Contract-First Synthetic Data Factory**:
A synthetic data construction method where a structured behavioral contract is defined before generating DUTs, testbenches, checkers, variants, or faults. LLMs propose candidates, but contracts, independent checks, EVAS filtering, Spectre audit, and diversity controls determine whether data is admitted.
_Avoid_: Free-form synthetic data, self-consistent LLM oracle

**L2 Cold Start**:
A small but explicit SFT exposure to L2 mini-system examples before L2-heavy GRPO. It teaches output structure, module composition, and system patterns so GRPO has nonzero reward variance on L2 prompts.
_Avoid_: Direct cold-start L2 GRPO, L2-only SFT

**Spectre-Grounded Evaluation**:
Evaluation where Spectre is treated as the high-fidelity reference for executable Verilog-A behavior. It is the anchor for claims that require simulator realism.
_Avoid_: EVAS-only final judge, OpenVAF-only evaluation

**EVAS-Accelerated Training**:
Using the Rust EVAS evaluator as the fast training-time verifier for compile, simulate, and reward feedback after establishing EVAS/Spectre agreement on the relevant task slice. Its paper value is acceleration without abandoning Spectre-grounded credibility.
_Avoid_: Spectre replacement, heuristic reward

**Parity Gate**:
The requirement that EVAS and Spectre agree on the supported task slice before EVAS-derived training rewards or speedup claims are treated as credible. Failing the gate blocks strong training-paper claims.
_Avoid_: Informal smoke test, unchecked proxy

**Spectre Shadow Audit**:
A continuous audit process that samples EVAS-scored training or evaluation outputs and reruns them in Spectre to estimate and catch EVAS reward error. EVAS PASS / Spectre FAIL is treated as a zero-tolerance false positive.
_Avoid_: One-time validation, optional spot check

**EVAS False Negative Backlog**:
A deferred repair queue for cases where Spectre passes but EVAS fails. These cases reduce training coverage or efficiency, but they do not directly create reward false positives, so they may be batched and repaired after a fixed audit interval.
_Avoid_: Immediate blocker, reward exploit

**vaBench Probe**:
A post-training, separately gated evaluation of a frozen trained model on vaBench. It is not part of the training loop and is not used for model selection.
_Avoid_: Training eval, validation set, reward signal
