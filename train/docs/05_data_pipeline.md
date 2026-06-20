# 05 — Data Pipeline

How data flows from raw sources to SFT/RL/eval splits, with contamination firewall built in.

> Status note: this is the original data-pipeline sketch. The current
> contract-first synthetic generation protocol is `SYNTHETIC_DATA_FACTORY.md`.

## Flow recap (see also `00_overview.md`)

```
contracts/      raw/                synthesized/        verified/           sft/ rl/ eval/
task contract → EVAS examples   →   LLM-gen      →      EVAS/Spectre    →   packed
schema          skill templates     candidates          verified            for training
                textbook code                          + provenance
```

## Phase 1 implementation order

1. **Source inventory** (audit step) — build a starting catalog
2. **Seed dataset** — gather ~50 verified examples from safe sources
3. **Synthesis pipeline** — expand contract-reviewed seeds via multi-view LLM-generated candidates
4. **Verification pipeline** — drop anything EVAS rejects
5. **Trajectory builder** — decompose verified items into step-by-step CoT
6. **Contamination check** — gate before packing
7. **Pack and split** — SFT/RL/eval

## Source-by-source notes

### `EVAS/evas/examples/`

- **Status**: ⚠️ requires provenance and vaBench-overlap audit before promotion
- **Yield estimate**: ~20-40 example dirs, each with `.va` + `.scs`
- **Quality**: high, hand-curated by EVAS authors
- **Limitation**: small set, narrow circuit families
- **Action**: copy each to `data/raw/evas_examples/`, parse into seed catalog

### `veriloga-skills/`

- **Status**: ⚠️ requires provenance and vaBench-overlap audit before promotion
- **Yield estimate**: ~10-30 reference templates + worked examples
- **Quality**: high
- **Limitation**: even narrower than EVAS examples
- **Action**: pull reference VA snippets into `data/raw/veriloga_skills/`

### `behavioral-veriloga-eval/tasks/`

- **Status**: ⚠️ excluded by default as direct training data; use only for taxonomy/error-type inspiration unless an item receives explicit contamination clearance
- **Yield estimate**: potentially 100+ historical task entries
- **Quality**: variable
- **Audit step**: for each task ID, check `score_denominator_manifest.json::counted_in_score` — exclude any match
- **Action**: write `pipelines/audit_tasks.py` that does this check, output a whitelist

### Textbook / open IP (later phases)

- **Status**: ✅ allowed, cite source
- **Yield estimate**: dozens with effort
- **Quality**: high but Verilog-A may need adaptation
- **Action**: defer to Phase 1.5 if seed yield from above is < 100

### LLM-synthesized novel specs

- **Status**: ✅ allowed, must EVAS-verify
- **Yield estimate**: 5-10× the seed count
- **Quality**: variable, expect 30-50% to fail verification
- **Risk**: hallucinated Verilog-A that compiles but does wrong thing
- **Mitigation**: ALL synthesized items MUST pass through `verify_evas.py`; failed items kept only in `synthesized/` for diagnostics

## Synthesis strategy

Iterative expansion, not bulk generation:

```
seed (verified, ~50)
  │
  ├─ variant by parameter perturbation (gain=1 → gain=10)
  ├─ variant by port renaming + slight behavior change
  ├─ variant by combining two seed circuits
  └─ variant by spec-paraphrasing (same circuit, different prose spec)
       │
       └─ each variant runs through verify_evas.py
            │
            └─ keep verified, drop failed
```

Synthesis LLM choice: Claude Opus / GPT-4-class for quality; cheaper models only for paraphrasing tasks.

## Verification details

`pipelines/verify_evas.py` does:

1. **EVAS compile/elaboration**: invoke EVAS Rust frontend; capture stdout/stderr and diagnostic taxonomy
2. **EVAS simulate**: run the contract harness; success = exit 0, required observables present
3. **Sanity check on output**: trace not all-zero, not all-NaN, has at least 2 distinct values
4. **Reference comparison** (if seeded from a known example): compare to seed's `tran.csv` within tolerance

Outputs:
- `pass` / `fail` label
- full EVAS log
- compile + sim wall times
- trace summary (min, max, mean, length)

## Trajectory building

`pipelines/build_trajectory.py` (Phase 1) takes a verified item and writes step decomposition per `04_trajectory_format.md`.

Implementation sketch:
- Parse `.va` AST (use `pyverilog` or hand-written regex for module + ports)
- Extract port list → `<port>` step
- Extract analog block + key contributions → `<behavior>` step
- For tb tasks, parse `.scs` → `<testbench>` step

Where the parser can't extract structured info, fall back to **LLM-assisted** trajectory building (Claude/GPT-4 given the verified code, asked to write the trajectory in the schema). Then re-validate by checking the trajectory's step-1 ports match the actual `.va` ports.

## Contamination check

`pipelines/check_contamination.py`:

1. Walk `behavioral-veriloga-eval/benchmark-vabench-release-v1/` — collect every prompt text and every gold VA file.
2. Walk `train/data/verified/` — for each candidate prompt and gold code, compute:
   - Exact-string overlap (after normalization)
   - Fuzzy overlap (n-gram Jaccard > 0.8)
   - AST similarity (for VA: parsed module signature match)
3. Exit code 0 if zero hits, exit 1 + diff report if any hit.
4. Designed to be a **pre-commit hook** for any commit adding to `data/verified/`.

## Packing

`pipelines/pack_sft.py` produces `data/sft/train.jsonl` with entries:
```json
{
  "id": "verified_042",
  "prompt": "...spec...",
  "completion": "<think>...</think><answer>...</answer>",
  "level": "L1",
  "task_form": "dut",
  "category": "Comparator and Decision Circuits"
}
```

`pipelines/pack_rl.py` produces:
- `data/rl/prompts.jsonl`: just prompts (no completion; RL samples its own)
- `data/rl/ref_answers.jsonl`: reference answers indexed by prompt ID, for reflective learning

## Quantitative targets (Phase 1)

| Bucket | Target | Floor (acceptable) |
|---|---|---|
| Seed verified | 50 | 30 |
| Total verified | 300 | 200 |
| `sft/train.jsonl` | 250 | 180 |
| `sft/val.jsonl` | 25 | 20 |
| `rl/prompts.jsonl` | 250 | 180 |
| `eval/` | 50 | 30 |
| OOD subset of eval | 10-15 | 8 |

If we can't hit floor, escalate to user before relaxing contamination rules.

## Storage

All under `data/` is gitignored except:
- Schema files (`*.SCHEMA.md`, `*.schema.json`)
- README

Backups: rsync `data/verified/` and `data/sft/` to remote regularly (Phase 1+), but do not publish raw or generated training data in the design-sync repository.
