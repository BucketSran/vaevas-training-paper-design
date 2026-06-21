"""
Shared Phase 1 manifest schemas and local IO helpers.

The models in this file are the narrow contract between Phase 1 data-generation,
contamination, SFT packing, and GRPO prompt packing scripts. They intentionally
validate metadata and references only; they do not run EVAS or Spectre.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "phase1.v0.1"
SHA256_PREFIX = "sha256:"
SHA256_HEX_LEN = 64
EMPTY_SHA256 = f"{SHA256_PREFIX}{'0' * SHA256_HEX_LEN}"
FORBIDDEN_GRPO_COMPLETION_KEYS = {"output", "completion", "gold_completion", "answer"}

T = TypeVar("T", bound=BaseModel)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Producer(StrictModel):
    name: str
    version: str
    git_commit: str | None = None


class FixtureNotes(StrictModel):
    fixture_notes: list[str] = Field(default_factory=list)


class ProtectedItem(StrictModel):
    protected_id: str
    source_kind: str
    source_path: str
    task_id: str | None
    counted_in_score: bool
    asset_role: str
    raw_sha256: str
    normalized_sha256: str
    ngram_minhash: dict[str, Any] | None = None
    module_signature_hash: str | None = None
    property_signature_hash: str | None = None


class ProtectedIndex(FixtureNotes):
    schema_version: str
    protected_root_ref: str
    benchmark_release_id: str
    source_manifest_refs: list[str]
    indexed_at: str
    producer: Producer
    protected_index_hash: str
    items: list[ProtectedItem]


class SeedRef(StrictModel):
    seed_id: str
    source_tier: str
    source_kind: str
    source_ref: str
    license: str
    allowed_uses: list[str]
    forbidden_uses: list[str]


class SeedCatalogItem(SeedRef):
    intended_levels: list[Literal["L0", "L1", "L2"]]
    intended_task_forms: list[Literal["dut", "tb", "bugfix", "e2e", "conformance"]]
    category: str
    provenance_notes: list[str]
    reviewer: str
    status: Literal["seed_candidate", "accepted_seed", "rejected"]
    contamination_review: dict[str, Any]


class SeedCatalog(FixtureNotes):
    schema_version: str
    catalog_id: str
    purpose: str
    policy_version: str
    producer: Producer
    seed_catalog_hash: str
    items: list[SeedCatalogItem]


class CandidateItem(StrictModel):
    candidate_id: str
    contract_id: str
    contract_ref: str
    contract_sha256: str
    state: str
    level: Literal["L0", "L1", "L2"]
    task_form: Literal["dut", "tb", "bugfix", "e2e", "conformance"]
    category: str
    split_key: str
    seed_refs: list[SeedRef]
    artifact_refs: dict[str, str]
    artifact_hashes: dict[str, str]
    allowed_features: dict[str, Any]
    forbidden_constructs: list[str]
    reward_profile: str
    intended_uses: dict[str, bool]
    lineage: dict[str, Any]
    evidence_refs: dict[str, str]

    @model_validator(mode="after")
    def validate_use_flags(self) -> "CandidateItem":
        required = {"sft_gold", "grpo_prompt", "repair_data", "eval", "diagnostics"}
        missing = required - set(self.intended_uses)
        if missing:
            raise ValueError(f"candidate intended_uses missing {sorted(missing)}")
        if self.intended_uses["grpo_prompt"] and "gold_completion" in self.artifact_refs:
            raise ValueError("GRPO prompt candidates must not expose gold_completion")
        return self


class CandidateIndex(FixtureNotes):
    schema_version: str
    batch_id: str
    batch_plan_ref: str
    producer: Producer
    candidate_index_hash: str
    items: list[CandidateItem]


class StaticCheckItem(StrictModel):
    candidate_id: str
    decision: Literal["pass", "soft_fail", "hard_fail", "quarantine"]
    checks: dict[str, str]
    messages: list[str]


class StaticCheckReport(StrictModel):
    schema_version: str
    candidate_index_hash: str
    producer: Producer
    summary: dict[str, int]
    items: list[StaticCheckItem]
    static_check_report_hash: str


class EvasItem(StrictModel):
    candidate_id: str
    compile_status: dict[str, Any]
    simulate_status: dict[str, Any]
    property_status: dict[str, Any]
    trace_health: dict[str, Any]
    reward_components: dict[str, float]
    runtime_s: float
    log_ref: str
    artifact_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_compile_signal(self) -> "EvasItem":
        frontend = self.compile_status.get("frontend")
        if frontend != "evas_rust_parser_elaboration":
            raise ValueError("compile_status.frontend must be evas_rust_parser_elaboration")
        return self


class EvasVerificationReport(FixtureNotes):
    schema_version: str
    candidate_index_hash: str
    evas_commit: str
    evas_profile: dict[str, Any]
    checker_version: str
    producer: Producer
    summary: dict[str, int]
    items: list[EvasItem]
    evas_report_hash: str


class ContaminationItem(StrictModel):
    candidate_id: str
    decision: Literal["clean", "needs_review", "quarantined", "rejected"]
    strongest_match_tier: str
    matched_protected_ids: list[str]
    signals: dict[str, Any]
    reviewer: str | None = None
    notes: list[str]


class ContaminationReport(StrictModel):
    schema_version: str
    protected_index_hash: str
    candidate_index_hash: str
    policy_version: str
    producer: Producer
    summary: dict[str, int]
    split_report: dict[str, Any]
    items: list[ContaminationItem]
    contamination_report_hash: str


class SpectreShadowManifest(FixtureNotes):
    schema_version: str
    audit_id: str
    purpose: str
    selection_policy: dict[str, Any]
    candidate_index_hash: str
    evas_report_hash: str
    spectre_profile: dict[str, Any]
    items: list[dict[str, Any]]
    spectre_shadow_manifest_hash: str


class SpectreShadowReport(FixtureNotes):
    schema_version: str
    audit_id: str
    spectre_profile: dict[str, Any]
    producer: Producer
    summary: dict[str, int]
    claim_gate: dict[str, Any]
    items: list[dict[str, Any]]
    mismatches: list[dict[str, Any]]
    spectre_shadow_report_hash: str

    @model_validator(mode="after")
    def validate_no_false_positive_claim(self) -> "SpectreShadowReport":
        false_positive_count = self.summary.get("evas_false_positive", 0)
        gate_false_positive_count = self.claim_gate.get("evas_pass_spectre_fail_count", 0)
        if false_positive_count or gate_false_positive_count:
            raise ValueError("fixture must not contain EVAS PASS / Spectre FAIL evidence")
        return self


class DiversityReport(StrictModel):
    schema_version: str
    candidate_index_hash: str
    admission_pool_hash: str
    producer: Producer
    distribution: dict[str, Any]
    duplicate_clusters: dict[str, Any]
    decisions: list[dict[str, Any]]
    diversity_report_hash: str


class SplitManifest(StrictModel):
    schema_version: str
    batch_id: str
    policy_version: str
    source_pool_hash: str
    producer: Producer
    splits: dict[str, list[str]]
    ood_axes: dict[str, Any]
    leakage_checks: dict[str, Any]
    split_manifest_hash: str

    @model_validator(mode="after")
    def validate_no_eval_leakage(self) -> "SplitManifest":
        sft_train = set(self.splits.get("sft_train", []))
        eval_items = set(self.splits.get("validation", []))
        eval_items |= set(self.splits.get("clean_room_eval", []))
        eval_items |= set(self.splits.get("ood_eval", []))
        overlap = sft_train & eval_items
        if overlap:
            raise ValueError(f"sft_train overlaps eval-like splits: {sorted(overlap)}")
        return self


class AdmittedItem(StrictModel):
    item_id: str
    candidate_id: str
    contract_id: str
    split: str
    split_key: str
    level: Literal["L0", "L1", "L2"]
    task_form: Literal["dut", "tb", "bugfix", "e2e", "conformance"]
    category: str
    artifact_hashes: dict[str, str]
    evidence_refs: dict[str, str]
    reward_profile: str
    use_flags: dict[str, bool]
    pack_targets: list[str]


class AdmittedManifest(FixtureNotes):
    schema_version: str
    batch_id: str
    source_hashes: dict[str, str]
    producer: Producer
    admission_policy: dict[str, Any]
    summary: dict[str, Any]
    items: list[AdmittedItem]
    admitted_manifest_hash: str


class SftPackManifest(FixtureNotes):
    schema_version: str
    run_id: str
    admitted_manifest_hash: str
    producer: Producer
    output_files: dict[str, str]
    output_hashes: dict[str, str]
    format: str
    base_model_ref: str
    tokenizer_ref: str
    special_tokens: list[str]
    dataset_info_ref: str
    item_ids: dict[str, list[str]]
    pack_format_contract: dict[str, list[str]]


class GrpoPromptManifest(FixtureNotes):
    schema_version: str
    run_id: str
    admitted_manifest_hash: str
    producer: Producer
    output_files: dict[str, str]
    output_hashes: dict[str, str]
    format: str
    base_checkpoint_ref: str
    reward_runtime_ref: dict[str, Any]
    reward_profiles: list[str]
    group_size_hint: int
    item_ids: dict[str, list[str]]
    prompt_format_contract: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_no_completion_fields(self) -> "GrpoPromptManifest":
        forbidden = set(self.prompt_format_contract.get("forbidden_jsonl_keys", []))
        hidden = set(self.prompt_format_contract.get("model_hidden_fields", []))
        missing = FORBIDDEN_GRPO_COMPLETION_KEYS - forbidden
        if missing:
            raise ValueError(f"GRPO forbidden_jsonl_keys missing {sorted(missing)}")
        leaked = FORBIDDEN_GRPO_COMPLETION_KEYS & hidden
        if leaked:
            raise ValueError(f"GRPO model_hidden_fields still include completion keys {sorted(leaked)}")
        return self


class BatchPlan(StrictModel):
    schema_version: str
    batch_id: str
    purpose: str
    target_counts: dict[str, int]
    categories: dict[str, int]
    ood_holdout: dict[str, Any]
    admission_policy: dict[str, Any]
    notes: list[str]


class PilotBatchPlan(BatchPlan):
    seed_catalog_ref: str
    seed_catalog_hash: str
    planned_seed_ids: list[str]
    split_policy: dict[str, Any]
    generation_budget: dict[str, Any]
    verification_queue: dict[str, Any]
    pilot_plan_hash: str


class SynthesisRunPlan(FixtureNotes):
    schema_version: str
    run_id: str
    purpose: str
    producer: Producer
    seed_catalog_ref: str
    seed_catalog_hash: str
    pilot_plan_ref: str
    pilot_plan_hash: str
    prompt_template_dir: str
    prompt_selection: dict[str, Any]
    model_policy: dict[str, Any]
    output_policy: dict[str, Any]
    review_policy: dict[str, Any]
    validation_policy: dict[str, Any]
    remote_handoff: dict[str, Any]
    synthesis_run_hash: str


class GeneratedContractItem(StrictModel):
    generated_contract_id: str
    contract_ref: str
    contract_sha256: str
    source_prompt_id: str
    prompt_sha256: str | None = None
    seed_ids: list[str]
    category: str
    level: Literal["L0", "L1", "L2"]
    task_form: Literal["dut", "tb", "bugfix", "e2e", "conformance"]
    split_key: str
    model_ref: str
    generation_kind: str
    mechanical_review: dict[str, Any]
    intended_next_stage: str
    notes: list[str]


class GeneratedContractIndex(FixtureNotes):
    schema_version: str
    run_id: str
    synthesis_run_hash: str
    producer: Producer
    source_refs: dict[str, str]
    summary: dict[str, Any]
    items: list[GeneratedContractItem]
    generated_contract_index_hash: str


class ContractReviewItem(StrictModel):
    generated_contract_id: str
    contract_ref: str
    decision: Literal["accept_for_generation", "needs_revision", "quarantine", "reject"]
    reviewer: str
    blocking_findings: list[str]
    required_edits: list[str]
    allowed_next_stage: Literal["artifact_generation", "contract_revision", "none"]
    rationale: str


class ContractReviewManifest(FixtureNotes):
    schema_version: str
    run_id: str
    generated_contract_index_hash: str
    producer: Producer
    review_policy: dict[str, Any]
    summary: dict[str, Any]
    items: list[ContractReviewItem]
    contract_review_manifest_hash: str


class FixtureBundle(StrictModel):
    batch_plan: BatchPlan
    protected_index: ProtectedIndex
    candidate_index: CandidateIndex
    static_check_report: StaticCheckReport
    evas_verification_report: EvasVerificationReport
    contamination_report: ContaminationReport
    spectre_shadow_manifest: SpectreShadowManifest
    spectre_shadow_report: SpectreShadowReport
    diversity_report: DiversityReport
    split_manifest: SplitManifest
    admitted_manifest: AdmittedManifest
    sft_pack_manifest: SftPackManifest
    grpo_prompt_manifest: GrpoPromptManifest

    @field_validator("*", mode="after")
    @classmethod
    def validate_schema_version(cls, value: Any) -> Any:
        schema_version = getattr(value, "schema_version", None)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unexpected schema_version={schema_version!r}")
        return value

    @model_validator(mode="after")
    def validate_cross_refs(self) -> "FixtureBundle":
        candidate_hash = self.candidate_index.candidate_index_hash
        candidate_ids = {item.candidate_id for item in self.candidate_index.items}
        admitted_items = {item.item_id for item in self.admitted_manifest.items}
        admitted_candidate_ids = {item.candidate_id for item in self.admitted_manifest.items}

        require_sha256("protected_index_hash", self.protected_index.protected_index_hash)
        require_sha256("candidate_index_hash", candidate_hash)

        if admitted_candidate_ids - candidate_ids:
            raise ValueError(f"admitted candidates missing from candidate index: {sorted(admitted_candidate_ids - candidate_ids)}")

        for report_name, report_hash in {
            "static_check_report": self.static_check_report.static_check_report_hash,
            "evas_verification_report": self.evas_verification_report.evas_report_hash,
            "contamination_report": self.contamination_report.contamination_report_hash,
            "spectre_shadow_report": self.spectre_shadow_report.spectre_shadow_report_hash,
            "diversity_report": self.diversity_report.diversity_report_hash,
            "split_manifest": self.split_manifest.split_manifest_hash,
        }.items():
            require_sha256(report_name, report_hash)
            if self.admitted_manifest.source_hashes.get(report_name) != report_hash:
                raise ValueError(f"admitted_manifest source_hashes[{report_name}] mismatch")

        expected_candidate_hash_docs = {
            "static_check_report": self.static_check_report.candidate_index_hash,
            "evas_verification_report": self.evas_verification_report.candidate_index_hash,
            "contamination_report": self.contamination_report.candidate_index_hash,
            "spectre_shadow_manifest": self.spectre_shadow_manifest.candidate_index_hash,
            "diversity_report": self.diversity_report.candidate_index_hash,
        }
        for name, value in expected_candidate_hash_docs.items():
            if value != candidate_hash:
                raise ValueError(f"{name}.candidate_index_hash mismatch")

        if self.contamination_report.protected_index_hash != self.protected_index.protected_index_hash:
            raise ValueError("contamination_report.protected_index_hash mismatch")
        if self.spectre_shadow_manifest.evas_report_hash != self.evas_verification_report.evas_report_hash:
            raise ValueError("spectre_shadow_manifest.evas_report_hash mismatch")
        if self.admitted_manifest.source_hashes.get("candidate_index") != candidate_hash:
            raise ValueError("admitted_manifest source_hashes[candidate_index] mismatch")
        if self.sft_pack_manifest.admitted_manifest_hash != self.admitted_manifest.admitted_manifest_hash:
            raise ValueError("sft_pack_manifest admitted_manifest_hash mismatch")
        if self.grpo_prompt_manifest.admitted_manifest_hash != self.admitted_manifest.admitted_manifest_hash:
            raise ValueError("grpo_prompt_manifest admitted_manifest_hash mismatch")

        sft_ids = set(self.sft_pack_manifest.item_ids.get("train", []))
        sft_ids |= set(self.sft_pack_manifest.item_ids.get("val", []))
        grpo_ids = set(self.grpo_prompt_manifest.item_ids.get("prompts", []))
        if sft_ids - admitted_items:
            raise ValueError(f"SFT pack contains non-admitted item IDs: {sorted(sft_ids - admitted_items)}")
        if grpo_ids - admitted_items:
            raise ValueError(f"GRPO pack contains non-admitted item IDs: {sorted(grpo_ids - admitted_items)}")

        self._validate_report_candidate_ids(candidate_ids)
        self._validate_evidence_refs_exist()
        self._validate_no_protected_hash_reuse()
        return self

    def _validate_report_candidate_ids(self, candidate_ids: set[str]) -> None:
        report_ids = {
            "static_check_report": {item.candidate_id for item in self.static_check_report.items},
            "evas_verification_report": {item.candidate_id for item in self.evas_verification_report.items},
            "contamination_report": {item.candidate_id for item in self.contamination_report.items},
            "spectre_shadow_manifest": {str(item["candidate_id"]) for item in self.spectre_shadow_manifest.items},
            "spectre_shadow_report": {str(item["candidate_id"]) for item in self.spectre_shadow_report.items},
            "diversity_report": {str(item["candidate_id"]) for item in self.diversity_report.decisions},
        }
        for name, ids in report_ids.items():
            missing = ids - candidate_ids
            if missing:
                raise ValueError(f"{name} references unknown candidate IDs: {sorted(missing)}")

    def _validate_evidence_refs_exist(self) -> None:
        examples_root = Path("train")
        local_refs: list[str] = [self.candidate_index.batch_plan_ref]
        for item in self.candidate_index.items:
            local_refs.append(item.contract_ref)
            local_refs.extend(item.evidence_refs.values())
        for item in self.admitted_manifest.items:
            local_refs.extend(item.evidence_refs.values())

        missing = sorted(ref for ref in set(local_refs) if not (examples_root / ref).exists())
        if missing:
            raise ValueError(f"local manifest refs do not exist: {missing}")

    def _validate_no_protected_hash_reuse(self) -> None:
        protected_hashes = {item.raw_sha256 for item in self.protected_index.items}
        protected_hashes |= {item.normalized_sha256 for item in self.protected_index.items}
        candidate_hashes: set[str] = set()
        for item in self.candidate_index.items:
            candidate_hashes |= set(item.artifact_hashes.values())
            candidate_hashes.add(item.contract_sha256)
        overlap = protected_hashes & candidate_hashes
        if overlap:
            raise ValueError(f"candidate hashes overlap protected hashes: {sorted(overlap)}")


FIXTURE_FILES: dict[str, tuple[str, type[BaseModel]]] = {
    "batch_plan": ("batch_plan.synth-batch-toy-0001.yaml", BatchPlan),
    "protected_index": ("protected_index.yaml", ProtectedIndex),
    "candidate_index": ("candidate_index.synth-batch-toy-0001.yaml", CandidateIndex),
    "static_check_report": ("static_check_report.synth-batch-toy-0001.yaml", StaticCheckReport),
    "evas_verification_report": ("evas_verification_report.synth-batch-toy-0001.yaml", EvasVerificationReport),
    "contamination_report": ("contamination_report.synth-batch-toy-0001.yaml", ContaminationReport),
    "spectre_shadow_manifest": ("spectre_shadow_manifest.audit-toy-0001.yaml", SpectreShadowManifest),
    "spectre_shadow_report": ("spectre_shadow_report.audit-toy-0001.yaml", SpectreShadowReport),
    "diversity_report": ("diversity_report.synth-batch-toy-0001.yaml", DiversityReport),
    "split_manifest": ("split_manifest.synth-batch-toy-0001.yaml", SplitManifest),
    "admitted_manifest": ("admitted_manifest.synth-batch-toy-0001.yaml", AdmittedManifest),
    "sft_pack_manifest": ("sft_pack_manifest.toy-run-0001.yaml", SftPackManifest),
    "grpo_prompt_manifest": ("grpo_prompt_manifest.toy-run-0001.yaml", GrpoPromptManifest),
}


def is_sha256(value: str) -> bool:
    if not value.startswith(SHA256_PREFIX):
        return False
    payload = value[len(SHA256_PREFIX) :]
    return len(payload) == SHA256_HEX_LEN and all(ch in "0123456789abcdef" for ch in payload)


def require_sha256(name: str, value: str) -> None:
    if not is_sha256(value):
        raise ValueError(f"{name} must be sha256:<64 lowercase hex chars>")


def sha256_bytes(data: bytes) -> str:
    return f"{SHA256_PREFIX}{hashlib.sha256(data).hexdigest()}"


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalize_for_matching(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"#.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


def normalized_sha256_file(path: Path) -> str:
    text = path.read_bytes().decode("utf-8", errors="replace")
    return sha256_text(normalize_for_matching(text))


def canonical_payload_hash(payload: dict[str, Any], hash_field: str | None = None) -> str:
    stable_payload = copy.deepcopy(payload)
    if hash_field:
        stable_payload.pop(hash_field, None)
    encoded = json.dumps(stable_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(encoded)


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a YAML mapping")
    return data


def write_yaml_mapping(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def load_manifest(path: Path, model: type[T]) -> T:
    return model.model_validate(read_yaml_mapping(path))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return sha256_file(path)


def train_relative_ref(path: Path, train_root: Path = Path("train")) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(train_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_train_ref(ref: str, train_root: Path = Path("train")) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    return train_root / path


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "item"


def load_fixture_bundle(examples_dir: Path) -> FixtureBundle:
    loaded: dict[str, Any] = {}
    for key, (filename, model) in FIXTURE_FILES.items():
        path = examples_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        loaded[key] = model.model_validate(read_yaml_mapping(path))
    return FixtureBundle.model_validate(loaded)


def validate_fixture_dir(examples_dir: Path) -> FixtureBundle:
    return load_fixture_bundle(examples_dir)
