# Research Integrity Considerations

This document outlines potential issues that could raise red flags with academic reviewers, along with implemented/planned mitigations.

---

## 1. Data Contamination & Leakage

### Risk
Using the same problems for training and evaluation invalidates results. Even partial overlap can inflate performance metrics.

### Current Gaps
- ❌ No tracking of which problems appear in which training runs
- ❌ No strict holdout set enforcement
- ❌ Problem usage history not recorded

### Required Implementation

```python
# Problem usage tracking schema
@dataclass
class ProblemUsage:
    item_id: str              # Stable problem identifier
    run_id: str               # Which run used this problem
    split: str                # "train" | "probe" | "holdout"
    step_first_seen: int      # First training step
    step_last_seen: int       # Last training step
    exposure_count: int       # How many times seen in training
```

### Mitigation Plan
1. **Problem Registry**: Maintain a central registry of all problems with immutable IDs
2. **Usage Ledger**: Log every problem→run assignment before training starts
3. **Holdout Lock**: Pre-declare holdout set; abort if any training touches it
4. **Contamination Check**: Tool to verify no overlap between train/test splits

---

## 2. Classification Reproducibility

### Risk
LLM-based classification is non-deterministic. Different runs may produce different labels, making the experiment unreproducible.

### Current Gaps
- ⚠️ LLM model version not logged with classifications
- ⚠️ No temperature/seed control for classification calls
- ❌ No confidence scores or reasoning traces
- ❌ Single-rater classification (no inter-rater reliability)

### Required Implementation

```python
@dataclass
class ClassificationRecord:
    item_id: str
    family_id: str
    confidence: float              # 0.0-1.0
    reasoning: str                 # LLM's explanation
    model_id: str                  # e.g., "gpt-4o-mini-2024-07-18"
    temperature: float             # Should be 0.0 for reproducibility
    timestamp: datetime
    prompt_hash: str               # SHA256 of exact prompt used
```

### Mitigation Plan
1. **Fixed Model Version**: Pin model version in classification calls
2. **Temperature 0**: Use deterministic decoding
3. **Log Everything**: Store full prompt, response, and metadata
4. **Multi-Rater Option**: Run classification 3x, use majority vote
5. **Confidence Threshold**: Flag items with low confidence for review

---

## 3. Cherry-Picking & Selection Bias

### Risk
Manually selecting which problems to include, or reclassifying after seeing results, can bias findings.

### Current Gaps
- ⚠️ Problem selection criteria not formally pre-registered
- ❌ No audit trail for manual overrides

### Mitigation Plan
1. **Pre-Registration**: Document selection criteria before classification
2. **No Manual Override**: Classification is final unless documented
3. **Override Log**: Any manual corrections must be logged with justification
4. **Blind Analysis**: Family assignments should be frozen before analyzing results

---

## 4. P-Hacking & Multiple Comparisons

### Risk
Running many configurations and reporting only successful ones inflates false positive rate.

### Current Gaps
- ⚠️ No formal hypothesis pre-registration
- ❌ No correction for multiple comparisons

### Mitigation Plan
1. **Pre-Register Hypotheses**: Document expected outcomes before experiments
2. **Report All Runs**: Track and report all attempted configurations
3. **Bonferroni Correction**: Apply when comparing across many families
4. **Effect Size Focus**: Report effect sizes, not just p-values

---

## 5. Reproducibility Infrastructure

### Current State (Partial)
- ✅ Dataset hashing (SHA256 in manifest)
- ✅ Run IDs with seeds
- ⚠️ Environment not fully captured
- ❌ No `requirements.txt` lock file

### Required Additions

```yaml
# Each run's meta.json should include:
environment:
  python_version: "3.11.5"
  torch_version: "2.2.0"
  cuda_version: "12.1"
  package_lock_hash: "abc123..."  # Hash of requirements.lock
  git_commit: "def456..."

data:
  train_manifest_hash: "..."
  holdout_manifest_hash: "..."
  classification_version: "v1.0"
```

---

## 6. Recommended Unified Classifier

The current two-phase classification (target families → control families) adds complexity and potential inconsistency. A unified approach is preferable.

### Proposed Single-Step Classifier

```python
@dataclass
class UnifiedClassification:
    """Single-step classification into all categories."""

    # Primary classification
    family_id: str  # Target family, control family, or "unclassified"
    family_type: Literal["target", "control", "other"]

    # Confidence and alternatives
    confidence: float
    top_3_alternatives: List[Tuple[str, float]]  # (family_id, confidence)

    # Provenance
    reasoning: str
    model_id: str
    prompt_version: str
    timestamp: datetime
```

### All Families in One Prompt
```
Classify this math problem into ONE of these categories:

TARGET FAMILIES (hypothesized high-impact):
1. modular_arithmetic - Problems involving modular arithmetic, remainders, cyclicity
2. cyclic_groups - Discrete log, group order, cyclic group structure
3. bijection_counting - Counting via explicit bijections
4. graph_recurrence - Path/graph counting via recurrence relations
5. functional_equations - Finding functions satisfying given equations
6. invariant_monovariant - Invariants or monovariants to prove impossibility/termination

CONTROL FAMILIES (baseline comparison):
7. geometry_euclidean - Classical Euclidean geometry
8. algebra_polynomials - Polynomial manipulation, roots, factoring
9. inequalities - Proving or using inequalities
10. calculus_analysis - Limits, derivatives, integrals
11. probability_expectation - Probability and expected value
12. combinatorics_basic - Standard counting (permutations, combinations)
13. number_theory_basic - Divisibility, primes, GCD (not modular)
14. sequences_series - Arithmetic/geometric sequences
15. logic_proof - Pure logic or proof techniques
16. optimization - Finding maxima/minima

OTHER:
17. other - Does not fit any category above

Output ONLY the number (1-17).
```

---

## 7. Problem Usage Tracking System

### Schema

```python
# problem_registry.parquet
@dataclass
class ProblemRecord:
    item_id: str          # Primary key (e.g., "omr:cot:12345")
    content_hash: str     # SHA256 of problem text
    family_id: str        # Classified family
    classification_record_id: str  # Link to full classification details
    created_at: datetime

# usage_ledger.parquet
@dataclass
class UsageRecord:
    item_id: str
    run_id: str
    split: str            # "train" | "probe_fixed" | "probe_extended" | "holdout"
    assigned_at: datetime

# At training time, check:
# - No holdout items appear in train/probe
# - Log all assignments to ledger
```

### Enforcement

```python
def validate_no_contamination(run_id: str, train_items: List[str], holdout_items: List[str]) -> None:
    """Abort if any holdout items appear in training set."""
    overlap = set(train_items) & set(holdout_items)
    if overlap:
        raise ContaminationError(f"Run {run_id} would use {len(overlap)} holdout items!")
```

---

## 8. Action Items

### Immediate (Before Next Training Run)
- [ ] Implement unified classifier
- [ ] Add classification metadata logging (model version, timestamp)
- [ ] Create problem registry schema
- [ ] Implement usage ledger

### Before Publication
- [ ] Pre-register hypotheses
- [ ] Lock holdout set
- [ ] Environment capture (pip freeze)
- [ ] Git commit tagging for each experiment
- [ ] Multi-rater classification validation

### Documentation
- [ ] Data collection methodology
- [ ] Family definitions with examples
- [ ] Inclusion/exclusion criteria
- [ ] Analysis pre-registration

---

## 9. External Review Checklist

Before sharing results externally, verify:

1. **Data Integrity**
   - [ ] All problem IDs traceable to source
   - [ ] Classification decisions logged
   - [ ] No train/test contamination

2. **Reproducibility**
   - [ ] Complete environment specified
   - [ ] Random seeds logged
   - [ ] Data manifests with hashes

3. **Statistical Rigor**
   - [ ] Pre-registered hypotheses (or clearly exploratory)
   - [ ] Multiple comparison correction if applicable
   - [ ] Effect sizes reported

4. **Transparency**
   - [ ] All runs reported (not just successful)
   - [ ] Negative results included
   - [ ] Limitations clearly stated
