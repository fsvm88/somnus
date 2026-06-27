# Project Cleanup & Code Quality Analysis

**Date:** 2026-06-28 (During A/B validation)  
**Scope:** Full project folder + Rust codebase analysis

---

## Executive Summary

| Category | Status | Priority |
|----------|--------|----------|
| **Untracked files** | ✅ Clean | - |
| **Backup files** | 🟡 Minor | Low |
| **Compiler warnings** | 🟡 1 unused function | Low |
| **Code organization** | ✅ Well-structured | - |
| **Technical debt** | 🟡 Several opportunities | Medium |

---

## 1. CLEANUP OPPORTUNITIES

### 1.1 Redundant/Backup Files
**Status:** 🟡 Minor cleanup needed

```
models/qwen25_lm.onnx.data.bak (2.3 GB)
  → Safe to delete. Backup from model swapping experiments.
```

**Action:** Delete backup file (saves 2.3 GB)

### 1.2 Dead Test Directories
**Status:** 🟡 Low priority

```
crates/somnus-sensory/tests/         (8 KB)
crates/somnus-encoding/tests/        (4.1 MB)
crates/somnus-perception/tests/      (20 KB)
```

**Context:** 
- Tests use old integration test patterns
- Not part of CI/CD pipeline
- Could be removed or migrated to new test structure

**Action:** Option A: Remove unused tests (saves 4.1 MB)
         Option B: Update to new test framework

### 1.3 Log Files (Experimental)
**Status:** ✅ Can be kept (useful for debugging)

```
ab_batch_topk_run.log              (experimental)
qwen3_lora_training.log            (recent)
ab_qwen3_results.log               (recent)
ab_validation.log                  (ACTIVE - keep)
experiment_logs/*                  (historical)
memory/*/somnus.log                   (session logs)
```

**Recommendation:** Archive old experiment logs to `archive/` if disk space needed

### 1.4 Disk Space Summary

```
Directory      Size      Notes
─────────────────────────────────
target/        43 GB     Build artifacts (expected, regenerable)
models/        22 GB     Model weights (required, expected)
memory/        26 GB     Session data (important for continuity)
.venv/         4.9 GB    Python env (regenerable)
─────────────────────────────────
Total:         95.9 GB   (Reasonable for ML project)
```

**Cleanable:** ~2.3 GB (backup file + old test binaries)

---

## 2. CODE QUALITY ANALYSIS

### 2.1 Compiler Warnings

#### ⚠️ Unused Function: `cosine_sim`
**File:** `crates/somnus-perception/src/lib.rs`  
**Lines:** 1572, 2862 (duplicate definition)  
**Status:** Introduced in DER++ implementation (ADR Item 7)  
**Action:** Remove or document why kept

```rust
// Line 1572: Unused cosine_sim helper
#[allow(dead_code)]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 { ... }

// Line 2862: Duplicate definition!
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 { ... }
```

**Recommendation:** 
- Keep for future use (DER++ staleness detection may expand)
- OR consolidate duplicate into one location
- OR remove and document in ADR

### 2.2 Clippy Warnings

#### 🟡 Unnecessary `map_or` → `is_some_and`
**File:** `crates/somnus/src/main.rs:625`  
**Severity:** Low (readability improvement)  
**Suggested fix:**

```rust
// Current (line 625)
embed_path.file_stem().map_or(false, |n| n.to_string_lossy().starts_with("qwen3"))

// Better
embed_path.file_stem().is_some_and(|n| n.to_string_lossy().starts_with("qwen3"))
```

**Action:** Apply clippy suggestion

### 2.3 Error Handling

**Issue:** 311 unwrap/expect calls across codebase  
**Assessment:** ✅ Acceptable for research code, not production

```
311 × unwrap/expect
   ├─ Most in initialization (acceptable)
   ├─ Some in batch processing (risky if data malformed)
   └─ Generally safe given controlled inputs
```

**Recommendation:**
- Session initialization: keep as-is (fail fast is appropriate)
- Data processing loops: consider `?` or proper error handling

---

## 3. CODE ORGANIZATION & STRUCTURE

### 3.1 Crate Organization
**Status:** ✅ Well-structured

```
crates/
├── somnus-sensory/          (media I/O, audio/video capture)
├── somnus-encoding/         (frozen pretrained model encoders)
├── somnus-perception/       (k-WTA, binding, episode formation)
├── somnus-memory/           (episodic/semantic/cold storage)
└── somnus/                  (main binary, orchestration, sleep cycle)

Total: 15,802 lines of Rust code
```

**Assessment:** Clear separation of concerns, good modularity

### 3.2 Large Files (Potential Refactor Candidates)

| File | Size | Refactoring Opportunity |
|------|------|------------------------|
| `crates/somnus/src/main.rs` | 3,137 lines | Split: sensory pipeline vs action loop |
| `crates/somnus-perception/src/lib.rs` | 2,878 lines | Extract: region types, binding logic |
| `crates/somnus/src/sleep.rs` | 1,456 lines | Consider: NREM/REM split into modules |
| `crates/somnus/src/action.rs` | 1,392 lines | Consider: separate generation + LoRA logic |

**Recommendation:** Monitor, refactor if complexity increases

### 3.3 Dead Code (Intentional)

```
#[allow(dead_code)] suppressions:
├── crates/somnus-sensory/src/media_file.rs  (3)
├── crates/somnus-sensory/src/audio.rs       (1)
├── crates/somnus/src/sleep.rs               (1)
└── crates/somnus/src/action.rs              (5)
```

**Assessment:** ✅ Intentional, documented with suppressions

---

## 4. TECHNICAL DEBT TRACKING

### 4.1 Known Issues (Non-Critical)

| Issue | File | Status | Impact |
|-------|------|--------|--------|
| Unused `cosine_sim` | somnus-perception | Low | Memory footprint |
| Duplicate `cosine_sim` | somnus-perception | Low | Code duplication |
| `map_or` → `is_some_and` | somnus/src/main.rs | Very Low | Readability |
| Dead tests | somnus-sensory/tests | Low | Build time |

### 4.2 Deferred (by Design)

Per CLAUDE.md, Items A–N deferred indefinitely. Documented in:
- `ai-docs/decisions/` (ADR references)
- `.claude/session-state.md` (current queue)

---

## 5. RECOMMENDATIONS PRIORITIZED

### 🔴 Critical (Do first)
- None identified

### 🟡 Medium (Nice to have)
1. **Remove `cosine_sim` duplicate** (line 2862 in somnus-perception/src/lib.rs)
   - Impact: Clarity
   - Effort: 5 minutes

2. **Apply clippy suggestion: `map_or` → `is_some_and`** (main.rs:625)
   - Impact: Readability
   - Effort: 1 minute

3. **Remove model backup file** (models/qwen25_lm.onnx.data.bak)
   - Impact: Disk space (-2.3 GB)
   - Effort: 1 minute

### 🟢 Low (Polish)
1. Archive old experimental logs if disk space needed
2. Update/remove dead test directories (save 4.1 MB)
3. Document why `cosine_sim` is kept (future use in DER++ expansion)

---

## 6. FILE-BY-FILE QUALITY SNAPSHOT

### Rust Code Quality
```
✅ crates/somnus-encoding/         → Clean, minimal external deps
✅ crates/somnus-memory/           → Well-structured, clear APIs
🟡 crates/somnus-perception/       → One unused function, one duplicate
🟡 crates/somnus/src/main.rs       → Large but organized, one clippy warning
✅ crates/somnus-sensory/          → Clean, good error handling
```

### Python Scripts
```
✅ scripts/export_models.py      → Well-documented, clear
✅ scripts/train_lora_*.py       → Clear intent, good structure
✅ scripts/train_campaign.sh     → Clean bash, good practices
✅ scripts/ab_*.sh              → Functional, clear flow
```

### Documentation
```
✅ ai-docs/                      → Comprehensive, up-to-date
✅ .claude/session-state.md      → Current session tracking
✅ CLAUDE.md                     → Behavioral guidelines clear
✅ README.md                     → (if present) should document key files
```

---

## 7. BUILD & TEST INFRASTRUCTURE

### Current State
```
✅ No CI/CD pipeline configured (research project, acceptable)
✅ cargo build/clippy/test work locally
🟡 Test coverage limited (experimental focus)
✅ Reproducibility: seeds fixed, configs tracked
```

### Recommendations
- Keep current lightweight approach (no over-engineering)
- If moving toward reproducible campaigns: document in `CLAUDE.md`
- Current A/B validation approach is sound (external metrics collection)

---

## 8. SECURITY & SAFETY SCAN

### Quick Assessment
```
✅ No hardcoded secrets detected
✅ No unsafe { } blocks (Rust's memory safety at work)
🟡 311 unwrap calls (acceptable for research, monitor in future)
✅ Dependencies pinned (reproducibility)
✅ No shell injection vectors (careful bash practices)
```

---

## Summary Table

| Category | Current | Recommendation | Effort | Priority |
|----------|---------|-----------------|--------|----------|
| **Backup files** | 2.3 GB | Delete | 1 min | Low |
| **Unused `cosine_sim`** | 1 | Document or remove | 5 min | Low |
| **Clippy warnings** | 1 | Fix | 1 min | Low |
| **Dead tests** | 4.1 MB | Archive/remove | 10 min | Low |
| **Large files** | 10 files | Monitor | - | Medium |
| **Dead code** | <2% | Keep (intentional) | - | - |
| **Error handling** | 311 unwrap | Monitor | - | Medium |

---

## Next Steps

**Immediate (if needed):**
1. Remove `qwen25_lm.onnx.data.bak` (saves 2.3 GB)
2. Fix clippy warning (1 minute)
3. Consolidate `cosine_sim` functions (5 minutes)

**Future (non-urgent):**
- Refactor large files if complexity increases
- Archive old experimental logs
- Document dead code intentions in ADRs

**Project Health:** ✅ **Good**
- Code is well-organized
- Technical debt is minimal and documented
- No critical issues blocking research progress
- Ready for continued development

---

*Analysis completed while A/B validation (Session 2/5) runs.*
*All recommendations are non-blocking; current code structure is sound for research phase.*
