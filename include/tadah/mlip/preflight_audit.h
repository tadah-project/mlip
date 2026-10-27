#ifndef TADAH_MLIP_PREFLIGHT_AUDIT_H
#define TADAH_MLIP_PREFLIGHT_AUDIT_H

// Pre-flight design-matrix audit.
//
// The audit walks every configured descriptor's `meta()`, runs the
// per-parameter and cross-parameter HealthChecks declared by the
// descriptor's metadata schema, and emits per-finding
// verdicts at OK / WARN / FAIL levels.
//
// Audit modes (driven by the AUDIT context key, default `off`):
//   - off   : skip the audit entirely; return an empty report.
//   - warn  : emit WARNINGs via the Logger; never gate execution.
//   - error : emit WARNINGs via the Logger; throw on any FAIL finding.
//
// The dataset scan that backs the audit is sampled by default (256
// structures, deterministic) — opt out with `AUDIT_SAMPLE 0` to use
// the entire StructureDB. The sample size is read from the context by
// `run_preflight_from_db` and forwarded to `compute_data_stats`.
//
// The audit is intentionally "soft": it depends on `DataStats` and the
// seed `Context`, never on a built `DesignMatrix`. This lets it run
// without a realised matrix (e.g. inside `tadah hpo` before NOISE
// CALIBRATION) and keeps the invocation cost bounded by the dataset
// scan that already happens for stats.

#include <iosfwd>
#include <string>
#include <vector>

#include <tadah/core/context.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/models/descriptors/param_meta.h>

namespace tadah {
namespace mlip {

class StructureDB;

/** \brief Audit verbosity / gating mode. */
enum class AuditMode { Off, Warn, Error };

/** \brief Parse the AUDIT context value. Throws on unknown values
 *  (the user's typo should not silently degrade to "off").
 */
AuditMode parse_audit_mode(const std::string& s);

/** \brief Read AUDIT from the context, defaulting to Off when absent.
 *
 *  AUDIT defaults are deliberately Off so that large training DBs do
 *  not pay the data-stats scan when the user has not opted in. Set
 *  `AUDIT warn` (or `error`) in the master config to re-enable. */
AuditMode audit_mode_from_context(const tadah::core::Context& ctx);

/** \brief One finding emitted by the audit. */
struct AuditMessage {
  enum class Level { Ok, Warn, Fail };
  Level       level{Level::Ok};
  std::string descriptor_label{};   ///< e.g. "DM_mBlipEnv"
  std::string param_key{};          ///< e.g. "CGRIDMB" — empty for whole-descriptor checks
  std::string text{};               ///< human-readable explanation
};

/** \brief Aggregate of audit messages. */
struct AuditReport {
  std::vector<AuditMessage> messages{};

  bool any_fail() const noexcept;
  bool any_warn() const noexcept;
  bool empty()    const noexcept { return messages.empty(); }

  /** Pretty-print the report to `os` for human inspection. */
  void print(std::ostream& os) const;

  /** Emit each message at its corresponding Logger level. WARN/FAIL
   *  go to WARNING; OK is dropped (information-only audit lines are
   *  noise in the common case where everything is fine). */
  void log() const;
};

/** \brief Soft audit. Walks `meta()` for every descriptor configured
 *  in `seed` (TYPE2B and TYPEMB), runs its HealthChecks against
 *  `data` and `seed`, accumulates verdicts. Each descriptor of a joined
 *  line (D2_mJoin, DM_mJoin) is checked on its own context
 *  (joined_descriptors(): its channels, its cutoff), and its messages
 *  name an entry by its place in the whole key.
 *
 *  Honours `mode`:
 *    Off   -> returns an empty report immediately
 *    Warn  -> returns the report; caller logs
 *    Error -> returns the report; caller decides whether to throw on
 *             any FAIL finding (see `enforce_audit_mode`).
 *
 *  Never builds a design matrix. Never instantiates the descriptor in
 *  a way that requires a matched-with-seed Context.
 */
AuditReport run_preflight_soft(
    const tadah::core::stats::DataStats& data,
    const tadah::core::Context&          seed,
    AuditMode                            mode);

/** \brief Test-friendly overload. Uses an externally-supplied list of
 *  `(label, DescriptorMeta)` pairs instead of looking them up via the
 *  Registry. This avoids the static-initializer-elision pitfall when
 *  the audit is exercised from a unit test that does not link the
 *  full descriptor registration chain.
 */
AuditReport run_preflight_soft_with_metas(
    const tadah::core::stats::DataStats& data,
    const tadah::core::Context&          seed,
    const std::vector<std::pair<std::string, tadah::models::DescriptorMeta>>& metas,
    AuditMode                            mode);

/** \brief Test-friendly overload mirror of `run_critical_optim_audit`. */
AuditReport run_critical_optim_audit_with_metas(
    const tadah::core::Context&     seed,
    const std::vector<std::string>& optim_keys,
    const std::vector<std::pair<std::string, tadah::models::DescriptorMeta>>& metas);

/** \brief Convenience wrapper. Computes DataStats from `stdb` (no
 *  per-descriptor ρ stats) and runs `run_preflight_soft`. */
AuditReport run_preflight_from_db(
    StructureDB&                stdb,
    const tadah::core::Context& seed,
    AuditMode                   mode);

/** \brief Critical-fixed-not-optimised diagnosis. Emits one WARN per
 *  parameter that is (a) declared `Criticality::Critical` in the meta()
 *  of a configured descriptor that has a key in the search space, (b)
 *  present and constant in the seed, and (c) absent from every OPTIM
 *  block (cutoffs derived from knots or fixed by bump supports under a
 *  hard cutoff, and radial centres at the origin, excepted). Each
 *  descriptor of a joined line (D2_mJoin, DM_mJoin) is judged by its own
 *  values of a key (joined_descriptors(): every centre at the origin); a
 *  key in the search space counts for every descriptor that has it. The
 *  cutoff of a knot channel, and of a bump channel under a hard cutoff, is
 *  excepted where tadah hpo derives it from the largest knot or holds the
 *  bump supports within it: on a line of that descriptor alone, not on a
 *  joined line, whose cutoffs are fixed values of the search. Always emits
 *  at WARN (never FAIL): this is a configuration smell, not a hard error.
 *
 *  `optim_keys` is the canonical set of parameter names under HPO
 *  control (the keys of `HPOTargets::optimap`). A descriptor none of
 *  whose keys is among them is not searched, and its fixed keys are not
 *  flagged: an empty set, or a search of FIXWEIGHT alone, gives an empty
 *  report. `tadah hpo` runs it under AUDIT warn and error only.
 */
AuditReport run_critical_optim_audit(
    const tadah::core::Context& seed,
    const std::vector<std::string>& optim_keys = {});

/** \brief If `mode == Error` and `report.any_fail()`, throw a
 *  `std::runtime_error` whose message lists every FAIL finding. Else
 *  returns silently.
 */
void enforce_audit_mode(const AuditReport& report, AuditMode mode);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_PREFLIGHT_AUDIT_H
