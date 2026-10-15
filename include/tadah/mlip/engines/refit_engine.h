#ifndef TADAH_MLIP_ENGINES_REFIT_ENGINE_H
#define TADAH_MLIP_ENGINES_REFIT_ENGINE_H

#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>
#include <tadah/models/eam/eam_refit.h>
#include <tadah/models/eam/eam_setfl_io.h>

#include <memory>
#include <string>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Refit a tabulated EAM (setfl) potential into a native Tadah M_BLR model.
 *
 * Reads a single-element setfl file, expands its pair / density / embedding
 * functions in a flexible basis (D2_Blip + DM_REAM), writes a standard
 * pot.tadah, the round-trip setfl built from the native fitted functions, a
 * goodness-of-fit table, a tabulated comparison dump for plotting, and an
 * HPO-ready target file with OPTIM bounds.
 *
 * Two fitting modes (config key REFIT_MODE, in any letter case):
 *  - "dense" (this engine): region-uniform blip grids, least squares, basis
 *    auto-escalation against the compression-aware crystal self-test.
 *  - "compact" (the default): optimised blip centres/widths (ladder +
 *    analytic-gradient Ceres polish). Implemented by the HPO library's
 *    engine, which overrides the "REFIT" registration and the
 *    has_compact_fitter() / compact_fit() hooks below. This base has no
 *    compact fitter, so validate() and run() refuse REFIT_MODE compact,
 *    written or the default, before the EAM file is read ("tadah refit:
 *    REFIT_MODE compact (the default) needs the compact fitter, which this
 *    tadah was built without ..."); it fell back to the dense fit with a
 *    WARNING.
 * validate() and run() refuse a key that the configuration's mode does not
 * read, before the numbers (check_refit_mode_keys, refit_keys.h: e.g. N2B
 * under compact, the REFINE block under dense), and under --edit and
 * --retarget a key of a refit of an EAM file, which those modes do not read
 * (check_refit_output_mode_keys: e.g. N2B, REFIT_TOL_AMBIENT, a REFINE
 * block; HPOFILEOUT under --retarget), before a file is read or written.
 *
 * The configuration must declare the density form of the written DM_REAM
 * potential, RHOFORM SIGNED | NONNEG | MONOTONE (validate() and run() throw
 * the migration text without it). The density is fitted, self-tested
 * (EAMRefit::rho_fit) and written in that form, in both modes: the compact
 * fitter (non-negative amplitudes under NONNEG / MONOTONE) and the dense
 * windowed LSQ (NNLS under NONNEG / MONOTONE). MONOTONE refuses the knot
 * families (Knot3/Knot5/KnotMix), including the defaulted compact Knot5, and
 * a cutoff off the MONOTONE whitelist. NONNEG with a knot family locks the
 * density to non-increasing convex shapes: a HINT when REFIT_BASIS names the
 * family, a WARNING naming REFIT_BASIS Blip5 when the knot basis is the
 * REFIT_MODE compact default (a table with a concave core can then fail
 * the self-test).
 */
class RefitEngine : public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;

  /** Compression-aware crystal E-V self-test result (also the escalation
   *  judge). Energies in eV/atom; the table side uses EAMRefit::F_ref, so
   *  under EMBF0 ZERO they are relative to the isolated atom. */
  struct SelfTest {
    bool valid = false;
    double e_orig = 0, e_fit = 0;
    double rmse_a = 0, mx_a = 0, rmse_c = 0, mx_c = 0;
    double worst_s = 1.0;
    double contrib[3] = {0, 0, 0};  ///< |dE| at worst_s from phi / rho / F
  };

protected:
  /** True when this engine has the compact fitter (compact_fit()): HPO's
   *  engine in a build with Ceres. The base has none and returns false, so
   *  REFIT_MODE compact is refused. */
  virtual bool has_compact_fitter() const;

  /** Compact-mode hook, called only when has_compact_fitter(). Returns true
   *  when it produced a fit (refit filled, self-test computed), false when
   *  the fitter declines this table, having logged a WARNING that says why:
   *  run() then fits it in dense mode. HPO's engine declines no table: it
   *  returns true or throws (a reachability analysis whose r_reach is below
   *  the table's dr, which run()'s crystal gate refuses, is
   *  std::logic_error), so with it run() never falls back to the dense fit.
   *  The base throws std::logic_error (it has no compact fitter, and run()
   *  refuses compact mode before calling it). */
  virtual bool compact_fit(const tadah::core::Context &c,
                           const models::EAMSetflIO &eam,
                           models::EAMRefitConfig &cfg,
                           std::unique_ptr<models::EAMRefit> &refit,
                           SelfTest &st);

  /** Run the E-V self-test of `rf` against the original tables (the
   *  embedding as rf.F_ref, the reference rf fits): scan a/a0 in
   *  [smin, 1.10], ambient/compressed split + per-function error attribution
   *  at the worst scale. */
  SelfTest run_self_test(const models::EAMSetflIO &eam,
                         const models::EAMRefit &rf, double smin) const;
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_REFIT_ENGINE_H
