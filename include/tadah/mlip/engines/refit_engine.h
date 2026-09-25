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
 * Two fitting modes (config key REFIT_MODE):
 *  - "dense" (this engine): region-uniform blip grids, least squares, basis
 *    auto-escalation against the compression-aware crystal self-test.
 *  - "compact": optimised blip centres/widths (ladder + analytic-gradient
 *    Ceres polish). Implemented by the HPO library's engine, which overrides
 *    the "REFIT" registration and the compact_fit() hook below; in an
 *    HPO-less build this base falls back to dense with a warning.
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
  /** Compact-mode hook. Returns true when it produced a fit (refit filled,
   *  self-test computed). The base (dense-only build) returns false and the
   *  caller falls back to dense fitting with a WARNING. */
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
