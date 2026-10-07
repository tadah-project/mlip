#include <tadah/mlip/engines/refit_engine.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <tadah/mlip/engines/refit_keys.h>
#include <tadah/mlip/engines/refit_log.h>
#include <tadah/mlip/engines/refit_modes.h>
#include <tadah/mlip/engines/train_engine.h>

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/number_format.h>
#include <tadah/models/descriptors/dm/embf0.h>
#include <tadah/models/descriptors/dm/rhoform.h>
#include <tadah/models/eam/eam_setfl_io.h>
#include <tadah/models/eam/eam_refit.h>
#include <tadah/models/eam/eam_lattice.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {
namespace engines {

namespace {

using tadah::core::Context;

std::string get_str(const Context &c, const std::string &k, const std::string &def) {
  return c.contains(k) && c.size(k) > 0 ? c.get<std::string>(k) : def;
}
// The configuration's numbers come from read_refit_numbers (refit_keys.h).

// crystal_epa now lives in MODELS (eam_lattice.h) so the HPO compact engine
// can share it.
using tadah::models::eam_lattice::crystal_epa;

// Prefix of the missing-RHOFORM message: the refit configuration, not a
// potential, must declare the form of the DM_REAM potential it writes.
const char *const kRefitRhoFormWho = "tadah refit configuration";

// The one-sided knot families in the spellings REFIT_BASIS accepts: Knot5,
// Knot3 and KnotMix (alias Mixed).
bool is_knot_family(const std::string &b) {
  return b == "Knot5" || b == "KNOT5" || b == "knot5" || b == "Knot3" ||
         b == "KNOT3" || b == "knot3" || b == "KnotMix" || b == "KNOTMIX" ||
         b == "knotmix" || b == "Mixed" || b == "MIXED";
}

// The RHOFORM x REFIT_BASIS x REFIT_MODE matrix, checked before any fitting.
// `knot_family`: Knot3/Knot5/KnotMix; `basis_user_set`: REFIT_BASIS given
// (else `basis` is the mode's default); `compact`: REFIT_MODE compact;
// `rctype`: the many-body cutoff the refit writes (EAMRefitConfig::rctype).
// Every form fits in both modes (the compact fitter of the HPO build, the
// dense windowed LSQ); what remains are MONOTONE's own conditions, then the
// NONNEG x knot convex lock (a HINT for a chosen knot basis, a WARNING for
// the defaulted one).
void check_rhoform_matrix(models::RhoForm form, const std::string &basis,
                          bool knot_family, bool basis_user_set, bool compact,
                          const std::string &rctype) {
  // MONOTONE is defined for the bump families only; a defaulted knot basis
  // (REFIT_MODE compact) is named as such, since the remedy is to set one.
  if (form == models::RhoForm::Monotone && knot_family)
    throw std::runtime_error(
        "tadah refit: RHOFORM MONOTONE is not defined for REFIT_BASIS " +
        basis +
        (basis_user_set ? " (the one-sided knot family)"
                        : " (the REFIT_MODE compact default, a one-sided "
                          "knot family)") +
        ": use a bump basis (Blip, Blip5) or RHOFORM NONNEG (" +
        models::monotone_knot_refusal_text() + ").");
  if (form == models::RhoForm::Monotone &&
      !models::rhoform_cutoff_whitelisted(rctype))
    throw std::runtime_error(
        "tadah refit: RHOFORM MONOTONE requires a non-negative, "
        "non-increasing many-body cutoff (" +
        models::rhoform_cutoff_whitelist_text() + "); the refit writes " +
        rctype + ".");
  // The knot families fit in compact mode only (dense + knot is refused
  // later), so the lock is reported for a refit that goes ahead: a HINT when
  // the user chose the knot family, a WARNING when it is the compact default
  // (on a table with a concave core, such as the Ravelo Ta setfl, the
  // defaulted refit fails its self-test; Blip5 fits it in NONNEG).
  if (form == models::RhoForm::NonNeg && knot_family && compact) {
    if (basis_user_set)
      core::Logger::instance().log(core::LogLevel::WARNING,
          "HINT: RHOFORM NONNEG with REFIT_BASIS " + basis +
          ": a knot density with every amplitude >= 0 is non-increasing and "
          "convex (knots <= RCUTMB, non-increasing cutoff), so the fit is "
          "locked to such shapes and a table with a concave core is fitted "
          "poorly. RHOFORM SIGNED or a bump basis lifts the lock.");
    else
      core::Logger::instance().log(core::LogLevel::WARNING,
          "tadah refit: RHOFORM NONNEG with the REFIT_MODE compact default "
          "basis " + basis + ": a knot density with every amplitude >= 0 "
          "is non-increasing and convex (knots <= RCUTMB, non-increasing "
          "cutoff), so a table whose density is concave anywhere in the fit "
          "window (e.g. near the core) is fitted poorly and the self-test "
          "can fail. Set REFIT_BASIS Blip5 (a non-negative bump density "
          "without the convex lock), or use RHOFORM SIGNED.");
  }
}

// REFIT_MODE compact needs the compact fitter (RefitEngine::compact_fit):
// an engine without one (this base; HPO's without Ceres) refuses it, written
// or the default, instead of fitting the table in dense mode.
void require_fitter(const Context &c, RefitMode mode, bool has_compact_fitter) {
  if (mode != RefitMode::Compact || has_compact_fitter) return;
  throw std::runtime_error(
      std::string("tadah refit: REFIT_MODE compact") +
      (c.has_stored("REFIT_MODE") ? "" : " (the default)") +
      " needs the compact fitter, which this tadah was built without (it "
      "needs TADAH_ENABLE_NFP and TADAH_ENABLE_CERES); set REFIT_MODE dense");
}

void apply_level(const std::string &level, int &n2b, int &nrho, int &nF) {
  // nF (embedding) dominates the energy fidelity — see EAMRefitConfig. Energy
  // accuracy is material-dependent (the embedding curvature at the realised
  // rho_i differs per element); the crystal-energy report below tells the user
  // when to raise the level.
  if (level == "coarse")        { n2b = 24; nrho = 20; nF = 60;  }
  else if (level == "fine")     { n2b = 60; nrho = 48; nF = 130; }
  else if (level == "accurate") { n2b = 80; nrho = 64; nF = 180; }
  else                          { n2b = 44; nrho = 36; nF = 100; }  // balanced
}

// The hpotarget emitter (emit_optim_group/emit_optim_scalar/write_hpotarget)
// lives in hpotarget_writer.{h,cpp}: it is shared with the --edit mode's
// target regeneration and the --retarget bound rewriter.

// One plot-friendly file per function: value, 1st and 2nd derivatives
// (central differences on the dense dump grid — inspection only), residual,
// and an in_window flag marking the window the fit used
// (EAMRefit::fit_window_r_lo / fit_window_rho_hi, refit_range_report.dat's
// fit_window_lo..fit_window_hi). Header documents the columns.
void write_curve_file(const std::string &path, const std::string &fname,
                      double lo, double hi,
                      const std::function<double(double)> &orig,
                      const std::function<double(double)> &fit,
                      double win_lo, double win_hi) {
  std::ofstream os(path);
  if (!os) return;
  const int N = 1500;
  const double h = (hi - lo) / (N - 1);
  std::vector<double> yo(N), yf(N);
  for (int i = 0; i < N; ++i) {
    const double x = lo + h * i;
    yo[i] = orig(x);
    yf[i] = fit(x);
  }
  os << "# tadah refit inspection curve: " << fname
     << " (original tabulated vs native fit)\n"
     << "# derivatives: central finite differences on this grid (plot aid)\n"
     << "# in_window: 1 = inside the fit window (refit_range_report.dat's "
        "fit_window_lo..fit_window_hi), 0 = outside\n"
     << "# x  v_orig  v_fit  d1_orig  d1_fit  d2_orig  d2_fit  resid  in_window\n";
  os << std::scientific << std::setprecision(10);
  for (int i = 1; i + 1 < N; ++i) {
    const double x = lo + h * i;
    const double d1o = (yo[i + 1] - yo[i - 1]) / (2 * h);
    const double d1f = (yf[i + 1] - yf[i - 1]) / (2 * h);
    const double d2o = (yo[i + 1] - 2 * yo[i] + yo[i - 1]) / (h * h);
    const double d2f = (yf[i + 1] - 2 * yf[i] + yf[i - 1]) / (h * h);
    os << x << " " << yo[i] << " " << yf[i] << " " << d1o << " " << d1f << " "
       << d2o << " " << d2f << " " << (yf[i] - yo[i]) << " "
       << ((x >= win_lo && x <= win_hi) ? 1 : 0) << "\n";
  }
}

void write_curve_files(const models::EAMSetflIO &eam,
                       const models::EAMRefit &refit) {
  const double rcut = eam.rcut();
  const double r_lo = 0.15 * rcut;
  const double rwin = refit.fit_window_r_lo();
  const double rhow = refit.fit_window_rho_hi();
  write_curve_file("refit_curve_phi.dat", "pair phi(r)", r_lo, rcut,
                   [&](double r) { return eam.phi_tab(r); },
                   [&](double r) { return refit.phi_fit(r); }, rwin, rcut);
  write_curve_file("refit_curve_rho.dat", "density rho(r)", r_lo, rcut,
                   [&](double r) { return eam.rho_tab(r); },
                   [&](double r) { return refit.rho_fit(r); }, rwin, rcut);
  // The table in the reference the native F represents (EMBF0 ZERO:
  // F_tab - F_tab(0); FREE: F_tab verbatim).
  write_curve_file("refit_curve_F.dat",
                   refit.embf0_zero() ? "embedding F(rho) - F(0) (EMBF0 ZERO)"
                                      : "embedding F(rho)",
                   0.0, eam.rhomax(),
                   [&](double x) { return refit.F_ref(x); },
                   [&](double x) { return refit.F_fit(x); }, 0.0, rhow);
}

// Legacy combined dump (kept: the plotting script and tests read it).
void write_compare_dump(const std::string &path, const models::EAMSetflIO &eam,
                        const models::EAMRefit &refit) {
  std::ofstream os(path);
  if (!os) return;
  const int N = 500;
  const double rcut = eam.rcut();
  const double r_lo = 0.15 * rcut;
  const double rhomax = eam.rhomax();
  os << "# tadah refit comparison: original (tabulated) vs native fit\n"
     << "# r_phi phi_tab phi_fit  r_rho rho_tab rho_fit  rho F_tab F_fit\n";
  if (refit.embf0_zero())
    os << "# EMBF0 ZERO: F_tab = table F(rho) - F(0), the reference the "
          "native embedding represents\n";
  os << std::scientific << std::setprecision(10);
  for (int i = 0; i < N; ++i) {
    const double rr   = r_lo + (rcut - r_lo) * i / (N - 1);
    const double rhx  = rhomax * i / (N - 1);
    os << rr << " " << eam.phi_tab(rr) << " " << refit.phi_fit(rr) << "  "
       << rr << " " << eam.rho_tab(rr) << " " << refit.rho_fit(rr) << "  "
       << rhx << " " << refit.F_ref(rhx) << " " << refit.F_fit(rhx) << "\n";
  }
}

} // namespace

bool RefitEngine::has_compact_fitter() const {
  return false;  // the HPO engine overrides this hook and compact_fit()
}

bool RefitEngine::compact_fit(const tadah::core::Context &,
                              const models::EAMSetflIO &,
                              models::EAMRefitConfig &,
                              std::unique_ptr<models::EAMRefit> &,
                              SelfTest &) {
  throw std::logic_error(
      "tadah refit: RefitEngine::compact_fit called on an engine without a "
      "compact fitter (internal error: REFIT_MODE compact is refused before "
      "the fit)");
}

RefitEngine::SelfTest RefitEngine::run_self_test(const models::EAMSetflIO &eam,
                                                 const models::EAMRefit &rf,
                                                 double smin) const {
  SelfTest st;
  const std::string lat = eam.lattice();
  const double a0 = eam.lattice_param();
  const double rc = eam.rcut();
  auto ophi = [&](double r) { return eam.phi_tab(r); };
  auto orho = [&](double r) { return eam.rho_tab(r); };
  // the table embedding in the fit's reference (EMBF0 ZERO: F_tab - F_tab(0),
  // energies relative to the isolated atom; FREE: F_tab verbatim) — also the
  // reference of the per-function attribution below
  auto oF = [&](double x) { return rf.F_ref(x); };
  auto fphi = [&](double r) { return rf.phi_fit(r); };
  auto frho = [&](double r) { return rf.rho_fit(r); };
  auto fF = [&](double x) { return rf.F_fit(x); };
  const double e0 = crystal_epa(lat, a0, rc, ophi, orho, oF);
  if (std::isnan(e0)) return st;
  st.valid = true;
  st.e_orig = e0;
  st.e_fit = crystal_epa(lat, a0, rc, fphi, frho, fF);
  // E-V scan across the PRESSURE REGIMES the potential may be used in:
  // a/a0 from reach_smin (default 0.75, V/V0 ~ 0.42) to 1.10.
  // An ambient-only scan provably misses compression failures.
  double se_a = 0, se_c = 0, worst = -1.0;
  int n_a = 0, n_c = 0;
  for (int k = 0; k <= 28; ++k) {
    const double s = smin + (1.10 - smin) * k / 28.0;
    const double d = crystal_epa(lat, s * a0, rc, fphi, frho, fF) -
                     crystal_epa(lat, s * a0, rc, ophi, orho, oF);
    if (s >= 0.95) {
      se_a += d * d; st.mx_a = std::max(st.mx_a, std::fabs(d)); ++n_a;
    } else {
      se_c += d * d; st.mx_c = std::max(st.mx_c, std::fabs(d)); ++n_c;
    }
    if (std::fabs(d) > worst) { worst = std::fabs(d); st.worst_s = s; }
  }
  st.rmse_a = std::sqrt(se_a / std::max(n_a, 1));
  st.rmse_c = std::sqrt(se_c / std::max(n_c, 1));
  // attribution at the worst scale: swap one fitted function at a time
  const double aw = st.worst_s * a0;
  const double ew = crystal_epa(lat, aw, rc, ophi, orho, oF);
  st.contrib[0] = std::fabs(crystal_epa(lat, aw, rc, fphi, orho, oF) - ew);
  st.contrib[1] = std::fabs(crystal_epa(lat, aw, rc, ophi, frho, oF) - ew);
  st.contrib[2] = std::fabs(crystal_epa(lat, aw, rc, ophi, orho, fF) - ew);
  return st;
}

void RefitEngine::validate(const tadah::mlip::Task &task) {
  core::Logger::instance().log(core::LogLevel::INFO, "Validating RefitEngine...");
  const auto &c = task.getContext();
  const bool edit = c.contains("EDIT") && c.size("EDIT") > 0;
  const bool retarget = c.contains("RETARGET") && c.size("RETARGET") > 0;
  if (edit || retarget) {
    // CLI needs/excludes only guard flag usage; the config-file path (and
    // combinations the CLI cannot express) are re-validated here, with the
    // keys of a refit of an EAM file, which neither mode reads, and the one
    // configuration number of these modes, OPTIM_PCT.
    validate_refit_mode_keys(c, task.getKeyProvenance());
    (void)read_optim_pct(c);
    return;
  }
  if (!c.contains("EAM") || c.size("EAM") == 0) {
    throw std::runtime_error(
        "RefitEngine requires an input EAM file: declare `EAM <file>` in the "
        "config file passed with -c (or use --edit/--retarget to operate on "
        "existing refit outputs). Supported: single-element setfl "
        "(eam/alloy or eam/fs).");
  }
  // The written DM_REAM potential declares its density form, which only the
  // configuration can choose (no default).
  models::require_rhoform(c, kRefitRhoFormWho);
  // The mode rule: a key the configuration's REFIT_MODE does not read is
  // refused (before its number would be checked); then every configuration
  // number, strictly and with its range; then whether this engine fits the
  // mode. run() repeats these before it reads the EAM file.
  const RefitMode mode = read_refit_mode(c);
  check_refit_mode_keys(c, mode, task.getKeyProvenance());
  (void)read_refit_numbers(c);
  require_fitter(c, mode, has_compact_fitter());
}

int RefitEngine::run(tadah::mlip::Task &task) {
  core::Context c = task.getContext();

  // ---- editing modes on existing refit outputs (no EAM input) ----
  if (c.contains("RETARGET") && c.size("RETARGET") > 0)
    return run_refit_retarget(c, task.getKeyProvenance());
  if (c.contains("EDIT") && c.size("EDIT") > 0)
    return run_refit_edit(c, task.getKeyProvenance());

  // ---- RHOFORM: the density form fitted, self-tested and written ----
  // Mandatory in the configuration (no default, never inserted); checked
  // before the table is read.
  const models::RhoForm rho_form = models::require_rhoform(c, kRefitRhoFormWho);
  // REFIT_MODE and the mode rule, the configuration numbers (strictly and
  // with their ranges; the compact fitter reads them with the same reader)
  // and whether this engine fits the mode, before the table is read, as in
  // validate().
  const RefitMode mode = read_refit_mode(c);
  check_refit_mode_keys(c, mode, task.getKeyProvenance());
  const RefitNumbers nums = read_refit_numbers(c);
  require_fitter(c, mode, has_compact_fitter());
  const bool compact = (mode == RefitMode::Compact);

  const std::string eam_path  = c.get<std::string>("EAM");
  const std::string out_pot   = get_str(c, "OUTFILE", "pot.tadah");
  const std::string setfl_out = get_str(c, "SETFL_OUT", "refit.eam");
  const std::string hpo_out   = get_str(c, "HPOFILEOUT", "refit.hpotarget");
  const std::string level     = get_str(c, "REFIT_LEVEL", "balanced");
  // Default basis: Knot5 in compact mode (the measured-best family: worst-
  // case |dCij| 1.6 GPa vs Blip's 2.3 across the 5 reference EAMs, every
  // E-V observable in tolerance); Blip in dense mode and when the compact
  // fitter declines the table (below). An EXPLICIT REFIT_BASIS always wins
  // (an explicit knot family in a dense fit is a hard error rather than a
  // silent substitution).
  std::string basis = get_str(c, "REFIT_BASIS", "");
  const bool basis_user_set = !basis.empty();
  if (!basis_user_set)
    basis = compact ? "Knot5" : "Blip";
  const double optim_pct      = nums.optim_pct;
  const double grid_scale     = nums.output_grid_scale;

  // ---- read the tabulated EAM ----
  models::EAMSetflIO eam;
  eam.read_setfl(eam_path);
  core::Logger::instance().log(core::LogLevel::INFO,
      "tadah refit: read " + eam_path + " element=" + eam.element() +
      " nr=" + std::to_string(eam.nr()) + " nrho=" + std::to_string(eam.nrho()) +
      " rcut=" + std::to_string(eam.rcut()));

  // ---- crystal gate (fail fast, before any fitting work): the self-test
  //      (and, in compact mode, the rung-selection metric and reachability
  //      window) needs a recognised ground-state lattice. The setfl label is
  //      normalised to lower case on read, so FCC/BCC/HCP are accepted;
  //      anything else (or a0<=0, e.g. a "NONE 0.0" header) is a hard stop with
  //      a fix hint rather than a silently-skipped self-test. Shared by dense
  //      and compact modes (compact_fit runs later).
  models::eam_lattice::require_crystal(eam.lattice_raw(), eam.lattice_param());

  // ---- the refit.eam grid (OUTPUT_GRID_SCALE), before any fitting: a
  //      scale whose point counts a setfl table cannot have is refused here,
  //      not after the fit has written pot.tadah ----
  models::EAMGrid out_grid = eam.input_grid();
  if (grid_scale != 1.0) {
    try {
      out_grid = eam.scaled_grid(grid_scale);
    } catch (const std::invalid_argument &e) {
      throw std::runtime_error(
          "tadah refit: OUTPUT_GRID_SCALE '" +
          c.stored_value("OUTPUT_GRID_SCALE")
              .value_or(core::format_roundtrip(grid_scale)) +
          "' for " + setfl_out + ": " + e.what());
    }
  }

  eam.build_splines();

  // ---- configure + fit ----
  models::EAMRefitConfig cfg;
  cfg.basis = basis;
  apply_level(level, cfg.n2b, cfg.nrho, cfg.nF);
  cfg.n2b  = nums.n2b.value_or(cfg.n2b);
  cfg.nrho = nums.nrho.value_or(cfg.nrho);
  cfg.nF   = nums.nF.value_or(cfg.nF);
  cfg.placement    = get_str(c, "REFIT_PLACEMENT", cfg.placement);
  cfg.reach_smin   = nums.reach_smin.value_or(cfg.reach_smin);
  cfg.reach_margin = nums.reach_margin.value_or(cfg.reach_margin);
  cfg.r_lo_frac    = nums.r_lo_frac.value_or(cfg.r_lo_frac);
  if (c.contains("REFIT_CONSTRAIN_F0") && c.size("REFIT_CONSTRAIN_F0") > 0)
    cfg.constrain_F0 = c.get<bool>("REFIT_CONSTRAIN_F0");

  // ---- EMBF0: the embedding reference fitted, self-tested and written ----
  // Non-inserting lookup (absent -> the default ZERO); dense and compact
  // modes both read cfg.embf0_zero.
  cfg.embf0_zero = (models::stored_embf0(c).value_or(models::EmbF0Mode::Zero) ==
                    models::EmbF0Mode::Zero);
  if (cfg.embf0_zero) {
    if (!cfg.constrain_F0)  // the blueprint default is true: set by the user
      core::Logger::instance().log(core::LogLevel::WARNING,
          "tadah refit: REFIT_CONSTRAIN_F0 false has no effect under EMBF0 "
          "ZERO — F(0) = 0 exactly by construction (zero-referenced "
          "embedding). Use EMBF0 FREE for an unpinned raw fit.");
    // The one constant the zero-referenced fit drops from every atom's energy.
    const double F0 = eam.F_tab(0.0);
    const std::string xs = core::to_string(F0);
    core::Logger::instance().log(
        F0 != 0.0 ? core::LogLevel::WARNING : core::LogLevel::INFO,
        "tadah refit: EMBF0 ZERO: table F(0) = " + xs + " eV removed; the "
        "native potential and " + setfl_out + " give energies = table - " +
        (F0 < 0.0 ? "(" + xs + ")" : xs) + " per atom (forces and cohesive "
        "energies unchanged)");
  }

  // ---- the basis family (REFIT_BASIS, else the mode's default) ----
  const bool basis_blip = (cfg.basis == "Blip" || cfg.basis == "blip");
  const bool basis_blip5 =
      (cfg.basis == "Blip5" || cfg.basis == "BLIP5" || cfg.basis == "blip5" ||
       cfg.basis == "M6");
  const bool basis_knotfam = is_knot_family(cfg.basis);

  // ---- RHOFORM: form x basis x mode, then the fit in the declared form ----
  check_rhoform_matrix(rho_form, cfg.basis, basis_knotfam, basis_user_set,
                       compact, cfg.rctype);
  cfg.rho_form = rho_form;
  core::Logger::instance().log(core::LogLevel::INFO,
      std::string("tadah refit: RHOFORM ") + models::rhoform_name(rho_form) +
      (rho_form == models::RhoForm::Signed
           ? ": the density rho = f_c sum_n a_n psi_n(r) with signed "
             "amplitudes (the density before tadah 1.6.0)"
       : rho_form == models::RhoForm::NonNeg
           ? ": the density amplitudes a_n of rho = f_c sum_n a_n psi_n(r) "
             "are fitted by non-negative least squares (rho >= 0)"
           : ": the density rho = f_c sum_n a_n S_n(r) in integrated-bump "
             "steps S_n, amplitudes fitted by non-negative least squares "
             "(rho >= 0 and non-increasing)") +
      "; the potential declares RHOFORM " + models::rhoform_name(rho_form) +
      ".");

  // ---- faithfulness tolerances driving automatic basis escalation ----
  // The compression-aware crystal self-test below is the judge; when it fails,
  // the dominant offender (phi / rho / F, attributed by swapping one fitted
  // function at a time into the ORIGINAL crystal sum) gets a bigger basis and
  // the fit reruns. This adapts the basis to material physics no fixed preset
  // can anticipate (e.g. Ti1's steep F' ~ +15 eV/rho at compressed densities
  // demands drho ~ 1e-5, i.e. far more rho basis than other elements).
  const double tol_amb = nums.tol_ambient;     // meV/atom
  const double tol_cmp = nums.tol_compressed;  // meV/atom
  const int    max_esc = nums.max_escalations;
  const int    n_cap   = 1024;  // per-function basis-size safety cap (and
                                // the range of N2B/NRHO_BASIS/NF_BASIS)

  const std::string lat = eam.lattice();
  const double a0 = eam.lattice_param();
  const double rc = eam.rcut();

  std::unique_ptr<models::EAMRefit> refit_ptr;
  SelfTest st;

  // ---- compact mode (optimised blip centres/widths; HPO engine hook) ----
  bool fitted = false;
  if (compact) {
    // The compact fitter must optimise the SAME family the receiving
    // EAMRefit evaluates/serialises, else the pot's evaluation basis would
    // silently differ from the fit (measured hazard). Compact supports the
    // blip bumps and the one-sided quintic knot family.
    if (!basis_blip && !basis_knotfam && !basis_blip5)
      throw std::runtime_error(
          "tadah refit: REFIT_MODE compact supports REFIT_BASIS Blip, "
          "Blip5, Knot3, Knot5 or KnotMix (got '" + cfg.basis + "'); use "
          "REFIT_MODE dense for Gaussian, or drop REFIT_BASIS.");
    // (require_fitter refused compact mode without a compact fitter.)
    // false: the fitter could not fit this table and said why in a WARNING;
    // the dense fit runs instead.
    fitted = compact_fit(c, eam, cfg, refit_ptr, st);
    if (!fitted && !basis_user_set) {
      cfg.basis = "Blip";  // defaulted Knot5 has no dense path
      core::Logger::instance().log(core::LogLevel::WARNING,
          "tadah refit: defaulted basis falls back Knot5 -> Blip for the "
          "dense path (set REFIT_BASIS explicitly to control this).");
    }
  }

  // (cfg.basis, not basis_knotfam: a defaulted Knot5 fell back to Blip)
  if (!fitted && is_knot_family(cfg.basis))
    throw std::runtime_error(
        "tadah refit: REFIT_BASIS " + cfg.basis + " requires REFIT_MODE "
        "compact on an HPO (Ceres) build — the one-sided knot families have "
        "no dense windowed-LSQ path. Use REFIT_BASIS Blip/Blip5 for dense "
        "mode.");

  if (!fitted && !basis_blip && !basis_blip5)  // dense + Gaussian: demoted
                                               // (knot families threw above)
    core::Logger::instance().log(core::LogLevel::WARNING,
        "tadah refit: REFIT_BASIS '" + cfg.basis + "' is EXPERIMENTAL and "
        "UNVALIDATED for refit (no LAMMPS faithfulness battery has gated "
        "it; the hard cutoff truncates the Gaussian tail, leaving a value "
        "discontinuity at rcut). Prefer Blip (default), Blip5 or the Knot* "
        "families.");

  RefitLog dlog;
  if (!fitted) {
    dlog.open(get_str(c, "REFIT_LOG", "refit_opt.log"), /*append=*/false);
    dlog.raw("# tadah refit optimisation log (dense windowed-LSQ path)");
    dlog.line("input " + eam_path + "  element=" + eam.element() +
              "  basis=" + cfg.basis + "  mode=dense  rhoform=" +
              models::rhoform_name(cfg.rho_form));
  }
  if (!fitted)
  for (int attempt = 0; ; ++attempt) {
    refit_ptr = std::make_unique<models::EAMRefit>(eam, cfg);
    refit_ptr->fit();
    st = run_self_test(eam, *refit_ptr, cfg.reach_smin);
    if (dlog.ok())
      dlog.line("attempt " + std::to_string(attempt) + ": n2b/nrho/nF = " +
                std::to_string(cfg.n2b) + "/" + std::to_string(cfg.nrho) +
                "/" + std::to_string(cfg.nF) +
                (st.valid ? "  self-test amb/cmp max = " +
                     core::to_string(1000.0 * st.mx_a, 3) + "/" +
                     core::to_string(1000.0 * st.mx_c, 3) + " meV/atom"
                          : "  (no self-test: unknown lattice)"));
    if (!st.valid) break;  // unknown lattice: no internal judge, no escalation
    const bool pass =
        1000.0 * st.mx_a <= tol_amb && 1000.0 * st.mx_c <= tol_cmp &&
        std::fabs(1000.0 * (st.e_fit - st.e_orig)) <= tol_amb;
    if (pass || attempt >= max_esc) break;
    const int which =
        (st.contrib[0] >= st.contrib[1] && st.contrib[0] >= st.contrib[2]) ? 0
        : (st.contrib[1] >= st.contrib[2] ? 1 : 2);
    int *n = which == 0 ? &cfg.n2b : which == 1 ? &cfg.nrho : &cfg.nF;
    if (*n >= n_cap) break;
    const char *nm = which == 0 ? "phi" : which == 1 ? "rho" : "F";
    *n = std::min(n_cap, static_cast<int>(std::ceil(*n * 1.6)));
    core::Logger::instance().log(core::LogLevel::WARNING,
        "tadah refit: self-test above tolerance (ambient max " +
        core::to_string(1000.0 * st.mx_a, 3) + " meV vs " +
        core::to_string(tol_amb, 2) + "; compressed max " +
        core::to_string(1000.0 * st.mx_c, 3) + " meV vs " +
        core::to_string(tol_cmp, 2) + ") — dominant error: " + nm +
        " (contrib phi/rho/F = " + core::to_string(1000.0 * st.contrib[0], 3) +
        "/" + core::to_string(1000.0 * st.contrib[1], 3) + "/" +
        core::to_string(1000.0 * st.contrib[2], 3) +
        " meV at a/a0=" + core::to_string(st.worst_s, 3) + ")");
    core::Logger::instance().log(core::LogLevel::WARNING,
        "  escalating basis -> n2b=" + std::to_string(cfg.n2b) +
        " nrho=" + std::to_string(cfg.nrho) + " nF=" + std::to_string(cfg.nF) +
        "  (control: REFIT_TOL_AMBIENT / REFIT_TOL_COMPRESSED / "
        "REFIT_MAX_ESCALATIONS, or pin N2B / NRHO_BASIS / NF_BASIS)");
  }
  models::EAMRefit &refit = *refit_ptr;

  // ---- report the automatic reachability/placement decisions ----
  // (the lattice is guaranteed recognised by the crystal gate above). The
  // compact fit always fits on the reachable windows; REFIT_PLACEMENT and
  // the small uniform tail basis are the dense fit's.
  if (fitted || cfg.placement != "uniform") {
    core::Logger::instance().log(core::LogLevel::WARNING,
        fitted ? "tadah refit: fit windows (reachability-driven):"
               : "tadah refit: windowed basis placement (reachability-driven):");
    core::Logger::instance().log(core::LogLevel::WARNING,
        "  reachable pair distances r >= " + core::to_string(refit.r_reach(), 4) +
        " A (compression to a/a0=" + core::to_string(cfg.reach_smin, 2) +
        ", V/V0~" + core::to_string(std::pow(cfg.reach_smin, 3), 2) + ")");
    core::Logger::instance().log(core::LogLevel::WARNING,
        "  embedding fit window rho <= " + core::to_string(refit.rho_reach(), 4) +
        " (table rhomax=" + core::to_string(eam.rhomax(), 4) +
        (fitted ? ")" : "; the unreachable tail keeps a small uniform basis)"));
    core::Logger::instance().log(core::LogLevel::WARNING,
        fitted ? "  (control: REFIT_REACH_SMIN, REFIT_REACH_MARGIN)"
               : "  (control: REFIT_PLACEMENT window|uniform, REFIT_REACH_SMIN, "
                 "REFIT_REACH_MARGIN)");
    if (refit.rho_reach() >= 0.999 * eam.rhomax())
      core::Logger::instance().log(core::LogLevel::WARNING,
          "tadah refit: the SOURCE table's rhomax leaves no compression "
          "headroom (reachable rho ~ rhomax). Dense phases / strong "
          "compression beyond the reachability sweep probe the original's "
          "own out-of-table extrapolation — neither the original nor any "
          "refit is defined there (e.g. Al99 bcc@400 GPa).");
  }

  // ---- goodness-of-fit (per function) ----
  // Peak-to-peak amplitude of each tabulated function over its fit range, for
  // a relative (%) read-out alongside the absolute RMSE.
  const double r_lo = 0.20 * eam.rcut();
  auto amp = [](auto f, double a, double b, int N) {
    double lo = 1e300, hi = -1e300;
    for (int i = 0; i < N; ++i) {
      const double v = f(a + (b - a) * i / (N - 1));
      lo = std::min(lo, v); hi = std::max(hi, v);
    }
    return (hi - lo) > 0 ? (hi - lo) : 1.0;
  };
  const double phi_amp = amp([&](double r){ return eam.phi_tab(r); }, r_lo, eam.rcut(), 400);
  const double rho_amp = amp([&](double r){ return eam.rho_tab(r); }, r_lo, eam.rcut(), 400);
  const double F_amp   = amp([&](double x){ return eam.F_tab(x);  }, 0.0, eam.rhomax(), 400);
  auto line = [](const std::string &name, const models::FitStats &s, double a) {
    return "  " + name + " rmse=" + core::to_string(s.rmse, 5) +
           " (" + core::to_string(100.0 * s.rmse / a, 3) + "% of range)" +
           " mae=" + core::to_string(s.mae, 5) +
           " max=" + core::to_string(s.maxabs, 5);
  };
  core::Logger::instance().log(core::LogLevel::INFO,
      "tadah refit: goodness-of-fit (native fit vs tabulated):");
  core::Logger::instance().log(core::LogLevel::INFO, line("pair  phi(r):", refit.phi_stats(), phi_amp));
  core::Logger::instance().log(core::LogLevel::INFO, line("dens  rho(r):", refit.rho_stats(), rho_amp));
  core::Logger::instance().log(core::LogLevel::INFO, line("embed F(rho):", refit.F_stats(), F_amp));

  // ---- end-to-end fidelity report (computed by the escalation loop above) ----
  // st is always valid here: the crystal gate guarantees a recognised lattice.
  if (st.valid) {
    core::Logger::instance().log(core::LogLevel::INFO,
        "tadah refit: crystal energy vs original EAM (" + lat + ", a0=" +
        core::to_string(a0, 4) + " A; basis n2b=" + std::to_string(cfg.n2b) +
        " nrho=" + std::to_string(cfg.nrho) + " nF=" + std::to_string(cfg.nF) +
        "):");
    // EMBF0 ZERO: the table reference is F_tab - F_tab(0), so these are
    // energies relative to the isolated atom
    core::Logger::instance().log(core::LogLevel::INFO,
        std::string(refit.embf0_zero()
                        ? "  cohesive E/atom (relative to the isolated atom): "
                          "original="
                        : "  cohesive E/atom: original=") +
        core::to_string(st.e_orig, 5) +
        " refit=" + core::to_string(st.e_fit, 5) +
        " delta=" + core::to_string(1000.0 * (st.e_fit - st.e_orig), 3) +
        " meV/atom");
    core::Logger::instance().log(core::LogLevel::INFO,
        "  E-V ambient (a/a0 0.95-1.10):    RMSE=" +
        core::to_string(1000.0 * st.rmse_a, 3) + " meV/atom  max=" +
        core::to_string(1000.0 * st.mx_a, 3) + " meV/atom");
    core::Logger::instance().log(core::LogLevel::INFO,
        "  E-V compressed (a/a0 0.75-0.95): RMSE=" +
        core::to_string(1000.0 * st.rmse_c, 3) + " meV/atom  max=" +
        core::to_string(1000.0 * st.mx_c, 3) + " meV/atom  (V/V0 down to ~0.42)");
    // The advice names the keys of the fit that ran: the compact fitter's
    // basis ceiling, or the dense fit's escalation and basis sizes.
    if (1000.0 * st.mx_a > tol_amb || 1000.0 * st.mx_c > tol_cmp ||
        std::fabs(1000.0 * (st.e_fit - st.e_orig)) > tol_amb) {
      core::Logger::instance().log(core::LogLevel::WARNING,
          fitted ? "  WARNING: still above tolerance — raise "
                   "REFIT_COMPACT_NMAX or EXPLORE MAX_ADD (a larger basis), "
                   "or relax REFIT_TOL_* if this regime is irrelevant for "
                   "your use."
                 : "  WARNING: still above tolerance after escalation — "
                   "raise REFIT_MAX_ESCALATIONS, pin a larger "
                   "N2B/NRHO_BASIS/NF_BASIS, or relax REFIT_TOL_* if this "
                   "regime is irrelevant for your use.");
    }
  }

  // ---- write pot.tadah ----
  core::Context pot = refit.build_pot_context();
  // A NUMERIC in the refit configuration sets the precision of pot.tadah
  // like it does for train/hpo: dumpPotFile honours it (WARNING below 17)
  // and never writes it. Non-inserting lookup: absent -> 18, exact.
  if (const std::optional<std::string> numeric = c.stored_value("NUMERIC")) {
    pot.remove("NUMERIC");
    pot.add("NUMERIC", *numeric);
  }
  TrainEngine::dumpPotFile(out_pot, pot);

  // ---- write the round-trip setfl from the NATIVE fitted functions ----
  eam.write_setfl(setfl_out, out_grid,
                  [&](double rho){ return refit.F_setfl(rho); },
                  [&](double r){ return refit.rho_setfl(r); },
                  [&](double r){ return refit.z2_setfl(r); },
                  "Generated by tadah refit from " + eam_path);
  core::Logger::instance().log(core::LogLevel::INFO,
      "tadah refit: wrote setfl -> " + setfl_out);
  core::Logger::instance().log(core::LogLevel::WARNING,
      "tadah refit: below the inner fit bound r_lo=" +
      core::to_string(refit.r_lo(), 4) + " A the setfl carries the ORIGINAL "
      "table (offset-matched to the fit at r_lo); the native pot.tadah is "
      "valid for r >= r_lo only. Lower REFIT_R_LO_FRAC in the config to "
      "extend the fitted range if your application compresses harder.");

  // ---- comparison dump for the (not shipped) python plotting script ----
  write_compare_dump("refit_compare.dat", eam, refit);
  write_curve_files(eam, refit);

  // ---- range / window report (machine- and human-readable) ----
  {
    std::ofstream os("refit_range_report.dat");
    if (os) {
      os << "# tadah refit range report — how the automatic decisions split "
            "each function's domain\n"
         << "# function  full_lo  full_hi  fit_window_lo  fit_window_hi  "
            "notes\n";
      os << std::scientific << std::setprecision(8);
      // the windows the fit used: the compact fit's [r_lo, rcut] and
      // [0, rho_reach]; the dense fit's radial window (its wall edge, at
      // most 0.8 r_reach, and never below r_lo) and [0, rho_reach], or
      // [0, rhomax] under REFIT_PLACEMENT uniform
      const double rwin = refit.fit_window_r_lo();
      os << "phi " << refit.r_lo() << " " << rc << " " << rwin << " " << rc
         << "  # below r_lo the setfl carries the ORIGINAL table "
            "(offset-matched); control: REFIT_R_LO_FRAC, REFIT_REACH_SMIN\n";
      os << "rho " << refit.r_lo() << " " << rc << " " << rwin << " " << rc
         << "  # same conventions as phi\n";
      // The F note describes the fit that ran: the compact fit covers
      // [0, rho_reach] and refit.eam joins the table above it
      // (EAMRefit::set_rho_join); the dense fit's windowed placement keeps a
      // small uniform basis on the tail above it, and REFIT_PLACEMENT
      // uniform fits one uniform basis on the whole range (rho_reach 0).
      os << "F   " << 0.0 << " " << eam.rhomax() << " " << 0.0 << " "
         << refit.fit_window_rho_hi()
         << (fitted ? "  # above fit_window_hi the setfl carries the ORIGINAL "
                      "table (offset-matched); control: REFIT_REACH_SMIN, "
                      "REFIT_REACH_MARGIN\n"
             : cfg.placement == "uniform"
                 ? "  # one uniform basis fitted on the full range "
                   "(REFIT_PLACEMENT uniform); control: REFIT_PLACEMENT\n"
                 : "  # tail keeps a small uniform basis; control: "
                   "REFIT_REACH_MARGIN\n");
      os << "# reachability: r_1nn-compressed=" << refit.r_reach()
         << " A  rho_max-reachable=" << refit.rho_reach()
         << " (lattice " << (refit.lattice_known() ? lat : lat + "?")
         << ", a/a0 >= " << cfg.reach_smin << ")\n";
    }
  }
  core::Logger::instance().log(core::LogLevel::INFO,
      "tadah refit: inspection curves -> refit_curve_{phi,rho,F}.dat "
      "(value/1st/2nd derivative, residual, in_window flag)");

  // ---- HPO-ready target ----
  write_hpotarget(hpo_out, pot, optim_pct);
  core::Logger::instance().log(core::LogLevel::INFO,
      "tadah refit: wrote pot=" + out_pot + " hpotarget=" + hpo_out +
      " compare=refit_compare.dat");

  return EXIT_SUCCESS;
}

} // namespace engines
} // namespace mlip
} // namespace tadah
