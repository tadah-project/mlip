#include <tadah/mlip/engines/refit_engine.h>
#include <tadah/mlip/engines/refit_log.h>
#include <tadah/mlip/engines/train_engine.h>

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
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
int get_int(const Context &c, const std::string &k, int def) {
  return c.contains(k) && c.size(k) > 0 ? c.get<int>(k) : def;
}
double get_dbl(const Context &c, const std::string &k, double def) {
  return c.contains(k) && c.size(k) > 0 ? c.get<double>(k) : def;
}

// crystal_epa now lives in MODELS (eam_lattice.h) so the HPO compact engine
// can share it.
using tadah::models::eam_lattice::crystal_epa;

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

// One `OPTIM KEY (i) lo hi` line per parameter, each bound = v ± pct%·|v| with
// an absolute floor (relative to the group's scale) so near-zero/negative
// parameters don't collapse to a degenerate window.
void emit_optim_group(std::ostream &os, const std::string &key,
                      const tadah::core::v_type &vals, double pct,
                      bool positive_only, bool commented = false) {
  if (vals.empty()) return;
  double gmax = 0.0;
  for (double v : vals) gmax = std::max(gmax, std::fabs(v));
  const double floor = std::max(1e-8, 1e-3 * gmax);
  const char *pre = commented ? "# " : "";
  os << std::defaultfloat;
  os << "# " << key << "  (+-" << pct << "%, " << vals.size() << " params"
     << (commented ? ", DISABLED by default — uncomment to search" : "")
     << ")\n";
  os << std::scientific << std::setprecision(8);
  for (std::size_t i = 0; i < vals.size(); ++i) {
    const double v = vals[i];
    const double d = std::max(pct / 100.0 * std::fabs(v), floor);
    double lo = v - d, hi = v + d;
    if (positive_only && lo <= 0.0) lo = std::max(1e-6, 0.5 * std::fabs(v));
    if (hi <= lo) hi = lo + std::max(floor, 1e-6);
    os << pre << "OPTIM " << key << " (" << (i + 1) << ") " << lo << " " << hi
       << "\n";
  }
  os << "\n";
}

void emit_optim_scalar(std::ostream &os, const std::string &key, double v,
                       double pct, double lo_clamp, bool commented = false) {
  const double d = std::max(pct / 100.0 * std::fabs(v), 1e-6);
  double lo = std::max(lo_clamp, v - d);
  double hi = v + d;
  if (hi <= lo) hi = lo + 1e-3;
  os << std::scientific << std::setprecision(8);
  os << (commented ? "# " : "") << "OPTIM " << key << " (1) " << lo << " "
     << hi << "\n\n";
}

void write_hpotarget(const std::string &path, const Context &pot,
                     double rcut, double pct) {
  std::ofstream os(path);
  if (!os) {
    core::Logger::instance().log(core::LogLevel::WARNING,
        "tadah refit: cannot write hpotarget file: " + path);
    return;
  }
  os << "# HPO starter target written by `tadah refit`. Run:\n"
     << "#   tadah hpo -c <config> --hpotarget <this file> --validation <val>\n"
     << "# <config>: the refit potential with a DBFILE line appended naming a\n"
     << "# training dataset; <val>: a file with DBFILE line(s) naming the\n"
     << "# validation data the loss is measured on. See the Tadah docs (HPO).\n"
     << "#\n"
     << "# Search space: the active OPTIM lines below. Bounds are +-" << pct
     << "%\n"
     << "# around the fitted values (config key OPTIM_PCT). By default only\n"
     << "# the model's linear coefficients (FIXWEIGHT) are searched; the\n"
     << "# commented blocks add descriptor-shape parameters. Each uncommented\n"
     << "# line adds one dimension to a derivative-free search — raise\n"
     << "# MAXEVAL accordingly.\n\n";

  os << "OPTIMIZER\n"
     << "  STRATEGY\n"
     << "    TYPE PLAIN\n"
     << "    LIB NLOPT\n"
     << "    ALGO LN_BOBYQA\n"   // derivative-free local polish of a smooth fit
     << "    MAXEVAL 200\n"
     << "  ENDSTRATEGY\n"
     << "ENDOPTIMIZER\n\n";

  os << "LOSS L2\n"
     << "FAILSCORE 1e16\n"
     << "# Energy RMSE on the validation data (<target> <weight>):\n"
     << "PC_ERMSE 0 1\n"
     << "# Force RMSE — needs a dataset with forces and a force-trained\n"
     << "# model; the refit potential is fitted to energies only.\n"
     << "# PC_FRMSE 0 1\n"
     << "# LAMMPS-computed observables can be scored with PC_LAMMPS lines\n"
     << "# (see the docs).\n\n";

  os << "# ---- pair function (two-body) shape ----\n";
  emit_optim_scalar(os, "RCUT2B", rcut, pct, 0.0, true);
  auto read_vec = [&pot](const std::string &k) {
    if (!pot.contains(k)) return tadah::core::v_type{};  // knot pots carry
    tadah::core::v_type v(pot.size(k));                  // no width keys
    pot.get(k, v);
    return v;
  };
  // Knot families carry NO width keys (eta is a pure gauge — the pot has
  // no SGRID2B/SGRIDMB/SEMBFUNC entries); emit_optim_group skips empty
  // vectors, so no degenerate OPTIM blocks can appear.
  emit_optim_group(os, "CGRID2B", read_vec("CGRID2B"), pct, false, true);
  emit_optim_group(os, "SGRID2B", read_vec("SGRID2B"), pct, true, true);

  os << "# ---- density function (many-body) shape ----\n";
  emit_optim_scalar(os, "RCUTMB", rcut, pct, 0.0, true);
  emit_optim_group(os, "CGRIDMB", read_vec("CGRIDMB"), pct, false, true);
  emit_optim_group(os, "SGRIDMB", read_vec("SGRIDMB"), pct, true, true);
  emit_optim_group(os, "AMPGRIDMB", read_vec("AMPGRIDMB"), pct, false, true);

  os << "# ---- embedding function shape ----\n";
  emit_optim_group(os, "CEMBFUNC", read_vec("CEMBFUNC"), pct, false, true);
  emit_optim_group(os, "SEMBFUNC", read_vec("SEMBFUNC"), pct, true, true);

  os << "# ---- model linear coefficients (the default search space) ----\n";
  emit_optim_group(os, "FIXWEIGHT", read_vec("FIXWEIGHT"), pct, false);

  os << "OUTPUT 1\nBOUTPUT 1\n";
}

// One plot-friendly file per function: value, 1st and 2nd derivatives
// (central differences on the dense dump grid — inspection only), residual,
// and an in_window flag marking the physically reachable region the fit
// concentrates on. Header documents the columns.
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
     << "# in_window: 1 = physically reachable region (fit emphasis), 0 = tail\n"
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
  const double rwin = refit.r_reach() > 0 ? 0.9 * refit.r_reach() : r_lo;
  const double rhow = refit.rho_reach() > 0 ? refit.rho_reach() : eam.rhomax();
  write_curve_file("refit_curve_phi.dat", "pair phi(r)", r_lo, rcut,
                   [&](double r) { return eam.phi_tab(r); },
                   [&](double r) { return refit.phi_fit(r); }, rwin, rcut);
  write_curve_file("refit_curve_rho.dat", "density rho(r)", r_lo, rcut,
                   [&](double r) { return eam.rho_tab(r); },
                   [&](double r) { return refit.rho_fit(r); }, rwin, rcut);
  write_curve_file("refit_curve_F.dat", "embedding F(rho)", 0.0, eam.rhomax(),
                   [&](double x) { return eam.F_tab(x); },
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
  os << std::scientific << std::setprecision(10);
  for (int i = 0; i < N; ++i) {
    const double rr   = r_lo + (rcut - r_lo) * i / (N - 1);
    const double rhx  = rhomax * i / (N - 1);
    os << rr << " " << eam.phi_tab(rr) << " " << refit.phi_fit(rr) << "  "
       << rr << " " << eam.rho_tab(rr) << " " << refit.rho_fit(rr) << "  "
       << rhx << " " << eam.F_tab(rhx) << " " << refit.F_fit(rhx) << "\n";
  }
}

} // namespace

bool RefitEngine::compact_fit(const tadah::core::Context &,
                              const models::EAMSetflIO &,
                              models::EAMRefitConfig &,
                              std::unique_ptr<models::EAMRefit> &,
                              SelfTest &) {
  return false;  // dense-only build: the HPO engine overrides this hook
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
  auto oF = [&](double x) { return eam.F_tab(x); };
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
  if (!c.contains("EAM") || c.size("EAM") == 0) {
    throw std::runtime_error(
        "RefitEngine requires an input EAM file: declare `EAM <file>` in the "
        "config file passed with -c. Supported: single-element setfl "
        "(eam/alloy or eam/fs).");
  }
}

int RefitEngine::run(tadah::mlip::Task &task) {
  core::Context c = task.getContext();

  const std::string eam_path  = c.get<std::string>("EAM");
  const std::string out_pot   = get_str(c, "OUTFILE", "pot.tadah");
  const std::string setfl_out = get_str(c, "SETFL_OUT", "refit.eam");
  const std::string hpo_out   = get_str(c, "HPOFILEOUT", "refit.hpotarget");
  const std::string level     = get_str(c, "REFIT_LEVEL", "balanced");
  // Default basis: Knot5 in compact mode (the measured-best family: worst-
  // case |dCij| 1.6 GPa vs Blip's 2.3 across the 5 reference EAMs, every
  // E-V observable in tolerance); Blip in dense mode and as the fallback
  // when the compact fitter is unavailable (no Ceres). An EXPLICIT
  // REFIT_BASIS always wins (and explicit knot families on a Ceres-less
  // build remain a hard error rather than a silent substitution).
  const std::string mode_for_basis = get_str(c, "REFIT_MODE", "compact");
  std::string basis = get_str(c, "REFIT_BASIS", "");
  const bool basis_user_set = !basis.empty();
  if (!basis_user_set)
    basis = (mode_for_basis == "compact") ? "Knot5" : "Blip";
  const double optim_pct      = get_dbl(c, "OPTIM_PCT", 5.0);
  const double grid_scale     = get_dbl(c, "OUTPUT_GRID_SCALE", 1.0);

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

  eam.build_splines();

  // ---- configure + fit ----
  models::EAMRefitConfig cfg;
  cfg.basis = basis;
  apply_level(level, cfg.n2b, cfg.nrho, cfg.nF);
  cfg.n2b  = get_int(c, "N2B", cfg.n2b);
  cfg.nrho = get_int(c, "NRHO_BASIS", cfg.nrho);
  cfg.nF   = get_int(c, "NF_BASIS", cfg.nF);
  cfg.placement    = get_str(c, "REFIT_PLACEMENT", cfg.placement);
  cfg.reach_smin   = get_dbl(c, "REFIT_REACH_SMIN", cfg.reach_smin);
  cfg.reach_margin = get_dbl(c, "REFIT_REACH_MARGIN", cfg.reach_margin);
  cfg.r_lo_frac    = get_dbl(c, "REFIT_R_LO_FRAC", cfg.r_lo_frac);
  if (c.contains("REFIT_CONSTRAIN_F0") && c.size("REFIT_CONSTRAIN_F0") > 0)
    cfg.constrain_F0 = c.get<bool>("REFIT_CONSTRAIN_F0");

  // ---- faithfulness tolerances driving automatic basis escalation ----
  // The compression-aware crystal self-test below is the judge; when it fails,
  // the dominant offender (phi / rho / F, attributed by swapping one fitted
  // function at a time into the ORIGINAL crystal sum) gets a bigger basis and
  // the fit reruns. This adapts the basis to material physics no fixed preset
  // can anticipate (e.g. Ti1's steep F' ~ +15 eV/rho at compressed densities
  // demands drho ~ 1e-5, i.e. far more rho basis than other elements).
  const double tol_amb = get_dbl(c, "REFIT_TOL_AMBIENT", 0.1);     // meV/atom
  const double tol_cmp = get_dbl(c, "REFIT_TOL_COMPRESSED", 1.0);  // meV/atom
  const int    max_esc = get_int(c, "REFIT_MAX_ESCALATIONS", 4);
  const int    n_cap   = 1024;  // per-function basis-size safety cap

  const std::string lat = eam.lattice();
  const double a0 = eam.lattice_param();
  const double rc = eam.rcut();

  std::unique_ptr<models::EAMRefit> refit_ptr;
  SelfTest st;

  // ---- compact mode (optimised blip centres/widths; HPO engine hook) ----
  const std::string mode = get_str(c, "REFIT_MODE", "compact");
  const bool basis_blip = (cfg.basis == "Blip" || cfg.basis == "blip");
  const bool basis_knot5 =
      (cfg.basis == "Knot5" || cfg.basis == "KNOT5" || cfg.basis == "knot5");
  const bool basis_blip5 =
      (cfg.basis == "Blip5" || cfg.basis == "BLIP5" || cfg.basis == "blip5" ||
       cfg.basis == "M6");
  const bool basis_knot3 =
      (cfg.basis == "Knot3" || cfg.basis == "KNOT3" || cfg.basis == "knot3");
  const bool basis_knotmix =
      (cfg.basis == "KnotMix" || cfg.basis == "KNOTMIX" ||
       cfg.basis == "knotmix" || cfg.basis == "Mixed" || cfg.basis == "MIXED");
  const bool basis_knotfam = basis_knot5 || basis_knot3 || basis_knotmix;
  bool fitted = false;
  if (mode == "compact") {
    // The compact fitter must optimise the SAME family the receiving
    // EAMRefit evaluates/serialises, else the pot's evaluation basis would
    // silently differ from the fit (measured hazard). Compact supports the
    // blip bumps and the one-sided quintic knot family.
    if (!basis_blip && !basis_knotfam && !basis_blip5)
      throw std::runtime_error(
          "tadah refit: REFIT_MODE compact supports REFIT_BASIS Blip, "
          "Blip5, Knot3, Knot5 or KnotMix (got '" + cfg.basis + "'); use "
          "REFIT_MODE dense for Gaussian, or drop REFIT_BASIS.");
    fitted = compact_fit(c, eam, cfg, refit_ptr, st);
    if (!fitted) {
      core::Logger::instance().log(core::LogLevel::WARNING,
          "tadah refit: REFIT_MODE compact requires the HPO build (Ceres); "
          "falling back to dense fitting.");
      if (!basis_user_set) {
        cfg.basis = "Blip";  // defaulted Knot5 has no dense path
        core::Logger::instance().log(core::LogLevel::WARNING,
            "tadah refit: defaulted basis falls back Knot5 -> Blip for the "
            "dense path (set REFIT_BASIS explicitly to control this).");
      }
    }
  } else if (mode != "dense") {
    core::Logger::instance().log(core::LogLevel::WARNING,
        "tadah refit: unknown REFIT_MODE '" + mode + "' — using dense.");
  }

  const bool dense_is_knot =
      (cfg.basis == "Knot5" || cfg.basis == "KNOT5" || cfg.basis == "knot5" ||
       cfg.basis == "Knot3" || cfg.basis == "KNOT3" || cfg.basis == "knot3" ||
       cfg.basis == "KnotMix" || cfg.basis == "KNOTMIX" ||
       cfg.basis == "knotmix" || cfg.basis == "Mixed" || cfg.basis == "MIXED");
  if (!fitted && dense_is_knot)
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
              "  basis=" + cfg.basis + "  mode=dense");
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
  // (the lattice is guaranteed recognised by the crystal gate above)
  if (cfg.placement != "uniform") {
    core::Logger::instance().log(core::LogLevel::WARNING,
        "tadah refit: windowed basis placement (reachability-driven):");
    core::Logger::instance().log(core::LogLevel::WARNING,
        "  reachable pair distances r >= " + core::to_string(refit.r_reach(), 4) +
        " A (compression to a/a0=" + core::to_string(cfg.reach_smin, 2) +
        ", V/V0~" + core::to_string(std::pow(cfg.reach_smin, 3), 2) + ")");
    core::Logger::instance().log(core::LogLevel::WARNING,
        "  embedding fit window rho <= " + core::to_string(refit.rho_reach(), 4) +
        " (table rhomax=" + core::to_string(eam.rhomax(), 4) +
        "; the unreachable tail keeps a small uniform basis)");
    core::Logger::instance().log(core::LogLevel::WARNING,
        "  (control: REFIT_PLACEMENT window|uniform, REFIT_REACH_SMIN, "
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
    core::Logger::instance().log(core::LogLevel::INFO,
        "  cohesive E/atom: original=" + core::to_string(st.e_orig, 5) +
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
    if (1000.0 * st.mx_a > tol_amb || 1000.0 * st.mx_c > tol_cmp ||
        std::fabs(1000.0 * (st.e_fit - st.e_orig)) > tol_amb) {
      core::Logger::instance().log(core::LogLevel::WARNING,
          "  WARNING: still above tolerance after escalation — raise "
          "REFIT_MAX_ESCALATIONS, pin a larger N2B/NRHO_BASIS/NF_BASIS, or "
          "relax REFIT_TOL_* if this regime is irrelevant for your use.");
    }
  }

  // ---- write pot.tadah ----
  core::Context pot = refit.build_pot_context();
  TrainEngine::dumpPotFile(out_pot, pot);

  // ---- write the round-trip setfl from the NATIVE fitted functions ----
  models::EAMGrid grid = (grid_scale == 1.0) ? eam.input_grid()
                                             : eam.scaled_grid(grid_scale);
  eam.write_setfl(setfl_out, grid,
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
      const double rwin = refit.r_reach() > 0 ? 0.9 * refit.r_reach()
                                              : refit.r_lo();
      os << "phi " << refit.r_lo() << " " << rc << " " << rwin << " " << rc
         << "  # below r_lo the setfl carries the ORIGINAL table "
            "(offset-matched); control: REFIT_R_LO_FRAC, REFIT_REACH_SMIN\n";
      os << "rho " << refit.r_lo() << " " << rc << " " << rwin << " " << rc
         << "  # same conventions as phi\n";
      os << "F   " << 0.0 << " " << eam.rhomax() << " " << 0.0 << " "
         << (refit.rho_reach() > 0 ? refit.rho_reach() : eam.rhomax())
         << "  # tail keeps a small uniform basis; control: "
            "REFIT_REACH_MARGIN\n";
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
  write_hpotarget(hpo_out, pot, eam.rcut(), optim_pct);
  core::Logger::instance().log(core::LogLevel::INFO,
      "tadah refit: wrote pot=" + out_pot + " hpotarget=" + hpo_out +
      " compare=refit_compare.dat");

  return EXIT_SUCCESS;
}

} // namespace engines
} // namespace mlip
} // namespace tadah
