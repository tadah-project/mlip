#include <tadah/mlip/engines/hpotarget_writer.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <ostream>

#include <tadah/core/logger.h>
#include <tadah/models/eam/pot_curve_edit.h>

namespace tadah {
namespace mlip {
namespace engines {

const std::array<OptimGroupSpec, 8> &hpotarget_group_specs() {
  static const std::array<OptimGroupSpec, 8> specs = {{
      {"CGRID2B", false},
      {"SGRID2B", true},
      {"CGRIDMB", false},
      {"SGRIDMB", true},
      {"AMPGRIDMB", false},
      {"CEMBFUNC", false},
      {"SEMBFUNC", true},
      {"FIXWEIGHT", false},
  }};
  return specs;
}

double optim_group_floor(double gmax) {
  const double f = std::max(1e-8, 1e-3 * gmax);
  const double mag = std::pow(10.0, std::floor(std::log10(f)) - 5.0);
  return std::round(f / mag) * mag;  // 6 significant digits
}

double optim_bound_quantum(double centre, double halfwidth) {
  const double scale = std::fabs(centre) + std::fabs(halfwidth);
  if (scale <= 0.0) return 1e-8;
  return std::pow(10.0, std::floor(std::log10(scale)) - 6.0);
}

double quantize_optim(double x, double q) { return std::round(x / q) * q; }

// One `OPTIM KEY (i) lo hi` line per parameter, each bound = v ± pct%·|v| with
// an absolute floor (relative to the group's scale) so near-zero/negative
// parameters don't collapse to a degenerate window.
void emit_optim_group(std::ostream &os, const std::string &key,
                      const tadah::core::v_type &vals, double pct,
                      bool positive_only, bool commented) {
  if (vals.empty()) return;
  double gmax = 0.0;
  for (double v : vals) gmax = std::max(gmax, std::fabs(v));
  const double floor = optim_group_floor(gmax);
  const char *pre = commented ? "# " : "";
  os << std::defaultfloat;
  os << "# " << key << "  (+-" << pct << "%, " << vals.size() << " params"
     << (commented ? ", DISABLED by default — uncomment to search" : "")
     << ")\n";
  os << std::scientific << std::setprecision(8);
  for (std::size_t i = 0; i < vals.size(); ++i) {
    // centre and half-width snapped to the bound grid: the printed bounds
    // are then exact decimals, so --retarget round-trips byte-for-byte
    const double d_raw = std::max(pct / 100.0 * std::fabs(vals[i]), floor);
    const double q = optim_bound_quantum(vals[i], d_raw);
    const double v = quantize_optim(vals[i], q);
    const double d =
        quantize_optim(std::max(pct / 100.0 * std::fabs(v), floor), q);
    double lo = v - d, hi = v + d;
    if (positive_only && lo <= 0.0) lo = std::max(1e-6, 0.5 * std::fabs(v));
    if (hi <= lo) hi = lo + std::max(floor, 1e-6);
    os << pre << "OPTIM " << key << " (" << (i + 1) << ") " << lo << " " << hi
       << "\n";
  }
  os << "\n";
}

void emit_optim_scalar(std::ostream &os, const std::string &key, double v,
                       double pct, double lo_clamp, bool commented) {
  const double d_raw = std::max(pct / 100.0 * std::fabs(v), 1e-6);
  const double q = optim_bound_quantum(v, d_raw);
  const double vq = quantize_optim(v, q);
  const double d =
      quantize_optim(std::max(pct / 100.0 * std::fabs(vq), 1e-6), q);
  double lo = std::max(lo_clamp, vq - d);
  double hi = vq + d;
  if (hi <= lo) hi = lo + 1e-3;
  os << std::scientific << std::setprecision(8);
  os << (commented ? "# " : "") << "OPTIM " << key << " (1) " << lo << " "
     << hi << "\n\n";
}

void write_hpotarget(const std::string &path, const tadah::core::Context &pot,
                     double pct) {
  namespace potedit = tadah::models::potedit;
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

  auto read_vec = [&pot](const std::string &k) {
    if (!pot.contains(k)) return tadah::core::v_type{};  // knot pots carry
    tadah::core::v_type v(pot.size(k));                  // no width keys
    pot.get(k, v);
    return v;
  };
  // For a KNOT channel the effective cutoff is the largest knot: RCUT is a
  // DERIVED quantity, so no OPTIM scalar is offered — the knot positions
  // (CGRID*) are the search knobs. Bump channels keep the commented scalar.
  auto emit_rcut = [&](const char *suffix, const std::string &key) {
    const auto derived = potedit::derived_knot_rcut(pot, suffix);
    if (derived) {
      os << "# " << key << " is DERIVED for the knot family (= the largest "
         << "CGRID" << suffix << " knot,\n"
         << "# currently " << std::scientific << std::setprecision(8)
         << *derived << "): do not search it —\n"
         << "# vary the CGRID" << suffix
         << " knot positions to change the effective cutoff.\n\n";
    } else if (pot.contains(key) && pot.size(key) > 0) {
      emit_optim_scalar(os, key, pot.get<double>(key), pct, 0.0, true);
    }
  };

  os << "# ---- pair function (two-body) shape ----\n";
  emit_rcut("2B", "RCUT2B");
  // Knot families carry NO width keys (eta is a pure gauge — the pot has
  // no SGRID2B/SGRIDMB/SEMBFUNC entries); emit_optim_group skips empty
  // vectors, so no degenerate OPTIM blocks can appear.
  emit_optim_group(os, "CGRID2B", read_vec("CGRID2B"), pct, false, true);
  emit_optim_group(os, "SGRID2B", read_vec("SGRID2B"), pct, true, true);

  os << "# ---- density function (many-body) shape ----\n";
  emit_rcut("MB", "RCUTMB");
  emit_optim_group(os, "CGRIDMB", read_vec("CGRIDMB"), pct, false, true);
  emit_optim_group(os, "SGRIDMB", read_vec("SGRIDMB"), pct, true, true);
  emit_optim_group(os, "AMPGRIDMB", read_vec("AMPGRIDMB"), pct, false, true);

  os << "# ---- embedding function shape ----\n";
  emit_optim_group(os, "CEMBFUNC", read_vec("CEMBFUNC"), pct, false, true);
  emit_optim_group(os, "SEMBFUNC", read_vec("SEMBFUNC"), pct, true, true);

  // `OPTIM FIXWEIGHT (i)` indices name MODEL WEIGHTS. The refit pot pins
  // every coefficient with FIXINDEX 1-K, so weight index == FIXWEIGHT slot
  // here and the emitted 1..K lines are valid under both readings.
  os << "# ---- model linear coefficients (the default search space) ----\n";
  emit_optim_group(os, "FIXWEIGHT", read_vec("FIXWEIGHT"), pct, false);

  os << "OUTPUT 1\nBOUTPUT 1\n";
}

} // namespace engines
} // namespace mlip
} // namespace tadah
