#include <tadah/mlip/load_transforms.h>

#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/structure_transformations.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr double kB_eV_per_K = 8.617333262e-5;

template <typename Stream>
std::string fmt_g(Stream &&s, double x, int prec = 6) {
  s.precision(prec);
  s << x;
  return s.str();
}

std::string two_decimal(double x) {
  std::ostringstream s;
  s.setf(std::ios::fixed);
  s.precision(4);
  s << x;
  return s.str();
}

// In-place Gaussian elimination with partial pivoting on a K x (K+1) augmented
// matrix.  Returns false if the system is rank-deficient.
bool gauss_solve(std::vector<double> &aug, int k) {
  for (int i = 0; i < k; ++i) {
    int pivot = i;
    double pivot_abs = std::fabs(aug[i * (k + 1) + i]);
    for (int r = i + 1; r < k; ++r) {
      double v = std::fabs(aug[r * (k + 1) + i]);
      if (v > pivot_abs) {
        pivot = r;
        pivot_abs = v;
      }
    }
    if (pivot_abs < 1e-15) return false;
    if (pivot != i) {
      for (int c = 0; c <= k; ++c) {
        std::swap(aug[i * (k + 1) + c], aug[pivot * (k + 1) + c]);
      }
    }
    const double inv_piv = 1.0 / aug[i * (k + 1) + i];
    for (int r = 0; r < k; ++r) {
      if (r == i) continue;
      const double f = aug[r * (k + 1) + i] * inv_piv;
      for (int c = i; c <= k; ++c) {
        aug[r * (k + 1) + c] -= f * aug[i * (k + 1) + c];
      }
    }
  }
  // Now read out solution: aug[i*(k+1)+k] / aug[i*(k+1)+i].
  for (int i = 0; i < k; ++i) {
    aug[i * (k + 1) + k] /= aug[i * (k + 1) + i];
  }
  return true;
}

}  // namespace

namespace tadah {
namespace mlip {

void apply_eshift(StructureDB &stdb,
                  const std::map<int, double> &eshift_per_Z) {
  if (eshift_per_Z.empty()) return;
  for (auto &st : stdb) {
    auto natoms_map = st.get_natoms_map();
    double total = 0.0;
    for (const auto &p : natoms_map) {
      auto it = eshift_per_Z.find(p.first);
      if (it == eshift_per_Z.end()) continue;
      total += it->second * static_cast<double>(p.second);
    }
    st.energy -= total;  // subtract per Tadah's unified ESHIFT convention
  }
}

std::map<int, double>
derive_eshift_from_isolated_atoms(const StructureDB &stdb,
                                  double spread_warn_threshold_eV) {
  // Z -> list of energies of isolated-atom configs of that species
  std::map<int, std::vector<double>> by_Z;
  for (const auto &st : stdb) {
    if (st.natoms() == 1) {
      by_Z[st.atoms[0].Z].push_back(st.energy);
    }
  }
  std::map<int, double> out;
  std::set<core::Element> elements = stdb.get_unique_elements();
  for (const auto &e : elements) {
    auto it = by_Z.find(e.Z);
    if (it == by_Z.end()) {
      core::Logger::instance().log(
          core::LogLevel::WARNING,
          "ESHIFT_ATOM: no isolated-atom configuration in dataset for element ",
          e.symbol, " (Z=", std::to_string(e.Z),
          "); using ESHIFT[", e.symbol, "]=0.0");
      out[e.Z] = 0.0;
      continue;
    }
    const auto &energies = it->second;
    double mean = 0.0;
    for (double v : energies) mean += v;
    mean /= static_cast<double>(energies.size());
    if (energies.size() > 1) {
      double mn = *std::min_element(energies.begin(), energies.end());
      double mx = *std::max_element(energies.begin(), energies.end());
      if ((mx - mn) > spread_warn_threshold_eV) {
        core::Logger::instance().log(
            core::LogLevel::INFO,
            "ESHIFT_ATOM: ", std::to_string(energies.size()),
            " isolated-", e.symbol, " configs with spread ",
            two_decimal(mx - mn), " eV; using mean ",
            two_decimal(mean), " eV/atom.");
      }
    }
    out[e.Z] = mean;
  }
  return out;
}

std::map<int, double>
derive_eshift_by_lsq_fit(const StructureDB &stdb) {
  // Collect all unique species and assign each an index k = 0..K-1.
  std::set<core::Element> elements = stdb.get_unique_elements();
  if (elements.empty() || stdb.size() == 0) return {};
  const int K = static_cast<int>(elements.size());
  std::vector<int> Z_by_k;
  std::map<int, int> k_by_Z;
  Z_by_k.reserve(K);
  for (const auto &e : elements) {
    k_by_Z[e.Z] = static_cast<int>(Z_by_k.size());
    Z_by_k.push_back(e.Z);
  }

  // Build M^T M (K x K) and M^T y (K) directly to avoid materialising M.
  std::vector<double> MtM(K * K, 0.0);
  std::vector<double> Mty(K, 0.0);
  std::vector<double> row_counts(K, 0.0);

  for (const auto &st : stdb) {
    std::fill(row_counts.begin(), row_counts.end(), 0.0);
    auto natoms_map = st.get_natoms_map();
    for (const auto &p : natoms_map) {
      auto it = k_by_Z.find(p.first);
      if (it == k_by_Z.end()) continue;
      row_counts[it->second] += static_cast<double>(p.second);
    }
    const double y = st.energy;
    for (int i = 0; i < K; ++i) {
      Mty[i] += row_counts[i] * y;
      for (int j = i; j < K; ++j) {
        MtM[i * K + j] += row_counts[i] * row_counts[j];
      }
    }
  }
  // Symmetrise lower triangle.
  for (int i = 0; i < K; ++i)
    for (int j = 0; j < i; ++j) MtM[i * K + j] = MtM[j * K + i];

  // Augmented [MtM | Mty] for gauss_solve.
  std::vector<double> aug(K * (K + 1), 0.0);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) aug[i * (K + 1) + j] = MtM[i * K + j];
    aug[i * (K + 1) + K] = Mty[i];
  }
  if (!gauss_solve(aug, K)) {
    core::Logger::instance().log(
        core::LogLevel::WARNING,
        "ESHIFT_DBATOM: normal-equations matrix is singular; "
        "dataset compositions may be linearly dependent. Setting all ESHIFT to 0.");
    std::map<int, double> zero;
    for (int z : Z_by_k) zero[z] = 0.0;
    return zero;
  }

  // Compute residuals for diagnostic.
  std::vector<double> beta(K);
  for (int i = 0; i < K; ++i) beta[i] = aug[i * (K + 1) + K];

  double sse_before = 0.0;
  double sse_after = 0.0;
  for (const auto &st : stdb) {
    std::fill(row_counts.begin(), row_counts.end(), 0.0);
    auto natoms_map = st.get_natoms_map();
    for (const auto &p : natoms_map) {
      auto it = k_by_Z.find(p.first);
      if (it == k_by_Z.end()) continue;
      row_counts[it->second] += static_cast<double>(p.second);
    }
    const double y = st.energy;
    double yhat = 0.0;
    for (int i = 0; i < K; ++i) yhat += row_counts[i] * beta[i];
    sse_before += y * y;
    sse_after += (y - yhat) * (y - yhat);
  }
  const double n = static_cast<double>(stdb.size());
  if (n > 0) {
    core::Logger::instance().log(
        core::LogLevel::INFO,
        "ESHIFT_DBATOM: fitted per-element baselines; energy RMSE before=",
        two_decimal(std::sqrt(sse_before / n)), " eV  after=",
        two_decimal(std::sqrt(sse_after / n)), " eV.");
  }

  std::map<int, double> out;
  for (int k = 0; k < K; ++k) out[Z_by_k[k]] = beta[k];
  return out;
}

OutlierCounts apply_outlier_filter(StructureDB &stdb,
                                   std::optional<double> emin_per_atom,
                                   std::optional<double> emax_per_atom,
                                   std::optional<double> fmax_per_atom) {
  OutlierCounts c;
  if (!emin_per_atom && !emax_per_atom && !fmax_per_atom) return c;

  // We need to update dbidx as we drop structures. Approach: rebuild structures
  // and dbidx in place by walking the existing dbidx ranges.
  std::vector<size_t> old_dbidx = stdb.dbidx;
  if (old_dbidx.empty()) {
    // No dataset boundaries known (e.g. structures were added one-by-one).
    // Treat as a single dataset.
    old_dbidx = {0, stdb.size()};
  }

  std::vector<Structure> new_structures;
  new_structures.reserve(stdb.size());
  std::vector<size_t> new_dbidx;
  new_dbidx.reserve(old_dbidx.size());
  new_dbidx.push_back(0);

  for (size_t d = 0; d + 1 < old_dbidx.size(); ++d) {
    const size_t start = old_dbidx[d];
    const size_t stop = old_dbidx[d + 1];
    for (size_t i = start; i < stop; ++i) {
      const Structure &st = stdb(i);
      const std::size_t n = st.natoms();
      if (n == 0) {
        new_structures.push_back(st);
        continue;
      }
      bool drop = false;
      const double e_per_atom = st.energy / static_cast<double>(n);
      if (emin_per_atom && e_per_atom < *emin_per_atom) {
        drop = true;
        ++c.energy;
      }
      if (!drop && emax_per_atom && e_per_atom > *emax_per_atom) {
        drop = true;
        ++c.energy;
      }
      if (!drop && fmax_per_atom) {
        for (const auto &a : st.atoms) {
          if (a.force.norm() > *fmax_per_atom) {
            drop = true;
            ++c.force;
            break;
          }
        }
      }
      if (!drop) new_structures.push_back(st);
    }
    new_dbidx.push_back(new_structures.size());
  }
  stdb.structures = std::move(new_structures);
  stdb.dbidx = std::move(new_dbidx);
  return c;
}

void apply_per_db_weights(StructureDB &stdb,
                          const std::vector<double> &w) {
  if (w.empty()) return;
  std::vector<size_t> dbidx = stdb.dbidx;
  if (dbidx.empty()) {
    dbidx = {0, stdb.size()};
  }
  if (w.size() + 1 != dbidx.size()) {
    throw std::runtime_error(
        "apply_per_db_weights: weight vector size " +
        std::to_string(w.size()) + " does not match number of datasets " +
        std::to_string(dbidx.size() ? dbidx.size() - 1 : 0));
  }
  for (size_t d = 0; d + 1 < dbidx.size(); ++d) {
    const double f = w[d];
    if (f == 1.0) continue;
    for (size_t i = dbidx[d]; i < dbidx[d + 1]; ++i) {
      stdb(i).eweight *= f;
      stdb(i).fweight *= f;
      stdb(i).sweight *= f;
    }
  }
}

std::vector<double>
auto_balance_db_weights(const StructureDB &stdb, double alpha) {
  if (alpha < 0.0 || alpha > 1.0) {
    throw std::invalid_argument(
        "WDBFILE_AUTO alpha must be in [0, 1]; got " + std::to_string(alpha));
  }
  std::vector<size_t> dbidx = stdb.dbidx;
  if (dbidx.empty()) {
    dbidx = {0, stdb.size()};
  }
  std::vector<double> out(dbidx.size() ? dbidx.size() - 1 : 0, 1.0);
  if (alpha == 0.0) return out;
  for (size_t d = 0; d + 1 < dbidx.size(); ++d) {
    const double n = static_cast<double>(dbidx[d + 1] - dbidx[d]);
    out[d] = (n > 0.0) ? std::pow(n, -alpha) : 0.0;
  }
  return out;
}

void apply_boltzmann_eweight(StructureDB &stdb, double T_kelvin) {
  if (T_kelvin <= 0.0) {
    throw std::invalid_argument("EWEIGHT_TEMP must be > 0; got " +
                                std::to_string(T_kelvin));
  }
  if (stdb.size() == 0) return;

  double e_min = std::numeric_limits<double>::infinity();
  for (const auto &st : stdb) {
    if (st.natoms() == 0) continue;
    const double e_per = st.energy / static_cast<double>(st.natoms());
    if (e_per < e_min) e_min = e_per;
  }
  if (!std::isfinite(e_min)) return;

  const double inv_kT = 1.0 / (kB_eV_per_K * T_kelvin);
  double w_min = std::numeric_limits<double>::infinity();
  double w_max = -std::numeric_limits<double>::infinity();
  for (auto &st : stdb) {
    if (st.natoms() == 0) continue;
    const double e_per = st.energy / static_cast<double>(st.natoms());
    const double f = std::exp(-(e_per - e_min) * inv_kT);
    st.eweight *= f;
    if (st.eweight < w_min) w_min = st.eweight;
    if (st.eweight > w_max) w_max = st.eweight;
  }
  core::Logger::instance().log(
      core::LogLevel::INFO,
      "EWEIGHT_TEMP: Boltzmann reweighting at T=", std::to_string(T_kelvin),
      " K (kB*T=", two_decimal(kB_eV_per_K * T_kelvin),
      " eV); eweight in [", two_decimal(w_min), ", ",
      two_decimal(w_max), "].");
}

void apply_com_force_zero(StructureDB &stdb) {
  std::size_t n_applied = 0;
  double max_residual = 0.0;
  for (auto &st : stdb) {
    if (st.natoms() == 0) continue;
    core::Vec3d sum(0.0, 0.0, 0.0);
    for (const auto &a : st.atoms) sum += a.force;
    const double n = static_cast<double>(st.natoms());
    core::Vec3d mean = sum * (1.0 / n);
    if (mean.norm() > max_residual) max_residual = mean.norm();
    if (mean.norm() == 0.0) continue;
    for (auto &a : st.atoms) a.force -= mean;
    ++n_applied;
  }
  core::Logger::instance().log(
      core::LogLevel::INFO,
      "ZERO_COM_FORCE: zeroed COM force on ", std::to_string(n_applied),
      " configurations (max residual ", two_decimal(max_residual), " eV/Å).");
}

namespace {

// Helper: detect which ESHIFT source(s) are configured.
struct EshiftConfig {
  bool user = false;
  bool atom = false;
  bool dbatom = false;
};
EshiftConfig read_eshift_flags(const tadah::core::Context &ctx) {
  EshiftConfig c;
  c.user = ctx.contains("ESHIFT") && ctx.size("ESHIFT") > 0;
  c.atom = ctx.contains("ESHIFT_ATOM") && ctx.get<bool>("ESHIFT_ATOM");
  c.dbatom = ctx.contains("ESHIFT_DBATOM") && ctx.get<bool>("ESHIFT_DBATOM");
  return c;
}

std::map<int, double> read_explicit_eshift(const tadah::core::Context &ctx,
                                           const std::set<core::Element> &elements) {
  std::vector<double> vals(ctx.size("ESHIFT"));
  ctx.get("ESHIFT", vals);
  if (vals.size() != elements.size()) {
    throw std::runtime_error(
        "ESHIFT size (" + std::to_string(vals.size()) +
        ") does not match number of unique elements (" +
        std::to_string(elements.size()) + ").");
  }
  std::map<int, double> out;
  size_t i = 0;
  for (const auto &e : elements) out[e.Z] = vals[i++];
  return out;
}

}  // anonymous namespace

LoadTransformReport
apply_load_transforms(StructureDB &stdb, const tadah::core::Context &context) {
  LoadTransformReport report;

  // 1. Outlier filter (run first so subsequent steps see a clean dataset).
  std::optional<double> emin, emax, fmax;
  if (context.contains("EFILTER") && context.size("EFILTER") == 2) {
    emin = context.get<double>("EFILTER", 0);
    emax = context.get<double>("EFILTER", 1);
  }
  if (context.contains("FFILTER") && context.size("FFILTER") > 0) {
    fmax = context.get<double>("FFILTER");
  }
  if (emin || emax || fmax) {
    OutlierCounts c = apply_outlier_filter(stdb, emin, emax, fmax);
    report.n_dropped_energy = c.energy;
    report.n_dropped_force = c.force;
    if (c.energy + c.force > 0) {
      core::Logger::instance().log(
          core::LogLevel::INFO,
          "Outlier filter: dropped ", std::to_string(c.energy),
          " by energy, ", std::to_string(c.force), " by force.");
      report.applied = true;
    }
  }

  // 2. Length rescale.
  if (context.contains("LSCALE")) {
    double s = context.get<double>("LSCALE");
    if (s != 1.0) {
      if (s <= 0.0) {
        throw std::runtime_error("LSCALE must be > 0; got " +
                                 std::to_string(s));
      }
      for (auto &st : stdb) StructureTransformation::rescaleLength(st, s);
      report.lscale = s;
      report.applied = true;
      core::Logger::instance().log(
          core::LogLevel::INFO,
          "LSCALE: rescaling positions/cell by ", two_decimal(s),
          ", forces by 1/s; energy and stress invariant.");
    }
  }

  // 3-4. ESHIFT determination + application.
  EshiftConfig esh = read_eshift_flags(context);
  if (esh.user || esh.atom || esh.dbatom) {
    if ((static_cast<int>(esh.user) + static_cast<int>(esh.atom) +
         static_cast<int>(esh.dbatom)) > 1) {
      core::Logger::instance().log(
          core::LogLevel::WARNING,
          "Multiple ESHIFT sources set; precedence: explicit > ATOM > DBATOM.");
      if (esh.user) {
        esh.atom = false;
        esh.dbatom = false;
      } else if (esh.atom) {
        esh.dbatom = false;
      }
    }
    std::set<core::Element> elements = stdb.get_unique_elements();
    if (esh.user) {
      report.eshift = read_explicit_eshift(context, elements);
      report.eshift_source = LoadTransformReport::EshiftSource::User;
    } else if (esh.atom) {
      report.eshift = derive_eshift_from_isolated_atoms(stdb);
      report.eshift_source = LoadTransformReport::EshiftSource::Atom;
    } else if (esh.dbatom) {
      report.eshift = derive_eshift_by_lsq_fit(stdb);
      report.eshift_source = LoadTransformReport::EshiftSource::DBAtom;
    }
    apply_eshift(stdb, report.eshift);
    report.applied = true;

    // Report.
    std::ostringstream summary;
    summary << "ESHIFT applied (";
    switch (report.eshift_source) {
      case LoadTransformReport::EshiftSource::User: summary << "explicit"; break;
      case LoadTransformReport::EshiftSource::Atom: summary << "ESHIFT_ATOM"; break;
      case LoadTransformReport::EshiftSource::DBAtom: summary << "ESHIFT_DBATOM"; break;
      default: summary << "?"; break;
    }
    summary << "):";
    for (const auto &p : report.eshift) {
      summary << " "
              << core::PeriodicTable::find_by_Z(p.first).symbol << "="
              << two_decimal(p.second);
    }
    core::Logger::instance().log(core::LogLevel::INFO, summary.str());
  }

  // 5. Boltzmann eweight (after ESHIFT so it uses the shifted energies).
  if (context.contains("EWEIGHT_TEMP")) {
    double T = context.get<double>("EWEIGHT_TEMP");
    if (T > 0.0) {
      apply_boltzmann_eweight(stdb, T);
      report.eweight_temp = T;
      report.applied = true;
    }
  }

  // 6. COM-force zeroing.
  if (context.contains("ZERO_COM_FORCE") &&
      context.get<bool>("ZERO_COM_FORCE")) {
    apply_com_force_zero(stdb);
    report.zero_com_force = true;
    report.applied = true;
  }

  // 7. Per-dataset weights (composition of WDBFILE * WDBFILE_AUTO).
  std::vector<double> w_user;
  if (context.contains("WDBFILE") && context.size("WDBFILE") > 0) {
    w_user.assign(context.size("WDBFILE"), 1.0);
    context.get("WDBFILE", w_user);
  }
  double alpha = 0.0;
  if (context.contains("WDBFILE_AUTO")) {
    alpha = context.get<double>("WDBFILE_AUTO");
  }
  if (!w_user.empty() || alpha > 0.0) {
    std::vector<size_t> dbidx = stdb.dbidx;
    if (dbidx.empty()) dbidx = {0, stdb.size()};
    const std::size_t n_db = dbidx.size() ? dbidx.size() - 1 : 0;
    std::vector<double> w_auto = (alpha > 0.0)
                                     ? auto_balance_db_weights(stdb, alpha)
                                     : std::vector<double>(n_db, 1.0);
    if (!w_user.empty() && w_user.size() != n_db) {
      throw std::runtime_error(
          "WDBFILE size (" + std::to_string(w_user.size()) +
          ") does not match number of datasets (" + std::to_string(n_db) +
          ").");
    }
    std::vector<double> w_final(n_db, 1.0);
    std::vector<std::size_t> counts(n_db, 0);
    for (std::size_t d = 0; d < n_db; ++d) {
      const double user = w_user.empty() ? 1.0 : w_user[d];
      w_final[d] = user * w_auto[d];
      counts[d] = dbidx[d + 1] - dbidx[d];
    }
    apply_per_db_weights(stdb, w_final);
    report.wdbfile_applied = w_final;
    report.wdbfile_counts = counts;
    report.applied = true;
    std::ostringstream s;
    s << "WDBFILE applied:";
    for (std::size_t d = 0; d < n_db; ++d) {
      s << " ds" << d << "(N=" << counts[d] << ", w=" << two_decimal(w_final[d])
        << ")";
    }
    if (alpha > 0.0) s << "  [WDBFILE_AUTO alpha=" << two_decimal(alpha) << "]";
    core::Logger::instance().log(core::LogLevel::INFO, s.str());
  }

  // 8. Equilibrium-volume diagnostic.
  if (stdb.size() > 0) {
    double e_min = std::numeric_limits<double>::infinity();
    const Structure *eq = nullptr;
    for (const auto &st : stdb) {
      if (st.natoms() == 0) continue;
      const double e_per = st.energy / static_cast<double>(st.natoms());
      if (e_per < e_min) {
        e_min = e_per;
        eq = &st;
      }
    }
    if (eq) {
      const double v = eq->get_volume();
      report.eq_label = eq->label;
      report.eq_volume_per_atom = v / static_cast<double>(eq->natoms());
      report.eq_energy_per_atom = e_min;
      core::Logger::instance().log(
          core::LogLevel::INFO,
          "Reference state (min-energy config): label='", eq->label,
          "' E/N=", two_decimal(e_min), " eV  V/N=",
          two_decimal(report.eq_volume_per_atom), " Å³.");
    }
  }

  return report;
}

void stamp_into_param_file(tadah::core::Context &param_file,
                           const LoadTransformReport &report) {
  if (report.lscale != 1.0) {
    param_file.remove("LSCALE");
    param_file.add("LSCALE", core::to_string(report.lscale));
  }
  if (!report.eshift.empty()) {
    std::vector<int> Z_sorted;
    Z_sorted.reserve(report.eshift.size());
    for (const auto &p : report.eshift) Z_sorted.push_back(p.first);
    std::sort(Z_sorted.begin(), Z_sorted.end());
    param_file.remove("ESHIFT");
    for (int Z : Z_sorted) {
      param_file.add("ESHIFT", core::to_string(report.eshift.at(Z)));
    }
  }
}

}  // namespace mlip
}  // namespace tadah
