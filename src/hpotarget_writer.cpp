#include <tadah/mlip/engines/hpotarget_writer.h>
#include <tadah/models/descriptors/dm/embed_spec.h>

#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <vector>

#include <tadah/core/logger.h>
#include <tadah/core/number_format.h>
#include <tadah/models/descriptors/dm/embf0.h>
#include <tadah/models/descriptors/dm/rhoform.h>
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
double quantize_optim_down(double x, double q) { return std::floor(x / q) * q; }
double quantize_optim_up(double x, double q) { return std::ceil(x / q) * q; }

// centre and half-width snapped to the bound grid: the printed bounds
// are then exact decimals, so --retarget round-trips byte-for-byte
OptimBound optim_group_bound(double val, double pct, double floor,
                             bool positive_only,
                             models::OptimLowerBound lower) {
  OptimBound b;
  const double d_raw = std::max(pct / 100.0 * std::fabs(val), floor);
  b.q = optim_bound_quantum(val, d_raw);
  b.v = quantize_optim(val, b.q);
  const double d =
      quantize_optim(std::max(pct / 100.0 * std::fabs(b.v), floor), b.q);
  b.lo = b.v - d;
  b.hi = b.v + d;
  if (positive_only && b.lo <= 0.0)
    b.lo = std::max(1e-6, 0.5 * std::fabs(b.v));
  // RHOFORM NONNEG / MONOTONE amplitudes: lo = max(0, v - d), exact on the
  // bound grid, so a 0 amplitude keeps the searchable box [0, d]
  if (lower == models::OptimLowerBound::NonNegative && b.lo < 0.0) b.lo = 0.0;
  if (b.hi <= b.lo) b.hi = b.lo + std::max(floor, 1e-6);
  return b;
}

std::string rhoform_bounds_line(models::RhoForm form) {
  return std::string("# RHOFORM ") + models::rhoform_name(form) +
         ": AMPGRIDMB bounds are non-negative (lo = max(0, v - d))";
}

std::string knot_support_cap_line(const std::string &path, double radius) {
  return "# " + path + " knot-support cap: RADIUS = " +
         core::format_roundtrip(radius) +
         " (CUTOFF TYPE NONE: every knot ends within RADIUS, so the knot "
         "boxes end there)";
}

namespace {

void emit_group_header(std::ostream &os, const std::string &key,
                       std::size_t n, double pct, bool commented) {
  os << std::defaultfloat;
  os << "# " << key << "  (+-" << pct << "%, " << n << " params"
     << (commented ? ", DISABLED by default — uncomment to search" : "")
     << ")\n";
  os << std::scientific << std::setprecision(8);
}

void emit_group_line(std::ostream &os, const std::string &key, std::size_t i,
                     const OptimBound &b, bool commented) {
  os << (commented ? "# " : "") << "OPTIM " << key << " (" << (i + 1) << ") "
     << b.lo << " " << b.hi << "\n";
}

double group_floor_of(const tadah::core::v_type &vals) {
  double gmax = 0.0;
  for (double v : vals) gmax = std::max(gmax, std::fabs(v));
  return optim_group_floor(gmax);
}

}  // namespace

// One `OPTIM KEY (i) lo hi` line per parameter, each bound = v ± pct%·|v| with
// an absolute floor (relative to the group's scale) so near-zero/negative
// parameters don't collapse to a degenerate window.
void emit_optim_group(std::ostream &os, const std::string &key,
                      const tadah::core::v_type &vals, double pct,
                      bool positive_only, bool commented,
                      models::OptimLowerBound lower) {
  if (vals.empty()) return;
  const double floor = group_floor_of(vals);
  emit_group_header(os, key, vals.size(), pct, commented);
  for (std::size_t i = 0; i < vals.size(); ++i)
    emit_group_line(
        os, key, i,
        optim_group_bound(vals[i], pct, floor, positive_only, lower),
        commented);
  os << "\n";
}

void emit_optim_group_pair(std::ostream &os, const std::string &ckey,
                           const tadah::core::v_type &cvals,
                           const std::string &skey,
                           const tadah::core::v_type &svals, double pct,
                           bool commented, double w, double rcut) {
  if (cvals.empty() || cvals.size() != svals.size() || rcut <= 0.0) {
    // not a well-formed bump channel: fall back to the plain emitters
    emit_optim_group(os, ckey, cvals, pct, false, commented);
    emit_optim_group(os, skey, svals, pct, true, commented);
    return;
  }
  const double cfloor = group_floor_of(cvals);
  const double sfloor = group_floor_of(svals);
  const double tol = tadah::models::bump_rcut_tol(rcut);
  std::vector<OptimBound> cb(cvals.size()), sb(svals.size());
  std::string spilled;
  for (std::size_t i = 0; i < cvals.size(); ++i) {
    cb[i] = optim_group_bound(cvals[i], pct, cfloor, false);
    sb[i] = optim_group_bound(svals[i], pct, sfloor, true);
    /* Bounds are emitted at FULL width: `tadah hpo` (BUMPGATE CANDIDATE)
     * accepts partially infeasible boxes and rejects violating candidates
     * per evaluation, so pre-shrinking the box here would only re-create
     * the near-cutoff spike problem the per-candidate gate removed. A pot
     * whose own bump spills is still worth a warning: hpo's startup SEED
     * check rejects it. */
    if (sb[i].v <= 0.0 || cb[i].v + w / sb[i].v > rcut + tol)
      spilled += (spilled.empty() ? "" : ", ") + std::to_string(i + 1);
  }
  if (!spilled.empty()) {
    core::Logger::instance().log(core::LogLevel::WARNING,
        "tadah refit: " + ckey + " bump(s) " + spilled + " of the potential "
        "itself reach past the hard cutoff — `tadah hpo` rejects a spilling "
        "seed pot at startup until it is repaired (move/narrow the bump "
        "with `tadah refit --edit`, or raise RCUT in the pot).");
  }
  emit_group_header(os, ckey, cvals.size(), pct, commented);
  for (std::size_t i = 0; i < cvals.size(); ++i)
    emit_group_line(os, ckey, i, cb[i], commented);
  os << "\n";
  emit_group_header(os, skey, svals.size(), pct, commented);
  for (std::size_t i = 0; i < svals.size(); ++i)
    emit_group_line(os, skey, i, sb[i], commented);
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
  // The density form of a DM_REAM pot (mandatory, checked before the file
  // is opened): NONNEG and MONOTONE keep every AMPGRIDMB box non-negative,
  // recorded in a parsed header line because --retarget sees only this
  // file, never the pot.
  std::optional<tadah::models::RhoForm> form;
  if (tadah::models::context_has_ream(pot))
    form = tadah::models::require_rhoform(pot, "tadah refit (hpotarget)");
  const auto components=tadah::models::embed_specs(pot);
  const bool ream=tadah::models::context_has_ream(pot);
  bool zero_reference=!components.empty() && std::all_of(components.begin(),components.end(),[](const auto &s){return s.reference==tadah::models::EmbF0Mode::Zero;});
  if(ream)zero_reference=(components.empty() || zero_reference) && tadah::models::stored_embf0(pot).value_or(tadah::models::EmbF0Mode::Zero)==tadah::models::EmbF0Mode::Zero;
  // The zero reference is named by the potential's own keys: a DM_REAM
  // term's EMBF0, a DM_Embed component's EMBEDDING REFERENCE.
  const std::string zero_name=components.empty()?"EMBF0 ZERO"
                             :ream?"EMBF0 ZERO and REFERENCE ZERO":"REFERENCE ZERO";
  std::ofstream os(path);
  if (!os)
    throw std::runtime_error("tadah refit: cannot open the hpotarget file " +
                             path + " for writing.");
  os << "# Starter target file of tadah hpo, written by tadah refit. It runs "
        "with:\n"
     << "#   tadah hpo -c <config> --hpotarget <this file> --validation <val>\n"
     << "# <config>: the refit potential with a DBFILE line appended naming a\n"
     << "# training dataset; <val>: a file with DBFILE line(s) naming the\n"
     << "# validation data the loss is measured on (the Nested Fitting\n"
     << "# chapter of the documentation).\n";
  // EMBF0 ZERO (DM_REAM default; an absent key reads as ZERO): F(0) = 0, so
  // with the refit pot's BIAS false the model has no per-atom constant.
  // <config> is the refit potential (it has WEIGHTS), whose ESHIFT_ATOM and
  // ESHIFT_DBATOM lines a run does not read (potential_line_not_read): the
  // run's own toggle derives the shift; an ESHIFT line of <config> is read.
  if (zero_reference)
    os << "# " << zero_name << ": F(0) = 0 exactly, so with BIAS false the model has no\n"
       << "# per-atom energy constant. For absolute reference energies, such as\n"
       << "# DFT totals, tadah hpo takes --eshift-atom or --eshift-dbatom (an\n"
       << "# ESHIFT_ATOM or ESHIFT_DBATOM line of <config>, a potential, is not\n"
       << "# read), or an ESHIFT line in <config>.\n"
       << "# (Not BIAS true: its intercept columns, one per ATOM species,\n"
       << "# precede the weights that FIXINDEX/FIXWEIGHT and the OPTIM\n"
       << "# FIXWEIGHT lines below index.)\n";
  if (form && tadah::models::rhoform_is_positive(*form))
    os << rhoform_bounds_line(*form) << "\n";
  os << "#\n"
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
     << "# RMSEs on the validation data (<target> <weight>). Under LOSS L2\n"
     << "# each weight is 1/a^2, a the accuracy of the default error scale\n"
     << "# of the quantity: 1 meV/atom (ESIGMA), 0.1 eV/A (FSIGMA) and\n"
     << "# 0.5 GPa (SSIGMA), so each term counts its RMSE in units of a.\n"
     << "PC_ERMSE 0 1\n"
     << "# Force RMSE — needs validation data with reference forces; the\n"
     << "# validation predicts forces for it although the refit potential\n"
     << "# is fitted to energies only.\n"
     << "# PC_FRMSE 0 100\n"
     << "# Stress RMSE in GPa (the nine entries of each tensor) — needs\n"
     << "# validation data with reference stresses.\n"
     << "# PC_SRMSE_GPA 0 4\n"
     << "# LAMMPS-computed observables can be scored with PC_LAMMPS lines\n"
     << "# (see the docs).\n\n";

  auto read_vec = [&pot](const std::string &k) {
    if (!pot.contains(k)) return tadah::core::v_type{};  // knot pots carry
    tadah::core::v_type v(pot.size(k));                  // no width keys
    pot.get(k, v);
    return v;
  };
  // Bump channel under a hard (Cut_Dummy) cutoff: every candidate `tadah
  // hpo` trains keeps c + w/eta <= RCUT (BUMPGATE CANDIDATE rejects the
  // rest per evaluation), so the CGRID/SGRID bounds are emitted at full
  // width; the contract (with the half-width w) is recorded as a parsed
  // comment so --retarget can keep warning about spilling seed pots.
  auto bump_gate = [&](const char *suffix,
                       const std::string &rkey) -> std::optional<double> {
    const auto w = potedit::bump_channel_halfwidth(pot, suffix);
    if (!w || !potedit::channel_cutoff_is_discontinuous(pot, suffix))
      return std::nullopt;
    if (!pot.contains(rkey) || pot.size(rkey) == 0) return std::nullopt;
    return w;
  };
  // For a KNOT channel the effective cutoff is the largest knot: RCUT is a
  // DERIVED quantity, so no OPTIM scalar is offered — the knot positions
  // (CGRID*) are the search knobs. Bump channels keep the commented scalar.
  auto emit_rcut = [&](const char *suffix, const std::string &key,
                       const std::optional<double> &bump_w) {
    const auto derived = potedit::derived_knot_rcut(pot, suffix);
    if (derived) {
      os << "# " << key << " is DERIVED for the knot family (= the largest "
         << "CGRID" << suffix << " knot,\n"
         << "# currently " << std::scientific << std::setprecision(8)
         << *derived << "): do not search it —\n"
         << "# vary the CGRID" << suffix
         << " knot positions to change the effective cutoff.\n\n";
    } else if (pot.contains(key) && pot.size(key) > 0) {
      if (bump_w) {
        os << std::defaultfloat
           << "# " << key << " bump-support contract: w = " << *bump_w
           << " (every candidate trained keeps c + w/eta <= " << key << ";\n"
           << "# combinations outside it are rejected per evaluation, so the "
           << "OPTIM CGRID" << suffix << "/SGRID" << suffix << "\n"
           << "# bounds below are emitted at full width — BUMPGATE STRICT "
           << "restores the startup box check)\n";
      }
      emit_optim_scalar(os, key, pot.get<double>(key), pct, 0.0, true);
    }
  };

  os << "# ---- pair function (two-body) shape ----\n";
  const auto w2b = bump_gate("2B", "RCUT2B");
  emit_rcut("2B", "RCUT2B", w2b);
  // Knot families carry NO width keys (eta is a pure gauge — the pot has
  // no SGRID2B/SGRIDMB/SEMBFUNC entries); emit_optim_group skips empty
  // vectors, so no degenerate OPTIM blocks can appear.
  if (w2b) {
    emit_optim_group_pair(os, "CGRID2B", read_vec("CGRID2B"), "SGRID2B",
                          read_vec("SGRID2B"), pct, true, *w2b,
                          pot.get<double>("RCUT2B"));
  } else {
    emit_optim_group(os, "CGRID2B", read_vec("CGRID2B"), pct, false, true);
    emit_optim_group(os, "SGRID2B", read_vec("SGRID2B"), pct, true, true);
  }

  // ---- DM_Embed components: their named parameters, one section each ----
  // A knot density under CUTOFF TYPE NONE must end within its fixed RADIUS
  // (every knot <= RADIUS): a box constraint of each knot centre alone, so
  // the boxes end at RADIUS (the top on the bound grid, at most RADIUS),
  // recorded in a parsed comment --retarget keeps. Bump channels keep
  // full-width boxes: their support c + w/eta couples two parameters,
  // which tadah hpo's per-candidate support gate checks.
  {
    const auto bindings=tadah::models::embed_bindings(pot);
    for(const auto &spec:components) {
      const std::string id=tadah::core::to_upper(spec.id);
      os<<"# ---- "<<id<<" component (DM_Embed) density and embedding shape ----\n";
      const bool knot_cap=spec.density.basis==tadah::models::EmbedBasis::Knot5 &&
                          tadah::core::to_upper(spec.cutoff)=="NONE";
      const std::string centres=id+".DENSITY.CENTRES";
      if(knot_cap) os<<knot_support_cap_line(centres,spec.radius)<<"\n";
      for(const auto &binding:bindings) {
        if(tadah::core::to_upper(binding.owner)!=id) continue;
        const bool amplitude=binding.path.find(".AMPLITUDES")!=std::string::npos;
        os<<"# COMPONENT_PARAMETER "<<binding.path<<" "<<(binding.positive?(amplitude?"NONNEG":"POSITIVE"):"FREE")<<"\n";
        const bool positive_only=binding.positive && !amplitude;
        const auto lower=binding.positive && amplitude?tadah::models::OptimLowerBound::NonNegative:tadah::models::OptimLowerBound::Free;
        if(!(knot_cap && binding.path==centres)) {
          emit_optim_group(os,binding.path,binding.values,pct,positive_only,true,lower);
          continue;
        }
        if(binding.values.empty()) continue;
        const double floor=group_floor_of(binding.values);
        emit_group_header(os,binding.path,binding.values.size(),pct,true);
        for(std::size_t i=0;i<binding.values.size();++i) {
          auto b=optim_group_bound(binding.values[i],pct,floor,positive_only,lower);
          b.hi=std::min(b.hi,quantize_optim_down(spec.radius,b.q));
          if(!(b.hi>b.lo))
            throw std::runtime_error("tadah refit: knot "+std::to_string(i+1)+" of "+centres+" at "+
              core::format_roundtrip(binding.values[i])+" is not inside the component's RADIUS "+
              core::format_roundtrip(spec.radius)+" (CUTOFF TYPE NONE)");
          emit_group_line(os,binding.path,i,b,true);
        }
        os<<"\n";
      }
    }
  }

  // Pure two-body potentials carry no many-body keys at all — skip the
  // section headers instead of printing them over nothing.
  if (pot.contains("CGRIDMB") && pot.size("CGRIDMB") > 0) {
    os << "# ---- density function (many-body) shape ----\n";
    const auto wmb = bump_gate("MB", "RCUTMB");
    emit_rcut("MB", "RCUTMB", wmb);
    if (wmb) {
      emit_optim_group_pair(os, "CGRIDMB", read_vec("CGRIDMB"), "SGRIDMB",
                            read_vec("SGRIDMB"), pct, true, *wmb,
                            pot.get<double>("RCUTMB"));
    } else {
      emit_optim_group(os, "CGRIDMB", read_vec("CGRIDMB"), pct, false, true);
      emit_optim_group(os, "SGRIDMB", read_vec("SGRIDMB"), pct, true, true);
    }
    emit_optim_group(os, "AMPGRIDMB", read_vec("AMPGRIDMB"), pct, false, true,
                     tadah::models::optim_key_lower_bound("AMPGRIDMB", form));
  }

  if (pot.contains("CEMBFUNC") && pot.size("CEMBFUNC") > 0) {
    os << "# ---- embedding function shape ----\n";
    emit_optim_group(os, "CEMBFUNC", read_vec("CEMBFUNC"), pct, false, true);
    emit_optim_group(os, "SEMBFUNC", read_vec("SEMBFUNC"), pct, true, true);
  }

  // `OPTIM FIXWEIGHT (i)` indices name MODEL WEIGHTS. The refit pot pins
  // every coefficient with FIXINDEX 1-K, so weight index == FIXWEIGHT slot
  // here and the emitted 1..K lines are valid under both readings.
  os << "# ---- model linear coefficients (the default search space) ----\n";
  emit_optim_group(os, "FIXWEIGHT", read_vec("FIXWEIGHT"), pct, false);

  os << "OUTPUT 1\nBOUTPUT 1\n";
  os.close();
  if (!os)
    throw std::runtime_error("tadah refit: cannot write the hpotarget file " +
                             path + ".");
}

} // namespace engines
} // namespace mlip
} // namespace tadah
