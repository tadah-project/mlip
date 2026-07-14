#include <tadah/mlip/engines/refit_modes.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/logger.h>
#include <tadah/mlip/engines/hpotarget_retarget.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/models/eam/pot_curve_edit.h>

namespace tadah {
namespace mlip {
namespace engines {

namespace {

namespace fs = std::filesystem;
namespace potedit = tadah::models::potedit;
using tadah::core::Context;

bool has(const Context &c, const std::string &k) {
  return c.contains(k) && c.size(k) > 0;
}

std::string get_or(const Context &c, const std::string &k,
                   const std::string &def) {
  return has(c, k) ? c.get<std::string>(k) : def;
}

void log_info(const std::string &m) {
  core::Logger::instance().log(core::LogLevel::INFO, m);
}
void log_warn(const std::string &m) {
  core::Logger::instance().log(core::LogLevel::WARNING, m);
}

/** Output name policy: honour an explicitly-set key; otherwise use the safe
 *  `edited` default so the input is never silently clobbered. A value equal
 *  to the blueprint default string counts as "not set". */
std::string pick_output(const Context &c, const std::string &key,
                        const std::string &blueprint_default,
                        const std::string &edited_default) {
  const std::string v = get_or(c, key, blueprint_default);
  return v == blueprint_default ? edited_default : v;
}

void guard_not_input(const std::string &out, const std::string &in,
                     const char *hint) {
  if (fs::exists(out) && fs::exists(in) && fs::equivalent(out, in))
    throw std::runtime_error(
        "tadah refit: refusing to overwrite the input file " + in + " — " +
        hint);
}

potedit::Channel parse_channel(std::string f) {
  std::transform(f.begin(), f.end(), f.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  if (f == "phi") return potedit::Channel::Phi;
  if (f == "rho") return potedit::Channel::Rho;
  if (f == "f") return potedit::Channel::F;
  throw std::runtime_error(
      "tadah refit --edit: --func must be phi, rho or F (got '" + f + "').");
}

}  // namespace

void validate_refit_mode_keys(const Context &c) {
  const bool edit = has(c, "EDIT");
  const bool retarget = has(c, "RETARGET");
  const bool eam = has(c, "EAM");
  if (edit && retarget)
    throw std::runtime_error(
        "tadah refit: --edit and --retarget are mutually exclusive.");
  if ((edit || retarget) && eam)
    throw std::runtime_error(
        "tadah refit: the EAM key selects a refit run and cannot be combined "
        "with --edit/--retarget.");
  if (edit) {
    if (!has(c, "FUNC"))
      throw std::runtime_error(
          "tadah refit --edit: --func {phi|rho|F} is required.");
    const bool add = has(c, "ADD"), rem = has(c, "REMOVE");
    if (add == rem)
      throw std::runtime_error(
          "tadah refit --edit: exactly one of --add <centre> or "
          "--remove <centre> is required.");
    if (has(c, "ETA") && !add)
      throw std::runtime_error("tadah refit --edit: --eta requires --add.");
    if (has(c, "DEG") && !add)
      throw std::runtime_error("tadah refit --edit: --deg requires --add.");
  }
  for (const char *k : {"FUNC", "ADD", "REMOVE", "ETA", "DEG"})
    if (!edit && has(c, k))
      throw std::runtime_error(std::string("tadah refit: the ") + k +
                               " key requires --edit <pot.tadah>.");
  for (const char *k : {"PCT", "SET", "RETARGET_RCUT2B", "RETARGET_RCUTMB"})
    if (!retarget && has(c, k))
      throw std::runtime_error(std::string("tadah refit: the ") + k +
                               " key requires --retarget <file.hpotarget>.");
  if (!retarget && has(c, "IN_PLACE") && c.get<bool>("IN_PLACE"))
    throw std::runtime_error(
        "tadah refit: --in-place requires --retarget <file.hpotarget>.");
}

// ---------------------------------------------------------------------------
// --edit
// ---------------------------------------------------------------------------

int run_refit_edit(Context &c) {
  validate_refit_mode_keys(c);
  const std::string in_pot = c.get<std::string>("EDIT");

  core::CoreUserConfigParser parser;
  Context pot = parser.parse(in_pot);
  // F2: a hand-tweaked knot pot may carry a stale RCUT — warn + use derived.
  potedit::normalize_knot_rcut(pot, "tadah refit --edit");

  potedit::PotEAMView view = potedit::parse_pot_view(pot);
  const potedit::Channel ch = parse_channel(c.get<std::string>("FUNC"));
  if (ch != potedit::Channel::Phi && view.mb != potedit::MBKind::Ream) {
    if (view.mb == potedit::MBKind::None)
      throw std::runtime_error(
          "tadah refit --edit: this is a pure two-body potential (no active "
          "many-body term), so there is no rho/F curve — only --func phi "
          "can be edited.");
    throw std::runtime_error(
        "tadah refit --edit: the many-body descriptor '" + view.mb_name +
        "' is not DM_REAM, so its rho/F curves cannot be edited — only the "
        "pair channel (--func phi) of this potential is editable.");
  }
  potedit::CurveView &cv = ch == potedit::Channel::Phi ? view.phi
                         : ch == potedit::Channel::Rho ? view.rho
                                                       : view.F;
  const bool radial = ch != potedit::Channel::F;
  const char *fname = ch == potedit::Channel::Phi ? "phi"
                    : ch == potedit::Channel::Rho ? "rho"
                                                  : "F";

  if (has(c, "ADD")) {
    potedit::AddSpec spec;
    spec.centre = c.get<double>("ADD");
    if (has(c, "ETA")) spec.eta = c.get<double>("ETA");
    if (has(c, "DEG")) spec.deg = c.get<int>("DEG");
    // Crossing RCUT is an error only under a hard (Cut_Dummy) cutoff; a
    // smooth cutoff damps the overhang to zero at RCUT (warn-only).
    const bool hard_cutoff =
        !radial || potedit::channel_cutoff_is_discontinuous(
                       pot, ch == potedit::Channel::Phi ? "2B" : "MB");
    const potedit::AddReport rep =
        potedit::add_basis(cv, spec, radial, hard_cutoff);
    for (const auto &w : rep.warnings) log_warn("tadah refit --edit: " + w);
    std::ostringstream msg;
    msg << "tadah refit --edit: added a basis function to " << fname
        << " at " << spec.centre << " (coefficient 0";
    if (cv.family != models::BasisKind::Knot5)
      msg << ", eta " << rep.eta_used;
    else
      msg << ", degree " << rep.deg_used;
    msg << ") — the edited potential predicts identically to the input "
           "until the new coefficient is varied (e.g. by `tadah hpo`).";
    log_info(msg.str());
  } else {
    const potedit::RemoveReport rep =
        potedit::remove_basis(cv, c.get<double>("REMOVE"), radial);
    for (const auto &w : rep.warnings) log_warn("tadah refit --edit: " + w);
    std::ostringstream msg;
    msg << "tadah refit --edit: removed the " << fname
        << " basis function at " << rep.removed_mu << " (coefficient "
        << rep.removed_coeff << ")";
    if (rep.compensated) {
      msg << "; adjusted " << rep.adjusted.size()
          << " neighbouring coefficient(s) — value-RMS deviation over the "
          << "affected region: " << rep.vresid_plain << " (plain) -> "
          << rep.vresid_comp << " (compensated)";
    }
    msg << ". Removal is lossy: re-validate the potential.";
    log_info(msg.str());
  }

  Context out = potedit::rebuild_pot_context(view);

  const std::string out_pot =
      pick_output(c, "OUTFILE", "pot.tadah", "pot_edited.tadah");
  const std::string out_hpo = pick_output(c, "HPOFILEOUT", "refit.hpotarget",
                                          "refit_edited.hpotarget");
  guard_not_input(out_pot, in_pot,
                  "pass -o <name> to choose a different output name");
  TrainEngine::dumpPotFile(out_pot, out);

  const double pct = has(c, "OPTIM_PCT") ? c.get<double>("OPTIM_PCT") : 5.0;
  write_hpotarget(out_hpo, out, pct);
  log_info("tadah refit --edit: wrote pot=" + out_pot +
           " hpotarget=" + out_hpo);
  return EXIT_SUCCESS;
}

// ---------------------------------------------------------------------------
// --retarget
// ---------------------------------------------------------------------------

int run_refit_retarget(Context &c) {
  validate_refit_mode_keys(c);
  const std::string in = c.get<std::string>("RETARGET");
  const bool in_place = has(c, "IN_PLACE") && c.get<bool>("IN_PLACE");

  RetargetOptions opts;
  opts.pct = has(c, "OPTIM_PCT") ? c.get<double>("OPTIM_PCT") : 5.0;

  // --pct tokens: a bare number sets the global percentage, KEY=N one group.
  if (has(c, "PCT")) {
    for (std::size_t i = 0; i < c.size("PCT"); ++i) {
      const std::string tok = c.get<std::string>("PCT", i);
      const auto eq = tok.find('=');
      try {
        if (eq == std::string::npos) {
          opts.pct = std::stod(tok);
          if (opts.pct <= 0.0) throw std::invalid_argument("non-positive");
        } else {
          std::string key = tok.substr(0, eq);
          std::transform(key.begin(), key.end(), key.begin(),
                         [](unsigned char ch) { return std::toupper(ch); });
          const double p = std::stod(tok.substr(eq + 1));
          if (p <= 0.0) throw std::invalid_argument("non-positive");
          opts.pct_per_key[key] = p;
        }
      } catch (const std::invalid_argument &) {
        throw std::runtime_error(
            "tadah refit --retarget: cannot parse --pct token '" + tok +
            "' — expected a positive number or KEY=<positive number> "
            "(e.g. --pct 10 FIXWEIGHT=2).");
      }
    }
  }

  // --rcut2b / --rcutmb: declared fixed cutoffs for the bump channels
  for (const auto &[key, sfx] : {std::pair<const char *, const char *>{
                                     "RETARGET_RCUT2B", "2B"},
                                 {"RETARGET_RCUTMB", "MB"}}) {
    if (!has(c, key)) continue;
    const double r = c.get<double>(key);
    if (r <= 0.0)
      throw std::runtime_error(std::string("tadah refit --retarget: --rcut") +
                               (std::string(sfx) == "2B" ? "2b" : "mb") +
                               " must be positive.");
    opts.rcut_override[sfx] = r;
  }

  // --set 'KEY(i)=lo,hi'
  if (has(c, "SET")) {
    static const std::regex set_re(
        R"(^([A-Za-z0-9_]+)\s*\(\s*(\d+)\s*\)\s*=\s*([^,]+),(.+)$)");
    for (std::size_t i = 0; i < c.size("SET"); ++i) {
      const std::string tok = c.get<std::string>("SET", i);
      std::smatch m;
      if (!std::regex_match(tok, m, set_re))
        throw std::runtime_error(
            "tadah refit --retarget: cannot parse --set token '" + tok +
            "' — expected 'KEY(i)=lo,hi' (e.g. --set 'CGRID2B(3)=2.1,2.4').");
      RetargetOptions::SetOverride s;
      s.key = m[1].str();
      std::transform(s.key.begin(), s.key.end(), s.key.begin(),
                     [](unsigned char ch) { return std::toupper(ch); });
      s.index = std::stoul(m[2].str());
      try {
        s.lo = std::stod(m[3].str());
        s.hi = std::stod(m[4].str());
      } catch (const std::exception &) {
        throw std::runtime_error(
            "tadah refit --retarget: non-numeric bounds in --set token '" +
            tok + "'.");
      }
      opts.sets.push_back(s);
    }
  }

  std::string out;
  if (in_place) {
    if (get_or(c, "OUTFILE", "pot.tadah") != "pot.tadah")
      throw std::runtime_error(
          "tadah refit --retarget: --in-place and -o are mutually "
          "exclusive.");
    out = in;
  } else {
    const fs::path p(in);
    const std::string derived =
        (p.parent_path() / (p.stem().string() + "_edited")).string() +
        p.extension().string();
    out = pick_output(c, "OUTFILE", "pot.tadah", derived);
    guard_not_input(out, in,
                    "pass --in-place to rewrite it, or -o with a different "
                    "name");
  }

  const RetargetReport rep = retarget_hpotarget_file(in, out, in_place, opts);
  for (const auto &w : rep.warnings) log_warn("tadah refit --retarget: " + w);
  std::ostringstream msg;
  msg << "tadah refit --retarget: rewrote " << rep.n_rewritten
      << " OPTIM line(s) (" << rep.n_active << " active, " << rep.n_commented
      << " commented) at +-" << opts.pct << "%";
  for (const auto &[k, p] : opts.pct_per_key) msg << ", " << k << "=" << p << "%";
  msg << " -> " << out;
  log_info(msg.str());
  return EXIT_SUCCESS;
}

} // namespace engines
} // namespace mlip
} // namespace tadah
