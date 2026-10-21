#include <tadah/mlip/engines/refit_modes.h>
#include <tadah/mlip/engines/refit_keys.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/logger.h>
#include <tadah/core/number_format.h>
#include <tadah/mlip/engines/hpotarget_retarget.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/fit_balance.h>
#include <tadah/mlip/training_record.h>
#include <tadah/models/descriptors/dm/embf0.h>
#include <tadah/models/descriptors/dm/rhoform.h>
#include <tadah/models/eam/pot_curve_edit.h>
#include <tadah/models/eam/embed_edit.h>

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

/** Output name policy: honour an explicitly-set key -- set on the command
 *  line or in a configuration or task file, whatever its value
 *  (refit_key_configured: stored, and not the blueprint default the task
 *  factory added); otherwise use the safe `edited` default so the input is
 *  never silently clobbered. A value equal to the blueprint default string
 *  counted as "not set", so -o pot.tadah wrote pot_edited.tadah. */
std::string pick_output(const Context &c, const KeyProvenance *provenance,
                        const std::string &key,
                        const std::string &edited_default) {
  return refit_key_configured(c, provenance, key) ? c.get<std::string>(key)
                                                  : edited_default;
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

void validate_refit_mode_keys(const Context &c,
                              const KeyProvenance *provenance) {
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
  for (const char *k : {"PCT", "SET", "RETARGET_RCUT2B", "RETARGET_RCUTMB",
                        "RETARGET_RHOFORM"})
    if (!retarget && has(c, k))
      throw std::runtime_error(std::string("tadah refit: the ") + k +
                               " key requires --retarget <file.hpotarget>.");
  // An hpotarget carries no potential, so a density-form or embedding key
  // in the configuration of --retarget would have no effect (stored values
  // only: EMBF0 has a blueprint default).
  if (retarget && c.has_stored("RHOFORM"))
    throw std::runtime_error(
        "tadah refit --retarget: RHOFORM has no effect on --retarget (an "
        "hpotarget carries no potential); remove it from the configuration, "
        "and pass --rhoform <SIGNED|NONNEG|MONOTONE> to rewrite the AMPGRIDMB "
        "boxes for the density form of the potential the file searches "
        "around.");
  if (retarget && c.has_stored("EMBF0"))
    throw std::runtime_error(
        "tadah refit --retarget: EMBF0 has no effect on --retarget (an "
        "hpotarget carries no potential, and its bounds do not depend on the "
        "embedding reference); remove it from the configuration.");
  if (!retarget && has(c, "IN_PLACE") && c.get<bool>("IN_PLACE"))
    throw std::runtime_error(
        "tadah refit: --in-place requires --retarget <file.hpotarget>.");
  // Neither mode reads the keys of a refit of an EAM file.
  if (edit || retarget)
    check_refit_output_mode_keys(
        c, edit ? RefitOutputMode::Edit : RefitOutputMode::Retarget,
        provenance);
}

// ---------------------------------------------------------------------------
// --edit
// ---------------------------------------------------------------------------

int run_refit_edit(Context &c, const KeyProvenance *provenance) {
  validate_refit_mode_keys(c, provenance);
  // OPTIM_PCT, the hpotarget's bounds, before any file is written
  const double pct = read_optim_pct(c);
  const std::string in_pot = c.get<std::string>("EDIT");

  // The input's lines of the keys of its training record but DBFILE --
  // a potential an earlier version wrote has some: the keys that prepared
  // its training data, its run's VERBOSE, NUMERIC, ... -- are
  // not read (potential_line_not_read; no parse stores a run setting of a
  // potential): the edit runs with its own verbosity and NUMERIC, and the
  // lines go into the edit's record below.
  core::CoreUserConfigParser parser;
  std::vector<core::SkippedKeyLine> not_read;
  Context pot = parser.parse(in_pot, potential_line_not_read, not_read);
  // The edit fits nothing: its weights are the input's, edited, so the
  // record of the fit that produced them -- the input's training record
  // and, for a potential an earlier version wrote, the keys of the record
  // among its key lines (its DBFILE, VERBOSE, ...: they are
  // not written as key lines, so they move into the record rather than
  // being dropped) -- is the edit's, after a line naming the input. The
  // edit's own inputs (-o, -v, NUMERIC) are not added, and neither is the
  // balance table of the input's fit (fit_balance.h): it describes the
  // input's coefficients, which the edit changes.
  TrainingRecord record;
  record.lines.push_back("edited by tadah refit --edit from " + in_pot);
  {
    TrainingRecord input;
    try {
      input = without_fit_balance(read_training_record(in_pot));
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("tadah refit --edit: ") +
                               e.what());
    }
    record.lines.insert(record.lines.end(), input.lines.begin(),
                        input.lines.end());
    const TrainingRecord keys = record_of_key_lines(pot, not_read);
    record.lines.insert(record.lines.end(), keys.lines.begin(),
                        keys.lines.end());
  }
  // EMBF0 and RHOFORM are properties of the potential (its weights and
  // amplitudes were fitted in them): a value in the configuration that
  // differs from the potential's is a conflict, as in tadah predict. Only
  // a key the configuration sets is checked here; the potential's own keys
  // are checked by parse_pot_view below.
  if (c.has_stored("EMBF0"))
    models::check_embf0_precedence(pot, c, in_pot);
  if (c.has_stored("RHOFORM"))
    models::check_rhoform_precedence(pot, c, in_pot);
  // F2: a hand-tweaked knot pot may carry a stale RCUT — warn + use derived.
  potedit::normalize_knot_rcut(pot, "tadah refit --edit");

  // NUMERIC shapes only the written file (significant digits; never
  // written into it): the run's own, as for every writer (the input's
  // NUMERIC line -- every 1.5.0 `tadah hpo` pot has an auto-inserted
  // NUMERIC 12 -- was not read). Below 17 the edit is exact only up to
  // rounding.
  const std::optional<std::string> run_numeric = c.stored_value("NUMERIC");
  if (run_numeric) pot.add("NUMERIC", *run_numeric);
  const core::PotDigits pot_digits = core::resolve_pot_digits(pot);
  const bool exact_output = pot_digits.digits >= core::kRoundTripDigits;

  if(!pot.blocks("DESCRIPTOR").empty() && core::to_upper(c.get<std::string>("FUNC"))!="PHI") {
    const auto channel=core::to_upper(c.get<std::string>("FUNC"));
    if(channel!="RHO" && channel!="F")throw std::runtime_error("DM_Embed edit FUNC must be phi, rho or F");
    auto out=models::edit_embed_potential(pot,channel=="RHO",c.get<double>(has(c,"ADD")?"ADD":"REMOVE"),
      has(c,"REMOVE"),has(c,"ETA")?std::optional<double>(c.get<double>("ETA")):std::nullopt,
      has(c,"DEG")?std::optional<int>(c.get<int>("DEG")):std::nullopt);
    const auto out_pot=pick_output(c,provenance,"OUTFILE","pot_edited.tadah");
    const auto out_hpo=pick_output(c,provenance,"HPOFILEOUT","refit_edited.hpotarget");
    guard_not_input(out_pot,in_pot,"pass -o <name> to choose a different output name");
    TrainEngine::dumpPotFile(out_pot,out,record);write_hpotarget(out_hpo,out,pct);
    if(has(c,"REMOVE"))log_warn("DM_Embed basis removal refits the remaining coefficients; revalidate the potential");
    log_info("tadah refit --edit: wrote pot="+out_pot+" hpotarget="+out_hpo);
    return EXIT_SUCCESS;
  }

  potedit::PotEAMView view = potedit::parse_pot_view(pot);
  if (view.mb == potedit::MBKind::Ream) {
    // EMBF0 is a property of the potential: the F curve is edited in the
    // reference DM_REAM evaluates, and the edited potential declares it.
    const bool zero = view.embf0 == models::EmbF0Mode::Zero;
    // A key-less input is (typically) a tadah <= 1.5.0 fit: once the edited
    // potential declares EMBF0 ZERO the load-time notice no longer fires, so
    // say it here at WARNING level.
    const bool declared = models::stored_embf0(pot).has_value();
    (declared ? log_info : log_warn)(std::string("tadah refit --edit: EMBF0 ") +
             models::embf0_name(view.embf0) +
             (declared
                  ? ""
                  : " (key absent -> default ZERO; add 'EMBF0 FREE' to the "
                    "input to keep the raw embedding of a tadah <= 1.5.0 fit)") +
             (zero ? ": F is the zero-referenced embedding, F(0) = 0"
                   : ": F is the raw embedding of tadah 1.5.0 and earlier") +
             "; the edited potential declares EMBF0 " +
             models::embf0_name(view.embf0) + ".");
    // RHOFORM is a property of the potential too (parse_pot_view required
    // it): the rho curve is edited in that form and the edited potential
    // keeps it.
    const models::RhoForm form = view.rho.form;
    log_info(std::string("tadah refit --edit: RHOFORM ") +
             models::rhoform_name(form) +
             (form == models::RhoForm::Signed
                  ? ": rho is sum_k a_k psi_k(r) with signed amplitudes"
              : form == models::RhoForm::NonNeg
                  ? ": rho is sum_k a_k psi_k(r) with every a_k >= 0, which "
                    "an edit keeps"
                  : ": rho is sum_k a_k S_k(r) in integrated-bump steps with "
                    "every a_k >= 0, which an edit keeps") +
             "; the edited potential declares RHOFORM " +
             models::rhoform_name(form) + ".");
  }
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
    msg << ")";
    if (exact_output)
      msg << " — the edited potential predicts identically to the input "
             "until the new coefficient is varied (e.g. by `tadah hpo`).";
    else
      msg << ".";
    log_info(msg.str());
  } else {
    const potedit::RemoveReport rep =
        potedit::remove_basis(cv, c.get<double>("REMOVE"), radial);
    for (const auto &w : rep.warnings) log_warn("tadah refit --edit: " + w);
    for (const auto &n : rep.notes) log_info("tadah refit --edit: " + n);
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

  if (!exact_output) {
    const std::string n = std::to_string(pot_digits.digits);
    log_warn("tadah refit --edit: NUMERIC " + n +
             " (set for this run): the edited potential is written with " +
             n +
             " significant digits, so it predicts identically to the input "
             "only up to that rounding. Omit NUMERIC (default 18, exact) "
             "for an exact edit.");
  }

  Context out = potedit::rebuild_pot_context(view);

  const std::string out_pot =
      pick_output(c, provenance, "OUTFILE", "pot_edited.tadah");
  const std::string out_hpo =
      pick_output(c, provenance, "HPOFILEOUT", "refit_edited.hpotarget");
  guard_not_input(out_pot, in_pot,
                  "pass -o <name> to choose a different output name");
  TrainEngine::dumpPotFile(out_pot, out, record);
  write_hpotarget(out_hpo, out, pct);
  log_info("tadah refit --edit: wrote pot=" + out_pot +
           " hpotarget=" + out_hpo);
  return EXIT_SUCCESS;
}

// ---------------------------------------------------------------------------
// --retarget
// ---------------------------------------------------------------------------

int run_refit_retarget(Context &c, const KeyProvenance *provenance) {
  validate_refit_mode_keys(c, provenance);
  const std::string in = c.get<std::string>("RETARGET");
  const bool in_place = has(c, "IN_PLACE") && c.get<bool>("IN_PLACE");

  RetargetOptions opts;
  opts.pct = read_optim_pct(c);

  // --pct tokens: a bare number sets the global percentage, KEY=N one group.
  // Numbers in the hpotarget grammar (core::parse_decimal_normal, the
  // grammar tadah hpo reads the written bounds with).
  if (has(c, "PCT")) {
    for (std::size_t i = 0; i < c.size("PCT"); ++i) {
      const std::string tok = c.get<std::string>("PCT", i);
      const auto eq = tok.find('=');
      const std::string num =
          eq == std::string::npos ? tok : tok.substr(eq + 1);
      double p = 0.0;
      if (!core::parse_decimal_normal(num, p) || !(p > 0.0))
        throw std::runtime_error(
            "tadah refit --retarget: cannot parse --pct token '" + tok +
            "' — expected a positive number or KEY=<positive number> in the "
            "number grammar of the hpotarget file ([+-]digits[.digits]"
            "[e[+-]digits]; e.g. --pct 10 FIXWEIGHT=2).");
      if (eq == std::string::npos) {
        opts.pct = p;
      } else {
        std::string key = tok.substr(0, eq);
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char ch) { return std::toupper(ch); });
        opts.pct_per_key[key] = p;
      }
    }
  }

  // --rhoform: the density form of the potential the file searches around
  if (const auto f = c.stored_value("RETARGET_RHOFORM"))
    opts.rhoform = models::parse_rhoform(*f);

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
        R"(^([A-Za-z0-9_.]+)\s*\(\s*(\d+)\s*\)\s*=\s*([^,]+),(.+)$)");
    for (std::size_t i = 0; i < c.size("SET"); ++i) {
      const std::string tok = c.get<std::string>("SET", i);
      std::smatch m;
      if (!std::regex_match(tok, m, set_re))
        throw std::runtime_error(
            "tadah refit --retarget: cannot parse --set token '" + tok +
            "' — expected 'KEY(i)=lo,hi', KEY an OPTIM key or component path "
            "of the file (e.g. --set 'CGRID2B(3)=2.1,2.4' or "
            "'EAM.DENSITY.CENTRES(2)=2.4,2.6').");
      RetargetOptions::SetOverride s;
      s.key = m[1].str();
      std::transform(s.key.begin(), s.key.end(), s.key.begin(),
                     [](unsigned char ch) { return std::toupper(ch); });
      s.index = std::stoul(m[2].str());
      for (const auto &[text, value] :
           {std::pair<std::string, double *>{m[3].str(), &s.lo},
            {m[4].str(), &s.hi}})
        if (!core::parse_decimal_normal(text, *value))
          throw std::runtime_error(
              "tadah refit --retarget: --set token '" + tok + "': '" + text +
              "' is not a number in the number grammar of the hpotarget "
              "file ([+-]digits[.digits][e[+-]digits]).");
      opts.sets.push_back(s);
    }
  }

  std::string out;
  if (in_place) {
    if (refit_key_configured(c, provenance, "OUTFILE"))
      throw std::runtime_error(
          "tadah refit --retarget: --in-place and -o are mutually "
          "exclusive.");
    out = in;
  } else {
    // <input>_retargeted<ext>: never a default name of --edit
    // (pot_edited.tadah, refit_edited.hpotarget) or of a refit, so neither
    // overwrites the other's output (with `_edited`, the default for
    // refit.hpotarget was the --edit hpotarget, refit_edited.hpotarget)
    const fs::path p(in);
    const std::string derived =
        (p.parent_path() / (p.stem().string() + "_retargeted")).string() +
        p.extension().string();
    out = pick_output(c, provenance, "OUTFILE", derived);
    guard_not_input(out, in,
                    "pass --in-place to rewrite it, or -o with a different "
                    "name");
  }

  const RetargetReport rep = retarget_hpotarget_file(in, out, in_place, opts);
  for (const auto &w : rep.warnings) log_warn("tadah refit --retarget: " + w);
  if (!rep.rhoform.empty())
    log_info("tadah refit --retarget: " +
             (opts.rhoform ? "--rhoform " + rep.rhoform
                           : in + " declares RHOFORM " + rep.rhoform) +
             ": the AMPGRIDMB bounds are non-negative (lo = max(0, v - d)), "
             "and the rewritten file declares the form.");
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
