#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/intercepts.h>
#include <tadah/mlip/key_provenance.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/models/eam/pot_curve_edit.h>
#include <tadah/models/descriptors/dm/embf0.h>
#include <tadah/models/descriptors/dm/rhoform.h>
#include <tadah/mlip/memory/imlip_workspace_manager.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/output/output.h>
#include <tadah/mlip/preflight_audit.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/training_datasets.h>
#include <tadah/mlip/weight_covariance.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/models/m_krr_core.h>

namespace tadah {
namespace mlip {
namespace engines {

namespace {
// EMBF0 ZERO pins the DM_REAM isolated-atom energy at 0. Without BIAS or an
// ESHIFT the model then has no free per-atom constant, which matters when the
// reference energies are absolute (e.g. DFT totals).
void hint_embf0_reference(const Trainer &tr,
                          const LoadTransformReport &report) {
  if (!tr.DCS.dmb) return;
  const auto mode = tr.DCS.dmb->embf0_mode();
  if (!mode || *mode != tadah::models::EmbF0Mode::Zero) return;
  if (tr.DCS.bias > 0 || !report.eshift.empty()) return;
  tadah::core::Logger::instance().log(
      tadah::core::LogLevel::WARNING,
      "HINT: EMBF0 ZERO fixes the DM_REAM isolated-atom energy at 0 and the "
      "model has no other per-atom constant (BIAS false, no ESHIFT). If the "
      "reference energies are absolute (e.g. DFT totals), set ESHIFT_ATOM, "
      "ESHIFT_DBATOM or ESHIFT (compatible with EAMFAST), or BIAS true "
      "(disables EAMFAST; its intercept columns, one per ATOM species, "
      "precede the descriptor weights, so FIXINDEX/FIXWEIGHT indices "
      "shift).");
}
// A WNOISE without E (a fit without energies of a non-zero weight): every
// --error prediction has an energy, so predict --error refuses the
// potential; said when it is written.
void warn_noise_without_energies(const core::Context &param_file) {
  if (!param_file.has_stored("WCOV") || !param_file.has_stored("WNOISE"))
    return;
  const WeightCovariance wc =
      read_weight_covariance(param_file, param_file.size("WEIGHTS"));
  if (wc.noise.set[kEnergyRow]) return;
  core::Logger::instance().log(
      core::LogLevel::WARNING,
      "tadah train: the fit has no energies with a non-zero weight (EWEIGHT "
      "0, or every structure's eweight 0), so the potential's WNOISE has no "
      "E: `tadah predict --error` and `tadah properties pairwise --error` "
      "refuse it (every prediction has an energy). Its WCOV, and the "
      "standard deviations of --uncertainty, are those of the fit.");
}
// Whether the task asks for weights.tadah (--uncertainty, UNCERTAINTY
// true). A potential given as the configuration (-c, a file with WEIGHTS:
// the retrain route) recorded the UNCERTAINTY of the run that trained it --
// tadah 1.5.x and 1.6.0 development builds wrote the key into every
// potential, get_param_file no longer does -- and that is not a request: a
// retrain with another OALGO was refused for it ("needs OALGO 3") although
// no --uncertainty was given. --uncertainty on the command line, and
// UNCERTAINTY true in a training configuration or a task file, are
// requests. With @p warn, the ignored record is said (a WARNING).
bool uncertainty_requested(const tadah::mlip::Task &task, bool warn) {
  const core::Context &c = task.getContext();
  if (!c.get<bool>("UNCERTAINTY")) return false;
  const KeyProvenance *prov = task.getKeyProvenance();
  if (!prov || !c.has_stored("WEIGHTS")) return true;
  const KeyOrigin o = prov->origin("UNCERTAINTY");
  if (o.kind != KeyOrigin::Kind::ConfigFile) return true;
  if (warn)
    core::Logger::instance().log(
        core::LogLevel::WARNING,
        "tadah train: the configuration file " + o.file + " is a potential "
        "(it has WEIGHTS), and its UNCERTAINTY true is the record of the run "
        "that trained it (tadah 1.5.x and 1.6.0 development builds wrote "
        "UNCERTAINTY into every potential), not a request: it is ignored and "
        "no weights.tadah is written. Give --uncertainty (OALGO 3) to write "
        "the weights with their standard deviations.");
  return false;
}

// " (UNCERTAINTY true: configuration file c.tadah)" when the request did
// not come from the command line, else "".
std::string uncertainty_source(const tadah::mlip::Task &task) {
  const KeyProvenance *prov = task.getKeyProvenance();
  if (!prov) return "";
  const KeyOrigin o = prov->origin("UNCERTAINTY");
  if (o.kind == KeyOrigin::Kind::CommandLine ||
      o.kind == KeyOrigin::Kind::NotRecorded)
    return "";
  return " (UNCERTAINTY true: " + o.describe("UNCERTAINTY") + ")";
}
}  // namespace

int TrainEngine::run(tadah::mlip::Task &task) {
  core::Context context = task.getContext();
  // The datasets to fit: a configuration without DBFILE (a potential, which
  // records none, given as it is) is refused, saying how to retrain, before
  // anything is read; one dataset listed twice would be fitted twice.
  require_training_datasets(context);
  require_distinct_training_datasets(context);
  // The SBASIS of a fit that draws basis vectors (M_KRR with a non-linear
  // kernel, implementation 2 with any kernel), refused naming the model
  // before any dataset is read; with BIAS true and no ATOM its bound counts
  // the species, which the datasets give, so it is checked once ATOM is set
  // from them.
  const bool draws_basis = fit_draws_basis(context);
  const bool sbasis_needs_atoms =
      draws_basis && context.get<bool>("BIAS") &&
      (!context.contains("ATOM") || context.size("ATOM") == 0);
  if (draws_basis && !sbasis_needs_atoms) checked_sbasis(context);
  models::D_mJoin::expand_grids(context);
  // Knot-family refit configs: RCUT2B/RCUTMB are derived from the largest
  // knot — a hand-tweaked/stale stated value is warned about and corrected
  // here, BEFORE the neighbour list is sized from RCUTMAX.
  models::potedit::normalize_knot_rcut(context, "tadah train");
  std::set<core::Element> unique_elements = StructureDB::find_unique_elements(context);
  StructureDB::check_atom_key(context, unique_elements);
  StructureDB::check_watom_key(context, unique_elements);
  check_bias_constraints(context);
  if (sbasis_needs_atoms) checked_sbasis(context);

  tadah::mlip::memory::MLIPWorkspaceManager workspaceManager;
  Trainer tr(context, workspaceManager);
  {
    // which weights are intercepts is the model function's (and needs
    // DSIZE); a non-linear kernel's, those of the basis its fit draws
    const std::string w = fixindex_intercept_warning(tr.context, *tr.fb);
    if (!w.empty())
      tadah::core::Logger::instance().log(tadah::core::LogLevel::WARNING, w);
  }
  {
    // FIXINDEX pinning every weight leaves the fit nothing to change (a
    // refit potential retrained as it is): said before any dataset is read.
    const std::string w = all_weights_pinned_warning(tr.context, *tr.fb);
    if (!w.empty())
      tadah::core::Logger::instance().log(tadah::core::LogLevel::WARNING, w);
  }

  // Load structures and provide feedback if verbose
  tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Loading structures...");
  StructureDB stdb(tr.context);

  // Apply load-time transforms (LSCALE, ESHIFT, EFILTER/FFILTER, WDBFILE etc.)
  // before any neighbour-finding or descriptor calculation, so downstream code
  // sees the already-transformed positions, energies, and weights.
  LoadTransformReport report = apply_load_transforms(stdb, tr.context);
  hint_embf0_reference(tr, report);
  // BIAS true: the energy-weighted compositions must determine the
  // intercepts the fit has (a non-linear kernel's: those of the intercept
  // vectors its basis draw starts with), refused before the neighbour lists
  // are built.
  require_intercepts_determined(stdb, tr.context, *tr.fb);

  // Calculate nearest neighbors and provide feedback
  tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "Finding nearest neighbours within: " +
      core::to_string(tr.context.get<double>("RCUTMAX"),5) +
      " cutoff distance...");
  tr.nnf.calc(stdb, tr.context);

  // Pre-flight audit. Default mode is `off` (AUDIT unset); users opt into `warn`, or `error`
  // for hard refusal-to-launch via the AUDIT context key. Reads
  // dataset stats (cheap — same NN list) and walks every configured
  // descriptor's meta() to flag ρ-collapse and out-of-range parameter
  // combinations (Phase 5 of the bound-validation pipeline).
  {
    AuditMode mode = audit_mode_from_context(tr.context);
    if (mode != AuditMode::Off) {
      auto report = run_preflight_from_db(stdb, tr.context, mode);
      report.log();
      try {
        enforce_audit_mode(report, mode);
      } catch (const std::exception& e) {
        // AUDIT=error and there are FAIL findings. Surface the message
        // and exit immediately. Scheduler::execute ignores engine
        // return codes, so a clean refusal-to-launch must short-circuit
        // the process here. EXIT_FAILURE is documented behaviour for
        // this mode (`AUDIT error` in the user's seed config).
        tadah::core::Logger::instance().log(
            tadah::core::LogLevel::ERROR, e.what());
        std::exit(EXIT_FAILURE);
      }
    }
  }

  // Begin the training process and inform if verbose
  tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Training start...");
  tr.model->train(stdb, tr.dc);

  // --uncertainty lists the weights with their standard deviations: a fit
  // that computed no covariance (its noise variance undefined) is refused
  // before the potential is written (owner decision D8), with the reason
  // the model gives and what to change.
  const bool uncertainty = uncertainty_requested(task, /*warn=*/true);
  if (uncertainty && !tr.model->has_covariance())
    throw std::runtime_error(
        "tadah train --uncertainty" + uncertainty_source(task) +
        ": no weight covariance: " +
        tr.model->covariance_unavailable() +
        ". No potential was written (train without --uncertainty to write "
        "it, without the weights' standard deviations).");

  tadah::core::Context param_file = tr.model->get_param_file();
  stamp_into_param_file(param_file, report);
  warn_noise_without_energies(param_file);
  dumpPotFile(context.get<std::string>("OUTFILE"), param_file);

  // The weights and their standard deviations (weights.tadah)
  if (uncertainty) {
    tadah::core::t_type weights = tr.model->get_weights();
    tadah::core::t_type unc = tr.model->get_weights_uncertainty();
    Output(param_file).print_train_unc(weights, unc);
  }
  return EXIT_SUCCESS;
}
void TrainEngine::validate(const tadah::mlip::Task &task) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Validating TrainEngine...");
  const auto &context = task.getContext();
  if (!context.get<bool>("INIT2B") && !context.get<bool>("INITMB")) {
    throw std::runtime_error(
        "TrainEngine requires INIT2B or/and INITMB parameters to be set");
  }
  // Only the SVD fit computes the weight covariance, and M_KRR
  // implementation 2 none: refused before any training or output.
  const bool uncertainty = uncertainty_requested(task, /*warn=*/false);
  if (uncertainty && context.get<int>("OALGO") != 3)
    throw std::runtime_error(
        "tadah train --uncertainty" + uncertainty_source(task) +
        " needs OALGO 3: only the SVD fit (OALGO 3) "
        "computes the weight covariance; the configuration has OALGO " +
        std::to_string(context.get<int>("OALGO")) + ".");
  if (uncertainty && is_krr_implementation_2(context))
    throw std::runtime_error(
        "tadah train --uncertainty" + uncertainty_source(task) + ": " +
        tadah::models::krr_implementation2_without_covariance() +
        ", or train without --uncertainty.");
}


// helper function to dump the potential file
void TrainEngine::dumpPotFile(const std::string &outfile, const core::Context &context) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "Saving parameters to file: " + outfile);
  // NUMERIC = significant digits of the numbers written below; absent ->
  // 18 (exact round trip). Looked up without default insertion, and never
  // written into the potential itself (write_potential skips it).
  const core::PotDigits pd = core::resolve_pot_digits(context);
  const std::string digits_warning = core::pot_digits_warning(pd);
  if (!digits_warning.empty())
    core::Logger::instance().log(core::LogLevel::WARNING,
        outfile + ": " + digits_warning);
  core::Context modContext = context;
  // A potential records the model's settings and its fit: neither the
  // inputs of the run (VERBOSE, NUMERIC, OUTFILE, the AUDIT and HEALTH_LOG
  // diagnostics, a tadah hpo run's HPOTARGET and VALIDATION, which a
  // best_pot.tadah used as the configuration had, the train task's
  // UNCERTAINTY, which a 1.5.x potential edited by refit --edit has) nor its
  // training data
  // (a refit --edit potential is a copy of its input's context, which may
  // be an M_BLR potential written with its DBFILE).
  remove_run_inputs(modContext);
  remove_training_datasets(modContext);
  tadah::models::normalise_embf0_key(modContext);
  // A DM_REAM potential must declare its density form (absent -> throws
  // before the file is opened, so nothing is written); the key is
  // canonicalised, or stripped from a potential without a DM_REAM block.
  tadah::models::normalise_rhoform_key(modContext);
  // A potential that cannot be written is an error: logging it and
  // returning let `tadah train -o <unwritable>` exit 0 without a potential.
  std::ofstream outstream(outfile);
  if (!outstream.is_open())
    throw std::runtime_error("Cannot open the potential file " + outfile +
                             " for writing.");
  modContext.write_potential(outstream, pd.digits);
  outstream << std::endl;
  outstream.close();
  if (!outstream)
    throw std::runtime_error("Cannot write the potential file " + outfile +
                             ".");
}

} // namespace engines
} // namespace mlip
} // namespace tadah
