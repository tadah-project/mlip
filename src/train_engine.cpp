#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/models/eam/pot_curve_edit.h>
#include <tadah/mlip/memory/imlip_workspace_manager.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/output/output.h>
#include <tadah/mlip/preflight_audit.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>

namespace tadah {
namespace mlip {
namespace engines {

namespace {

void check_bias_constraints(const core::Context &ctx) {
  const bool norm = ctx.contains("NORM") && ctx.get<bool>("NORM");
  const bool bias = ctx.contains("BIAS") && ctx.get<bool>("BIAS");
  const std::string model_bf =
      ctx.contains("MODEL") && ctx.size("MODEL") >= 2
          ? ctx.get<std::string>("MODEL", 1)
          : std::string{};

  if (norm && model_bf == "BF_Linear" && !bias) {
    throw std::runtime_error(
        "Configuration: NORM=true with MODEL=BF_Linear requires BIAS=true. "
        "Without intercept columns the train→nweights conversion would silently "
        "produce wrong LAMMPS predictions. Set BIAS=true or NORM=false.");
  }
  if (bias && (!ctx.contains("ATOM") || ctx.size("ATOM") == 0)) {
    throw std::runtime_error(
        "Configuration: BIAS=true requires a non-empty ATOM list "
        "(BIAS adds one intercept column per species).");
  }
}

}  // namespace


int TrainEngine::run(tadah::mlip::Task &task) {
  core::Context context = task.getContext();
  models::D_mJoin::expand_grids(context);
  // Knot-family refit configs: RCUT2B/RCUTMB are derived from the largest
  // knot — a hand-tweaked/stale stated value is warned about and corrected
  // here, BEFORE the neighbour list is sized from RCUTMAX.
  models::potedit::normalize_knot_rcut(context, "tadah train");
  std::set<core::Element> unique_elements = StructureDB::find_unique_elements(context);
  StructureDB::check_atom_key(context, unique_elements);
  StructureDB::check_watom_key(context, unique_elements);
  check_bias_constraints(context);

  tadah::mlip::memory::MLIPWorkspaceManager workspaceManager;
  Trainer tr(context, workspaceManager);

  // Load structures and provide feedback if verbose
  tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Loading structures...");
  StructureDB stdb(tr.context);

  // Apply load-time transforms (LSCALE, ESHIFT, EFILTER/FFILTER, WDBFILE etc.)
  // before any neighbour-finding or descriptor calculation, so downstream code
  // sees the already-transformed positions, energies, and weights.
  LoadTransformReport report = apply_load_transforms(stdb, tr.context);

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

  tadah::core::Context param_file = tr.model->get_param_file();
  stamp_into_param_file(param_file, report);
  dumpPotFile(context.get<std::string>("OUTFILE"), param_file);

  // Output uncertainty if requested
  if (context.get<bool>("UNCERTAINTY")) {
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
}

// helper function to dump the potential file
void TrainEngine::dumpPotFile(const std::string &outfile, const core::Context &context) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "Saving parameters to file: " + outfile);
  std::ofstream outstream(outfile);
  core::Context modContext = context;
  modContext.remove("VERBOSE");
  modContext.remove("NUMERIC");
  modContext.remove("OUTFILE");
  // Audit / health-monitor knobs are runtime-only and not part of the
  // trained model. The LAMMPS-side PairTadah parser does not register
  // them, so leaving them in pot.tadah triggers an "Invalid key"
  // error during pair_coeff and a downstream null-deref in calc_rho.
  modContext.remove("AUDIT");
  modContext.remove("HEALTH_LOG");
  if (outstream.is_open()) {
    outstream << modContext << std::endl;
  } else {
    tadah::core::Logger::instance().log(tadah::core::LogLevel::ERROR,
        "Error: Unable to open file: ", outfile);
  }
}

} // namespace engines
} // namespace mlip
} // namespace tadah
