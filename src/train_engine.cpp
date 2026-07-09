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

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

namespace tadah {
namespace mlip {
namespace engines {

namespace {

// Detect global-view features that are not yet supported under MPI training.
[[maybe_unused]] void check_mpi_unsupported_features(const core::Context &ctx) {
  std::vector<std::string> bad;
  if (ctx.contains("ESHIFT_ATOM") && ctx.get<bool>("ESHIFT_ATOM"))
    bad.emplace_back("ESHIFT_ATOM");
  if (ctx.contains("ESHIFT_DBATOM") && ctx.get<bool>("ESHIFT_DBATOM"))
    bad.emplace_back("ESHIFT_DBATOM");
  if (ctx.contains("EWEIGHT_TEMP") && ctx.get<double>("EWEIGHT_TEMP") > 0.0)
    bad.emplace_back("EWEIGHT_TEMP");
  if (ctx.contains("WDBFILE_AUTO") && ctx.get<double>("WDBFILE_AUTO") > 0.0)
    bad.emplace_back("WDBFILE_AUTO");
  if (!bad.empty()) {
    std::string msg =
        "The following load-time transforms require a global view of the "
        "dataset and are not yet supported in the MPI training build:";
    for (const auto &k : bad) msg += " " + k;
    msg += ". Run without MPI, set explicit ESHIFT, or omit these keys.";
    throw std::runtime_error(msg);
  }
}

[[maybe_unused]] void check_bias_constraints(const core::Context &ctx) {
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

#ifdef TADAH_BUILD_MPI
  // MPI: workers load chunks incrementally, so global-view transforms are not
  // yet supported. Local-only transforms (LSCALE, explicit ESHIFT, EFILTER,
  // FFILTER, ZERO_COM_FORCE, WDBFILE) would still need TrainerWorker plumbing,
  // which is a separate change. For now, fail fast if any of these are set.
  check_mpi_unsupported_features(context);
  if (context.contains("LSCALE") && context.get<double>("LSCALE") != 1.0) {
    throw std::runtime_error(
        "LSCALE is not yet wired into the MPI training path. Run without MPI.");
  }
  if (context.contains("ESHIFT") && context.size("ESHIFT") > 0) {
    throw std::runtime_error(
        "ESHIFT is not yet wired into the MPI training path. Run without MPI.");
  }
#endif

#ifdef TADAH_BUILD_MPI
  int rank = 0;
  int ncpu = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
  // Execute host or worker code based on rank
  if (rank == 0) { // Host code
    if (ncpu<2) {
      core::Logger::instance().log(
          tadah::core::LogLevel::ERROR,
          "Minimum number of cpus for an MPI version is 2");
      return EXIT_FAILURE;
    }

    // The uncertainty flag is incompatible with MPI
    if (context.get<bool>("UNCERTAINTY")) {
      core::Logger::instance().log(
          tadah::core::LogLevel::ERROR,
          "The --uncertainty flag is not supported by MPI build.");
      return EXIT_FAILURE;
    }

    // Start training timer if verbose mode is enabled
    core::Logger::instance().log(tadah::core::LogLevel::INFO, "Training...");
    // Initialize host MPI trainer
    tadah::mlip::memory::MLIPWorkspaceManager workspaceManager;
    tadah::mlip::memory::MLIPWorkspaceManager workspaceManager2;
    TrainerHost host(context, workspaceManager, workspaceManager2);
    host.run();

    // Perform the final computation and save parameters
    host.solve();

    tadah::core::Context param_file = host.model->get_param_file();
    dumpPotFile(context.get<std::string>("OUTFILE"), param_file);
  } else { // Worker code
           // Initialize worker MPI trainer
    tadah::mlip::memory::MLIPWorkspaceManager workspaceManager;
    tadah::mlip::memory::MLIPWorkspaceManager workspaceManager2;
    TrainerWorker worker(context, workspaceManager, workspaceManager2);
    worker.run();
    worker.solve(); // Perform worker computations
  }
#else // Non-MPI version

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

  // Pre-flight audit. Default mode is `warn`; users opt into `error`
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
#endif
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
