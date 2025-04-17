#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/memory/imlip_workspace_manager.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/output/output.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>

namespace tadah {
namespace mlip {
namespace engines {


int TrainEngine::run(tadah::mlip::Task &task) {
  try {
    tadah::core::Context context = task.getContext();
#ifdef TADAH_BUILD_MPI
    int rank = 0;
    int ncpu = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
    // Execute host or worker code based on rank
    if (rank == 0) { // Host code
      // Ensure there are enough CPUs for MPI
      if (ncpu < 2) {
        std::cerr << "Minimum number of CPUs for an MPI version is 2"
                  << std::endl;
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
      tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Training...");
      // Initialize host MPI trainer
      tadah::mlip::memory::MLIPWorkspaceManager workspaceManager;
      TrainerHost host(context, rank, ncpu, workspaceManager);
      host.run();

      // Perform the final computation and save parameters
      host.solve();

      tadah::core::Context param_file = host.model->get_param_file();
      dumpPotFile(context.get<std::string>("OUTFILE"), param_file);
    } else { // Worker code
      // Initialize worker MPI trainer
      tadah::mlip::memory::MLIPWorkspaceManager workspaceManager;
      TrainerWorker worker(context, rank, ncpu, workspaceManager);
      worker.run();
      worker.solve(); // Perform worker computations
    }
#else // Non-MPI version

    tadah::mlip::memory::MLIPWorkspaceManager workspaceManager;
    Trainer tr(context, workspaceManager);

    // Load structures and provide feedback if verbose
    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Loading structures...");
    StructureDB stdb(tr.context);

    // Calculate nearest neighbors and provide feedback
    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO,
                           "Finding nearest neighbours within: " +
                               core::to_string(tr.context.get<double>("RCUTMAX"),5) +
                               " cutoff distance...");
    tr.nnf.calc(stdb);

    // Begin the training process and inform if verbose
    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Training start...");
    tr.model->train(stdb, tr.dc);

    tadah::core::Context param_file = tr.model->get_param_file();
    dumpPotFile(context.get<std::string>("OUTFILE"), param_file);

    // Output uncertainty if requested
    if (context.contains("UNCERTAINTY")) {
      tadah::core::t_type weights = tr.model->get_weights();
      tadah::core::t_type unc = tr.model->get_weights_uncertainty();
      Output(param_file).print_train_unc(weights, unc);
    }
#endif
  } catch (const std::exception &e) {
    std::cerr << "TrainEngine error: " << e.what() << std::endl;
    return EXIT_FAILURE;
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
  if (outstream.is_open()) {
    outstream << context << std::endl;
  } else {
    tadah::core::Logger::instance().log(tadah::core::LogLevel::ERROR,
        "Error: Unable to open file: ", outfile);
  }
}


} // namespace engines
} // namespace mlip
} // namespace tadah
