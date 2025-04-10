#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/memory/MLIPWorkspaceManager.h>
#include <tadah/mlip/output/output.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>

namespace tadah {
namespace mlip {
namespace engines {

int TrainEngine::run(tadah::mlip::Task &task) {
  try {
    int rank = 0;
    int ncpu = 1;

    tadah::core::Context context = task.getContext();

#ifdef TADAH_BUILD_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
    // Execute host or worker code based on rank
    if (rank == 0) { // Host code
      // Ensure there are enough CPUs for MPI
      if (ncpu < 2) {
        std::cerr << "Minimum number of CPUs for an MPI version is 2"
                  << std::endl;
        return;
      }

      // The uncertainty flag is incompatible with MPI
      if (context.contains(
              "--uncertainty")) { // TODO Context does not have this key
        std::cout << "-----------------------------------------------------"
                  << std::endl;
        std::cout << "The --uncertainty flag is not supported by MPI build."
                  << std::endl;
        std::cout << "-----------------------------------------------------"
                  << std::endl;
        return;
      }

      // Start training timer if verbose mode is enabled
      tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Training...");
      // Initialize host MPI trainer
      TrainerHost host(context, rank, ncpu);
      host.run();

      // Perform the final computation and save parameters
      host.solve();

      tadah::core::Context param_file = host.model->get_param_file();
      std::ofstream outfile("pot.tadah");
      if (outfile.is_open()) {
        outfile << param_file << std::endl;
      } else {
        std::cerr << "Error: Unable to open file pot.tadah" << std::endl;
      }
    } else { // Worker code
      // Initialize worker MPI trainer
      TrainerWorker worker(context, rank, ncpu);
      worker.run();
      worker.solve(); // Perform worker computations
    }
#else // Non-MPI version
    tadah::mlip::memory::MLIPWorkspaceManager workspaceManager;
    Trainer tr(context, workspaceManager);

    // Load structures and provide feedback if verbose
    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Loading structures...");
    StructureDB stdb(tr.context);
    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Done");

    // Calculate nearest neighbors and provide feedback
    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO,
                           "Finding nearest neighbours within: " +
                               tr.context.get<std::string>("RCUTMAX") +
                               " cutoff distance...");
    tr.nnf.calc(stdb);
    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Done!");

    // Begin the training process and inform if verbose
    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Training start...");
    tr.model->train(stdb, tr.dc);
    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Done!");

    // Save the trained model parameters to a file
    tadah::core::Context param_file = tr.model->get_param_file();
    std::ofstream outfile("pot.tadah");
    if (outfile.is_open()) {
      outfile << param_file << std::endl;
    } else {
      tadah::core::Logger::instance().log(tadah::core::LogLevel::ERROR,
                             "Error: Unable to open file pot.tadah");
    }

    // Output uncertainty if requested
    if (context.contains("--uncertainty")) {
      tadah::core::t_type weights = tr.model->get_weights();
      tadah::core::t_type unc = tr.model->get_weights_uncertainty();
      Output(param_file, false).print_train_unc(weights, unc);
    }
#endif
  } catch (const std::exception &e) {
    std::cerr << "TrainEngine error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
void TrainEngine::validate(const tadah::mlip::Task &task) {
  core::Logger::instance().log(core::LogLevel::INFO,
      "TrainEngine validate() called. No mandatory keys in stub.");
}


} // namespace engines
} // namespace mlip
} // namespace tadah
