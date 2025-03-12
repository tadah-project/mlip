#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tadah/core/config.h>
#include <tadah/core/logger.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/memory/MLIPWorkspaceManager.h>
#include <tadah/mlip/output/output.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>

namespace tadah {
namespace mlip {
namespace engines {

int TrainEngine::train(tadah::core::Config &config) {
  try {
    int rank = 0;
    int ncpu = 1;

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
      if (config.exists(
              "--uncertainty")) { // TODO Config does not have this key
        std::cout << "-----------------------------------------------------"
                  << std::endl;
        std::cout << "The --uncertainty flag is not supported by MPI build."
                  << std::endl;
        std::cout << "-----------------------------------------------------"
                  << std::endl;
        return;
      }

      // Start training timer if verbose mode is enabled
      Logger::instance().log(LogLevel::INFO, "Training...");
      // Initialize host MPI trainer
      TrainerHost host(config, rank, ncpu);
      host.run();

      // Perform the final computation and save parameters
      host.solve();

      tadah::core::Config param_file = host.model->get_param_file();
      param_file.check_pot_file();
      std::ofstream outfile("pot.tadah");
      if (outfile.is_open()) {
        outfile << param_file << std::endl;
      } else {
        std::cerr << "Error: Unable to open file pot.tadah" << std::endl;
      }
    } else { // Worker code
      // Initialize worker MPI trainer
      TrainerWorker worker(config, rank, ncpu);
      worker.run();
      worker.solve(); // Perform worker computations
    }
#else // Non-MPI version
    tadah::mlip::memory::MLIPWorkspaceManager workspaceManager;
    Trainer tr(config, workspaceManager);

    // Load structures and provide feedback if verbose
    Logger::instance().log(LogLevel::INFO, "Loading structures...");
    StructureDB stdb(tr.config);
    Logger::instance().log(LogLevel::INFO, "Done");

    // Calculate nearest neighbors and provide feedback
    Logger::instance().log(LogLevel::INFO,
                           "Finding nearest neighbours within: " +
                               tr.config.get<std::string>("RCUTMAX") +
                               " cutoff distance...");
    tr.nnf.calc(stdb);
    Logger::instance().log(LogLevel::INFO, "Done!");

    // Begin the training process and inform if verbose
    Logger::instance().log(LogLevel::INFO, "Training start...");
    tr.model->train(stdb, tr.dc);
    Logger::instance().log(LogLevel::INFO, "Done!");

    // Save the trained model parameters to a file
    tadah::core::Config param_file = tr.model->get_param_file();
    param_file.check_pot_file();
    std::ofstream outfile("pot.tadah");
    if (outfile.is_open()) {
      outfile << param_file << std::endl;
    } else {
      Logger::instance().log(LogLevel::ERROR,
                             "Error: Unable to open file pot.tadah");
    }

    // Output uncertainty if requested
    if (config.exist("--uncertainty")) {
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

} // namespace engines
} // namespace mlip
} // namespace tadah
