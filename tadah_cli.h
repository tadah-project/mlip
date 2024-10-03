#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

#ifndef TADAH_CLI_H
#define TADAH_CLI_H

#include "../LIBS/CLI11/CLI11.hpp"
#include <string>
#include <vector>

namespace fs = std::filesystem;
class TadahCLI {
  private:
    bool verbose=false;
    std::string logo=
"                                                                       \n\
           /$$$$$$$$                      /$$           /$$       /$$   \n\
          |__  $$__/                     | $$          | $$      | $$   \n\
             | $$  /$$$$$$           /$$$$$$$  /$$$$$$ | $$$$$$$ | $$   \n\
             | $$ |____  $$ /$$$$$$ /$$__  $$ |____  $$| $$__  $$| $$   \n\
             | $$  /$$$$$$$|______/| $$  | $$  /$$$$$$$| $$  \\ $$|__/  \n\
             | $$ /$$__  $$        | $$  | $$ /$$__  $$| $$  | $$       \n\
             | $$|  $$$$$$$        |  $$$$$$$|  $$$$$$$| $$  | $$ /$$   \n\
             |__/ \\_______/         \\_______/ \\_______/|__/  |__/|__/\n\
                                                                        \n\
        Library for Machine Learning Interatomic Potentials Development.\n\
                     https://ta-dah.readthedocs.io/en/latest            \n\
                                                                        \n\
                    Type `ta-dah -h SUBCOMMAND` for more info.          \n\
                                                                        \n\
";
  public:
    TadahCLI();
    void run(int argc, char** argv);
    CLI::App app;

    // subcommands:
    CLI::App *train;
    CLI::App *predict;
    CLI::App *hpo;

    // PREDICTION
    std::string pot_file;
    std::string config_file;
    std::string config_tpred;
    std::string target_file;
    std::string targets_dir;
    std::string targets_out_dir;
    std::vector<std::string> datasets;
    void subcommand_train();
    void subcommand_predict();
    void subcommand_hpo(int argc, char**argv);
    void set_verbose();
    bool is_verbose();
    void flag_version();
    std::vector<fs::path> read_targets(std::string path);

    /* Return a vector of absolute paths to every file in a directory.
     * Subdirectories are ommited.
     */
    std::vector<fs::path> read_files(std::string path);

#ifdef TADAH_BUILD_MPI
    const static int WAIT_TAG = 3;
    const static int RELEASE_TAG = 2;
    const static int DATA_TAG = 1;
    const static int WORK_TAG = 0;
#endif
};
#endif

