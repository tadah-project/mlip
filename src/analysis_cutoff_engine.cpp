#include <tadah/mlip/engines/analysis_cutoff_engine.h>
#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/models/dc_selector.h>
#include <iostream>

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

namespace tadah {
namespace mlip {
namespace engines {

int AnalysisCutoffEngine::run(mlip::Task &task) {
  #ifdef TADAH_BUILD_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank!=0) return EXIT_SUCCESS;
  #endif

  auto &taskContext = task.getContext();

  std::string type = taskContext.get<std::string>("TYPE",0);

  // generate range
  double start = taskContext.get<double>("RANGE",0);
  double stop = taskContext.get<double>("RANGE",1);
  size_t N = taskContext.get<size_t>("RANGE",2);
  core::v_type range = core::linspace(start,stop,N);
  models::Cut_Base *fcut = nullptr;
  try {
    fcut = core::factory<models::Cut_Base,double&>( type, range.back() );
  } catch (const std::exception &e) {
    core::Logger::instance().log(tadah::core::LogLevel::ERROR, "Requested cutoff function does not exist: ", std::string(e.what()));
    using RegistryType = tadah::core::Registry<tadah::models::Cut_Base, double&>;
    std::cout << "Available functions stored in the registry:" << std::endl;
    for (const auto& entry : RegistryType::registry) {
      std::cout << "Function name: " << entry.first << std::endl;
    }
    throw e;
  }
  core::Logger::instance().log(tadah::core::LogLevel::INFO, "Generating output...");
  std::ostringstream oss;

  size_t numeric = taskContext.get<size_t>("NUMERIC");
  if(taskContext.get<bool>("DERIVATIVE")) {
    for (size_t i=0; i<range.size(); ++i) {
      oss << std::fixed << std::setprecision(numeric)
        << std::setw(numeric+5)
        << range[i] << "    "
        << std::setw(numeric+5) << fcut->calc_prime(range[i]) << std::endl;
    }
    oss << std::endl << std::endl;
  }
  else {
    for (size_t i=0; i<range.size(); ++i) {
      oss << std::fixed << std::setprecision(numeric)
        << std::setw(numeric+5)
        << range[i] << "    "
        << std::setw(numeric+5) << fcut->calc(range[i]) << std::endl;
    }
    oss << std::endl << std::endl;
  }

  core::Logger::instance().log(tadah::core::LogLevel::INFO, "Dumping output...");
  if (taskContext.contains("OUTFILE")) {
    std::string outFile = taskContext.get<std::string>("OUTFILE",0);
    std::ofstream outfile(outFile);
    if (!outfile.is_open()) {
      throw std::runtime_error("Could not open file: " + outFile);
    }
    outfile << oss.str();
    outfile.close();
  } else {
    std::cout << oss.str();
  }

  core::Logger::instance().log(tadah::core::LogLevel::INFO, "AnalysisCutoffEngine finished.");
  return EXIT_SUCCESS;
}
void AnalysisCutoffEngine::validate(const tadah::mlip::Task &task) {
  const auto &context = task.getContext();
  if (context.contains("OUTFILE") && context.size("OUTFILE")!=1) {
    throw std::runtime_error("AnalysisCutoffEngine requires either none or a single OUTFILE.");
  }
  if (context.size("TYPE") > 1) {
    throw std::runtime_error("AnalysisCutoffEngine requires one value for TYPE.");
  }
  if (context.size("RANGE") != 3) {
    throw std::runtime_error("AnalysisCutoffEngine requires three numbers for RANGE.");
  }
}

} // namespace engines
} // namespace mlip
} // namespace tadah
