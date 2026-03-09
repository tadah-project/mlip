#include <tadah/mlip/engines/analysis_bfunc_engine.h>
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

int AnalysisBfuncEngine::run(mlip::Task &task) {
  #ifdef TADAH_BUILD_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank!=0) return EXIT_SUCCESS;
  #endif

  auto &taskContext = task.getContext();
  if (taskContext.contains("POTENTIAL") || taskContext.contains("CONFIG")) {
    const std::string inputKey =
      taskContext.contains("POTENTIAL") ? "POTENTIAL" : "CONFIG";

    const std::string inFile = taskContext.get<std::string>(inputKey);
    core::Context tempContext = core::CoreUserConfigParser().parse(inFile);
    taskContext = tadah::core::Context::merge(taskContext, tempContext);
  }

  std::string types[2];
  types[0] = core::to_upper(taskContext.get<std::string>("TYPE",0));
  if (taskContext.size("TYPE") > 1) {
    types[1] = core::to_upper(taskContext.get<std::string>("TYPE",1));
  } else {
    types[1] = "N";
  }

  models::Cut_Base *fcut = nullptr;
  double (*fb)(double,double,double) = nullptr;
  std::string cgridtype;
  std::string sgridtype;
  std::string dtype;
  if (types[0]=="2B") {
    cgridtype="CGRID2B";
    sgridtype="SGRID2B";
    dtype = taskContext.get<std::string>("TYPE2B");
    if (types[1]=="Y" || taskContext.get<bool>("RESCALE")) {
      double rcut = taskContext.get<double>("RCUT2BMAX");
      fcut = core::factory<models::Cut_Base,double&>( taskContext.get<std::string>("RCTYPE2B"), rcut);
    }
  } else if (types[0]=="MB") {
    cgridtype="CGRIDMB";
    sgridtype="SGRIDMB";
    dtype = taskContext.get<std::string>("TYPEMB");
    if (types[1]=="Y" || taskContext.get<bool>("RESCALE")) {
      double rcut = taskContext.get<double>("RCUTMBMAX");
      fcut = core::factory<models::Cut_Base,double&>( taskContext.get<std::string>("RCTYPEMB"), rcut);
    }
  } else {
    throw std::runtime_error("Unsupported -t, --type: " + types[0] + " " + types[1]);
  }

  core::v_type cgrid;
  core::v_type sgrid;
  models::D_Base::get_grid(taskContext, cgridtype, cgrid);
  models::D_Base::get_grid(taskContext, sgridtype, sgrid);

  models::DC_Selector DCS(taskContext);
  // only required for DM_mEAD types testing
  models::D_Base *dm = nullptr;
  if (types[0]=="MB") {
    dm = DCS.dmb;
  }

  if (dtype == "D2_Blip" || dtype == "DM_Blip") {
    if (taskContext.get<bool>("DERIVATIVE")) { fb = &models::dB;} else { fb = &models::B; }
    // we have to test DM_mEAD types (for example DM_mRLR) by generic label
    // so we do not have to modify line below each time we add new type
  } else if (dtype == "D2_BP" || dtype == "DM_EAD" || dm->label() == "DM_mEAD") {
    if (taskContext.get<bool>("DERIVATIVE")) { fb = &models::dG;} else { fb = &models::G; }
  } else {
    throw std::runtime_error("Unsupported: " + dtype);
  }

  std::vector<size_t> indices;
  if (taskContext.contains("INDEX")) {
    std::vector<std::string> indicesStrVec(taskContext.size("INDEX"));
    taskContext.get<>("INDEX", indicesStrVec);
    indices = core::parse_indices(indicesStrVec);
    // prune indices if too large values are given by the user
    for (auto it = indices.begin(); it != indices.end(); ) {
      if (*it > cgrid.size()) { // remember indices start from 1
        core::Logger::instance().log(tadah::core::LogLevel::WARNING, "Index " + std::to_string(*it) + " is out of range. This index will be removed from the list.");
        it = indices.erase(it);
      } else {
        ++it;
      }
    }
  }
  else {
    for (size_t j=0; j < cgrid.size(); ++j) {
      indices.push_back(j+1);
    }
  }

  // generate range
  double start = taskContext.get<double>("RANGE",0);
  double stop = taskContext.get<double>("RANGE",1);
  size_t N = taskContext.get<size_t>("RANGE",2);
  core::v_type range = core::linspace(start,stop,N);

  size_t numeric = taskContext.get<size_t>("NUMERIC");
  core::v_type cutoff_values (range.size(), 1.0);
  if (taskContext.get<bool>("RESCALE")) {
    // only for rescale
    if(taskContext.get<bool>("DERIVATIVE")) {
      for (size_t i=0; i<range.size(); ++i) {
        cutoff_values[i] = fcut->calc_prime(range[i]);
      }
    } else {
      for (size_t i=0; i<range.size(); ++i) {
        cutoff_values[i] = fcut->calc(range[i]);
      }
    }
  }

  core::Logger::instance().log(tadah::core::LogLevel::INFO, "Generating output...");
  std::ostringstream oss;
  for (const auto &i : indices) {
    double c = cgrid[i-1];
    double s = sgrid[i-1];
    for (size_t j=0; j<range.size(); ++j) {
      oss << std::fixed << std::setprecision(numeric)
        << std::setw(numeric+5)
        << range[j] << "    "
        << std::setw(numeric+5) << fb(range[j],s,c)*cutoff_values[j] << std::endl;
    }
    oss << std::endl << std::endl;
  }

  if (types[1]=="Y") {
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

  core::Logger::instance().log(tadah::core::LogLevel::INFO, "AnalysisBfuncEngine finished.");
  return EXIT_SUCCESS;
}
void AnalysisBfuncEngine::validate(const tadah::mlip::Task &task) {
  const auto &context = task.getContext();
  // std::cout << "Validating context for AnalysisBfuncEngine..." << std::endl;
  // std::cout << "Context keys: ";
  // std::cout << context << std::endl;
  // CONFIG key triggers nested parsing (POTENTIAL does not) such that CONFIG disappears from the context therfore we do not check either.
  // if (!context.contains("POTENTIAL") && !context.contains("CONFIG")) {
  //   throw std::runtime_error("AnalysisBfuncEngine requires POTENTIAL or CONFIG.");
  // }
  if (context.contains("OUTFILE") && context.size("OUTFILE")!=1) {
    throw std::runtime_error("AnalysisBfuncEngine requires either none or a single OUTFILE.");
  }
  if (context.size("TYPE") > 2) {
    throw std::runtime_error("AnalysisBfuncEngine requires either one or two values for TYPE.");
  }
  if (context.size("RANGE") != 3) {
    throw std::runtime_error("AnalysisBfuncEngine requires three numbers for RANGE.");
  }
}

} // namespace engines
} // namespace mlip
} // namespace tadah

