#include <tadah/mlip/engines/analysis_bfunc_engine.h>
#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/descriptors/basis_dispatch.h>
#include <iostream>
#include <optional>

namespace tadah {
namespace mlip {
namespace engines {

int AnalysisBfuncEngine::run(mlip::Task &task) {
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

  std::vector<tadah::core::Context> contexts = D_mJoin::parse_context(taskContext, "TYPE"+types[0]);

  std::ostringstream oss;
  for (size_t i=0; i<contexts.size(); ++i) {
    tadah::core::Context &singleContext = contexts[i];

    models::Cut_Base *fcut = nullptr;
    std::string cgridtype;
    std::string sgridtype;
    std::string dtype;
    if (types[0]=="2B") {
      cgridtype="CGRID2B";
      sgridtype="SGRID2B";
      dtype = singleContext.get<std::string>("TYPE2B");
      if (types[1]=="Y" || singleContext.get<bool>("RESCALE")) {
        double rcut = singleContext.get<double>("RCUT2BMAX");
        fcut = core::factory<models::Cut_Base,double&>( singleContext.get<std::string>("RCTYPE2B"), rcut);
      }
    } else if (types[0]=="MB") {
      cgridtype="CGRIDMB";
      sgridtype="SGRIDMB";
      dtype = singleContext.get<std::string>("TYPEMB");
      if (types[1]=="Y" || singleContext.get<bool>("RESCALE")) {
        double rcut = singleContext.get<double>("RCUTMBMAX");
        fcut = core::factory<models::Cut_Base,double&>( singleContext.get<std::string>("RCTYPEMB"), rcut);
      }
    } else {
      throw std::runtime_error("Unsupported -t, --type: " + types[0] + " " + types[1]);
    }

    core::v_type cgrid;
    core::v_type sgrid;
    models::D_Base::get_grid(singleContext, cgridtype, cgrid);
    models::D_Base::get_grid(singleContext, sgridtype, sgrid);

    models::DC_Selector DCS(singleContext);

    // The descriptor reports its own radial basis family; the basis-analysis
    // engine no longer branches on the descriptor NAME string (which forced a
    // code edit here for every new descriptor). A descriptor with no analytic
    // radial basis (tabulated / fixed) returns nullopt and is rejected exactly
    // as the old "else throw" path did.
    const std::optional<models::BasisKind> bk =
        (types[0] == "2B") ? DCS.d2b->basis_kind() : DCS.dmb->basis_kind();
    if (!bk) {
      throw std::runtime_error(
          "Unsupported: " + dtype +
          " (descriptor has no analytic radial basis for bfunc analysis).");
    }
    const bool deriv = singleContext.get<bool>("DERIVATIVE");

    std::vector<size_t> indices;
    if (singleContext.contains("INDEX")) {
      std::vector<std::string> indicesStrVec(singleContext.size("INDEX"));
      singleContext.get<>("INDEX", indicesStrVec);
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
    double start = singleContext.get<double>("RANGE",0);
    double stop = singleContext.get<double>("RANGE",1);
    size_t N = singleContext.get<size_t>("RANGE",2);
    core::v_type range = core::linspace(start,stop,N);

    size_t numeric = singleContext.get<size_t>("NUMERIC");
    core::v_type cutoff_values (range.size(), 1.0);
    if (singleContext.get<bool>("RESCALE")) {
      // only for rescale
      if(singleContext.get<bool>("DERIVATIVE")) {
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
    for (const auto &i : indices) {
      double c = cgrid[i-1];
      double s = sgrid[i-1];
      for (size_t j=0; j<range.size(); ++j) {
        oss << std::fixed << std::setprecision(numeric)
          << std::setw(numeric+5)
          << range[j] << "    "
          << std::setw(numeric+5)
          << (deriv ? models::radial_deriv_bare(*bk, range[j], s, c)
                    : models::radial_value_bare(*bk, range[j], s, c))
             * cutoff_values[j]
          << std::endl;
      }
      oss << std::endl;
    }


    if (types[1]=="Y") {
      if(singleContext.get<bool>("DERIVATIVE")) {
        for (size_t i=0; i<range.size(); ++i) {
          oss << std::fixed << std::setprecision(numeric)
            << std::setw(numeric+5)
            << range[i] << "    "
            << std::setw(numeric+5) << fcut->calc_prime(range[i]) << std::endl;
        }
        oss << std::endl;
      }
      else {
        for (size_t i=0; i<range.size(); ++i) {
          oss << std::fixed << std::setprecision(numeric)
            << std::setw(numeric+5)
            << range[i] << "    "
            << std::setw(numeric+5) << fcut->calc(range[i]) << std::endl;
        }
        oss << std::endl;
      }
    }
    oss << std::endl;
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
  // CONFIG key triggers nested parsing (POTENTIAL does not) such that CONFIG disappears from the context therefore we do not check either.
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

