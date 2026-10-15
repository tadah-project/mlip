#include <tadah/mlip/engines/analysis_bfunc_engine.h>
#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/core/registry.h>
#include <tadah/models/descriptors/d2/d2_base.h>
#include <tadah/models/descriptors/dm/dm_base.h>
#include <tadah/models/descriptors/basis_dispatch.h>
#include <tadah/models/descriptors/d_mjoin.h>
#include <tadah/models/descriptors/dm/embf0.h>
#include <tadah/models/descriptors/dm/rhoform.h>
#include <iostream>
#include <memory>
#include <optional>
#include <iomanip>

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
    // EMBF0 and RHOFORM belong to the potential, as in tadah predict and
    // analysis descriptor: a task value that differs, or a task RHOFORM for
    // a DM_REAM potential without one, is refused before the merge.
    tadah::models::check_embf0_precedence(tempContext, taskContext, inFile);
    tadah::models::check_rhoform_precedence(tempContext, taskContext, inFile);
    taskContext = tadah::core::Context::merge(taskContext, tempContext);
  }

  std::string types[2];
  types[0] = core::to_upper(taskContext.get<std::string>("TYPE",0));
  if (taskContext.size("TYPE") > 1) {
    types[1] = core::to_upper(taskContext.get<std::string>("TYPE",1));
  } else {
    types[1] = "N";
  }

  std::vector<tadah::core::Context> contexts = models::D_mJoin::parse_context(taskContext, "TYPE"+types[0]);

  std::ostringstream oss;
  for (size_t i=0; i<contexts.size(); ++i) {
    tadah::core::Context &singleContext = contexts[i];
    if(types[0]=="MB") {
      std::unique_ptr<models::DM_Base> descriptor(core::factory<models::DM_Base,const core::Context&>(singleContext.get<std::string>("TYPEMB"),singleContext));
      const std::string channel=types[1]=="EMBEDDING"?"EMBEDDING":"DENSITY";
      const auto count=descriptor->component_basis_size(channel);
      if(count) {
        // The component functions (DM_Embed, DM_REAM) with the meaning of
        // the radial curves below: the bare functions; --rescale tapers
        // them by the channel's spatial cutoff (the product rule under
        // --derivative); Y appends a block with the cutoff (its derivative
        // under --derivative). An embedding has no spatial cutoff.
        const bool rescale=singleContext.get<bool>("RESCALE");
        double probe_fc=0,probe_dfc=0;
        const bool has_cutoff=descriptor->component_cutoff(channel,0,probe_fc,probe_dfc);
        if(rescale && !has_cutoff)
          throw std::runtime_error("tadah analysis bfunc: --rescale tapers radial functions by their spatial cutoff, and the "+
            core::to_lower(channel)+" functions of "+descriptor->label()+" have none (the embedding's argument is a density); "
            "remove --rescale");
        std::vector<std::size_t> indices;
        if(singleContext.has_stored("INDEX")) indices=core::parse_one_based_indices(singleContext("INDEX"),count,"tadah analysis bfunc","component basis");
        else for(std::size_t j=1;j<=count;++j) indices.push_back(j);
        auto range=core::linspace(singleContext.get<double>("RANGE",0),singleContext.get<double>("RANGE",1),singleContext.get<std::size_t>("RANGE",2));
        const bool derivative=singleContext.get<bool>("DERIVATIVE");
        const auto numeric=singleContext.get<std::size_t>("NUMERIC");
        auto write=[&](double q,double v) {
          oss<<std::fixed<<std::setprecision(numeric)<<std::setw(numeric+5)<<q<<"    "
             <<std::setw(numeric+5)<<v<<std::endl;
        };
        for(auto index:indices) {
          for(double q:range) {double v,d;descriptor->component_basis_curve(channel,index-1,q,rescale,v,d);write(q,derivative?d:v);}
          oss<<std::endl;
        }
        if(types[1]=="Y") {
          for(double q:range) {double fc=0,dfc=0;descriptor->component_cutoff(channel,q,fc,dfc);write(q,derivative?dfc:fc);}
          oss<<std::endl;
        }
        oss<<std::endl;
        continue;
      }
    }

    std::unique_ptr<models::Cut_Base> fcut;
    std::string cgridtype;
    std::string sgridtype;
    std::string dtype;
    if (types[0]=="2B") {
      cgridtype="CGRID2B";
      sgridtype="SGRID2B";
      dtype = singleContext.get<std::string>("TYPE2B");
      if (types[1]=="Y" || singleContext.get<bool>("RESCALE")) {
        double rcut = singleContext.get<double>("RCUT2BMAX");
        fcut.reset(core::factory<models::Cut_Base,double&>( singleContext.get<std::string>("RCTYPE2B"), rcut));
      }
    } else if (types[0]=="MB") {
      cgridtype="CGRIDMB";
      sgridtype="SGRIDMB";
      dtype = singleContext.get<std::string>("TYPEMB");
      if (types[1]=="Y" || singleContext.get<bool>("RESCALE")) {
        double rcut = singleContext.get<double>("RCUTMBMAX");
        fcut.reset(core::factory<models::Cut_Base,double&>( singleContext.get<std::string>("RCTYPEMB"), rcut));
      }
    } else {
      throw std::runtime_error("Unsupported -t, --type: " + types[0] + " " + types[1]);
    }

    core::v_type cgrid;
    core::v_type sgrid;
    models::D_Base::get_grid(singleContext, cgridtype, cgrid);
    if(singleContext.contains(sgridtype))models::D_Base::get_grid(singleContext, sgridtype, sgrid);

    // The descriptor reports its own radial basis family; the basis-analysis
    // engine no longer branches on the descriptor NAME string (which forced a
    // code edit here for every new descriptor). A descriptor with no analytic
    // radial basis (tabulated / fixed) returns nullopt and is rejected exactly
    // as the old "else throw" path did. Only the descriptor is built: its
    // radial basis does not depend on the rest of a model's descriptor (the
    // intercept columns of BIAS true need ATOM, which a training
    // configuration may leave to its DBFILE).
    std::optional<models::BasisKind> bk;
    if (types[0] == "2B") {
      std::unique_ptr<models::D2_Base> d(
          core::factory<models::D2_Base, const core::Context&>(dtype, singleContext));
      bk = d->basis_kind();
    } else {
      std::unique_ptr<models::DM_Base> d(
          core::factory<models::DM_Base, const core::Context&>(dtype, singleContext));
      bk = d->basis_kind();
    }
    if (!bk) {
      throw std::runtime_error(
          "Unsupported: " + dtype +
          " (descriptor has no analytic radial basis for bfunc analysis).");
    }
    if(*bk==models::BasisKind::Knot5)sgrid.assign(cgrid.size(),1);
    if(sgrid.size()!=cgrid.size())throw std::runtime_error("bfunc: centre/scale size mismatch");
    core::v_type degrees;
    const std::string degree_key=types[0]=="2B"?"KDEG2B":"KDEGMB";
    if(singleContext.contains(degree_key))models::D_Base::get_grid(singleContext,degree_key,degrees);
    if(degrees.empty())degrees.assign(cgrid.size(),5);
    if(degrees.size()!=cgrid.size())throw std::runtime_error("bfunc: centre/degree size mismatch");
    const bool deriv = singleContext.get<bool>("DERIVATIVE");

    std::vector<size_t> indices;
    if (singleContext.contains("INDEX")) {
      std::vector<std::string> indicesStrVec(singleContext.size("INDEX"));
      singleContext.get<>("INDEX", indicesStrVec);
      // 1-based basis-function indices, parsed strictly and checked against
      // the descriptor's number of functions.
      indices = core::parse_one_based_indices(
          indicesStrVec, cgrid.size(), "tadah analysis bfunc",
          "the " + dtype + " basis has " + std::to_string(cgrid.size()) +
              " functions");
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
    core::Logger::instance().log(tadah::core::LogLevel::INFO, "Generating output...");
    for (const auto &i : indices) {
      double c = cgrid[i-1];
      double s = sgrid[i-1];
      for (size_t j=0; j<range.size(); ++j) {
        oss << std::fixed << std::setprecision(numeric)
          << std::setw(numeric+5)
          << range[j] << "    "
          << std::setw(numeric+5)
          << [&] {
            double value=models::radial_value_bare(*bk,range[j],s,c,static_cast<int>(degrees[i-1]));
            double derivative=models::radial_deriv_bare(*bk,range[j],s,c,static_cast<int>(degrees[i-1]));
            if(singleContext.get<bool>("RESCALE")) {
              derivative=derivative*fcut->calc(range[j])+value*fcut->calc_prime(range[j]);
              value*=fcut->calc(range[j]);
            }
            return deriv?derivative:value;
          }()
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

