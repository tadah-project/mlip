#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/output/output.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>
#include <tadah/models/dc_selector.h>
#include <iostream>

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

namespace tadah {
namespace mlip {
namespace engines {

int PredictEngine::run(mlip::Task &task) {
  int rank = 0;
  #ifdef TADAH_BUILD_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  #endif
  if (rank!=0) return EXIT_SUCCESS;

  const auto &taskContext = task.getContext();
  std::string pot_file= taskContext.get<std::string>("POTENTIAL");
  core::Context potContext = core::CoreUserConfigParser().parse(pot_file);
  potContext  = tadah::core::Context::merge(potContext, taskContext);

  models::DC_Selector DCS(potContext);
  core::Logger::instance().log(tadah::core::LogLevel::INFO, "Finding nearest neighbours within: " +
      std::to_string(potContext.get<double>("RCUTMAX")) + " cutoff distance...");
  NNFinder nnf;

  // StructureDB stdb(taskContext);

  StructureDB stdb;
  if(taskContext.contains("DBFILE")) {
    stdb.add(taskContext);
  }
  if(taskContext.contains("STRUCTURE")) {
    std::vector<std::string> structures(taskContext.size("STRUCTURE"));
    taskContext.get<>("STRUCTURE", structures);
    for (std::string &struc : structures) {
      auto reader = StructureReaderSelector::getReader(struc);
      reader->read(struc);
      Structure s = reader->getStructure();
      stdb.add(s);
    }
  }


  nnf.calc(stdb,potContext);

  DescriptorsCalc<> dc(potContext,*DCS.d2b,*DCS.dmb,
                       *DCS.c2b,*DCS.cmb);
  
  tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Prediction...");
  DM_Function_Base *fb = tadah::core::factory<DM_Function_Base,const core::Context&>(
      potContext.get<std::string>("MODEL",1),potContext);
  M_Tadah_Base *modelp = tadah::core::factory<M_Tadah_Base,DM_Function_Base&,core::Context&>(
      potContext.get<std::string>("MODEL",0),*fb,potContext);
  StructureDB stpred;
  tadah::core::aed_type predicted_error;
  if (taskContext.get<bool>("ERROR")) {
    stpred = modelp->predict(potContext,stdb,dc,predicted_error);
  }
  else {
    stpred = modelp->predict(potContext,stdb,dc);
  }
  
  core::Logger::instance().log(core::LogLevel::INFO, "Dumping output...");
  Output output(taskContext);
  output.print_predict_all(stdb,stpred,predicted_error);
  if(taskContext.get<bool>("ANALYTICS")) {
    core::Logger::instance().log(core::LogLevel::INFO, "Calculating analytics...");
    Analytics a(stdb,stpred);
    std::cout << "Energy MAE (meV/atom): " << 1000*a.calc_e_mae() << std::endl;
    std::cout << "Energy RMSE (meV/atom): " << 1000*a.calc_e_rmse() << std::endl;
    std::cout << "Energy R^2: " << a.calc_e_r_sq() << std::endl;
  
    if (taskContext.get<bool>("FORCE")) {
      std::cout << "Force MAE (eV/A): "<< a.calc_f_mae() << std::endl;
      std::cout << "Force RMSE (eV/A): "<< a.calc_f_rmse() << std::endl;
      std::cout << "Force R^2: " << a.calc_f_r_sq() << std::endl;
    }
  
    if (taskContext.get<bool>("STRESS")) {
      std::cout << "Stress MAE (eV/A^3): "<< a.calc_s_mae() << std::endl;
      std::cout << "Stress RMSE (eV/A^3): "<< a.calc_s_rmse() << std::endl;
      std::cout << "Stress R^2: " << a.calc_s_r_sq() << std::endl;
    }
  }

  delete modelp;
  delete fb;

  return EXIT_SUCCESS;
}
void PredictEngine::validate(const tadah::mlip::Task &task) {
  const auto &context = task.getContext();
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "Validating PredictEngine...");
  if (!context.contains("POTENTIAL")) {
    throw std::runtime_error("PredictEngine requires a potential file to be set.");
  }
  // require context OUTFILE to be size 1 if no force or stress
  // size 2 if force or stress
  // size 3 if force and stress
  // use logger to inform user do not throw error
  if (context.get<bool>("FORCE") && context.get<bool>("STRESS")) {
    if (context.size("OUTFILE") != 3) {
      core::Logger::instance().log(tadah::core::LogLevel::ERROR,
          "Provide 3 output files for energy forces and stress.");
    }
  } else if (context.get<bool>("FORCE") || context.get<bool>("STRESS")) {
    if (context.size("OUTFILE") < 2) {
      std::string which = context.get<bool>("FORCE") ? "forces" : "stress";
      core::Logger::instance().log(tadah::core::LogLevel::ERROR,
          "Provide 2 output files: one for energy and one for " + which + ".");
    }
  } else  if (context.size("OUTFILE") != 1 && context.size("OUTFILE") != 3) {
    core::Logger::instance().log(tadah::core::LogLevel::ERROR,
        "Provide single output file for energy.");
  }
}


} // namespace engines
} // namespace mlip
} // namespace tadah

