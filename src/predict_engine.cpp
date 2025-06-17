#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/core/context.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/output/output.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/models/dc_selector.h>
#include <iostream>

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

namespace tadah {
namespace mlip {
namespace engines {

int PredictEngine::predict(const tadah::core::Context &pot_context, const StructureDB &stdb) {
  int rank = 0;
  #ifdef TADAH_BUILD_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  #endif
  if (rank!=0) return EXIT_SUCCESS;

  // tadah::core::Context pot_context(pot_file);


  tadah::models::DC_Selector DCS(pot_context);


  tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Finding nearest neighbours within: " +
    std::to_string(pot_context.get<double>("RCUTMAX")) + " cutoff distance...");
  NNFinder nnf(pot_context);
  nnf.calc(stdb);

  tadah::models::DescriptorsCalc<> dc(pot_context,*DCS.d2b,*DCS.d3b,*DCS.dmb,
                       *DCS.c2b,*DCS.c3b,*DCS.cmb);

  tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Prediction...");
  DM_Function_Base *fb = tadah::core::factory<DM_Function_Base,Context&>(
    pot_context.get<std::string>("MODEL",1),pot_context);
  M_Tadah_Base *modelp = tadah::core::factory<M_Tadah_Base,DM_Function_Base&,Context&>(
    pot_context.get<std::string>("MODEL",0),*fb,pot_context);

  StructureDB stpred;
  tadah::core::aed_type predicted_error;
  if (predict->count("--error")) {
    stpred = modelp->predict(pot_context,stdb,dc,predicted_error);
  }
  else {
    stpred = modelp->predict(pot_context,stdb,dc);
  }

  tada::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Dumping output...");

  Output output(pot_context,predict->count("--error"));

  if (predict->count("--numeric"))
    output.set_numeric(outprec);

  output.print_predict_all(stdb,stpred,predicted_error);


  tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, timer_tot.to_string());

  if(predict->count("--analytics")) {
    Analytics a(stdb,stpred);

    std::cout << "Energy MAE (meV/atom): " << 1000*a.calc_e_mae() << std::endl;
    std::cout << "Energy RMSE (meV/atom): " << 1000*a.calc_e_rmse() << std::endl;
    std::cout << "Energy R^2: " << a.calc_e_r_sq() << std::endl;

    if (predict->count("--Force")) {
      std::cout << "Force MAE (eV/A): "<< a.calc_f_mae() << std::endl;
      std::cout << "Force RMSE (eV/A): "<< a.calc_f_rmse() << std::endl;
      std::cout << "Force R^2: " << a.calc_f_r_sq() << std::endl;
    }

    if (predict->count("--Stress")) {
      std::cout << "Stress MAE (eV/A^3): "<< a.calc_s_mae() << std::endl;
      std::cout << "Stress RMSE (eV/A^3): "<< a.calc_s_rmse() << std::endl;
      std::cout << "Stress R^2: " << a.calc_s_r_sq() << std::endl;
    }
  }

  if(modelp)
    delete modelp;
  if(fb)
    delete fb;

  return 0;
  return EXIT_SUCCESS;
}

} // namespace engines
} // namespace mlip
} // namespace tadah

