#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/core/config.h>
#include <tadah/mlip/structure_db.h>
#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

int PredictEngine::predict(const tadah::core::Config &pot_config, const StructureDB &stdb) {
  if (rank!=0) return EXIT_SUCCESS;

  tadah::core::Config pot_config(pot_file);


  DC_Selector DCS(pot_config);


  if (is_verbose()) std::cout << "Finding nearest neighbours within: " <<
    pot_config.get<double>("RCUTMAX") << " cutoff distance..." << std::flush;
  NNFinder nnf(pot_config);
  nnf.calc(stdb);
  if (is_verbose()) std::cout << "Done!" << std::endl;

  DescriptorsCalc<> dc(pot_config,*DCS.d2b,*DCS.d3b,*DCS.dmb,
                       *DCS.c2b,*DCS.c3b,*DCS.cmb);

  if (is_verbose()) std::cout << "Prediction..." << std::flush;
  DM_Function_Base *fb = tadah::core::factory<DM_Function_Base,Config&>(
    pot_config.get<std::string>("MODEL",1),pot_config);
  M_Tadah_Base *modelp = tadah::core::factory<M_Tadah_Base,DM_Function_Base&,Config&>(
    pot_config.get<std::string>("MODEL",0),*fb,pot_config);

  StructureDB stpred;
  aed_type predicted_error;
  if (predict->count("--error")) {
    stpred = modelp->predict(pot_config,stdb,dc,predicted_error);
  }
  else {
    stpred = modelp->predict(pot_config,stdb,dc);
  }

  if (is_verbose()) std::cout << "Done!" << std::endl;

  if (is_verbose()) std::cout << "Dumping output..." << std::flush;

  Output output(pot_config,predict->count("--error"));

  if (predict->count("--numeric"))
    output.set_numeric(outprec);

  output.print_predict_all(stdb,stpred,predicted_error);

  if (is_verbose()) std::cout << "Done!" << std::endl;
  if (is_verbose()) std::cout << timer_tot.to_string() << std::endl;

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

