#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/mlip/load_transforms.h>
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

  // Honour --no-lscale / --no-eshift cancellation flags by removing the
  // corresponding keys from the merged context BEFORE apply_load_transforms
  // runs. The user is asserting "the dataset I'm passing in is already at the
  // model's scale / baseline".
  if (taskContext.contains("NO_LSCALE") && taskContext.get<bool>("NO_LSCALE")) {
    potContext.remove("LSCALE");
    core::Logger::instance().log(core::LogLevel::INFO,
        "--no-lscale: ignoring LSCALE recorded in pot.tadah.");
  }
  if (taskContext.contains("NO_ESHIFT") && taskContext.get<bool>("NO_ESHIFT")) {
    potContext.remove("ESHIFT");
    core::Logger::instance().log(core::LogLevel::INFO,
        "--no-eshift: ignoring ESHIFT recorded in pot.tadah.");
  }

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
    // STRUCTURE inputs were not loaded via DBFILE iteration, so dbidx is empty
    // for those configs. Mark the whole stdb as a single dataset so that
    // any transforms that consult dbidx see consistent boundaries.
    if (stdb.dbidx.empty()) {
      stdb.dbidx = {0, stdb.size()};
    }
  }

  // Apply load-time transforms (LSCALE, ESHIFT, EFILTER/FFILTER, ZERO_COM_FORCE)
  // using the merged potContext, so the input dataset is reshaped to match
  // what the model was trained on. NB: at predict time the model has already
  // been trained, so derived sources (ESHIFT_ATOM, ESHIFT_DBATOM) and
  // training-only knobs (WDBFILE, WDBFILE_AUTO, EWEIGHT_TEMP) are no-ops here.
  apply_load_transforms(stdb, potContext);

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
    Analytics a(stdb, stpred);

    const int col_width = 16;
    const int val_width = 16;

    // Prepare a sample vector to determine the number of values per metric
    auto sample_values = a.calc_e_mae();
    size_t n_vals = sample_values.size();

    // Print header
    std::cout << std::setw(col_width) << std::left << "Metric"
      << std::setw(val_width) << std::right << "Mean";
    for (size_t i = 0; i < n_vals; ++i) {
      std::cout << std::setw(val_width) << std::right << ("Val" + std::to_string(i + 1));
    }
    std::cout << std::endl;

    // Print separator
    std::cout << std::setw(col_width) << std::setfill('-') << ""
      << std::setw(val_width) << ""
      << std::setfill('-');
    for (size_t i = 0; i < n_vals; ++i) {
      std::cout << std::setw(val_width) << "";
    }
    std::cout << std::setfill(' ') << std::endl;

    auto print_metric = [col_width, val_width](const std::string& name, const auto& values, double scale = 1.0) {
      double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
      std::cout << std::setw(col_width) << std::left << name
        << std::setw(val_width) << std::right << std::scientific << std::setprecision(6) << scale * mean;
      for (const auto& v : values) {
        std::cout << std::setw(val_width) << std::right << std::scientific << std::setprecision(6) << scale * v;
      }
      std::cout << std::endl;
    };

    // Energy
    print_metric("Energy MAE", a.calc_e_mae(), 1000);
    print_metric("Energy RMSE", a.calc_e_rmse(), 1000);
    print_metric("Energy R^2", a.calc_e_r_sq());

    if (taskContext.get<bool>("FORCE")) {
      std::cout << std::endl;
      print_metric("Force MAE", a.calc_f_mae());
      print_metric("Force RMSE", a.calc_f_rmse());
      print_metric("Force R^2", a.calc_f_r_sq());
    }

    if (taskContext.get<bool>("STRESS")) {
      std::cout << std::endl;
      print_metric("Stress MAE", a.calc_s_mae());
      print_metric("Stress RMSE", a.calc_s_rmse());
      print_metric("Stress R^2", a.calc_s_r_sq());
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

