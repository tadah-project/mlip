#include <tadah/models/descriptors/dm/embf0.h>
#include <tadah/models/descriptors/dm/rhoform.h>
#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/mlip/stream_format_guard.h>
#include <tadah/mlip/command_options.h>
#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/intercepts.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/output/output.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>
#include <tadah/mlip/training_record.h>
#include <tadah/core/blueprint_parser.h>
#include <tadah/core/utils.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/eam/pot_curve_edit.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {
namespace engines {

int PredictEngine::run(mlip::Task &task) {
  const auto &taskContext = task.getContext();
  // The model is the potential's, as it was fitted: a run key that is no
  // option of tadah predict is refused before anything is read.
  require_command_options(taskContext, "PREDICT");
  std::string pot_file= taskContext.get<std::string>("POTENTIAL");
  // No parse stores a run setting of a potential (core::run_setting_keys():
  // VERBOSE, NUMERIC, OUTFILE, ...): a VERBOSE line of a potential of an
  // earlier version sets no log level, predict's own -v applies throughout.
  core::Context potContext = core::CoreUserConfigParser().parse(pot_file);
  // EMBF0 and RHOFORM belong to the potential, and the run gives neither:
  // a DM_REAM potential without RHOFORM throws, and either key of a
  // potential without a DM_REAM block gives its notice.
  tadah::models::check_embf0_precedence(potContext, core::Context(), pot_file);
  tadah::models::check_rhoform_precedence(potContext, core::Context(),
                                          pot_file);
  // The keys that prepared the training data (EFILTER, FFILTER, the ESHIFT
  // toggles, the weights, ZERO_COM_FORCE) are not the potential's: a
  // potential of an earlier version has them as key lines, and they are
  // ignored. The potential's LSCALE
  // and ESHIFT, the model's length scale and energy reference, stay.
  {
    const std::string ignored =
        ignore_potential_dataset_keys(potContext, pot_file);
    if (!ignored.empty())
      core::Logger::instance().log(core::LogLevel::INFO, ignored);
  }
  // The options of the run over the potential: the data predicted is
  // prepared with the filters predict is given (EFILTER, FFILTER,
  // ZERO_COM_FORCE) and its LSCALE and ESHIFT override the potential's.
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

  // Knot-family refit pots: RCUT2B/RCUTMB are derived from the largest knot
  // — correct a stale stated value (warns) before the descriptors and the
  // neighbour list read RCUT*MAX.
  models::potedit::normalize_knot_rcut(potContext, "tadah predict");

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

  // Apply load-time transforms using the merged potContext: the
  // potential's LSCALE and ESHIFT (or predict's --lscale / --eshift), so
  // the data predicted is at the model's length scale and energy reference,
  // and the filters predict was given (--efilter, --ffilter,
  // --zero-com-force). No shift is derived from the data predicted: after
  // --no-eshift the reference energies are compared as they are.
  apply_load_transforms(stdb, potContext);

  // The energy, force and stress metrics leave out structures with energy,
  // force or stress priority 0; a dataset made only of such structures is not
  // measured (nan, left out of the mean). A run in which no dataset is
  // measured by the force (stress) metrics it asks for (-F, -S) is refused
  // before the prediction; the energy metrics, which every run prints, are
  // nan then (ANALYTICS excludes STRUCTURE, so the datasets are the DBFILE
  // files).
  if (taskContext.get<bool>("ANALYTICS")) {
    std::vector<std::string> files;
    if (taskContext.contains("DBFILE")) files = taskContext("DBFILE");
    for (size_t d = 0; d + 1 < stdb.dbidx.size(); ++d)
      if (stdb.dbidx[d] == stdb.dbidx[d + 1])
        throw std::runtime_error(
            "tadah predict -A: dataset " + std::to_string(d + 1) + " of " +
            std::to_string(stdb.dbidx.size() - 1) +
            (d < files.size() ? " (" + files[d] + ")" : std::string()) +
            " has no structures after loading and filtering (EFILTER, "
            "FFILTER): it has no metrics");
    if (taskContext.get<bool>("FORCE"))
      require_metric_structures(stdb, MetricQuantity::Forces, "tadah predict -A", files);
    if (taskContext.get<bool>("STRESS"))
      require_metric_structures(stdb, MetricQuantity::Stress, "tadah predict -A", files);
  }

  nnf.calc(stdb,potContext);

  DescriptorsCalc<> dc(potContext,*DCS.d2b,*DCS.dmb,
                       *DCS.c2b,*DCS.cmb);
  require_descriptor_layout(DCS, potContext, "tadah predict");
  
  tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Prediction...");
  DM_Function_Base *fb = tadah::core::factory<DM_Function_Base,const core::Context&>(
      potContext.get<std::string>("MODEL",1),potContext);
  M_Tadah_Base *modelp = tadah::core::factory<M_Tadah_Base,DM_Function_Base&,core::Context&>(
      potContext.get<std::string>("MODEL",0),*fb,potContext);
  StructureDB stpred = modelp->predict(potContext,stdb,dc);

  core::Logger::instance().log(core::LogLevel::INFO, "Dumping output...");
  Output output(taskContext);
  output.print_predict_all(stdb,stpred);
  if(taskContext.get<bool>("ANALYTICS")) {
    core::Logger::instance().log(core::LogLevel::INFO, "Calculating analytics...");
    Analytics a(stdb, stpred);
    // The table formats std::cout its way (std::scientific, precision 6):
    // the next writer of std::cout finds it as it was.
    const StreamFormatGuard keep_format(std::cout);

    const int col_width = 24;  // the longest row name and a blank
    const int val_width = 16;

    std::vector<std::string> files;
    if (taskContext.contains("DBFILE")) files = taskContext("DBFILE");
    const std::vector<bool> measured_e = measured_datasets(stdb, MetricQuantity::Energy);
    const std::vector<bool> measured_f = measured_datasets(stdb, MetricQuantity::Forces);
    const std::vector<bool> measured_s = measured_datasets(stdb, MetricQuantity::Stress);

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

    // The mean column of a metric runs over the datasets it measures. A
    // row names its unit ("Stress RMSE (GPa)"), padded to the column by
    // the characters a terminal shows (the Å of eV/Å is two bytes).
    auto print_metric = [col_width, val_width](const std::string& name, const auto& values,
                                               double scale = 1.0,
                                               const std::vector<bool>* measured = nullptr) {
      double mean = measured ? measured_mean(values, *measured)
                             : std::accumulate(values.begin(), values.end(), 0.0) / values.size();
      std::size_t shown = 0;
      for (unsigned char ch : name)
        if ((ch & 0xC0) != 0x80) ++shown;
      std::cout << name
        << std::string(shown < static_cast<std::size_t>(col_width)
                           ? static_cast<std::size_t>(col_width) - shown
                           : 1,
                       ' ')
        << std::setw(val_width) << std::right << std::scientific << std::setprecision(6) << scale * mean;
      for (const auto& v : values) {
        std::cout << std::setw(val_width) << std::right << std::scientific << std::setprecision(6) << scale * v;
      }
      std::cout << std::endl;
    };

    // An R^2 has no value (NaN, Statistics::r_sq) for a dataset whose
    // reference values are all equal; its mean runs over the measured
    // datasets that have one, and a note names the others. NaN is told by
    // its bits (the build uses -ffast-math).
    struct Undefined { std::size_t dataset; const char *what; const char *metric; };
    std::vector<Undefined> undefined;
    auto r_sq_defined = [&](const tadah::core::t_type &values, const std::vector<bool> &measured,
                            const char *what, const char *metric) {
      std::vector<bool> defined(measured);
      for (std::size_t d = 0; d < defined.size() && d < values.size(); ++d) {
        const double v = values[d];
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof bits);
        const bool nan = (bits & 0x7ff0000000000000ull) == 0x7ff0000000000000ull &&
                         (bits & 0x000fffffffffffffull) != 0;
        if (defined[d] && nan) {
          defined[d] = false;
          undefined.push_back({d, what, metric});
        }
      }
      return defined;
    };

    // Energy
    print_metric("Energy MAE (meV/atom)", a.calc_e_mae(), 1000, &measured_e);
    print_metric("Energy RMSE (meV/atom)", a.calc_e_rmse(), 1000, &measured_e);
    {
      const auto r = a.calc_e_r_sq();
      const auto defined = r_sq_defined(r, measured_e, "energy per atom", "Energy");
      print_metric("Energy R^2", r, 1.0, &defined);
    }

    if (taskContext.get<bool>("FORCE")) {
      std::cout << std::endl;
      print_metric("Force MAE (eV/Å)", a.calc_f_mae(), 1.0, &measured_f);
      print_metric("Force RMSE (eV/Å)", a.calc_f_rmse(), 1.0, &measured_f);
      const auto r = a.calc_f_r_sq();
      const auto defined = r_sq_defined(r, measured_f, "force component", "Force");
      print_metric("Force R^2", r, 1.0, &defined);
    }

    if (taskContext.get<bool>("STRESS")) {
      std::cout << std::endl;
      // the nine entries of each tensor, in GPa
      print_metric("Stress MAE (GPa)", a.calc_s_mae_gpa(), 1.0, &measured_s);
      print_metric("Stress RMSE (GPa)", a.calc_s_rmse_gpa(), 1.0, &measured_s);
      const auto r = a.calc_s_r_sq();
      const auto defined = r_sq_defined(r, measured_s, "stress component", "Stress");
      print_metric("Stress R^2", r, 1.0, &defined);
    }

    // The datasets a requested metric does not measure, named.
    auto note = [&](const std::vector<bool> &measured, const char *data, const char *weight,
                    const char *metrics) {
      for (size_t d = 0; d < measured.size(); ++d)
        if (!measured[d])
          std::cout << "\nNote: dataset " << d + 1
                    << (d < files.size() ? " (" + files[d] + ")" : std::string())
                    << " has no structure with " << data << " (" << weight
                    << " 0 everywhere): its " << metrics
                    << " metrics are not measured (nan) and left out of the mean." << std::endl;
    };
    note(measured_e, "an energy", "energy priority", "energy");
    if (taskContext.get<bool>("FORCE")) note(measured_f, "forces", "force priority", "force");
    if (taskContext.get<bool>("STRESS"))
      note(measured_s, "a stress tensor", "stress priority", "stress");
    for (const auto &u : undefined)
      std::cout << "\nNote: dataset " << u.dataset + 1
                << (u.dataset < files.size() ? " (" + files[u.dataset] + ")" : std::string())
                << ": every reference " << u.what << " has the same value, so its "
                << u.metric << " R^2 has no value (nan) and is left out of the mean."
                << std::endl;
  }

  delete modelp;
  delete fb;

  return EXIT_SUCCESS;
}
void PredictEngine::validate(const tadah::mlip::Task &task) {
  const auto &context = task.getContext();
  require_command_options(context, "PREDICT");
  if (!context.contains("POTENTIAL")) {
    throw std::runtime_error(
        "tadah predict needs a potential file (--potential)");
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

