#include <tadah/mlip/engines/properties_pairwise_engine.h>
#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
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

// Helper function for printing data tables
static void printPotentialData(
  std::ofstream& outfile,
  const std::string& title,
  const std::string& energyUnit,
  bool hasForce,
  bool hasError,
  size_t colWidth,
  size_t numeric,
  const core::v_type& distances,
  const StructureDB& predictions,
  const tadah::core::aed_type& errors
);

int PropertiesPairwiseEngine::run(mlip::Task &task) {
#ifdef TADAH_BUILD_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank!=0) return EXIT_SUCCESS;
#endif

  const auto &taskContext = task.getContext();
  std::string pot_file= taskContext.get<std::string>("POTENTIAL");
  core::Context potContext = core::CoreUserConfigParser().parse(pot_file);
  potContext  = tadah::core::Context::merge(potContext, taskContext);

  // generate range
  double start = taskContext.get<double>("RANGE",0);
  double stop = taskContext.get<double>("RANGE",1);
  size_t N = taskContext.get<size_t>("RANGE",2);
  core::v_type range = core::linspace(start,stop,N);

  models::DC_Selector DCS(potContext);
  double cutoff = potContext.get<double>("RCUTMAX");
  double cell_v = range.back() > cutoff? 4*range.back() : 4*cutoff;
  core::Matrix3d cell;
  cell.set_zero();
  cell(0,0) = cell(1,1) = cell(2,2) = cell_v;
  std::string e1 = taskContext.get<std::string>("ATOMPAIR",0);
  std::string e2 = taskContext.get<std::string>("ATOMPAIR",1);
  Atom a1(core::Element(e1),0,0,0,0,0,0);
  Atom a2(core::Element(e2),0,0,0,0,0,0);
  StructureDB stdb;
  for (const auto& r: range) {
    Structure s;
    s.cell = cell;
    a2.position[0]=r;
    s.add_atom(a1);
    s.add_atom(a2);
    stdb.add(s);
  }

  core::Logger::instance().log(tadah::core::LogLevel::INFO, "Finding nearest neighbours within: " +
      std::to_string(potContext.get<double>("RCUTMAX")) + " cutoff distance...");
  NNFinder nnf(potContext);
  nnf.calc(stdb);

  tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Computing pairwise interactions...");
  DescriptorsCalc<> dc(potContext,*DCS.d2b,*DCS.dmb,
      *DCS.c2b,*DCS.cmb);
  DM_Function_Base *fb = tadah::core::factory<DM_Function_Base,core::Context&>(
      potContext.get<std::string>("MODEL",1),potContext);
  M_Tadah_Base *modelp = tadah::core::factory<M_Tadah_Base,DM_Function_Base&,core::Context&>(
      potContext.get<std::string>("MODEL",0),*fb,potContext);

  bool is_2b = potContext.get<bool>("INIT2B");
  bool is_mb = potContext.get<bool>("INITMB");

  StructureDB stpred;
  StructureDB stpred_2b;
  StructureDB stpred_mb;
  tadah::core::aed_type predicted_error;
  tadah::core::aed_type predicted_error_2b;
  tadah::core::aed_type predicted_error_mb;
  core::Context potContext_2b = potContext;
  core::Context potContext_mb = potContext;
  potContext_2b.remove("INITMB");
  potContext_2b.add("INITMB",false);
  potContext_mb.remove("INIT2B");
  potContext_mb.add("INIT2B",false);

  DescriptorsCalc<> dc_2b(potContext_2b,*DCS.d2b,*DCS.dmb,
      *DCS.c2b,*DCS.cmb);
  DM_Function_Base *fb_2b = tadah::core::factory<DM_Function_Base,core::Context&>(
      potContext_2b.get<std::string>("MODEL",1),potContext_2b);
  M_Tadah_Base *modelp_2b = tadah::core::factory<M_Tadah_Base,DM_Function_Base&,core::Context&>(
      potContext_2b.get<std::string>("MODEL",0),*fb_2b,potContext_2b);

  DescriptorsCalc<> dc_mb(potContext_mb,*DCS.d2b,*DCS.dmb,
      *DCS.c2b,*DCS.cmb);
  DM_Function_Base *fb_mb = tadah::core::factory<DM_Function_Base,core::Context&>(
      potContext_mb.get<std::string>("MODEL",1),potContext_mb);
  M_Tadah_Base *modelp_mb = tadah::core::factory<M_Tadah_Base,DM_Function_Base&,core::Context&>(
      potContext_mb.get<std::string>("MODEL",0),*fb_mb,potContext_mb);

  if (taskContext.get<bool>("ERROR")) {
    stpred = modelp->predict(potContext,stdb,dc,predicted_error);
    if (is_2b && is_mb) {
      stpred_2b = modelp_2b->predict(potContext_2b,stdb,dc_2b,predicted_error_2b);
      stpred_mb = modelp_mb->predict(potContext_mb,stdb,dc_mb,predicted_error_mb);
    }
  }
  else {
    stpred = modelp->predict(potContext,stdb,dc);
    if (is_2b && is_mb) {
      stpred_2b = modelp_2b->predict(potContext_2b,stdb,dc_2b);
      stpred_mb = modelp_mb->predict(potContext_mb,stdb,dc_mb);
    }
  }

  if (taskContext.contains("ESHIFT")) {
    double eshift = taskContext.get<double>("ESHIFT");
    if (eshift == 0) eshift = - stpred.structures.back().energy;
    for (auto &s : stpred) {
      s.energy += eshift;
    }
    if (is_2b && is_mb) {
      double eshift_2b = taskContext.get<double>("ESHIFT",1);
      if (eshift_2b == 0) eshift_2b = - stpred_2b.structures.back().energy;
      for (auto &s : stpred_2b) {
        s.energy += eshift_2b;
      }
      double eshift_mb = taskContext.get<double>("ESHIFT",2);
      if (eshift_mb == 0) eshift_mb = - stpred_mb.structures.back().energy;
      for (auto &s : stpred_mb) {
        s.energy += eshift_mb;
      }
    }
  }
  // energy is predicted per structure. Given that we have
  // two atoms, we need to divide by 2 unless user
  // specifically asked for bond energy
  std::string energyUnit;
  if (!taskContext.get<bool>("BONDENERGY")) {
    for (auto &s : stpred) {
      s.energy /= s.natoms();
    }
    energyUnit = "eV/atom";
  } else {
    energyUnit = "eV/bond";
  }

  bool force = taskContext.get<bool>("FORCE");
  bool error = taskContext.get<bool>("ERROR");
  size_t numeric = taskContext.get<size_t>("NUMERIC");
  core::Logger::instance().log(core::LogLevel::INFO, "Dumping output...");
  std::ofstream outfile(taskContext.get<std::string>("OUTFILE"));

  // Determine label width for headers and a separate column width for numeric data
  // The 'labelWidth' ensures headers fit, 'colWidth' ensures numeric fitting in scientific notation
  size_t labelWidth = 15;
  size_t colWidth = std::max(labelWidth, numeric + 10);

  printPotentialData(outfile, "Total potential", energyUnit, force,
      error, colWidth, numeric, range, stpred, predicted_error);

  if (is_2b && is_mb) {
    printPotentialData(outfile, "Two-body potential", energyUnit, force,
        error, colWidth, numeric, range, stpred_2b, predicted_error_2b);

    printPotentialData(outfile, "Many-body potential", energyUnit, force,
        error, colWidth, numeric, range, stpred_mb, predicted_error_mb);
  }

  delete modelp;
  delete fb;

  return EXIT_SUCCESS;
}
void PropertiesPairwiseEngine::validate(const tadah::mlip::Task &task) {
  const auto &taskContext = task.getContext();
  if (!taskContext.contains("POTENTIAL")) {
    throw std::runtime_error("PropertiesPairwiseEngine requires a potential file to be set.");
  }
  if (!taskContext.contains("RANGE")) {
    throw std::runtime_error("PropertiesPairwiseEngine requires a range to be set.");
  }
  if (!taskContext.contains("ATOMPAIR")) {
    throw std::runtime_error("PropertiesPairwiseEngine requires a atom pair to be set.");
  }
  if (taskContext.contains("ESHIFT")) {
    std::string pot_file= taskContext.get<std::string>("POTENTIAL");
    core::Context potContext = core::CoreUserConfigParser().parse(pot_file);
    bool is_2b = potContext.get<bool>("INIT2B");
    bool is_mb = potContext.get<bool>("INITMB");
    if (is_2b && is_mb && taskContext.size("ESHIFT")!=3) {
      throw std::runtime_error("Provide three ESHIFT values for total, 2B and MB energy.");
    }
  }
}

static void printPotentialData(
  std::ofstream& outfile,
  const std::string& title,
  const std::string& energyUnit,
  bool hasForce,
  bool hasError,
  size_t colWidth,
  size_t numeric,
  const core::v_type& distances,
  const StructureDB& predictions,
  const tadah::core::aed_type& errors
)
{
  // Print a header row for this data block
  outfile << "# " << title << std::endl;
  outfile << "# " << std::right
          << std::setw(colWidth-2) << "r (Å)"
          << std::setw(colWidth) << ("Energy " + energyUnit);
  if (hasForce) {
    outfile << std::setw(colWidth) << "Force (eV/Å)";
  }
  if (hasError) {
    outfile << std::setw(colWidth) << ("Error " + energyUnit);
  }
  outfile << std::endl;

  // Print the data rows
  for (size_t j = 0; j < distances.size(); ++j) {
    outfile << std::scientific << std::setprecision(numeric)
            << std::setw(colWidth) << distances[j]
            << std::setw(colWidth) << predictions(j).energy;
    if (hasForce) {
      outfile << std::setw(colWidth) << predictions(j)(1).force[0];
    }
    if (hasError) {
      outfile << std::setw(colWidth) << errors[j];
    }
    outfile << std::endl;
  }
  outfile << std::endl << std::endl;
}

} // namespace engines
} // namespace mlip
} // namespace tadah
