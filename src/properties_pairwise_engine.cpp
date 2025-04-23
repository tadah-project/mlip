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
  StructureDB stpred;
  tadah::core::aed_type predicted_error;
  if (taskContext.get<bool>("ERROR")) {
    stpred = modelp->predict(potContext,stdb,dc,predicted_error);
  }
  else {
    stpred = modelp->predict(potContext,stdb,dc);
  }

  if (taskContext.contains("ESHIFT")) {
    double eshift = taskContext.get<double>("ESHIFT");
    if (eshift == 0) eshift = stpred.structures.back().energy;
    for (auto &s : stpred) {
      s.energy -= eshift;
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
  size_t labelWidth = 19;
  size_t colWidth = std::max(labelWidth, numeric + 10);

  // Write header
  outfile << std::right
          << std::setw(colWidth) << "Distance (Angstrom)"
          << std::setw(colWidth) << ("Energy " + energyUnit);

  if (force) {
    outfile << std::setw(colWidth) << "Force (eV/A)";
  }
  if (error) {
    outfile << std::setw(colWidth) << ("Error " + energyUnit);
  }
  outfile << std::endl;

  // Write data
  for (size_t j = 0; j < range.size(); ++j) {
    outfile << std::scientific << std::setprecision(numeric)
            << std::setw(colWidth) << range[j];

    outfile << std::setw(colWidth) << stpred(j).energy;

    if (force) {
      outfile << std::setw(colWidth) << stpred(j)(1).force[0];
    }
    if (error) {
      outfile << std::setw(colWidth) << predicted_error[j];
    }
    outfile << std::endl;
  }
  outfile << std::endl << std::endl;

  delete modelp;
  delete fb;

  return EXIT_SUCCESS;
}
void PropertiesPairwiseEngine::validate(const tadah::mlip::Task &task) {
  const auto &context = task.getContext();
  if (!context.contains("POTENTIAL")) {
    throw std::runtime_error("PropertiesPairwiseEngine requires a potential file to be set.");
  }
  if (!context.contains("RANGE")) {
    throw std::runtime_error("PropertiesPairwiseEngine requires a range to be set.");
  }
  if (!context.contains("ATOMPAIR")) {
    throw std::runtime_error("PropertiesPairwiseEngine requires a atom pair to be set.");
  }
}

} // namespace engines
} // namespace mlip
} // namespace tadah
