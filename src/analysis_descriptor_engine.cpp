#include <tadah/mlip/engines/analysis_descriptor_engine.h>
#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/models/m_tadah_base.h>
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

int AnalysisDescriptorEngine::run(mlip::Task &task) {
  #ifdef TADAH_BUILD_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank!=0) return EXIT_SUCCESS;
  #endif

  const auto &taskContext = task.getContext();
  std::string pot_file= taskContext.get<std::string>("POTENTIAL");
  core::Context potContext = core::CoreUserConfigParser().parse(pot_file);
  potContext  = tadah::core::Context::merge(potContext, taskContext);

  std::vector<StructureDB> stdbs;
  if(taskContext.contains("DBFILE")) {
    if (taskContext.contains("INDEX")) {
      // assume single DBFILE, guaranteed by validate()
      stdbs.resize(1);
      std::vector<size_t> indices;
      std::vector<std::string> indicesStr;
      indicesStr.resize(taskContext.size("INDEX"));
      taskContext.get<>("INDEX", indicesStr);
      indices = core::parse_indices(indicesStr);
      size_t maxIndex = *std::max_element(indices.begin(), indices.end());
      if (maxIndex > taskContext.size("DBFILE")) {  // INDEX starts from 1
        throw std::runtime_error("Index out of bounds for DBFILE.");
      }
      StructureDB &stdb = stdbs[0];
      for (const auto idx : indices) {
        stdb.add(taskContext.get<std::string>("DBFILE", 0), idx-1);
      }
    } else {
      size_t numFiles = taskContext.size("DBFILE");
      stdbs.resize(numFiles);
      for (size_t i = 0; i < numFiles; ++i) {
        std::string dbFile = taskContext.get<std::string>("DBFILE", i);
        core::Logger::instance().log(tadah::core::LogLevel::INFO, "Reading file: " + dbFile);
        stdbs[i].add(dbFile);
        core::Logger::instance().log(tadah::core::LogLevel::INFO, stdbs[i].summary());
      }
    }
  }
  if(taskContext.contains("STRUCTURE")) {
    core::Logger::instance().log(tadah::core::LogLevel::INFO, "Reading structures...");
    stdbs.push_back(StructureDB());
    std::vector<std::string> structures(taskContext.size("STRUCTURE"));
    taskContext.get<>("STRUCTURE", structures);
    StructureDB &stdb = stdbs.back();
    for (std::string &struc : structures) {
      auto reader = StructureReaderSelector::getReader(struc);
      reader->read(struc);
      Structure s = reader->getStructure();
      stdb.add(s);
    }
  }

  models::DC_Selector DCS(potContext);
  core::Logger::instance().log(tadah::core::LogLevel::INFO, "Finding nearest neighbours within: " +
      std::to_string(potContext.get<double>("RCUTMAX")) + " cutoff distance...");
  NNFinder nnf(potContext);
  for (auto &stdb : stdbs) {
    nnf.calc(stdb);
  }

  tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Calculating descriptors...");
  DescriptorsCalc<> dc(potContext,*DCS.d2b,*DCS.dmb,
                       *DCS.c2b,*DCS.cmb);
  DM_Function_Base *fb = tadah::core::factory<DM_Function_Base,core::Context&>(
      potContext.get<std::string>("MODEL",1),potContext);
  M_Tadah_Base *modelp = tadah::core::factory<M_Tadah_Base,DM_Function_Base&,core::Context&>(
      potContext.get<std::string>("MODEL",0),*fb,potContext);

  std::vector<StDescriptorsDB> stDescDBs(stdbs.size());
  for (size_t i=0; i<stdbs.size(); ++i) {
    auto &stdb = stdbs[i];
    for (auto &st : stdb) {
      StDescriptors stDesc = dc.calc(st);
      stDescDBs[i].add(stDesc);
    }
  }

  std::vector<std::vector<std::string>> outputs(stDescDBs.size());
  size_t hw = 12;
  size_t numeric = taskContext.get<size_t>("NUMERIC");
  size_t w = numeric + 8;
  std::string dim[] = {"x","y","z"}; 
  size_t idxDB=0;
  for (auto &stDescDB : stDescDBs) {
    // for each dbfile
    size_t idxSt = 0;
    std::vector<std::string> &output = outputs[idxDB];
    for (auto &stDesc : stDescDB) {
      // for each structure
      std::ostringstream oss;
      Structure &st = stdbs[idxDB](idxSt);
      oss << "# Strcuture: " << st.label << std::endl;
      for (size_t a=0; a<st.natoms(); ++a) {  
        const core::aed_type &aed = stDesc.get_aed(a);
        oss << std::scientific << std::setprecision(numeric);
        for (size_t s=0; s<aed.size(); ++s) {
          oss << std::right << std::setw(w) <<  aed(s) << "  ";
        }
        oss << std::left << std::setw(hw) << ("# E " + std::to_string(a+1));
        oss<<std::endl;
      }
      if (taskContext.get<bool>("FORCE")) {
        for (size_t a1=0; a1<st.natoms(); ++a1) {
          for (size_t jj=0; jj<stDesc.fd[a1].size(); ++jj) {
            const size_t a2=st.near_neigh_idx[a1][jj];
            const core::fd_type &fij = stDesc.fd[a1][jj];
            for (size_t k=0; k<3; ++k) {
              oss << std::scientific << std::setprecision(numeric);
              for (size_t d=0; d<fij.rows(); ++d) {
                oss << std::right << std::setw(w) <<  fij(d,k) << "  ";
              }
              oss << std::left << std::setw(hw) << ("# F"+dim[k]+ " " + std::to_string(a1+1)) + " - " +  std::to_string(a2+1);
              oss<<std::endl;
            }
          }
        }
      }
      oss << std::endl;
      output.push_back(oss.str());
      idxSt++;
    }
    idxDB++;
  }
  core::Logger::instance().log(core::LogLevel::INFO, "Dumping output...");
  std::vector<std::string> outFiles(taskContext.size("OUTFILE"));
  taskContext.get<>("OUTFILE", outFiles);
  bool append = taskContext.get<bool>("APPEND");
  bool merge = taskContext.get<bool>("MERGE");
  if (merge) {
    std::string outFile = outFiles[0];
    std::ofstream ofs = append ? std::ofstream(outFile, std::ios::app) : std::ofstream(outFile);
    for (size_t i=0; i<outputs.size(); ++i) {
      ofs << "# Dataset: " << taskContext.get<std::string>("DBFILE", i) << std::endl;
      for (auto &line : outputs[i]) {
        ofs << line;
      }
    }
    ofs.close();
  } else {
    for (size_t i=0; i<outputs.size(); ++i) {
      std::string outFile = outFiles[i];
      std::ofstream ofs = append ? std::ofstream(outFile, std::ios::app) : std::ofstream(outFile);
      ofs << "# Dataset: " << taskContext.get<std::string>("DBFILE", i) << std::endl;
      for (auto &line : outputs[i]) {
        ofs << line;
      }
      ofs.close();
    }
  }

  delete modelp;
  delete fb;

  core::Logger::instance().log(tadah::core::LogLevel::INFO, "AnalysisDescriptorEngine finished.");
  return EXIT_SUCCESS;
}
void AnalysisDescriptorEngine::validate(const tadah::mlip::Task &task) {
  const auto &context = task.getContext();
  if (!context.contains("POTENTIAL")) {
    throw std::runtime_error("AnalysisDescriptorEngine requires a potential file to be set.");
  }
  if (context.contains("INDEX")) {
    if (context.size("DBFILE")!=1) {
      throw std::runtime_error("AnalysisDescriptorEngine requires a single DBFILE when using INDEX.");
    }
    if (context.contains("STRUCTURE")) {
      throw std::runtime_error("AnalysisDescriptorEngine INDEX and STRUCTURE cannot be used together.");
    }
  }
  if (context.get<bool>("MERGE") && context.size("OUTFILE") != 1) {
    throw std::runtime_error("AnalysisDescriptorEngine requires a single OUTFILE when using MERGE.");
  }
  if (!context.get<bool>("MERGE")) {
    if (context.size("OUTFILE") != context.size("DBFILE")) {
      throw std::runtime_error("AnalysisDescriptorEngine requires the same number of OUTFILE and DBFILE.");
    }
  }
}

} // namespace engines
} // namespace mlip
} // namespace tadah

