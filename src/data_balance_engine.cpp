#include <tadah/core/logger.h>
#include <tadah/mlip/engines/data_balance_engine.h>
#include <tadah/mlip/structure_db.h>


#include <iostream>
#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

namespace tadah {
namespace mlip {
namespace engines {

int DataBalanceEngine::run(tadah::mlip::Task &task) {
#ifdef TADAH_BUILD_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank!=0) return EXIT_SUCCESS;
#endif

  core::Context taskContext = task.getContext();

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataBalanceEngine started...");

  std::vector<StructureDB> inStdbs(taskContext.size("DBFILE")); // we have to keep them separate

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataBalanceEngine: Reading input files...");

  size_t stdbCounter = 0;
  std::vector<std::string> dbfilesPaths(taskContext.size("DBFILE"));
  taskContext.get("DBFILE", dbfilesPaths);
  for (std::string dbFile : dbfilesPaths) {
    inStdbs.at(stdbCounter++).add(dbFile);
  }

  std::map<int, double> eshiftMap;  // Z -> Eshift
  if (taskContext.contains("ESHIFT")) {
    // we have to do it before the main loop 
    // as not every stdb has the same elements
    std::set<core::Element> elements;
    for (auto &stdb : inStdbs) {
      std::set<core::Element> newElements = stdb.get_unique_elements();
      elements.insert(newElements.begin(), newElements.end());  // sorted by Z
    }
    std::vector<double> eshiftVec(taskContext.size("ESHIFT"));
    if (elements.size() != eshiftVec.size()) {
      throw std::runtime_error("ESHIFT size does not match number of unique elements.");
    }
    taskContext.get("ESHIFT", eshiftVec);
    size_t eCounter = 0;
    for (const auto &element : elements) {
      eshiftMap[element.Z] = eshiftVec[eCounter++];
    }
  }

  // floating point thresholds for rescaling
  std::vector<double> thresholdVec(3);  // energy, force, stress
  if (taskContext.get<bool>("RESCALE") && 
      taskContext.contains("THRESHOLD")) {
    taskContext.get("THRESHOLD", thresholdVec);
    // check if thresholds are positive
    for (const auto &threshold : thresholdVec) {
      if (threshold <= 0.0) {
        throw std::runtime_error("THRESHOLD values must be positive.");
      }
    }
  }
  bool force = taskContext.get<bool>("FORCE");
  bool stress = taskContext.get<bool>("STRESS");

  std::vector<bool> indicesBool;  // what structures to process
  size_t numStructures = 0;
  for (const auto &stdb : inStdbs) {
    numStructures += stdb.size();
  }
  indicesBool.resize(numStructures);

  if(taskContext.contains("INDEX")) {
    std::vector<size_t> indices;
    std::vector<std::string> indicesStrVec(taskContext.size("INDEX"));
    taskContext.get<>("INDEX", indicesStrVec);
    indices = core::parse_indices(indicesStrVec);
    for (const auto idx : indices) {
      if (idx >= numStructures) {
        throw std::runtime_error("Index out of range.");
      }
      indicesBool[idx-1] = true;
    }
  } else {
    std::fill(indicesBool.begin(), indicesBool.end(), true);
  }

  if (!eshiftMap.empty() && taskContext.contains("ESHIFT")) {
    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "DataBalanceEngine: Applying ESHIFT...");
    size_t currentIndex = 0;
    for (auto &stdb : inStdbs) {
      for (auto &st : stdb) {
        if (!indicesBool[currentIndex++]) continue;
        auto natomsMap = st.get_natoms_map();
        for (const auto &pair : natomsMap) {
          int Z = pair.first;
          size_t natomsZ = pair.second;
          st.energy += eshiftMap[Z] * natomsZ;
        }
      }
    }
  }

  if (taskContext.get<bool>("RESCALE")) {
    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "DataBalanceEngine: Applying RESCALE...");

    size_t currentIndex = 0;
    for (auto &stdb : inStdbs) {
      for (auto &s: stdb) {
        if (!indicesBool[currentIndex++]) continue;
        // ENERGY WEIGHT
        if (std::fabs(s.energy) < thresholdVec[0]) {
          s.eweight = 1.0;
        } else {
          s.eweight = 1.0 / std::fabs(s.energy);
        }

        // FORCE WEIGHT
        if (force) {
          double fweight = 0.0;
          for (auto &atom : s) {
            double fnorm = atom.force.norm();
            fweight += fnorm;
          }
          // If total force norm is near zero, set fweight safely
          if (std::fabs(fweight) < thresholdVec[1]) {
            s.fweight = 1.0;
          } else {
            s.fweight = 1.0 / fweight;
          }
        }

        // STRESS WEIGHT
        if (stress) {
          double sweight = s.stress.norm();
          if (std::fabs(sweight) < thresholdVec[2]) {
            s.sweight = 1.0;
          } else {
            s.sweight = 1.0 / sweight;
          }
        }
      }
    }
  }

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataBalanceEngine: Dumping output....");
  std::vector<std::string> outFiles(taskContext.size("OUTFILE"));
  taskContext.get("OUTFILE", outFiles);
  bool append = taskContext.get<bool>("APPEND");
  bool merge = taskContext.get<bool>("MERGE");
  size_t numeric = taskContext.get<size_t>("NUMERIC");

  if (merge) {  // merge all stdbs into one
    StructureDB stdb;
    for (const auto& _stdb : inStdbs) stdb.add(_stdb);
    stdb.dump_to_file(outFiles[0], append, numeric);
  } else {
    for (size_t i=0; i<inStdbs.size(); ++i) {
      inStdbs[i].dump_to_file(outFiles[i], append, numeric);
    }
  }

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataBalanceEngine finished.");

  return EXIT_SUCCESS;
}
void DataBalanceEngine::validate(const tadah::mlip::Task &task) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Validating DataBalanceEngine...");
  core::Context taskContext = task.getContext();
  bool doRescale = taskContext.get<bool>("RESCALE");
  bool doEshift = taskContext.contains("ESHIFT") && taskContext.size("ESHIFT");
  if (!doRescale && !doEshift) {
    throw std::runtime_error("DataBalanceEngine requires either RESCALE and/or ESHIFT option.");
  }

  size_t numOutFiles = taskContext.size("OUTFILE");
  size_t numInFiles = taskContext.size("DBFILE");
  bool merge = taskContext.get<bool>("MERGE");
  if (!merge && numInFiles != numOutFiles) {
    throw std::runtime_error("Number of input and output files must match.");
  } else if (merge && numOutFiles != 1) {
    throw std::runtime_error("Number of output files must be 1 when MERGE is true.");
  } else if (numInFiles != numOutFiles) {
    throw std::runtime_error("Number of input and output files must match.");
  }
}


} // namespace engines
} // namespace mlip
} // namespace tadah
