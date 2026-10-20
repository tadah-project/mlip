#include <tadah/core/logger.h>
#include <tadah/mlip/engines/data_balance_engine.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/output/writable_outputs.h>
#include <tadah/mlip/structure_db.h>


#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

int DataBalanceEngine::run(tadah::mlip::Task &task) {
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
    // 1-based, checked against 1 ... numStructures.
    indices = core::parse_dataset_indices(indicesStrVec, numStructures,
                                          "tadah data balance");
    for (const auto idx : indices) indicesBool[idx - 1] = true;
  } else {
    std::fill(indicesBool.begin(), indicesBool.end(), true);
  }

  if (!eshiftMap.empty() && taskContext.contains("ESHIFT")) {
    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "DataBalanceEngine: Applying ESHIFT (E -= sum_Z N_Z * eshift[Z])...");
    size_t currentIndex = 0;
    for (auto &stdb : inStdbs) {
      for (auto &st : stdb) {
        if (!indicesBool[currentIndex++]) continue;
        auto natomsMap = st.get_natoms_map();
        double total = 0.0;
        for (const auto &pair : natomsMap) {
          auto it = eshiftMap.find(pair.first);
          if (it == eshiftMap.end()) continue;
          total += it->second * static_cast<double>(pair.second);
        }
        st.energy -= total;  // unified Tadah! ESHIFT convention: subtract baseline
      }
    }
  }

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataBalanceEngine: Dumping output....");
  std::vector<std::string> outFiles(taskContext.size("OUTFILE"));
  taskContext.get("OUTFILE", outFiles);
  bool append = taskContext.get<bool>("APPEND");
  bool merge = taskContext.get<bool>("MERGE");
  size_t numeric = StructureDB::dataset_digits(taskContext);

  require_writable_outputs(outFiles, "tadah data balance");
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
  if (!(taskContext.contains("ESHIFT") && taskContext.size("ESHIFT") > 0))
    throw std::runtime_error(
        "tadah data balance shifts the energies of its input files by a "
        "reference energy per element: --eshift (ESHIFT) gives them, one per "
        "element of the files in the order of the atomic numbers, and the "
        "run has none.");
  const size_t numOutFiles = taskContext.size("OUTFILE");
  const size_t numInFiles = taskContext.size("DBFILE");
  const bool merge = taskContext.get<bool>("MERGE");
  if (merge && numOutFiles != 1)
    throw std::runtime_error(
        "tadah data balance --merge writes the structures of its " +
        std::to_string(numInFiles) + " input file(s) to one output file, "
        "and --outfile names " + std::to_string(numOutFiles) + ".");
  if (!merge && numInFiles != numOutFiles)
    throw std::runtime_error(
        "tadah data balance writes one output file per input file: " +
        std::to_string(numInFiles) + " input file(s) and " +
        std::to_string(numOutFiles) + " output file(s) (--merge writes "
        "them all to one).");
}


} // namespace engines
} // namespace mlip
} // namespace tadah
