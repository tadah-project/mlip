#include <tadah/core/logger.h>
#include <tadah/mlip/engines/data_print_engine.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>

#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

int DataPrintEngine::run(tadah::mlip::Task &task) {
  core::Context taskContext = task.getContext();


  StructureDB stdb;
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "Adding structures to the database...");

  // has_stored, not size(): size() throws for a key that is not set
  // (`data print -s` has no DBFILE).
  const bool has_dbfile = taskContext.has_stored("DBFILE");
  if(has_dbfile && !taskContext.contains("INDEX")) {
    // just print the summary and exit
    std::string dbFile = taskContext.get<std::string>("DBFILE");
    stdb.add(dbFile);
    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "Database file: " + dbFile);
    std::cout << stdb << std::endl;
    return EXIT_SUCCESS;
  }
  if(has_dbfile && taskContext.contains("INDEX")) {
    const std::string dbFile = taskContext.get<std::string>("DBFILE");
    std::vector<std::string> indicesStrVec(taskContext.size("INDEX"));
    taskContext.get<>("INDEX", indicesStrVec);
    // 1-based, parsed strictly and checked against the number of structures
    // before any is read; the file is then read once, and the structures
    // are printed in the given order.
    const size_t total = static_cast<size_t>(StructureDB::count(dbFile).first);
    std::vector<size_t> indices =
        core::parse_dataset_indices(indicesStrVec, total, "tadah data print");
    for (auto &i : indices) --i;
    stdb.add_selected(dbFile, indices);
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
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "Printing structures...");
  for (const auto &struc : stdb) {
    std::cout << struc << std::endl;
  }


  return EXIT_SUCCESS;
}
void DataPrintEngine::validate(const tadah::mlip::Task &) {
}


} // namespace engines
} // namespace mlip
} // namespace tadah
