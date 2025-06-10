#include <tadah/core/logger.h>
#include <tadah/mlip/engines/data_print_engine.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>

#include <iostream>

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

namespace tadah {
namespace mlip {
namespace engines {

int DataPrintEngine::run(tadah::mlip::Task &task) {
#ifdef TADAH_BUILD_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank!=0) return EXIT_SUCCESS;
#endif

  core::Context taskContext = task.getContext();

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataPrintEngine started...");

  StructureDB stdb;
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "Adding structures to the database...");

  if(taskContext.size("DBFILE") && !taskContext.contains("INDEX")) {
    // just print the summary and exit
    std::string dbFile = taskContext.get<std::string>("DBFILE");
    stdb.add(dbFile);
    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "Database file: " + dbFile);
    std::cout << stdb << std::endl;
    return EXIT_SUCCESS;
  }
  if(taskContext.size("DBFILE") && taskContext.contains("INDEX")) {
    std::vector<std::string> indicesStrVec;
    indicesStrVec.resize(taskContext.size("INDEX")); // required for DBFILE
    taskContext.get<>("INDEX", indicesStrVec);
    std::vector<size_t> indices =  core::parse_indices(indicesStrVec);
    // if indices contain 0, we throw as min index is 1
    if (std::find(indices.begin(), indices.end(), 0) != indices.end()) {
      throw std::runtime_error("Indices start from 1, not 0.");
    }
    std::string dbFile = taskContext.get<std::string>("DBFILE");
    for (auto i :indices) {
      stdb.add(dbFile,i-1);
    }
    if (stdb.size() == 0) {
      throw std::runtime_error("No structures found in the database with the given indices.");
    }
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
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Validating DataPrintEngine...");
}


} // namespace engines
} // namespace mlip
} // namespace tadah
