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
  if(taskContext.contains("DBFILE")) {
    std::vector<std::string> indicesStrVec;
    indicesStrVec.resize(taskContext.size("INDEX")); // required for DBFILE
    taskContext.get<>("INDEX", indicesStrVec);
    std::vector<size_t> indices =  core::parse_indices(indicesStrVec);
    std::string dbFile = taskContext.get<std::string>("DBFILE");
    for (size_t i=0; i<indices.size(); ++i) {
      stdb.add(dbFile,i);
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
