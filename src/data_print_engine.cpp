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
  int rank = 0;
#ifdef TADAH_BUILD_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  if (rank!=0) return EXIT_SUCCESS;

  try {
    core::Context taskContext = task.getContext();

    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "DataPrintEngine started...");

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
  }

  for (const auto &struc : stdb) {
    std::cout << struc << std::endl;
  }

  } catch (const std::exception &e) {
    std::cerr << "DataPrintEngine error: " << e.what() << std::endl;
    return EXIT_FAILURE;
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
