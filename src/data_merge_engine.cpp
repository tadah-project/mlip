#include <tadah/core/logger.h>
#include <tadah/mlip/engines/data_merge_engine.h>
#include <tadah/mlip/structure_db.h>


#include <iostream>
#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

namespace tadah {
namespace mlip {
namespace engines {

int DataMergeEngine::run(tadah::mlip::Task &task) {
  int rank = 0;
#ifdef TADAH_BUILD_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  if (rank!=0) return EXIT_SUCCESS;

  try {
    core::Context taskContext = task.getContext();

    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "DataMergeEngine started...");

    std::string outFile = taskContext.get<std::string>("OUTFILE");
    StructureDB stdb(taskContext);
    bool append = taskContext.get<bool>("APPEND");
    size_t numeric = taskContext.get<size_t>("NUMERIC");
    stdb.dump_to_file(outFile,append, numeric);

    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        stdb.summary());

  } catch (const std::exception &e) {
    std::cerr << "DataMergeEngine error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
void DataMergeEngine::validate(const tadah::mlip::Task &) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Validating DataMergeEngine...");
}


} // namespace engines
} // namespace mlip
} // namespace tadah
