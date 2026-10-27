#include <tadah/core/logger.h>
#include <tadah/mlip/engines/data_merge_engine.h>
#include <tadah/mlip/structure_db.h>


#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

// An error reaches the program, which prints it as every command's.
int DataMergeEngine::run(tadah::mlip::Task &task) {
  core::Context taskContext = task.getContext();
  std::string outFile = taskContext.get<std::string>("OUTFILE");
  StructureDB stdb(taskContext);
  bool append = taskContext.get<bool>("APPEND");
  size_t numeric = StructureDB::dataset_digits(taskContext);
  stdb.dump_to_file(outFile,append, numeric);

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      stdb.summary());
  return EXIT_SUCCESS;
}
void DataMergeEngine::validate(const tadah::mlip::Task &) {
}


} // namespace engines
} // namespace mlip
} // namespace tadah
