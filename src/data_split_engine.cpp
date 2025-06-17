#include <tadah/core/logger.h>
#include <tadah/mlip/engines/data_split_engine.h>
#include <tadah/mlip/structure_db.h>


#include <iostream>
#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

namespace tadah {
namespace mlip {
namespace engines {

int DataSplitEngine::run(tadah::mlip::Task &task) {
#ifdef TADAH_BUILD_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank!=0) return EXIT_SUCCESS;
#endif

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataSplitEngine started...");

  core::Context taskContext = task.getContext();
  std::vector<std::string> outFiles(taskContext.size("OUTFILE"));
  taskContext.get<>("OUTFILE",outFiles);
  bool append = taskContext.get<bool>("APPEND");
  size_t numeric = taskContext.get<size_t>("NUMERIC");

  if (!areDataLoaded) loadInput(taskContext);
  if (!areSizesCalculated) calculateSizes(taskContext);

  std::vector<StructureDB> outStdbs(outFiles.size());

  std::vector<size_t> indices(inStdb.size());
  std::iota(indices.begin(), indices.end(), 0);

  if (taskContext.get<bool>("SHUFFLE")) {
    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "Randomizing data...");
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
  }

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "Dumping data into ", outFiles.size(), " files.");
  size_t total = 0;
  for (size_t i=0; i < outStdbs.size(); ++i) {
    StructureDB &outStdb = outStdbs[i];
    for (size_t j=0; j < sizes[i]; ++j) {
      outStdb.add(inStdb(indices[total++]));
    }
    outStdb.dump_to_file(outFiles[i], append, numeric);
    // print outStdb summary
    std::string dumpOrAppend = append ? "Appended" : "Dumped";
    core::Logger::instance().log(tadah::core::LogLevel::INFO,
       dumpOrAppend+" ", outStdb.size(), " structures to ", outFiles[i],
        "\n", outStdb.summary());
  }

  // clear internal state in case this engine is used again
  resetState();
  return EXIT_SUCCESS;
}
void DataSplitEngine::validate(const tadah::mlip::Task &task) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Validating DataSplitEngine...");
  core::Context taskContext = task.getContext();
  size_t outFilesSize = taskContext.size("OUTFILE");
  if (!areDataLoaded) loadInput(taskContext);

  if (taskContext.contains("CHUNK")) {
    size_t numChunks = taskContext.size("CHUNK");
    std::vector<size_t> chunks(numChunks);
    taskContext.get<std::vector<size_t>>("CHUNK", chunks);
    if (outFilesSize != numChunks) throw std::runtime_error("Number of chunks does not match number of output files.");
    // check if all chunks are greater than 0
    for (const auto &chunk: chunks) {
      if (chunk<=0) throw std::runtime_error("Chunks must be greater than 0.");
    }
    // check if all chunks add up to the number of structures in the file
    size_t total=0;
    for (const auto &chunk: chunks) total+= chunk;
    if (total>inStdb.size()) throw std::runtime_error("Sum of chunks is greater than available number of structures in a file.");
  }
  else if (taskContext.contains("PERCENT")) {
    size_t numPercent = taskContext.size("PERCENT");
    std::vector<size_t> percent(numPercent);
    taskContext.get<std::vector<size_t>>("PERCENT", percent);
    if (outFilesSize != numPercent) throw std::runtime_error("Number of percentages does not match number of output files.");
    // check if all percentages are between 0 (exclusive) and 100.
    for (const auto &perc: percent) {
      if (perc<=0 || perc>100) throw std::runtime_error("Percentages must be between 1 and 100.");
    }
    // check if all percentages add up to 100
    size_t total=0;
    for (const auto &perc: percent) total+= perc;
    if (total!=100) throw std::runtime_error("Percentages do not add up to 100%. Total:" + std::to_string(total));
  }
  // else if (taskContext.contains("EVEN")) {} // do nothing as sizes are calculated from outFiles
}

void DataSplitEngine::loadInput(const core::Context &taskContext) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Loading input for DataSplitEngine...");
  inStdb.add(taskContext);
  areDataLoaded = true;
}

void DataSplitEngine::calculateSizes(const core::Context &taskContext) {
  size_t outFilesSize = taskContext.size("OUTFILE");

  if (!areDataLoaded) loadInput(taskContext);

  if (taskContext.get<bool>("EVEN")) {
    sizes = core::divide_into_parts(inStdb.size(),outFilesSize);
    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "Spliting evenly. Resulting sizes: ", sizes);
  }
  else if (taskContext.contains("PERCENT")) {
    sizes.resize(taskContext.size("PERCENT"));
    taskContext.get<std::vector<size_t>>("PERCENT", sizes);
    sizes = core::divide_by_percentage(inStdb.size(),sizes);
    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "Spliting by percentage. Resulting sizes: ", sizes);
  }
  else if (taskContext.contains("CHUNK")) {
    sizes.resize(outFilesSize);
    taskContext.get<std::vector<size_t>>("CHUNK", sizes);
    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "Requested chunks", sizes);
  }
  else {
    throw std::runtime_error("No valid splitting method provided.");
  }
  areSizesCalculated = true;
}

void DataSplitEngine::resetState() {
  inStdb.clear();
  sizes.clear();
  areDataLoaded = false;
  areSizesCalculated = false;
}

} // namespace engines
} // namespace mlip
} // namespace tadah
