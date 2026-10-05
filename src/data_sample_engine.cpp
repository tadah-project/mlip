#include <tadah/core/logger.h>
#include <tadah/mlip/engines/data_sample_engine.h>
#include <tadah/mlip/structure_db.h>


#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {
namespace engines {

int DataSampleEngine::run(tadah::mlip::Task &task) {
  core::Context taskContext = task.getContext();

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataSampleEngine started...");

  std::string outFile = taskContext.get<std::string>("OUTFILE");
  bool append = taskContext.get<bool>("APPEND");
  size_t numeric = StructureDB::dataset_digits(taskContext);

  if (!areDataLoaded) loadInput(taskContext);

  StructureDB outStdb;
  if (taskContext.contains("UNIFORM")) {
    // Every N-th structure, from the first (validate() refuses N = 0).
    const size_t every = taskContext.get<size_t>("UNIFORM");
    for (size_t j = 0; j < inStdb.size(); j += every) outStdb.add(inStdb(j));
  }
  else if (taskContext.contains("RANDOM")) {
    // N distinct structures, each with the same chance, written in their
    // dataset order: the first N of a random permutation of all of them.
    const size_t N = taskContext.get<size_t>("RANDOM");
    std::vector<size_t> all(inStdb.size());
    std::iota(all.begin(), all.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all.begin(), all.end(), g);
    all.resize(N);
    std::sort(all.begin(), all.end());
    for (size_t i : all) outStdb.add(inStdb(i));
  }
  else if (taskContext.contains("INDEX")) {
    if (!isIndexParsed) parseIndices(taskContext);
    for (const auto &i : indices) {
      outStdb.add(inStdb(i-1));
    }
  }

  outStdb.dump_to_file(outFile, append, numeric);
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataSampleEngine finished. Sampled data written to: " + outFile);
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      outStdb.summary());

  resetState();
  return EXIT_SUCCESS;
}
void DataSampleEngine::validate(const tadah::mlip::Task &task) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "Validating DataSampleEngine...");
  core::Context taskContext = task.getContext();

  if (!areDataLoaded) loadInput(taskContext);

  const size_t total = inStdb.size();
  const std::string structures = std::to_string(total) + " structures";
  if (taskContext.contains("UNIFORM")) {
    const size_t every = taskContext.get<size_t>("UNIFORM");
    if (every == 0)
      throw std::runtime_error(
          "tadah data sample: --uniform 0: the step must be at least 1 "
          "(--uniform N keeps every N-th structure)");
    if (every > total)
      throw std::runtime_error("tadah data sample: --uniform " +
                               std::to_string(every) +
                               " is larger than the dataset's " + structures);
  }
  if (taskContext.contains("RANDOM")) {
    const size_t N = taskContext.get<size_t>("RANDOM");
    if (N > total)
      throw std::runtime_error("tadah data sample: --random " + std::to_string(N) +
                               ": the dataset has only " + structures);
  }
  if (taskContext.contains("INDEX")) {
    if (!isIndexParsed) parseIndices(taskContext);
    if (indices.empty()) {
      throw std::runtime_error("Parsing of indices failed.");
    }
    // Indices are 1-based: 1 ... total (parse_indices drops duplicates).
    for (const size_t i : indices)
      if (i == 0 || i > total)
        throw std::runtime_error("tadah data sample: index " + std::to_string(i) +
                                 " is outside 1-" + std::to_string(total) +
                                 " (the dataset has " + structures + ")");
  }
}

void DataSampleEngine::loadInput(const core::Context &taskContext) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Loading input for DataSampleEngine...");
  inStdb.add(taskContext);
  areDataLoaded = true;
}

void DataSampleEngine::parseIndices(const core::Context &taskContext) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Parsing indices for DataSampleEngine...");
  if (taskContext.contains("INDEX")) {
    std::vector<std::string> indicesStrVec;
    indicesStrVec.resize(taskContext.size("INDEX"));
    taskContext.get<>("INDEX", indicesStrVec);
    indices = core::parse_indices(indicesStrVec);
  }
  isIndexParsed = true;

}
void DataSampleEngine::resetState() {
  inStdb.clear();
  areDataLoaded = false;
  indices.clear();
  isIndexParsed = false;
}


} // namespace engines
} // namespace mlip
} // namespace tadah
