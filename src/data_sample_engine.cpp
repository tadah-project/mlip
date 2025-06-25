#include <tadah/core/logger.h>
#include <tadah/mlip/engines/data_sample_engine.h>
#include <tadah/mlip/structure_db.h>


#include <iostream>
#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

namespace tadah {
namespace mlip {
namespace engines {

int DataSampleEngine::run(tadah::mlip::Task &task) {
#ifdef TADAH_BUILD_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank!=0) return EXIT_SUCCESS;
#endif

  core::Context taskContext = task.getContext();

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataSampleEngine started...");

  std::string outFile = taskContext.get<std::string>("OUTFILE");
  bool append = taskContext.get<bool>("APPEND");
  size_t numeric = taskContext.get<size_t>("NUMERIC");

  if (!areDataLoaded) loadInput(taskContext);

  StructureDB outStdb;
  if (taskContext.contains("UNIFORM")) {
    size_t every = taskContext.get<size_t>("UNIFORM");
    for (size_t j=0; j < inStdb.size(); ++j) {
      if (!(j++ % every)) outStdb.add(inStdb(j));
    }
  }
  else if (taskContext.contains("RANDOM")) {
    size_t N = taskContext.get<size_t>("RANDOM");
    indices.resize(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    for (size_t i : indices) {
      outStdb.add(inStdb(i));
      if (outStdb.size() >= N) break;
    }
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

  return 0;

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataSampleEngine finished.");

  resetState();
  return EXIT_SUCCESS;
}
void DataSampleEngine::validate(const tadah::mlip::Task &task) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "Validating DataSampleEngine...");
  core::Context taskContext = task.getContext();

  if (!areDataLoaded) loadInput(taskContext);

  if (taskContext.contains("UNIFORM")) {
    size_t N = taskContext.get<size_t>("UNIFORM");
    size_t total = inStdb.size();
    if (N>total) throw std::runtime_error(
        "Selected number is larger than total number of configurations availble.");
  }
  if (taskContext.contains("RANDOM")) {
    size_t N = taskContext.get<size_t>("RANDOM");
    size_t total = inStdb.size();
    if (N>total) throw std::runtime_error(
        "Selected number is larger than total number of configurations availble.");
  }
  if (taskContext.contains("INDEX")) {
    if (!isIndexParsed) parseIndices(taskContext);
    if (indices.empty()) {
      throw std::runtime_error("Parsing of indices failed.");
    }
    if (indices.size() >= inStdb.size()) {
      throw std::runtime_error("Number of indices is larger than the number of available structures.");
    }
    size_t max_index = *std::max_element(indices.begin(), indices.end());
    if (max_index>=inStdb.size())   // INDEX starts from 1
      throw std::runtime_error("Selected index is larger than the number of available structures.");
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
