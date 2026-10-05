#include <tadah/core/logger.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/engines/data_sample_engine.h>
#include <tadah/mlip/structure_db.h>


#include <algorithm>
#include <system_error>
#include <charconv>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {
namespace engines {

namespace {
/* The whole token `s` as an unsigned number, or false. */
bool whole_number(const std::string &s, size_t &value) {
  const auto r = std::from_chars(s.data(), s.data() + s.size(), value);
  return !s.empty() && r.ec == std::errc{} && r.ptr == s.data() + s.size();
}

/* The 1-based indices of the INDEX values (each a comma-separated list of
 * 'i', 'a-b' or 'a-b:step'; several values form one list), in order,
 * repeated indices removed. Every segment is checked, against 1 ... total
 * too, before its range is expanded; a segment that is not an index, a
 * descending range or a step of 0 is refused, naming it. */
std::vector<size_t> sample_indices(const std::vector<std::string> &values, size_t total) {
  auto check = [&](size_t i, const std::string &seg) {
    if (i == 0 || i > total)
      throw std::runtime_error(
          "tadah data sample: index " + std::to_string(i) +
          (seg == std::to_string(i) ? "" : " (in '" + seg + "')") +
          " is outside 1-" + std::to_string(total) + " (the dataset has " +
          std::to_string(total) + " structures)");
  };
  std::vector<size_t> out;
  std::vector<char> seen(total + 1, 0);
  for (const auto &value : values) {
    size_t pos = 0;
    while (true) {
      const size_t comma = value.find(',', pos);
      const std::string seg = core::trim(value.substr(
          pos, comma == std::string::npos ? std::string::npos : comma - pos));
      const auto bad = [&](const std::string &why) {
        return std::runtime_error("tadah data sample: index '" + seg + "' " + why +
                                  " (use i, a-b or a-b:step, 1-based)");
      };
      size_t a = 0, b = 0, step = 1;
      const size_t dash = seg.find('-');
      if (dash == std::string::npos) {
        if (!whole_number(seg, a)) throw bad("is not an index");
        b = a;
      } else {
        const size_t colon = seg.find(':', dash);
        const std::string end = seg.substr(
            dash + 1, colon == std::string::npos ? std::string::npos : colon - dash - 1);
        if (!whole_number(core::trim(seg.substr(0, dash)), a) ||
            !whole_number(core::trim(end), b) ||
            (colon != std::string::npos &&
             !whole_number(core::trim(seg.substr(colon + 1)), step)))
          throw bad("is not an index range");
        if (a > b) throw bad("is a descending range");
        if (step == 0) throw bad("has a step of 0");
      }
      check(a, seg);
      check(b, seg);
      for (size_t i = a;; i += step) {
        if (!seen[i]) {
          seen[i] = 1;
          out.push_back(i);
        }
        if (b - i < step) break;
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }
  return out;
}
}  // namespace

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

  // Exactly one selection: the command line excludes the others, and a
  // task file must too (run() would apply only the first).
  std::vector<std::string> given;
  for (const char *key : {"UNIFORM", "RANDOM", "INDEX"})
    if (taskContext.has_stored(key)) given.push_back(key);
  if (given.empty())
    throw std::runtime_error(
        "tadah data sample: say what to sample: --uniform N (every N-th "
        "structure), --random N (N structures at random) or --index "
        "<indices> (UNIFORM, RANDOM or INDEX in a task file)");
  if (given.size() > 1)
    throw std::runtime_error("tadah data sample: give one of UNIFORM, RANDOM "
                             "and INDEX, not " +
                             given[0] + " and " + given[1]);

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
    // Parsed against 1 ... total (sample_indices refuses what it cannot use).
    if (!isIndexParsed) parseIndices(taskContext);
    if (indices.empty())
      throw std::runtime_error("tadah data sample: --index selects nothing");
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
    indices = sample_indices(indicesStrVec, inStdb.size());
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
