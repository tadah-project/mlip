#include <tadah/core/logger.h>
#include <tadah/mlip/engines/data_dedup_engine.h>
#include <tadah/mlip/structure_db.h>


#include <iostream>
#include <stdexcept>
#include <string>

namespace tadah {
namespace mlip {
namespace engines {

int DataDedupEngine::run(tadah::mlip::Task &task) {
    core::Context taskContext = task.getContext();

    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "DataDedupEngine started...");
  std::vector<std::string> outFiles(taskContext.size("OUTFILE"));
  taskContext.get<>("OUTFILE",outFiles);
  bool append = taskContext.get<bool>("APPEND");
  bool merge = taskContext.get<bool>("MERGE");
  size_t numeric = StructureDB::dataset_digits(taskContext);
  // No default: how close two positions must be to count as equal depends
  // on how precisely the data were written.
  if (!taskContext.has_stored("THRESHOLD"))
    throw std::runtime_error(
        "tadah data dedup needs --threshold <Angstrom>: the largest "
        "difference, in each Cartesian component, at which two atom "
        "positions count as equal (e.g. --threshold 1e-6)");
  double threshold = taskContext.get<double>("THRESHOLD");

  std::vector<StructureDB> inStdbs(taskContext.size("DBFILE"));
  std::vector<std::string> inFiles(taskContext.size("DBFILE"));
  taskContext.get<>("DBFILE",inFiles);

  size_t total=0;
  std::vector<size_t> initSizes;
  initSizes.resize(inStdbs.size());
  for (size_t i=0; i < inStdbs.size(); ++i) {
    inStdbs[i].add(taskContext.get<std::string>("DBFILE",i));
    total += inStdbs[i].size();
    initSizes[i] = inStdbs[i].size();
  }

  std::vector<std::pair<size_t, size_t>> indices(total);
  size_t idx=0;
  for (size_t i=0; i < inStdbs.size(); ++i) {
    for (size_t j=0; j < inStdbs[i].size(); ++j) {
      indices[idx++] = std::make_pair(i,j);
    }
  }

  // In input order (the files in order), a structure is a duplicate when
  // it is the same as an earlier structure that is kept; the first of each
  // set of equal structures is kept. Pairs are (kept, removed).
  std::vector<std::pair<size_t, size_t>> duplicates;
  std::vector<size_t> kept;
  for (size_t j=0; j < indices.size(); ++j) {
    Structure &st_j = inStdbs[indices[j].first](indices[j].second);
    bool duplicate = false;
    for (size_t i : kept) {
      Structure &st_i = inStdbs[indices[i].first](indices[i].second);
      bool same;
      try {
        same = st_i.is_the_same(st_j,threshold);
      } catch (const std::invalid_argument &e) {
        // Name both structures by file and 1-based position.
        throw std::runtime_error(
            "tadah data dedup: comparing structure #" +
            std::to_string(indices[i].second + 1) + " of " +
            inFiles[indices[i].first] + " with structure #" +
            std::to_string(indices[j].second + 1) + " of " +
            inFiles[indices[j].first] + ": " + e.what());
      }
      if (same) {
        duplicates.push_back(std::make_pair(i,j));
        duplicate = true;
        break;
      }
    }
    if (!duplicate) kept.push_back(j);
  }

  if (core::Logger::instance().getLogLevel() == tadah::core::LogLevel::INFO) {
    size_t d=0;
    std::stringstream ss;
    for (const auto &db : inFiles) if (d<db.size()) d=db.size()+2;
    size_t w = 10;
    for (const auto &dup: duplicates) {
      size_t i1 = dup.first;
      size_t i2 = dup.second;
      ss  << "  " << std::setw(d) << std::left
        << inFiles[indices[i1].first]
        << " : #"
        << std::setw(w)
        << indices[i1].second + 1
        << "  "
        << std::setw(d)
        << inFiles[indices[i2].first]
        << " : #"
        << std::setw(w)
        << indices[i2].second + 1
        << std::endl;
    }
    // remove last newline from the stringstream
    if (duplicates.size() == 0) {
      core::Logger::instance().log(tadah::core::LogLevel::INFO,
          "No duplicates found");
    } else {
      std::string str = ss.str();
      if (str.size() > 0) str.pop_back();
      core::Logger::instance().log(tadah::core::LogLevel::INFO,
          "Duplicates found:\n", str);
    } 
  }

  // finally prune: each removed structure once, the last first, so that
  // removing one does not move the others of its file.
  for (size_t idx = duplicates.size(); idx-- > 0;) {
    const size_t removed = duplicates[idx].second;
    inStdbs[indices[removed].first].remove(indices[removed].second);
  }

  size_t newTotal=0;
  for (size_t i=0; i < inStdbs.size(); ++i) {
    newTotal += inStdbs[i].size();
  }

  if (merge) {
    StructureDB stdb;
    for (const auto& _stdb : inStdbs) stdb.add(_stdb);
    stdb.dump_to_file(outFiles[0], append, numeric);
  } else {
    for (size_t i=0; i<inStdbs.size(); ++i) {
      inStdbs[i].dump_to_file(outFiles[i], append, numeric);
      core::Logger::instance().log(tadah::core::LogLevel::INFO,
          "Dumped ", inStdbs[i].size(), " structures to ", outFiles[i]," (Removed: ",initSizes[i]-inStdbs[i].size(),")",
          "\n", "  "+inStdbs[i].summary());
    }
  }

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataDedupEngine finished. ", newTotal, " structures in ", outFiles.size(), " files.");

  return EXIT_SUCCESS;
}
void DataDedupEngine::validate(const tadah::mlip::Task &task) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Validating DataDedupEngine...");
  core::Context taskContext = task.getContext();
  size_t outFilesSize = taskContext.size("OUTFILE");
  size_t inFilesSize = taskContext.size("DBFILE");
  bool merge = taskContext.get<bool>("MERGE");
  if (merge && outFilesSize != 1) {
    throw std::runtime_error("When using MERGE, only one output file is allowed.");
  }
  else if (!merge && outFilesSize != inFilesSize) {
    throw std::runtime_error("Number of input files and output files must match.");
  }
}


} // namespace engines
} // namespace mlip
} // namespace tadah
