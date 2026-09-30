#include <tadah/mlip/engines/analysis_dataset_stats_engine.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/logger.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/core/stats/data_stats_io.h>
#include <tadah/mlip/data_stats_compute.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>

namespace tadah {
namespace mlip {
namespace engines {

void AnalysisDatasetStatsEngine::validate(const tadah::mlip::Task& /*task*/) {
  // Blueprint validation already enforces input-group requirements
  // (DBFILE | STRUCTURE | task). No additional pre-conditions here.
}

int AnalysisDatasetStatsEngine::run(tadah::mlip::Task& task) {
  auto& taskContext = task.getContext();

  // Build a working context. If the user passed `-c train.tadah`, the
  // CLI dispatcher will have run CoreUserConfigParser on it already
  // and merged the result into taskContext (so RCUTMAX is set).
  tadah::core::Context workCtx = taskContext;

  // Collect structures.
  StructureDB stdb;
  if (taskContext.contains("DBFILE")) {
    const std::size_t nf = taskContext.size("DBFILE");
    for (std::size_t i = 0; i < nf; ++i) {
      const std::string dbf = taskContext.get<std::string>("DBFILE", i);
      tadah::core::Logger::instance().log(
          tadah::core::LogLevel::INFO, "Reading file: " + dbf);
      stdb.add(dbf);
      tadah::core::Logger::instance().log(
          tadah::core::LogLevel::INFO, stdb.summary());
    }
  }
  if (taskContext.contains("STRUCTURE")) {
    const std::size_t ns = taskContext.size("STRUCTURE");
    for (std::size_t i = 0; i < ns; ++i) {
      const std::string s = taskContext.get<std::string>("STRUCTURE", i);
      auto reader = StructureReaderSelector::getReader(s);
      reader->read(s);
      stdb.add(reader->getStructure());
    }
  }

  if (stdb.size() == 0) {
    throw std::runtime_error(
        "analysis dataset_stats: no structures loaded — provide at "
        "least one DBFILE or STRUCTURE.");
  }

  // RCUTMAX must already be set by either the user's config or the
  // postprocessor of CoreUserConfigParser.
  if (!workCtx.contains("RCUTMAX") || workCtx.get<double>("RCUTMAX") <= 0) {
    throw std::runtime_error(
        "analysis dataset_stats: RCUTMAX is not set or <= 0. "
        "Set at least one of RCUT2B / RCUTMB in your config so the "
        "audit cutoff is well-defined.");
  }

  auto stats = compute_data_stats(stdb, workCtx, /*with_descriptor_stats=*/false);

  // Structures whose cell is not symmetric (max |C - C^T| > 0). A cell that
  // an older Tadah stored transposed (see the data-migration note of the
  // 1.6.0 changelog) can only be told apart from a correct one in these, so
  // they are listed exactly; nothing is guessed about any of them.
  {
    std::vector<std::size_t> nonsym;
    for (std::size_t s = 0; s < stdb.size(); ++s) {
      const auto &C = stdb(s).cell;
      double d = 0.0;
      for (int a = 0; a < 3; ++a)
        for (int b = a + 1; b < 3; ++b)
          d = std::max(d, std::fabs(C(a, b) - C(b, a)));
      if (d > 0.0) nonsym.push_back(s);
    }
    std::cout << nonsym.size() << " of " << stdb.size()
              << " structures have a non-symmetric cell";
    if (nonsym.empty()) {
      std::cout << "." << std::endl;
    } else {
      std::cout << " (only in these can a cell stored transposed by an older "
                   "Tadah be told apart from a correct one; see the "
                   "data-migration note in the 1.6.0 changelog):"
                << std::endl;
      for (std::size_t s : nonsym)
        std::cout << "  structure " << s + 1 << ": " << stdb(s).label
                  << std::endl;
    }
  }

  // Output.
  std::string outpath = "dataset_stats.tadah";
  if (taskContext.contains("OUTFILE")) {
    outpath = taskContext.get<std::string>("OUTFILE");
  }
  std::ofstream os(outpath);
  if (!os) {
    throw std::runtime_error(
        "analysis dataset_stats: could not open output file '" + outpath + "'");
  }
  tadah::core::stats::save(stats, os);

  tadah::core::Logger::instance().log(
      tadah::core::LogLevel::INFO,
      "analysis dataset_stats: wrote " + outpath +
          " (n_atoms=" + std::to_string(stats.n_atoms) +
          ", class=" + (stats.class_label.empty() ? "?" : stats.class_label) +
          ").");
  return EXIT_SUCCESS;
}

}  // namespace engines
}  // namespace mlip
}  // namespace tadah
