#include <tadah/mlip/engines/analysis_check_bounds_engine.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/mlip/check_bounds.h>
#include <tadah/mlip/structure_db.h>

namespace tadah {
namespace mlip {
namespace engines {

void AnalysisCheckBoundsEngine::validate(const tadah::mlip::Task& /*task*/) {
  // Blueprint enforces input-group constraints. Per-engine validation
  // happens inside run() so we can produce actionable messages with
  // the parsed file's line numbers.
}

int AnalysisCheckBoundsEngine::run(tadah::mlip::Task& task) {
  auto& taskCtx = task.getContext();
  if (!taskCtx.contains("HPOTARGET")) {
    throw std::runtime_error(
        "analysis check_bounds: HPOTARGET is required (set HPOTARGET in "
        "config or pass via the CLI).");
  }
  if (!taskCtx.contains("DBFILE")) {
    throw std::runtime_error(
        "analysis check_bounds: at least one DBFILE must be configured.");
  }

  const std::string hpo_path = taskCtx.get<std::string>("HPOTARGET");
  auto bounds = parse_hpo_target_bounds(hpo_path);
  if (bounds.empty()) {
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::WARNING,
        "analysis check_bounds: no OPTIM lines found in " + hpo_path);
    return EXIT_SUCCESS;
  }
  tadah::core::Logger::instance().log(
      tadah::core::LogLevel::INFO,
      "analysis check_bounds: parsed " + std::to_string(bounds.size()) +
      " bound entries from " + hpo_path);

  StructureDB stdb;
  const std::size_t nf = taskCtx.size("DBFILE");
  for (std::size_t i = 0; i < nf; ++i) {
    stdb.add(taskCtx.get<std::string>("DBFILE", i));
  }
  if (stdb.size() == 0) {
    throw std::runtime_error("analysis check_bounds: no structures loaded");
  }

  auto corners = sample_corners(bounds);
  tadah::core::Logger::instance().log(
      tadah::core::LogLevel::INFO,
      "analysis check_bounds: sampled " + std::to_string(corners.size()) +
      " corners (k=" + std::to_string(bounds.size()) + ")");

  auto res = check_bounds_at_corners(stdb, taskCtx, bounds, corners);
  return (res.n_corners_failing > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}

}  // namespace engines
}  // namespace mlip
}  // namespace tadah
