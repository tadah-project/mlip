#include <tadah/mlip/engines/analysis_check_bounds_engine.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/mlip/check_bounds.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/training_record.h>
#include <tadah/models/descriptors/d_mjoin.h>

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
  // A seed potential's lines of the keys of its training record but DBFILE
  // were not read (TaskFactory's parse: potential_line_not_read; a tadah hpo
  // seed of an earlier version has HPOTARGET, VERBOSE and NUMERIC lines):
  // the WARNING names those check_bounds reads, HPOTARGET and VERBOSE.
  const std::string ignored = ignore_configuration_potential_lines(
      taskCtx, task.getKeyProvenance(), "analysis check_bounds");
  if (!ignored.empty())
    tadah::core::Logger::instance().log(tadah::core::LogLevel::WARNING,
                                        ignored);
  if (!taskCtx.has_stored("HPOTARGET"))
    throw std::runtime_error(missing_run_setting(
        "analysis check_bounds", "HPOTARGET", "the target file",
        task.getKeyProvenance()));
  if (!taskCtx.contains("DBFILE")) {
    throw std::runtime_error(
        "analysis check_bounds: at least one DBFILE must be configured.");
  }

  // The radial grids as the descriptors read them: a grid generator
  // expanded into its values, as tadah hpo expands it, so that an OPTIM
  // index addresses the value tadah hpo searches and the checks read the
  // values as numbers.
  models::D_mJoin::expand_grids(taskCtx);

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
