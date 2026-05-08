#ifndef TADAH_MLIP_ENGINES_ANALYSIS_DATASET_STATS_ENGINE_H
#define TADAH_MLIP_ENGINES_ANALYSIS_DATASET_STATS_ENGINE_H

#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Compute and persist dataset statistics for use by the
 * HPO bound-validation pipeline.
 *
 * Reads DBFILEs (and optionally individual STRUCTUREs) from the task
 * context, computes per-element-pair distance percentiles + KDE peaks,
 * neighbour-count percentiles per element, energy/force scales, and a
 * material-class label. Writes the result to a text file consumable by
 * `tadah hpo --suggest-bounds` and the pre-flight audit (Phases 4/5).
 */
class AnalysisDatasetStatsEngine: public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif  // TADAH_MLIP_ENGINES_ANALYSIS_DATASET_STATS_ENGINE_H
