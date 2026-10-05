#ifndef TADAH_MLIP_ENGINES_SAMPLE_WRITE_ENGINE_H
#define TADAH_MLIP_ENGINES_SAMPLE_WRITE_ENGINE_H

#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>
#include <tadah/mlip/structure_db.h>

#include <vector>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Writes a subset of a dataset (`tadah data sample`).
 *
 * Exactly one selection: every N-th structure from the first (UNIFORM),
 * N distinct structures drawn uniformly from the whole dataset and written
 * in dataset order (RANDOM), or the structures at 1-based indices (INDEX,
 * parsed strictly against the dataset size). validate() refuses anything
 * else before run() writes the output.
 */
class DataSampleEngine : public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;

private:
  StructureDB inStdb; // so we do not load it twice by run and validate
  std::vector<std::size_t> indices; // calc them once
  bool areDataLoaded = false;
  bool isIndexParsed = false;
  void loadInput(const core::Context &taskContext);
  void parseIndices(const core::Context &taskContext);
  void resetState();

};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_SAMPLE_WRITE_ENGINE_H
