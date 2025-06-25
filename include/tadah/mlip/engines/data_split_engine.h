#ifndef TADAH_MLIP_ENGINES_SPLIT_WRITE_ENGINE_H
#define TADAH_MLIP_ENGINES_SPLIT_WRITE_ENGINE_H

#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>
#include <tadah/mlip/structure_db.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides dataset merging functionality.
 */
class DataSplitEngine : public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
private:
  StructureDB inStdb; // so we do not load it twice by run and validate
  std::vector<std::size_t> sizes; // calc them once
  bool areDataLoaded = false;
  void loadInput(const core::Context &taskContext);
  void calculateSizes(const core::Context &taskContext);
  bool areSizesCalculated = false;
  void resetState();
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_SPLIT_WRITE_ENGINE_H
