#ifndef TADAH_MLIP_ENGINES_TRAIN_ENGINE_H
#define TADAH_MLIP_ENGINES_TRAIN_ENGINE_H

#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides core training functionality.
 *
 * Delegates model training to the underlying Trainer.
 *
 * @param context The contexturation object.
 * @return int EXIT_SUCCESS or EXIT_FAILURE.
 */
class TrainEngine : public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
private:
  void dumpPotFile(const std::string &outfile, const core::Context &context);
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_TRAIN_ENGINE_H

