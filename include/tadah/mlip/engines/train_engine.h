#ifndef TADAH_MLIP_ENGINES_TRAIN_ENGINE_H
#define TADAH_MLIP_ENGINES_TRAIN_ENGINE_H

#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>
#include <tadah/mlip/training_record.h>

#include <string>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides core training functionality.
 */
class TrainEngine : public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
  /** Writes @p context as the potential file @p outfile: its training
   *  record @p record (the keys of the run that wrote it, a comment block
   *  no loader of a potential reads; nothing when it is empty), then the
   *  key lines of
   *  @p context with NUMERIC significant digits (resolved from @p context)
   *  and without the keys of training_record_keys(), EMBF0 and RHOFORM
   *  normalised (write_potential_with_record). The record is its caller's:
   *  TrainEngine::run, RefitEngine::run and refit --edit give the record of
   *  their run; a key of the list that @p context holds is not copied into
   *  it. A context with a DM_REAM block but no RHOFORM throws
   *  models::DensityFormError before the file is opened; a file that cannot
   *  be written throws std::runtime_error naming it. */
  static void dumpPotFile(const std::string &outfile,
                          const core::Context &context,
                          const TrainingRecord &record = TrainingRecord());
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_TRAIN_ENGINE_H

