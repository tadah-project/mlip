#ifndef TADAH_MLIP_ENGINES_BALANCE_WRITE_ENGINE_H
#define TADAH_MLIP_ENGINES_BALANCE_WRITE_ENGINE_H

#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief tadah data balance: shifts the energies of datasets by a
 * reference energy per element (ESHIFT), one output file per input file or,
 * with MERGE, one for all; the priorities of the structures are written as
 * they are.
 */
class DataBalanceEngine : public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_BALANCE_WRITE_ENGINE_H
