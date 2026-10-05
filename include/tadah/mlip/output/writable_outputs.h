#ifndef TADAH_MLIP_WRITABLE_OUTPUTS_H
#define TADAH_MLIP_WRITABLE_OUTPUTS_H

#include <string>
#include <vector>

namespace tadah {
namespace mlip {

/**
 * Checks that every file in @p paths can be written, for a command that
 * writes several files, before it writes any: a file that cannot be written
 * must not leave the others written (an --append rerun would append them
 * twice).
 *
 * An existing file is opened for appending, which leaves it unchanged; a new
 * one is created and removed again. Throws std::runtime_error naming the
 * first file that cannot be written, prefixed with @p who.
 */
void require_writable_outputs(const std::vector<std::string> &paths,
                              const std::string &who);

}  // namespace mlip
}  // namespace tadah

#endif
