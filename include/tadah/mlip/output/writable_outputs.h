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
 * Nothing is opened, created or removed (a FIFO is not opened, and a
 * symbolic link is followed, not replaced): an existing output, at the end of
 * any chain of symbolic links, must not be a directory and must be writable
 * (access(W_OK)); a new one needs an existing directory that takes new files.
 * A write that fails later (a full disk) is still refused by the writer.
 * Throws std::runtime_error naming the first file that cannot be written,
 * prefixed with @p who.
 */
void require_writable_outputs(const std::vector<std::string> &paths,
                              const std::string &who);

}  // namespace mlip
}  // namespace tadah

#endif
