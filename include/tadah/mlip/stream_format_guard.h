#ifndef TADAH_MLIP_STREAM_FORMAT_GUARD_H
#define TADAH_MLIP_STREAM_FORMAT_GUARD_H

#include <ios>
#include <ostream>

namespace tadah {
namespace mlip {

/** \brief Restores the format of a stream (flags, precision, width, fill)
 *  when it goes out of scope.
 *
 *  A writer that formats numbers its own way (std::scientific, a
 *  precision) on a stream it does not own -- std::cout above all -- holds
 *  one, so that the next writer finds the stream as it was: the settings
 *  line of a kernel copies the format of std::cout, and tadah runs several
 *  commands in one process when a program calls it so. Not copyable. */
class StreamFormatGuard {
public:
  explicit StreamFormatGuard(std::ostream &os) : os_(os), saved_(nullptr) {
    saved_.copyfmt(os);
  }
  ~StreamFormatGuard() { os_.copyfmt(saved_); }
  StreamFormatGuard(const StreamFormatGuard &) = delete;
  StreamFormatGuard &operator=(const StreamFormatGuard &) = delete;

private:
  std::ostream &os_;
  std::ios saved_;
};

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_STREAM_FORMAT_GUARD_H
