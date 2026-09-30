#ifndef TADAH_MLIP_DETAIL_STRUCTURE_ERRORS_H
#define TADAH_MLIP_DETAIL_STRUCTURE_ERRORS_H

// Internal helper, not part of the public API: carry an exception out of
// an OpenMP loop over the structures of a dataset. An exception may not
// leave an OpenMP structured block (it would call std::terminate), so each
// iteration stores its own in a slot of its index (no lock), and after
// the loop the lowest-numbered failure is rethrown with the structure
// named.

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>

namespace tadah {
namespace mlip {
namespace detail {

/** Rethrow @p e (from structure @p i of @p n, 0-based @p i) with the
 *  message prefixed "<who>: structure <i+1> of <n> ('<label>'): " (a
 *  leading "<who>: " of the original message is not repeated). An
 *  std::invalid_argument stays one, an std::logic_error stays one, any
 *  other std::exception becomes an std::runtime_error; anything else is
 *  rethrown unchanged. */
[[noreturn]] inline void rethrow_for_structure(std::exception_ptr e,
                                               std::size_t i, std::size_t n,
                                               const std::string &label,
                                               const std::string &who) {
  const std::string lead = who + ": ";
  const std::string prefix = lead + "structure " + std::to_string(i + 1) +
                             " of " + std::to_string(n) + " ('" + label +
                             "'): ";
  auto message = [&](const std::exception &x) {
    std::string m = x.what();
    if (m.compare(0, lead.size(), lead) == 0) m.erase(0, lead.size());
    return prefix + m;
  };
  try {
    std::rethrow_exception(e);
  } catch (const std::invalid_argument &x) {
    throw std::invalid_argument(message(x));
  } catch (const std::logic_error &x) {
    throw std::logic_error(message(x));
  } catch (const std::exception &x) {
    throw std::runtime_error(message(x));
  }
}

}  // namespace detail
}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_DETAIL_STRUCTURE_ERRORS_H
