#ifndef TADAH_MLIP_DETAIL_NN_GUARD_H
#define TADAH_MLIP_DETAIL_NN_GUARD_H

// Internal helper, not part of the public API: a consumer of a neighbour
// list checks that the list covers the cutoff it reads pairs to. A list
// built with a smaller cutoff silently lacks pairs, so this is a
// programming error of the caller (std::logic_error).

#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace tadah {
namespace mlip {
namespace detail {

/** Throw std::logic_error unless st.nn exists and was built with a cutoff
 *  of at least @p needed (Å). @p who names the consumer in the message. */
inline void require_nn_cutoff(const Structure &st, double needed,
                              const char *who) {
  if (!st.nn)
    throw std::logic_error(std::string(who) + ": structure '" + st.label +
                           "' has no neighbour list; build it with NNFinder "
                           "first");
  if (st.nn->cutoff < needed) {
    std::ostringstream o;
    o.precision(10);
    o << who << ": the neighbour list of structure '" << st.label
      << "' was built with cutoff " << st.nn->cutoff
      << " A, below the cutoff " << needed
      << " A this model reads pairs to; rebuild it with NNFinder at a "
         "cutoff of at least "
      << needed << " A";
    throw std::logic_error(o.str());
  }
}

/** require_nn_cutoff for every structure of @p stdb, in order. Called
 *  before a parallel loop over the structures, so the error is raised
 *  outside it and names the first such structure. */
inline void require_db_nn_cutoff(const StructureDB &stdb, double needed,
                                 const char *who) {
  for (std::size_t s = 0; s < stdb.size(); ++s)
    require_nn_cutoff(stdb(s), needed, who);
}

}  // namespace detail
}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_DETAIL_NN_GUARD_H
