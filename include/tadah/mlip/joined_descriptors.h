#ifndef TADAH_MLIP_JOINED_DESCRIPTORS_H
#define TADAH_MLIP_JOINED_DESCRIPTORS_H

// The descriptors of a joined TYPE2B or TYPEMB line (D2_mJoin, DM_mJoin),
// each with its own context, for the code that reads a descriptor's
// parameters from a configuration: tadah analysis suggest_bounds and the
// pre-flight audit (tadah train and tadah hpo with AUDIT, tadah analysis
// check_bounds).
//
// A joined line splits its RCUT and RCTYPE values (one per descriptor) and
// its list values (CGRID*, SGRID*, ...) among its descriptors, in the order
// of the line (D_mJoin::parse_context). A descriptor's parameters are read
// from its own share; an OPTIM line numbers a value by its place in the
// whole key, after the shares of the descriptors before it.

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include <tadah/core/context.h>

namespace tadah {
namespace mlip {

/** \brief One descriptor of a joined TYPE2B or TYPEMB line. */
struct JoinedDescriptor {
  /** The descriptor's label (D2_ZBL, DM_EAD, ...). */
  std::string label;
  /** Its 0-based place among the descriptors of the line. */
  std::size_t position = 0;
  /** Its own context: the configuration with its TYPE line, its RCTYPE
   *  and RCUT value and its share of each list key
   *  (D_mJoin::parse_context()). */
  tadah::core::Context context;
  /** For each key the line splits (its RCUT and RCTYPE keys and the list
   *  keys of its descriptors), the number of the key's values that come
   *  before the descriptor's own: those of the descriptors before it. */
  std::map<std::string, std::size_t> first_entries;
};

/** \brief Whether the @p type_key line (TYPE2B or TYPEMB) of @p c joins
 *  descriptors: its first value names D2_mJoin or DM_mJoin, and, for
 *  TYPEMB, no DESCRIPTOR block configures the line. */
bool is_joined_line(const tadah::core::Context &c,
                    const std::string &type_key);

/** \brief The descriptors of the joined @p type_key line of @p c, in the
 *  order of the line; empty when the line does not join descriptors
 *  (is_joined_line()). A list value given by a grid generator (LIN, GEOM)
 *  counts as the values it makes.
 *  @throws what D_mJoin::parse_context() throws for a line it refuses. */
std::vector<JoinedDescriptor> joined_descriptors(
    const tadah::core::Context &c, const std::string &type_key);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_JOINED_DESCRIPTORS_H
