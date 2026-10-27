#ifndef TADAH_MLIP_JOINED_DESCRIPTORS_H
#define TADAH_MLIP_JOINED_DESCRIPTORS_H

// The descriptors of a joined TYPE2B or TYPEMB line (D2_mJoin, DM_mJoin),
// or of a TYPEMB key that places descriptors among DESCRIPTOR blocks, each
// with its own context, for the code that reads a descriptor's
// parameters from a configuration: tadah analysis suggest_bounds and the
// pre-flight audit (tadah train and tadah hpo with AUDIT, tadah analysis
// check_bounds).
//
// A joined line splits its RCUT and RCTYPE values (one per descriptor) and
// its list values (CGRID*, SGRID*, ...) among its descriptors, in the order
// of the line (D_mJoin::parse_context). A descriptor's parameters are read
// from its own share; an OPTIM line numbers a value by its place in the
// whole key, after the shares of the descriptors before it; a value of a
// descriptor's TYPE line by the numbers of the whole line before it.

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
   *  before the descriptor's own: those of the descriptors before it; and,
   *  for the line's TYPE2B or TYPEMB key, the number of the numbers among
   *  the values of the lines before its own, as an OPTIM line numbers the
   *  values of a key (a descriptor name and an element are not numbers). */
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

/** \brief The descriptors of the TYPEMB key of @p c that DESCRIPTOR blocks
 *  configure, other than the blocks: the descriptors such as DM_EAD that
 *  the key places among blocks (TYPEMB DM_EAD 0 1 1 Ta Ta then TYPEMB @ta,
 *  with or without DM_mJoin), in the order of the key, each with its own
 *  context as the model builds it (models::embed_contexts(): its TYPEMB
 *  line, its RCTYPEMB and RCUTMB value and its share of each list key) and
 *  the first entries of the keys they split, as joined_descriptors() gives
 *  them (a block takes no value of those keys); @c position is the place
 *  among them. Empty when no DESCRIPTOR block configures @p c, or its
 *  TYPEMB key places no such descriptor.
 *  @throws what models::embed_contexts() throws for a key it refuses. */
std::vector<JoinedDescriptor> block_line_descriptors(
    const tadah::core::Context &c);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_JOINED_DESCRIPTORS_H
