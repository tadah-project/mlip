#ifndef TADAH_MLIP_CASTEP_SPECIES_H
#define TADAH_MLIP_CASTEP_SPECIES_H

#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>

#include <stdexcept>
#include <string>

namespace tadah {
namespace mlip {

/**
 * @brief The element of a CASTEP species label.
 *
 * CASTEP accepts a custom species: an element symbol with a tag after ':'
 * ('Fe:up', 'Si:1', for spin labels or isotopes). The element is the part
 * before ':'; the tag is dropped, since a Tadah atom has no field for it.
 * Used by the .castep, .md, .geom and .cell readers.
 *
 * @throws std::runtime_error naming the whole label when the part before
 *         ':' is not an element symbol.
 */
inline tadah::core::Element castep_species_element(const std::string &label) {
  const std::string symbol = label.substr(0, label.find(':'));
  try {
    return tadah::core::PeriodicTable::find_by_symbol(symbol);
  } catch (const std::exception &) {
    throw std::runtime_error("CASTEP species label '" + label +
                             "' does not start with an element symbol");
  }
}

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_CASTEP_SPECIES_H
