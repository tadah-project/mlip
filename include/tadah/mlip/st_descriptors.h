#ifndef ST_DESCRIPTORS_H
#define ST_DESCRIPTORS_H

#include <tadah/mlip/structure.h>
#include <tadah/core/config.h>
#include <tadah/core/core_types.h>

#include <vector>
namespace tadah {
namespace mlip {


/** \brief Container for a structure descriptors.
 *
 * Contains:
 * - Atomic Energy Descriptors
 * - Force Descriptors (Optional)
 * - Stress Descriptors (Optional)
 *
 * To fully initialise this object it needs to know:
 * - number of atoms in a structure (from Structure)
 * - number of nn for every atom (from Structure)
 * - dimension of the descriptor vector (tadah::core::Config)
 * - whether force and stress is being calculated (from tadah::core::Config)
 *
 *   \note
 *       Required tadah::core::Config keys:
 *       \ref FORCE \ref STRESS.
 *       \ref INTERNAL_KEY \ref DSIZE.
 *
 */
struct StDescriptors {

    /** This constructor fully initialise this object
     *
     * Requires:
     * - Structure st to have NN calculated
     * - tadah::core::Config c to contain keys: \ref DSIZE, \ref FORCE, \ref STRESS
     */
    StDescriptors(const Structure &s, const tadah::core::Config &c);

    /** Default constructor. Object is left uninitialised */
    StDescriptors();

    /** AED for all atoms
     *
     * This object will always be initialised given
     * provided Structure contains atoms.
     *
     * Dimensions:
     * [number of atoms, descriptor size]
     */
    tadah::core::aeds_type2 aeds;

    /** FD for all atoms
     *
     * Dimensions:
     * [number of atoms, number of atom NN, descriptor size, 3]
     */
    std::vector<std::vector<tadah::core::fd_type>> fd;

    /** SD for a Structure
     *
     * Dimensions:
     * [descriptor size, 6]
     */
    //sd_type sd;

    /** True if descriptors are normalised */
    bool normalised=false;

    tadah::core::rhos_type rhos;

    tadah::core::aed_type & get_aed(const size_t i);
    const tadah::core::aed_type &get_aed(const size_t i) const;
    tadah::core::rho_type& get_rho(const size_t i);

    size_t naed() const;
    size_t dim() const;

};
} // namespace mlip
} // namespace tadah
#endif
