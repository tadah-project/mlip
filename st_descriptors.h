#ifndef ST_DESCRIPTORS_H
#define ST_DESCRIPTORS_H

#include "structure.h"
#include "../CORE/config/config.h"
#include "../CORE/core_types.h"

#include <vector>


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
 * - dimension of the descriptor vector (Config)
 * - whether force and stress is being calculated (from Config)
 *
 *   \note
 *       Required Config keys:
 *       \ref FORCE \ref STRESS.
 *       \ref INTERNAL_KEY \ref DSIZE.
 *
 */
struct StDescriptors {

    /** This constructor fully initialise this object
     *
     * Requires:
     * - Structure st to have NN calculated
     * - Config c to contain keys: \ref DSIZE, \ref FORCE, \ref STRESS
     */
    StDescriptors(const Structure &s, const Config &c);

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
    aeds_type2 aeds;

    /** FD for all atoms
     *
     * Dimensions:
     * [number of atoms, number of atom NN, descriptor size, 3]
     */
    std::vector<std::vector<fd_type>> fd;

    /** SD for a Structure
     *
     * Dimensions:
     * [descriptor size, 6]
     */
    //sd_type sd;

    /** True if descriptors are normalised */
    bool normalised=false;

    rhos_type rhos;

    aed_type2 & get_aed(const size_t i);
    const aed_type2 &get_aed(const size_t i) const;
    rho_type& get_rho(const size_t i);

    size_t naed() const;
    size_t dim() const;

};
#endif
