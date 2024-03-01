#ifndef ST_DESCRIPTORS_DB_H
#define ST_DESCRIPTORS_DB_H

#include "../CORE/config/config.h"
#include "st_descriptors.h"
#include "structure_db.h"


/** \brief Container for StDescriptors.
 *
 *   \note
 *       Required Config keys:
 *       \ref FORCE \ref STRESS
 *       \ref INTERNAL_KEY \ref DSIZE.
 *
 */
struct StDescriptorsDB {
    std::vector<StDescriptors> st_descs;
    StDescriptorsDB() {};

    /** This constructor fully initialise this object
     *
     * Requires:
     * - StructureDB st to have all nearest neighbours calculated
     * - Config to contain keys: \ref FORCE, \ref STRESS
     *   and \ref INTERNAL_KEY \ref DSIZE
     */
    StDescriptorsDB(const StructureDB &stdb, Config &config);

    /** Return reference to the n-th StDescriptors
     * stored by this object
     */
    StDescriptors& operator()(const size_t n);

    /** Return number of StDescriptors stored */
    size_t size() const;

};
#endif
