#ifndef ST_DESCRIPTORS_DB_H
#define ST_DESCRIPTORS_DB_H

#include <tadah/core/context.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure_db.h>
namespace tadah {
namespace mlip {


/** \brief Container for StDescriptors.
 *
 *   \note
 *       Required tadah::core::Context keys:
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
     * - tadah::core::Context to contain keys: \ref FORCE, \ref STRESS
     *   and \ref INTERNAL_KEY \ref DSIZE
     */
  StDescriptorsDB(const StructureDB &stdb, tadah::core::Context &context);

  /** Return reference to the n-th StDescriptors
     * stored by this object
     */
  StDescriptors& operator()(const size_t n);

  /** Return number of StDescriptors stored */
  size_t size() const;

  /** Add StDescriptors to DB */
  void add(const StDescriptors &st_d);

  // Methods to enable range-based for loop
  std::vector<StDescriptors>::iterator begin();
  std::vector<StDescriptors>::iterator end();
  std::vector<StDescriptors>::const_iterator begin() const;
  std::vector<StDescriptors>::const_iterator end() const;

};
} // namespace mlip
} // namespace tadah
#endif
