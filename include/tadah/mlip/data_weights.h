#ifndef TADAH_MLIP_DATA_WEIGHTS_H
#define TADAH_MLIP_DATA_WEIGHTS_H

// The weights of the training data a fit reads: each structure's eweight,
// fweight and sweight (the line before its energy in a dataset file), and
// EWEIGHT, FWEIGHT and SWEIGHT of the configuration. A weight is a finite
// number >= 0; 0 leaves the structure's energy (forces, stress) out of the
// fit, and of the metrics and data statistics that read it.

#include <tadah/core/context.h>
#include <tadah/mlip/structure_db.h>

#include <cstddef>
#include <string>

namespace tadah {
namespace mlip {

/** "structure 2 of 3 ('cell 1') of dataset 1 (train.tadah)": structure
 *  @p s (0-based) of @p stdb, numbered 1-based among all its structures,
 *  with its label when it has one and the dataset that holds it (the
 *  dataset ranges of stdb.dbidx; the file of @p context's DBFILE when it
 *  lists one per dataset). Without dataset ranges, no dataset is named. */
std::string structure_in_data(const StructureDB &stdb, std::size_t s,
                              const tadah::core::Context *context = nullptr);

/** Refuses a weight of the training data that is not a finite number
 *  >= 0, before a fit reads it: each structure's eweight, fweight and
 *  sweight (the first such structure, named with structure_in_data()), then
 *  EWEIGHT, FWEIGHT and SWEIGHT of @p context. A negative weight is no
 *  weight: the fit multiplies its rows by the square root, so a weight
 *  below 0 is not a number, and two of them cancel. Called first by every
 *  fit of M_BLR and M_KRR (their train methods, before anything else reads
 *  the weights; implementation 2 of M_KRR included), by
 *  require_intercepts_determined (which tadah train and tadah hpo call
 *  before a fit) and by DesignMatrix::build of a fit.
 *  @throws std::runtime_error naming the weight, its value and the
 *  structure or key. */
void require_valid_weights(const StructureDB &stdb,
                           const tadah::core::Context &context);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_DATA_WEIGHTS_H
