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

/** The refusal of M_KRR implementation 2 (MODEL M_KRR <kernel> 2) for a
 *  setting it would ignore: "MODEL M_KRR <kernel> 2 fits the energies per
 *  atom of whole structures, without forces, stresses or weights: <what>
 *  would be ignored. Implementation 1 (MODEL M_KRR <kernel> 1) <fits>;
 *  implementation 2 fits <instead>." */
std::string krr_implementation2_ignores(const std::string &kernel,
                                        const std::string &what,
                                        const std::string &fits,
                                        const std::string &instead);

/** Refuses what M_KRR implementation 2 of @p context (MODEL M_KRR <kernel>
 *  2) would ignore, before it fits: it solves (K + LAMBDA I) w = t on the
 *  mean descriptors of whole structures and their energies per atom, so no
 *  force, stress or weight of the data enters it. FORCE true, STRESS true,
 *  EWEIGHT, FWEIGHT or SWEIGHT other than 1, a WDBFILE value other than 1,
 *  WDBFILE_AUTO other than 0, EWEIGHT_TEMP (any value: it reweights the
 *  energies), and a structure of @p stdb whose eweight is not 1 (its energy
 *  weighted, or left out with 0) are refused, each naming implementation
 *  1, which fits them (krr_implementation2_ignores()). A structure's
 *  fweight and sweight weight nothing here, as in implementation 1 with
 *  FORCE and STRESS false: a valid one (finite, >= 0) is accepted, and the
 *  fit refuses another before this (require_valid_weights), as every fit
 *  does. M_KRR::train2 calls it; tadah hpo calls it when a run starts.
 *  @throws std::runtime_error naming the key and its values (or the
 *  structure and its eweight) and MODEL M_KRR <kernel> 1. */
void refuse_what_krr_implementation2_ignores(
    const tadah::core::Context &context, const StructureDB &stdb);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_DATA_WEIGHTS_H
