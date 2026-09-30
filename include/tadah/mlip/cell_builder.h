#ifndef TADAH_MLIP_CELL_BUILDER_H
#define TADAH_MLIP_CELL_BUILDER_H

#include <tadah/mlip/structure.h>

#include <cstddef>
#include <string>

namespace tadah {
namespace mlip {

/** \brief Native cell-construction primitives for the typed
 *         PC_OBSERVABLE pipeline.
 *
 *  These build a Tadah ::Structure (with cell vectors, atoms, free/periodic
 *  boundaries left implicit — periodic is the default for all lattice
 *  builders; the isolated-atom builder uses a large vacuum-padded box that
 *  is effectively non-periodic at the cutoff scale).
 *
 *  None of these set up the nearest-neighbour table; callers must invoke
 *  NNFinder::calc(st, rcut) before passing the Structure to descriptor
 *  evaluation or relaxation.
 *
 *  MVP scope (PC_OBSERVABLE cohesive_energy): `bcc` and `isolated_atom`
 *  ship. Other lattice variants (`fcc`, `hcp`, vacancy / slab insertions)
 *  are declared but throw `std::logic_error` until their observable
 *  consumer lands.
 */
namespace cells {

/** Conventional 2-atom BCC unit cell.
 *  - cell vectors = a0 * I (orthorhombic, cubic)
 *  - atoms at (0,0,0) and (a0/2, a0/2, a0/2)
 *  - both atoms are of `species`
 *
 *  Compose with `StructureTransformation::replicate(st, nx, ny, nz)`
 *  (structure_transformations.h) for supercells.
 */
Structure bcc(const std::string& species, double a0);

/** Single atom of `species` at the centre of a cubic box of edge
 *  `box_size` (default 20 Å, i.e. ≥3× any reasonable rcut → effectively
 *  isolated). Cell vectors = box_size * I; atom at (box_size/2, ...).
 *
 *  For the cohesive-energy observable this gives `E_iso` once descriptors
 *  + a `pair`-only evaluation are run on it (it has no neighbours within
 *  any sensible rcut so its descriptor reduces to a constant per-species
 *  intercept).
 */
Structure isolated_atom(const std::string& species, double box_size = 20.0);

/** Stubs for the follow-up observables. Throw std::logic_error so accidental
 *  use during the MVP triggers loud failure rather than silent wrong cell. */
Structure fcc(const std::string& species, double a0);
Structure hcp(const std::string& species, double a, double c_over_a = 1.633);

}  // namespace cells
}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_CELL_BUILDER_H
