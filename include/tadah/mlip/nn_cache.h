#ifndef TADAH_MLIP_NN_CACHE_H
#define TADAH_MLIP_NN_CACHE_H

#include <tadah/core/core_types.h>
#include <tadah/mlip/nn_table.h>
#include <tadah/mlip/structure.h>

#include <vector>

namespace tadah {
namespace mlip {
namespace nn_cache {

/** \brief Skin-based neighbour-list cache state.
 *
 *  Mirrors LAMMPS's `neighbor SKIN bin` + `neigh_modify check yes`
 *  invalidation policy. The cache is caller-owned (typically a local
 *  in NativeRelaxer::relax()) so that NNFinder stays stateless and
 *  NNTable stays immutable.
 *
 *  The cache contains a snapshot of the positions and cell at the
 *  last NN build, plus the skin used and the resulting (immutable,
 *  thread-shared) NN list. `is_fresh` then computes the LAMMPS-style
 *  delta-distance check and either returns true (caller reuses
 *  `cached_nn`) or false (caller rebuilds via `NNFinder::calc(st,
 *  rcut + skin)` then calls `capture` to update the state).
 */
struct State {
    /** Atom positions (absolute, Cartesian) at the last rebuild.
     *  Empty until the first capture. */
    std::vector<tadah::core::Vec3d> xref;

    /** Cell matrix at the last rebuild. */
    tadah::core::Matrix3d cell_ref;

    /** Skin used. Caller must set this before the first build. The
     *  NN list itself is built at cutoff = rcut + skin so pairs that
     *  drift into the rcut sphere by up to skin/2 are already present. */
    double skin = 0.0;

    /** Cached NN list. Shared (immutable). Caller obtains it from
     *  `Structure::nn` after a rebuild and assigns via `capture`. */
    NNTablePtr cached_nn;
};

/** \brief Decide whether the cached NN list is still valid for the
 *         Structure's current positions and cell.
 *
 *  Returns true iff (a) the cache has been populated (xref non-empty,
 *  matching natoms), AND (b) the LAMMPS-style displacement check
 *  passes:
 *      max_i |position[i] - xref[i]|² ≤ ( max(0, skin/2 - 2·corner_shift) )²
 *  where `corner_shift` is the sum of the two largest box-corner
 *  displacements between the cached and current cell (8 corners of
 *  the [0,1]³ fractional box, mapped through cell and cell_ref).
 *
 *  On hit, assigns `st.nn = s.cached_nn` (the shared NNTablePtr) so
 *  the caller can proceed to descriptor evaluation without rebuilding.
 *  On miss, does NOT touch `st.nn`. */
bool is_fresh(Structure& st, const State& s);

/** \brief Snapshot the Structure's current state into the cache.
 *
 *  Captures positions, cell, and the current `st.nn` shared_ptr.
 *  Preserves `s.skin` (caller sets it once). The caller is expected
 *  to have just rebuilt `st.nn` via `NNFinder::calc(st, rcut + s.skin)`. */
void capture(const Structure& st, State& s);

}  // namespace nn_cache
}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_NN_CACHE_H
