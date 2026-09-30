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
 *  Like LAMMPS's `neighbor SKIN bin` + `neigh_modify check yes`: a list
 *  built with a skin is reused until it may miss a pair. The cache is
 *  caller-owned (typically a local in NativeRelaxer::relax()) so that
 *  NNFinder stays stateless and NNTable stays immutable.
 *
 *  The cache contains a snapshot of the positions and cell at the
 *  last NN build, plus the skin used and the resulting (immutable,
 *  thread-shared) NN list. `is_fresh` decides whether the list still
 *  holds every pair closer than rcut = cutoff − skin and either returns
 *  true (the caller uses `st.nn`) or false (caller rebuilds via
 *  `NNFinder::calc(st, rcut + skin)` then calls `capture` to update the
 *  state).
 */
struct State {
    /** Atom positions (absolute, Cartesian) at the last rebuild.
     *  Empty until the first capture. */
    std::vector<tadah::core::Vec3d> xref;

    /** Cell matrix at the last rebuild. */
    tadah::core::Matrix3d cell_ref;

    /** Skin used. Caller must set this before the first build. The
     *  NN list itself is built at cutoff = rcut + skin (NNFinder), so
     *  pairs that come within rcut as the atoms and the cell move are
     *  already present. */
    double skin = 0.0;

    /** Cached NN list. Shared (immutable). Caller obtains it from
     *  `Structure::nn` after a rebuild and assigns via `capture`. */
    NNTablePtr cached_nn;
};

/** \brief Decide whether the cached NN list is still valid for the
 *         Structure's current positions and cell.
 *
 *  The list was built by NNFinder at cutoff R = cached_nn->cutoff for the
 *  cell A0 (rows) and positions r0; the caller needs every pair closer
 *  than rc = R − skin at the current cell A and positions r. A pair's
 *  vector v now and v0 at the build satisfy
 *      |v0| <= |v| ||M||_2 + 2 e_max,   M = A^-1 A0,
 *      e_max = max_i |r_i M − r0_i|
 *  (row vectors, positions as stored, no wrapping), so the list is fresh
 *  iff
 *      (rc + delta) B + 2 e_max + tau <= R,
 *      B = min(sqrt(||M||_1 ||M||_inf), 1 + ||M − I||_F) >= ||M||_2,
 *  with delta and tau rounding bounds (about 1e-13 relative). This holds
 *  for any cell, any number of images and unwrapped atoms. For an
 *  unchanged cell (M = I) it is LAMMPS' test: no atom moved more than
 *  skin/2. Uniform expansion keeps a list fresh (B < 1); a compression,
 *  shear or thin cell costs what it moves the images.
 *
 *  Returns false when the cache is empty, the atom count changed, or the
 *  current cell is singular (the rebuild then reports it). Throws
 *  std::logic_error when rc <= 0 (a list not built at rcut + skin).
 *
 *  On a hit `st.nn` becomes the cached list with the image displacements
 *  of the current cell: `s.cached_nn` itself while the cell is bitwise
 *  unchanged, otherwise a copy with NNTable::disp recomputed (the cached
 *  table is shared and immutable). It holds every pair closer than rc;
 *  its NNTable::cutoff stays the build cutoff R.
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
