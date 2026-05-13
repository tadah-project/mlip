#include <tadah/mlip/nn_cache.h>

#include <algorithm>
#include <cmath>

namespace tadah {
namespace mlip {
namespace nn_cache {

using tadah::core::Vec3d;
using tadah::core::Matrix3d;

bool is_fresh(Structure& st, const State& s)
{
    /* Topology / first-build gate. xref empty until capture() runs. */
    if (s.xref.empty()) return false;
    if (s.xref.size() != st.natoms()) return false;
    if (!s.cached_nn) return false;

    /* Max single-atom squared displacement since last build. */
    double max_disp_sq = 0.0;
    for (std::size_t i = 0; i < st.natoms(); ++i) {
        const double dx = st.atoms[i].position[0] - s.xref[i][0];
        const double dy = st.atoms[i].position[1] - s.xref[i][1];
        const double dz = st.atoms[i].position[2] - s.xref[i][2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > max_disp_sq) max_disp_sq = r2;
    }

    /* Box-corner displacements between cached and current cell. The 8
     * corners of the [0,1]³ fractional box map through cell to absolute
     * coordinates; their displacement bounds how much pairs near
     * different corners can have approached each other since the cache
     * was captured. Mirrors LAMMPS Neighbor::check_distance corner shrink. */
    double corner_shift[8];
    for (int n = 0; n < 8; ++n) {
        const double a = (n & 1) ? 1.0 : 0.0;
        const double b = (n & 2) ? 1.0 : 0.0;
        const double c = (n & 4) ? 1.0 : 0.0;
        const double dx = a * (st.cell(0, 0) - s.cell_ref(0, 0))
                        + b * (st.cell(1, 0) - s.cell_ref(1, 0))
                        + c * (st.cell(2, 0) - s.cell_ref(2, 0));
        const double dy = a * (st.cell(0, 1) - s.cell_ref(0, 1))
                        + b * (st.cell(1, 1) - s.cell_ref(1, 1))
                        + c * (st.cell(2, 1) - s.cell_ref(2, 1));
        const double dz = a * (st.cell(0, 2) - s.cell_ref(0, 2))
                        + b * (st.cell(1, 2) - s.cell_ref(1, 2))
                        + c * (st.cell(2, 2) - s.cell_ref(2, 2));
        corner_shift[n] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    /* Two largest corner shifts. */
    double d1 = 0.0, d2 = 0.0;
    for (int n = 0; n < 8; ++n) {
        if (corner_shift[n] > d1)      { d2 = d1; d1 = corner_shift[n]; }
        else if (corner_shift[n] > d2) {           d2 = corner_shift[n]; }
    }

    /* Effective trigger distance: shrink skin/2 by the corner contribution.
     * If two corners moved more than skin/2 combined, the cache is
     * immediately stale regardless of per-atom displacement. */
    const double delta    = std::max(0.0, 0.5 * s.skin - (d1 + d2));
    const double deltasq  = delta * delta;

    if (max_disp_sq > deltasq) return false;

    /* Cache hit: hand the cached NN back to the caller via the
     * Structure's shared_ptr slot. */
    st.nn = s.cached_nn;
    return true;
}

void capture(const Structure& st, State& s)
{
    s.xref.resize(st.natoms());
    for (std::size_t i = 0; i < st.natoms(); ++i) {
        s.xref[i] = Vec3d{st.atoms[i].position[0],
                          st.atoms[i].position[1],
                          st.atoms[i].position[2]};
    }
    s.cell_ref  = st.cell;
    s.cached_nn = st.nn;
}

}  // namespace nn_cache
}  // namespace mlip
}  // namespace tadah
