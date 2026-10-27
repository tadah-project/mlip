#include <tadah/mlip/nn_cache.h>
#include <tadah/core/number_format.h>
#include <tadah/mlip/detail/cell_geometry.h>
#include <tadah/mlip/nn_finder.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace tadah {
namespace mlip {
namespace nn_cache {

using tadah::core::Vec3d;
using tadah::core::Matrix3d;

namespace {

bool same_bits(const Matrix3d &x, const Matrix3d &y)
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const double a = x(i, j), b = y(i, j);
            if (std::memcmp(&a, &b, sizeof(double)) != 0) return false;
        }
    return true;
}

}  // namespace

bool is_fresh(Structure& st, const State& s)
{
    /* Topology / first-build gate. xref empty until capture() runs. */
    if (s.xref.empty()) return false;
    if (s.xref.size() != st.natoms()) return false;
    if (!s.cached_nn) return false;

    /* The list holds every pair closer than R at the build geometry; the
     * caller needs every pair closer than rc = R - skin now, where its own
     * cutoff rcut gave R = fl(rcut + skin). That sum and the subtraction
     * each round by at most eps R, so rc is taken 4 eps R above R - skin:
     * never below the caller's rcut, even when skin > rcut. */
    const double R = s.cached_nn->cutoff;
    const double rc_exact = R - s.skin;
    if (!(rc_exact > 0.0))
        throw std::logic_error(
            "internal error: the cached neighbour list was built with "
            "cutoff " + tadah::core::format_roundtrip(R) + " A, which does "
            "not exceed the skin " + tadah::core::format_roundtrip(s.skin) +
            " A; a cached list is built at the cutoff plus the skin");
    const double rc = rc_exact + 4.0 * detail::kUnitRoundoff * R;

    /* Geometry of the current cell. A cell NNFinder would refuse is
     * stale: the rebuild then reports it. */
    double A[3][3], A0[3][3];
    for (int k = 0; k < 3; ++k)
        for (int c = 0; c < 3; ++c) {
            A[k][c]  = st.cell(k, c);
            A0[k][c] = s.cell_ref(k, c);
        }
    const detail::CellGeometry g = detail::cell_geometry(A);
    if (!(g.volume != 0.0) || !detail::is_finite_bits(g.G) ||
        !detail::is_finite_bits(g.Q))
        return false;

    /* A pair at image s has the vector v = (f_q - f_p + s) A now and
     * v0 = (f0_q - f0_p + s) A0 at the build, with f = r A^-1 from the
     * raw positions (no wrapping). Hence v0 = v M - (r_q M - r0_q)
     * + (r_p M - r0_p) with M = A^-1 A0, and
     *     |v0| <= |v| ||M||_2 + 2 e_max,   e_max = max_i |r_i M - r0_i|.
     * Every pair closer than rc now is in the list if the bound stays
     * below R. ||M||_2 <= B = min(sqrt(||M||_1 ||M||_inf), 1 + ||M - I||_F).
     * For an unchanged cell M = I and this is LAMMPS' |dr| <= skin/2. */
    // Row vectors: f_k = r . b*_k, so A^-1 has entries (A^-1)[c][k] =
    // b*_k[c], and M[i][j] = sum_k b*_k[i] A0[k][j]; (r M)_j =
    // sum_i r_i M[i][j].
    double M[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double m = 0.0;
            for (int k = 0; k < 3; ++k) m += g.bstar[k][i] * A0[k][j];
            M[i][j] = m;
        }
    double n1 = 0.0, ninf = 0.0, nF = 0.0;
    for (int i = 0; i < 3; ++i) {
        double row = 0.0, col = 0.0;
        for (int j = 0; j < 3; ++j) {
            row += std::fabs(M[i][j]);
            col += std::fabs(M[j][i]);
            const double d = M[i][j] - (i == j ? 1.0 : 0.0);
            nF += d * d;
        }
        ninf = std::max(ninf, row);
        n1 = std::max(n1, col);
    }
    const double B = std::min(std::sqrt(n1 * ninf), 1.0 + std::sqrt(nF));

    double e_max = 0.0, rho = 0.0, rho0 = 0.0;
    for (std::size_t i = 0; i < st.natoms(); ++i) {
        const auto &r = st.atoms[i].position;
        const auto &r0 = s.xref[i];
        double e2 = 0.0;
        for (int j = 0; j < 3; ++j) {
            const double rm = r[0] * M[0][j] + r[1] * M[1][j] + r[2] * M[2][j];
            const double d = rm - r0[j];
            e2 += d * d;
        }
        e_max = std::max(e_max, std::sqrt(e2));
        rho = std::max(rho, std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]));
        rho0 = std::max(rho0, std::sqrt(r0[0] * r0[0] + r0[1] * r0[1] +
                                        r0[2] * r0[2]));
    }

    /* Rounding: delta bounds a computed distance's error (a pair a
     * consumer computes below rc is truly below rc + delta); tau bounds
     * the error of e_max; the factor (1 + 32 Q eps) that of M. */
    const double eps = detail::kUnitRoundoff;
    const double delta = detail::distance_rounding_bound(g.G, rc, rho);
    const double tau = 16.0 * eps * g.Q * (B * rho + rho0);
    const double bound =
        (rc + delta) * B * (1.0 + 32.0 * g.Q * eps) + 2.0 * e_max + tau;
    if (!detail::is_finite_bits(bound) || !(bound <= R)) return false;

    /* Cache hit. The shifts stay valid; their displacements s.A depend on
     * the cell, so a changed cell gets a copy with disp recomputed (the
     * cached table is shared and immutable). */
    if (same_bits(st.cell, s.cell_ref)) {
        st.nn = s.cached_nn;
    } else {
        auto t = std::make_shared<NNTable>(*s.cached_nn);
        NNFinder().populate_disp(st, t);
        st.nn = t;
    }
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
