#include <tadah/mlip/cell_builder.h>

#include <tadah/core/element.h>
#include <tadah/mlip/atom.h>

#include <stdexcept>

namespace tadah {
namespace mlip {
namespace cells {

Structure bcc(const std::string& species, double a0)
{
    if (a0 <= 0.0)
        throw std::invalid_argument("cells::bcc: a0 must be > 0");

    Structure st;
    /* Conventional cubic cell: matches LAMMPS `lattice bcc a0`
     * + `region box block 0 1 0 1 0 1 units lattice` (2 atoms). */
    st.cell.load(a0,   0.0,  0.0,
                 0.0,  a0,   0.0,
                 0.0,  0.0,  a0);

    const tadah::core::Element el(species);
    st.add_atom(Atom(el, 0.0,      0.0,      0.0));
    st.add_atom(Atom(el, 0.5 * a0, 0.5 * a0, 0.5 * a0));
    return st;
}

Structure isolated_atom(const std::string& species, double box_size)
{
    if (box_size <= 0.0)
        throw std::invalid_argument("cells::isolated_atom: box_size must be > 0");

    Structure st;
    st.cell.load(box_size, 0.0,      0.0,
                 0.0,      box_size, 0.0,
                 0.0,      0.0,      box_size);

    const tadah::core::Element el(species);
    /* Atom at box centre. The exact position is immaterial under
     * periodic boundaries as long as no neighbours land within rcut
     * after the periodic image expansion — box_size = 20 Å clears any
     * reasonable rcut (≤ ~6 Å) by ≥3×. */
    st.add_atom(Atom(el, 0.5 * box_size, 0.5 * box_size, 0.5 * box_size));
    return st;
}

Structure fcc(const std::string&, double)
{
    throw std::logic_error(
        "cells::fcc: not implemented in MVP. "
        "First observable that needs FCC (e.g. phase_transition_energy) "
        "must land its consumer before this stub may be filled in.");
}

Structure hcp(const std::string&, double, double)
{
    throw std::logic_error(
        "cells::hcp: not implemented in MVP. "
        "First observable that needs HCP (e.g. phase_transition_energy) "
        "must land its consumer before this stub may be filled in.");
}

}  // namespace cells
}  // namespace mlip
}  // namespace tadah
