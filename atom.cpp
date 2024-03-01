#include "atom.h"
#include "../CORE/periodic_table.h"
#include <iomanip>

Atom::Atom() {}

Atom::Atom(const Element &element,
        const double px, const double py, const double pz,
        const double fx, const double fy, const double fz):
    Element(element),
    position(px,py,pz),
    force(fx,fy,fz)
{}

std::ostream& operator<<(std::ostream& os, const Atom& atom)
{
    os     << (Element&)atom;
    os     << "Coordinates:  " << std::left << atom.position << std::endl;
    os     << "Force:        " << std::left << atom.force << std::endl;
    return os;
}
bool Atom::operator==(const Atom &a) const {
    double EPSILON = std::numeric_limits<double>::epsilon();
    return 
        position.isApprox(a.position, EPSILON)
        && force.isApprox(a.force, EPSILON)
        && Element::operator==(a)
        ;
}
