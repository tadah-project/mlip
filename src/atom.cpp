#include <tadah/mlip/atom.h>
#include <tadah/mlip/stream_format_guard.h>
#include <tadah/core/periodic_table.h>
#include <iomanip>
namespace tadah {
namespace mlip {

Atom::Atom() {}

Atom::Atom(const tadah::core::Element &element,
           const double px, const double py, const double pz,
           const double fx, const double fy, const double fz):
  tadah::core::Element(element),
  position(px,py,pz),
  force(fx,fy,fz)
{}
Atom::Atom(const tadah::core::Element &element,
           const double px, const double py, const double pz):
  tadah::core::Element(element),
  position(px,py,pz),
  force(0,0,0)
{}

std::ostream& operator<<(std::ostream& os, const Atom& atom)
{
  const StreamFormatGuard keep(os);
  os     << (tadah::core::Element&)atom;
  os     << "Coordinates:  " << std::left << atom.position << std::endl;
  os     << "Force:        " << std::left << atom.force << std::endl;
  return os;
}
bool Atom::operator==(const Atom &a) const {
  double EPSILON = std::numeric_limits<double>::epsilon();
  return 
  position.isApprox(a.position, EPSILON)
  && force.isApprox(a.force, EPSILON)
  && tadah::core::Element::operator==(a)
  ;
}
bool Atom::is_the_same(const Atom &a, double thr) const {
  return 
  position.isApprox(a.position, thr)
  && tadah::core::Element::operator==(a)
  ;
}
} // namespace mlip
} // namespace tadah
