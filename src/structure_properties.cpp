#include <tadah/mlip/structure_properties.h>
#include <tadah/core/periodic_table.h>
#include <stdexcept>
#include <cmath>

double StructureProperties::getVolume(const Structure &st)
{
  // Columns of st.cell => a1, a2, a3
  Vec3d a1(st.cell(0,0), st.cell(1,0), st.cell(2,0));
  Vec3d a2(st.cell(0,1), st.cell(1,1), st.cell(2,1));
  Vec3d a3(st.cell(0,2), st.cell(1,2), st.cell(2,2));

  double volume = a1 * (a2.cross(a3));  // scalar triple product
  return volume; // [Å^3]
}

double StructureProperties::getDensity(const Structure &st)
{
  double volumeA3 = getVolume(st);
  if (volumeA3 <= 0.0) {
    // Negative or zero volume => invalid
    return 0.0; // or throw std::runtime_error("Volume <= 0.0, invalid cell.")
  }

  // Sum atomic mass in amu
  double totalMassAmu = 0.0;
  for (const auto &atom : st.atoms) {
    totalMassAmu += PeriodicTable::get_mass(atom.Z);
  }
  // Convert Å^3 => cm^3
  double volumeCm3 = volumeA3 * 1.0e-24;

  // Convert amu => grams
  double totalMassGrams = totalMassAmu * 1.66053906660e-24;

  // density = mass / volume
  double rho = totalMassGrams / volumeCm3; // [g/cm^3]
  return rho;
}

Vec3d StructureProperties::getCentreOfMass(const Structure &st)
{
  if (st.atoms.empty()) {
    return Vec3d(0.0, 0.0, 0.0);
  }
  double totalMass = 0.0;
  Vec3d com(0.0, 0.0, 0.0);

  for (const auto &atom : st.atoms) {
    double mass = PeriodicTable::get_mass(atom.Z); // amu
    com += atom.position * mass; 
    totalMass += mass;
  }
  if (totalMass > 1e-14) {
    com /= totalMass;
  }
  return com;
}

Vec3d StructureProperties::getGeometricCenter(const Structure &st)
{
  if (st.atoms.empty()) {
    return Vec3d(0.0, 0.0, 0.0);
  }
  Vec3d c(0.0, 0.0, 0.0);
  for (auto &atom : st.atoms) {
    c += atom.position;
  }
  c /= (double)st.atoms.size();
  return c;
}
