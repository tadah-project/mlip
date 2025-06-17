#ifndef STRUCTURE_PROPERTIES_H
#define STRUCTURE_PROPERTIES_H

#include <tadah/mlip/structure.h>
namespace tadah {
namespace mlip {

/**
 * @class StructureProperties
 * @brief Collects read-only property computations for a Structure, such as:
 *        - Volume (getVolume)
 *        - Density (getDensity)
 *        - Center of mass (getCentreOfMass)
 */
class StructureProperties
{
public:
  /**
   * @brief Computes the volume of the Structure from its lattice vectors.
   *
   * Interprets the columns of st.cell as vectors a1, a2, a3 and
   * computes the scalar triple product: volume = a1 · (a2 × a3).
   * Returns volume in Å^3.
   */
  static double getVolume(const Structure &st);

  /**
   * @brief Computes the density of the Structure in g/cm^3.
   * 
   * Procedure:
   *   - Accumulate atomic mass from tadah::core::PeriodicTable::get_mass(atom.Z) in amu.
   *   - Convert volume from Å^3 to cm^3 (1 Å^3 = 1e-24 cm^3).
   *   - Multiply total mass (amu) by 1.66053906660e-24 g/amu.
   *   - Divide by volume in cm^3 => density.
   * If the volume is non-positive, returns 0.0 or throws an exception (user’s choice).
   */
  static double getDensity(const Structure &st);

  /**
   * @brief Computes the center of mass in Cartesian coordinates.
   *
   * Sums (mass_i * position_i) for each atom, divides by total mass.
   * If no atoms exist, returns (0,0,0).
   */
  static tadah::core::Vec3d getCentreOfMass(const Structure &st);

  /**
   * @brief Computes the *geometric center* (centroid) of all atoms.
   * 
   * If no atoms exist, returns (0,0,0).
   */
  static tadah::core::Vec3d getGeometricCenter(const Structure &st);
};

} // namespace mlip
} // namespace tadah
#endif // STRUCTURE_PROPERTIES_H
