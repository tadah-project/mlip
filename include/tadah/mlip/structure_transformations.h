#ifndef STRUCTURE_TRANSFORMATION_H
#define STRUCTURE_TRANSFORMATION_H

#include <tadah/mlip/structure.h>
namespace tadah {
namespace mlip {

/**
 * @class StructureTransformation
 * @brief Methods that modify a Structure: scaling, shearing, rotating, replicating, etc.
 *        The lattice vectors are the rows of Structure::cell and move with the
 *        atoms (a_i' = M a_i, cell' = cell * M^T). For an orthogonal M (a
 *        rotation or a reflection) the transform is a rigid motion: forces
 *        move as F' = M F and the stress as S' = M * S * M^T, exactly. A
 *        shear or a scaling changes the structure itself; its stored energy,
 *        forces and stress no longer belong to it (shear applies
 *        S' = M * S * M^T, scale multiplies the stress by factor^2).
 *
 * New additions:
 *   - scaleVolumeToDensity: resizes the cell to achieve a target density.
 *   - addAtom / removeAtom: insert or remove atoms from st.
 *   - shakeAtoms: random displacement to mimic thermal agitation.
 *   - remapToCell: ensures all atoms lie in the principal box.
 */
class StructureTransformation
{
public:
  /**
   * Scales cell vectors and atomic positions by a factor > 0.
   * Virial stress => S' = factor^2 * S (common approach for isotropic scaling).
   */
  static void scale(Structure &st, double factor);

  /**
   * Uniform length rescale: the physical effect of viewing the same configuration
   * at a different length scale. Suitable for the LSCALE feature, where DFT data
   * is reinterpreted at the experimental scale.
   *
   *   cell    -> s * cell
   *   pos     -> s * pos
   *   energy  -> energy        (invariant)
   *   force   -> force / s     (chain rule on E(r))
   *   stress  -> stress        (Tadah stores the virial V*sigma_Cauchy with units
   *                             of energy; under V -> s^3 V and sigma_Cauchy -> sigma/s^3
   *                             the product is invariant)
   *
   * Throws std::invalid_argument if s <= 0, std::runtime_error if the resulting
   * cell is near-singular.
   */
  static void rescaleLength(Structure &st, double s);

  /**
   * Scales the cell to achieve a target volume [Å^3].
   * factor = cbrt(targetVolume / currentVolume).
   * Then calls scale(st, factor).
   */
  static void scaleToVolume(Structure &st, double targetVolume);

  /**
   * Scales the cell to achieve a target density [g/cm^3].
   *
   * 1) Compute total mass (amu -> grams).  
   * 2) volume = mass / density => in cm^3 => convert to Å^3.  
   * 3) scaleToVolume(...).
   */
  static void scaleVolumeToDensity(Structure &st, double targetDensity);

  /**
   * Rotates cell and atoms by a 3x3 matrix R. Every vector moves as
   * v' = R v; the lattice vectors are the rows of st.cell, so
   *   cell'  = cell * R^T   (a_i' = R a_i),
   *   pos'   = R * pos,
   *   force' = R * force,
   *   S'     = R * S * R^T,
   * and the atoms keep their fractional coordinates.
   */
  static void rotate(Structure &st, const tadah::core::Matrix3d &R);

  /**
   * Shears cell and atoms by shearMatrix S (lattice vectors are the rows
   * of st.cell):
   *   cell' = cell * S^T   (a_i' = S a_i),
   *   pos'  = S * pos,
   *   S'    = S * stress * S^T,
   * so the atoms keep their fractional coordinates.
   */
  static void shear(Structure &st, const tadah::core::Matrix3d &shearMatrix);

  /**
   * Replicates the structure nx, ny, nz times along the lattice vectors
   * a0, a1, a2 (the rows of st.cell): the supercell's rows are nx a0,
   * ny a1, nz a2, and copy (ix, iy, iz) of each atom sits at
   * r + ix a0 + iy a1 + iz a2 (loop order ix, iy, iz, then atoms).
   * The energy is multiplied by nx*ny*nz (it is extensive); the stress
   * (per volume) is unchanged and every copy keeps its atom's force. The
   * neighbour list is cleared. Throws std::invalid_argument if a factor
   * is below 1.
   */
  static void replicate(Structure &st, int nx, int ny, int nz);

  /**
   * Shifts all atomic positions by shiftVec. Cell and stress remain unchanged.
   */
  static void shift(Structure &st, const tadah::core::Vec3d &shiftVec);

  /**
   * Inversion through the origin: pos' = -pos and force' = -force. The
   * stress (S' = M S M^T with M = -I) and the cell (-a_i span the same
   * lattice) are unchanged.
   */
  static void mirrorOrigin(Structure &st);

  /**
   * Mirrors about the plane normal to normalVec (through the origin) with
   * M = I - 2 n_hat n_hat^T: positions and forces v' = M v, the lattice
   * vectors too (cell' = cell * M^T, so fractional coordinates stay; the
   * reflected basis has the opposite handedness), and S' = M*S*M^T.
   */
  static void mirrorPlane(Structure &st, const tadah::core::Vec3d &normalVec);

  /**
   * Mirrors about a plane with Miller indices (h,k,l) through origin.
   * The plane normal is computed from the reciprocal lattice; the
   * reflection is applied as in mirrorPlane.
   */
  static void mirrorHkl(Structure &st, int h, int k, int l);

  /**
   * Centers atoms so that either the geometric center or center of mass 
   * is placed at the origin. If shiftCell is true, also shift the cell origin 
   * (typical usage might not require that).
   */
  static void center(Structure &st, bool useCenterOfMass = false, bool shiftCell = false);

  /**
   * Remaps each atom into the principal unit cell by fractional wrapping.
   */
  static void remapToCell(Structure &st);

  /**
   * Adds an atom to the structure.
   * If useRandomPosition is true, places it randomly inside the cell.
   * If false, uses the provided pos argument.
   * Checks distance from all existing atoms. If below minDistance, prints warning.
   */
  static void addAtom(Structure &st, const Atom &atomTemplate, bool useRandomPosition = false,
                      const tadah::core::Vec3d &pos = tadah::core::Vec3d(0.0, 0.0, 0.0), double minDistance = 0.8);

  /**
   * Removes an atom with a given global index from the structure if valid.
   * If index is out of range, does nothing or throws a warning.
   */
  static void removeAtom(Structure &st, size_t index);

  /**
   * Applies random displacements to each atom’s position, up to maxDisplacement in each coordinate,
   * to simulate thermal agitation.
   * 
   * Example: shakeAtoms(st, 0.05) => random ±0.05 Å displacements in x,y,z.
   */
  static void shakeAtoms(Structure &st, double maxDisplacement);

private:
  /** Applies the reflection M to positions, forces, the cell rows and the
   *  stress (mirrorPlane, mirrorHkl). */
  static void reflect(Structure &st, const tadah::core::Matrix3d &M);
  static void transformStress(Structure &st, const tadah::core::Matrix3d &M);
  static void checkNearSingular(const Structure &st);

  /**
   * Helper that returns a uniform random real in [0,1].
   */
  static double rand01();
};

} // namespace mlip
} // namespace tadah
#endif // STRUCTURE_TRANSFORMATION_H
