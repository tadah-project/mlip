#ifndef STRUCTURE_TRANSFORMATION_H
#define STRUCTURE_TRANSFORMATION_H

#include <tadah/mlip/structure.h>

/**
 * @class StructureTransformation
 * @brief Methods that modify a Structure: scaling, shearing, rotating, replicating, etc.
 *        The stored virial stress is updated for linear transforms as S' = M * S * M^T 
 *        if the transform has a physically consistent interpretation (det(M) > 0).
 *        For negative determinant transforms (e.g., reflection), the code still does 
 *        S' = M*S*M^T, but it might be physically questionable.
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
   * Rotates cell and atoms by a 3x3 matrix R:
   *   cell' = R * cell,
   *   pos'  = R * pos.
   * Stress => S' = R * S * R^T.
   */
  static void rotate(Structure &st, const Matrix3d &R);

  /**
   * Shears cell and atoms by shearMatrix:
   *   cell' = shearMatrix * cell,
   *   pos'  = shearMatrix * pos,
   *   S'    = shearMatrix * S * (shearMatrix^T).
   */
  static void shear(Structure &st, const Matrix3d &shearMatrix);

  /**
   * Replicates the structure nx, ny, nz times along cell vectors.
   * Creates a supercell with cell dimension scaled accordingly and 
   * updates the atom list. Stress scaling is context-dependent.
   */
  static void replicate(Structure &st, int nx, int ny, int nz);

  /**
   * Shifts all atomic positions by shiftVec. Cell and stress remain unchanged.
   */
  static void shift(Structure &st, const Vec3d &shiftVec);

  /**
   * Mirrors about origin => pos' = -pos.
   * By default, leaves stress unchanged or uses M = -I => S' = S mathematically.
   */
  static void mirrorOrigin(Structure &st);

  /**
   * Mirrors about plane normal to normalVec (through origin).
   * Reflects positions via p' = p - 2*(p·n_hat) * n_hat.
   * Then S' = M*S*M^T, M = (I - 2n_hat n_hat^T).
   */
  static void mirrorPlane(Structure &st, const Vec3d &normalVec);

  /**
   * Mirrors about a plane with Miller indices (h,k,l) through origin.
   * The plane normal is computed from the reciprocal lattice.
   * Similar reflection for stress => S' = M*S*M^T (det(M)<0).
   */
  static void mirrorHkl(Structure &st, int h, int k, int l);

  /**
   * Centers atoms so that either the geometric center or center of mass 
   * is placed at the origin. If shiftCell is true, also shift the cell origin 
   * (typical usage might not require that).
   */
  static void center(Structure &st, bool useCenterOfMass = false, bool shiftCell = false);

  /**
   * Realigns the cell’s principal directions to the Cartesian axes 
   * (e.g., diagonalizing the metric). A typical approach:
   *   - compute G = cell^T * cell
   *   - eigen-decompose G
   *   - build rotation R from eigenvectors
   *   - rotate(st, R)
   * Provided as a stub.
   */
  static void alignAxes(Structure &st);

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
                      const Vec3d &pos = Vec3d(0.0, 0.0, 0.0), double minDistance = 0.8);

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
  static void transformStress(Structure &st, const Matrix3d &M);
  static void checkNearSingular(const Structure &st);

  /**
   * Helper that returns a uniform random real in [0,1].
   */
  static double rand01();
};

#endif // STRUCTURE_TRANSFORMATION_H
