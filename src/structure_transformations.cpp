#include <tadah/mlip/structure_transformations.h>
#include <tadah/mlip/structure_properties.h>
#include <random>
#include <iostream>
#include <stdexcept>
#include <cmath>
namespace tadah {
namespace mlip {

namespace {

/** Local utility function: multiply 3x3 matrices. */
tadah::core::Matrix3d matMul(const tadah::core::Matrix3d &A, const tadah::core::Matrix3d &B)
{
  tadah::core::Matrix3d C;
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      C(i,j) = A(i,0)*B(0,j) + A(i,1)*B(1,j) + A(i,2)*B(2,j);
    }
  }
  return C;
}

/** A simple global RNG. Replace with thread_local or a better approach if needed. */
std::mt19937 &globalRng()
{
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  return rng;
}

} // end unnamed namespace

double StructureTransformation::rand01()
{
  static std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(globalRng());
}

void StructureTransformation::transformStress(Structure &st, const tadah::core::Matrix3d &M)
{
  tadah::core::Matrix3d tmp = matMul(M, st.stress);
  tadah::core::Matrix3d Mt  = M.transpose();
  st.stress    = matMul(tmp, Mt);
}

void StructureTransformation::checkNearSingular(const Structure &st)
{
  double vol = StructureProperties::getVolume(st);
  if (std::fabs(vol) < 1e-14) {
    throw std::runtime_error("Cell volume is near zero after transformation. Invalid operation.");
  }
}

void StructureTransformation::scale(Structure &st, double factor)
{
  if (factor <= 0.0) {
    throw std::invalid_argument("Scale factor must be positive.");
  }
  st.cell = factor * st.cell;
  for (auto &atom : st.atoms) {
    atom.position = factor * atom.position;
  }
  // Common approach => stress is factor^2 * stress
  st.stress = (factor*factor) * st.stress;

  checkNearSingular(st);
}

void StructureTransformation::rescaleLength(Structure &st, double s)
{
  if (s <= 0.0) {
    throw std::invalid_argument("rescaleLength factor must be positive.");
  }
  st.cell = s * st.cell;
  const double inv_s = 1.0 / s;
  for (auto &atom : st.atoms) {
    atom.position = s * atom.position;
    atom.force    = inv_s * atom.force;
  }
  // Energy and stored virial stress are invariant under uniform length rescaling.
  // (Tadah's stress is the virial V * sigma_Cauchy with units of energy.)
  checkNearSingular(st);
}

void StructureTransformation::scaleToVolume(Structure &st, double targetVolume)
{
  if (targetVolume <= 0.0) {
    throw std::invalid_argument("Target volume must be positive.");
  }
  double currentVolume = StructureProperties::getVolume(st);
  if (currentVolume <= 0.0) {
    throw std::runtime_error("Current volume is invalid (<= 0); cannot scale.");
  }
  double factor = std::cbrt(targetVolume / currentVolume);
  scale(st, factor);
}

void StructureTransformation::scaleVolumeToDensity(Structure &st, double targetDensity)
{
  if (targetDensity <= 0.0) {
    throw std::invalid_argument("Target density must be positive.");
  }
  // total mass in grams
  double totalMassAmu = 0.0;
  for (auto &atom : st.atoms) {
    totalMassAmu += tadah::core::PeriodicTable::get_mass(atom.Z);
  }
  double totalMassGrams = totalMassAmu * 1.66053906660e-24; // amu->g

  // volume in cm^3 => mass/density
  double volumeCm3 = totalMassGrams / targetDensity;
  // convert to Å^3
  double volumeA3 = volumeCm3 * 1.0e24;

  scaleToVolume(st, volumeA3);
}

void StructureTransformation::rotate(Structure &st, const tadah::core::Matrix3d &R)
{
  st.cell = matMul(R, st.cell);
  for (auto &atom : st.atoms) {
    atom.position = R * atom.position;
  }
  transformStress(st, R);
  checkNearSingular(st);
}

void StructureTransformation::shear(Structure &st, const tadah::core::Matrix3d &shearMatrix)
{
  st.cell = matMul(shearMatrix, st.cell);
  for (auto &atom : st.atoms) {
    atom.position = shearMatrix * atom.position;
  }
  transformStress(st, shearMatrix);
  checkNearSingular(st);
}

void StructureTransformation::replicate(Structure &st, int nx, int ny, int nz)
{
  if (nx < 1) nx = 1;
  if (ny < 1) ny = 1;
  if (nz < 1) nz = 1;

  size_t origNatoms = st.atoms.size();
  std::vector<Atom> originalAtoms = st.atoms;
  tadah::core::Matrix3d oldCell = st.cell;

  // Scale cell
  st.cell(0,0) *= nx; 
  st.cell(1,1) *= ny; 
  st.cell(2,2) *= nz;

  // Clear and reserve memory
  st.atoms.clear();
  st.atoms.reserve(origNatoms * nx * ny * nz);

  for (int ix=0; ix<nx; ix++) {
    for (int iy=0; iy<ny; iy++) {
      for (int iz=0; iz<nz; iz++) {
        // shift vector in oldCell coords
        tadah::core::Vec3d shiftVec = 
           ix*tadah::core::Vec3d(oldCell(0,0), oldCell(1,0), oldCell(2,0)) +
           iy*tadah::core::Vec3d(oldCell(0,1), oldCell(1,1), oldCell(2,1)) +
           iz*tadah::core::Vec3d(oldCell(0,2), oldCell(1,2), oldCell(2,2));
        for (const auto &atom : originalAtoms) {
          Atom newAtom = atom;
          newAtom.position += shiftVec;
          st.atoms.push_back(newAtom);
        }
      }
    }
  }
  // Stress scaling is context dependent. E.g. st.stress might remain the same per cell, 
  // or scale by 1/(nx*ny*nz), etc. Not explicitly changed here.
  checkNearSingular(st);
}

void StructureTransformation::shift(Structure &st, const tadah::core::Vec3d &shiftVec)
{
  for (auto &atom : st.atoms) {
    atom.position += shiftVec;
  }
  // cell, stress unchanged
}

void StructureTransformation::mirrorOrigin(Structure &st)
{
  for (auto &atom : st.atoms) {
    atom.position = -atom.position;
  }
  // Reflection M = -I => S'=S. (No net effect if we do M*S*M^T.)
}

void StructureTransformation::mirrorPlane(Structure &st, const tadah::core::Vec3d &normalVec)
{
  tadah::core::Vec3d n = normalVec;
  double len = n.norm();
  if (len < 1e-14) {
    throw std::invalid_argument("Plane normal is near zero length.");
  }
  n /= len;

  for (auto &atom : st.atoms) {
    double proj = atom.position * n;
    atom.position -= 2.0 * proj * n;
  }

  // M = I - 2 n n^T
  tadah::core::Matrix3d M = tadah::core::Matrix3d::Identity();
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      M(i,j) -= 2.0 * n[i]*n[j];
    }
  }
  transformStress(st, M);
}

void StructureTransformation::mirrorHkl(Structure &st, int h, int k, int l)
{
  if ((h==0) && (k==0) && (l==0)) {
    throw std::invalid_argument("Miller indices (0,0,0) do not define a plane.");
  }
  // Build a1,a2,a3 from st.cell rows:
  tadah::core::Vec3d a1 = st.cell.row(0);
  tadah::core::Vec3d a2 = st.cell.row(1);
  tadah::core::Vec3d a3 = st.cell.row(2);

  tadah::core::Vec3d crossA2A3 = a2.cross(a3);
  double denomA1  = a1*crossA2A3;
  tadah::core::Vec3d b1        = crossA2A3 / denomA1;

  tadah::core::Vec3d crossA3A1 = a3.cross(a1);
  double denomA2  = a2*crossA3A1;
  tadah::core::Vec3d b2        = crossA3A1 / denomA2;

  tadah::core::Vec3d crossA1A2 = a1.cross(a2);
  double denomA3  = a3*crossA1A2;
  tadah::core::Vec3d b3        = crossA1A2 / denomA3;

  tadah::core::Vec3d g = (double)h*b1 + (double)k*b2 + (double)l*b3;
  double normG = g.norm();
  if (normG < 1e-14) {
    throw std::runtime_error("Plane normal is near zero. Check cell or (h,k,l).");
  }
  tadah::core::Vec3d n = g / normG;

  // reflection
  for (auto &atom : st.atoms) {
    double proj = atom.position*n;
    atom.position -= 2.0 * proj * n;
  }

  // reflection matrix
  tadah::core::Matrix3d M = tadah::core::Matrix3d::Identity();
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      M(i,j) -= 2.0 * n[i]*n[j];
    }
  }
  transformStress(st, M);
}

void StructureTransformation::center(Structure &st, bool useCenterOfMass, bool shiftCell)
{
  tadah::core::Vec3d c;
  if (useCenterOfMass) {
    c = StructureProperties::getCentreOfMass(st);
  } else {
    c = StructureProperties::getGeometricCenter(st);
  }

  for (auto &atom : st.atoms) {
    atom.position -= c;
  }

  if (shiftCell) {
    // The cell matrix typically has its origin at (0,0,0). 
    // Shifting it requires some reference. 
    // If st had an explicit 'origin' vector, we'd do e.g. st.origin -= c;
    // This snippet does nothing by default because st.cell is typically 
    // just directions from (0,0,0).
  }
}

void StructureTransformation::alignAxes(Structure &st)
{
  // Typical approach:
  //  1) G = cell^T * cell
  //  2) Eigen-decompose G => principal directions
  //  3) build rotation R
  //  4) rotate(st, R)
  // Left unimplemented here.
}

void StructureTransformation::remapToCell(Structure &st)
{
  double vol = StructureProperties::getVolume(st);
  if (std::fabs(vol) < 1e-14) {
    throw std::runtime_error("Volume is near zero, cannot remap to cell.");
  }
  // Rows-as-lattice-vectors convention: r = cell^T * frac, so frac = (cell^T)^{-1} * r.
  tadah::core::Matrix3d invCellT = st.cell.transpose().inverse();

  for (auto &atom : st.atoms) {
    tadah::core::Vec3d frac = invCellT * atom.position;
    for (int i=0; i<3; i++) {
      frac[i] -= std::floor(frac[i]);
    }
    // r[k] = sum_j frac[j] * cell(j, k)
    atom.position = st.cell.transpose() * frac;
  }
}

void StructureTransformation::addAtom(
  Structure &st,
  const Atom &atomTemplate,
  bool useRandomPosition,
  const tadah::core::Vec3d &pos,
  double minDistance
){
  Atom newAtom = atomTemplate;

  if (useRandomPosition) {
    // place randomly in [0,1) fractional, then convert => Cartesian
    double vol = StructureProperties::getVolume(st);
    if (std::fabs(vol) < 1e-14) {
      throw std::runtime_error("Cannot add atom randomly: volume near zero.");
    }

    tadah::core::Vec3d frac(rand01(), rand01(), rand01());
    // Rows-as-lattice-vectors: r = cell^T * frac.
    newAtom.position = st.cell.transpose() * frac;
  } else {
    newAtom.position = pos;
  }

  // Check distance to existing atoms
  for (const auto &a : st.atoms) {
    double dist = (newAtom.position - a.position).norm();
    if (dist < minDistance) {
      std::cerr << "[WARNING] New atom is only " << dist << " Å away from an existing atom. "
                << "Minimum suggested distance: " << minDistance << " Å.\n";
      break;
    }
  }

  st.atoms.push_back(newAtom);
}

void StructureTransformation::removeAtom(Structure &st, size_t index)
{
  if (index < st.atoms.size()) {
    st.atoms.erase(st.atoms.begin() + index);
  } else {
    std::cerr << "[WARNING] removeAtom: index out of range (" << index << "). No atom removed.\n";
  }
}

void StructureTransformation::shakeAtoms(Structure &st, double maxDisplacement)
{
  if (maxDisplacement <= 0.0) {
    std::cerr << "[WARNING] shakeAtoms: non-positive maxDisplacement => no action.\n";
    return;
  }

  for (auto &atom : st.atoms) {
    // random in [-maxDisplacement, +maxDisplacement]
    double rx = (2.0*rand01() - 1.0) * maxDisplacement;
    double ry = (2.0*rand01() - 1.0) * maxDisplacement;
    double rz = (2.0*rand01() - 1.0) * maxDisplacement;
    atom.position[0] += rx;
    atom.position[1] += ry;
    atom.position[2] += rz;
  }
}

} // namespace mlip
} // namespace tadah
