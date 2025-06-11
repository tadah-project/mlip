#include <tadah/mlip/nn_finder.h>
#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
namespace tadah {
namespace mlip {

double NNFinder::get_cutoff(const tadah::core::Context &context) const
{
  return context.get<double>("RCUTMAX");
}

// ---------------------------------------------------------------------------
bool NNFinder::check_box(const Structure &st, double cutoff) const
{
  double cutoff_sq = cutoff * cutoff;
  for (int i = 0; i < 3; i++) {
    double vx = st.cell(i, 0);
    double vy = st.cell(i, 1);
    double vz = st.cell(i, 2);
    double len2 = vx*vx + vy*vy + vz*vz;
    if (len2 < cutoff_sq) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Naive approach - store only neighbor indices and shifts, no local Atom copies
void NNFinder::calc_naive(const Structure &st, std::shared_ptr<NNTable> tbl, double cutoff) const
{
  double cutoff_sq = cutoff * cutoff;

  tbl->shift.resize(st.natoms());
  tbl->idx.resize(st.natoms());
  for (size_t i = 0; i < st.natoms(); i++) {
    tbl->shift[i].reserve(100);
    tbl->idx[i].reserve(100);
  }

  // Compute shift bounds
  int N[3];
  num_shifts(st, N, cutoff);

  // Precompute shifts (both real displacement and integer triple if needed)
  std::vector<tadah::core::Vec3d> shifts;
  std::vector<tadah::core::Vec3d> shiftIdx;
  shifts.reserve((2*N[0]+1)*(2*N[1]+1)*(2*N[2]+1));
  shiftIdx.reserve((2*N[0]+1)*(2*N[1]+1)*(2*N[2]+1));

  for (int n1 = -N[0]; n1 <= N[0]; n1++) {
    for (int n2 = -N[1]; n2 <= N[1]; n2++) {
      for (int n3 = -N[2]; n3 <= N[2]; n3++) {
        tadah::core::Vec3d disp;
        disp[0] = n1*st.cell(0,0) + n2*st.cell(1,0) + n3*st.cell(2,0);
        disp[1] = n1*st.cell(0,1) + n2*st.cell(1,1) + n3*st.cell(2,1);
        disp[2] = n1*st.cell(0,2) + n2*st.cell(1,2) + n3*st.cell(2,2);
        shifts.push_back(disp);
        shiftIdx.push_back(tadah::core::Vec3d(n1, n2, n3));
      }
    }
  }

  // Extract positions in contiguous arrays (optional but faster for distance checks)
  const size_t natoms = st.natoms();
  std::vector<double> xPos(natoms), yPos(natoms), zPos(natoms);
  for (size_t i = 0; i < natoms; i++) {
    xPos[i] = st(i).position[0];
    yPos[i] = st(i).position[1];
    zPos[i] = st(i).position[2];
  }

  // Distance checks
  for (size_t s = 0; s < shifts.size(); s++) {
    const tadah::core::Vec3d &disp = shifts[s];
    const tadah::core::Vec3d &dispIdxVal = shiftIdx[s];
    bool selfShift = (dispIdxVal[0] == 0 && dispIdxVal[1] == 0 && dispIdxVal[2] == 0);
    size_t startA2 = (selfShift ? 1ul : 0ul);

    for (size_t a1 = 0; a1 < natoms; a1++) {
      for (size_t a2 = a1 + startA2; a2 < natoms; a2++) {
        double dx = xPos[a1] - (xPos[a2] + disp[0]);
        double dy = yPos[a1] - (yPos[a2] + disp[1]);
        double dz = zPos[a1] - (zPos[a2] + disp[2]);
        double rij_sq = dx*dx + dy*dy + dz*dz;
        if (rij_sq < cutoff_sq) {
          // forward
          tbl->idx[a1].push_back(a2);
          tbl->shift[a1].push_back(dispIdxVal);

          // reverse
          tbl->idx[a2].push_back(a1);
          tbl->shift[a2].push_back(-dispIdxVal);
        }
      }
    }
  }

  // shrink neighbor arrays
  for (size_t i = 0; i < natoms; i++) {
    tbl->shift[i].shrink_to_fit();
    tbl->idx[i].shrink_to_fit();
  }
}
// void NNFinder::num_shifts(Structure &st, int N[3]) {
//   tadah::core::Matrix3d cell_inv = st.cell.inverse();
//
//   double l1 = cell_inv.row(0).norm();
//   double l2 = cell_inv.row(1).norm();
//   double l3 = cell_inv.row(2).norm();
//
//   double f1 = (l1 > 0) ? 1.0/l1 : 1.0;
//   double f2 = (l2 > 0) ? 1.0/l2 : 1.0;
//   double f3 = (l3 > 0) ? 1.0/l3 : 1.0;
//
//   int b1 = std::max(int(f1/cutoff),1);
//   int b2 = std::max(int(f2/cutoff),1);
//   int b3 = std::max(int(f3/cutoff),1);
//
//   N[0] = (int)std::round(0.5 + cutoff*b1/f1);
//   N[1] = (int)std::round(0.5 + cutoff*b2/f2);
//   N[2] = (int)std::round(0.5 + cutoff*b3/f3);
// }

void NNFinder::num_shifts(const Structure &st, int N[3], double cutoff)  const
{
  // 1) Invert the cell
  tadah::core::Matrix3d cellInv = st.cell.inverse();

  tadah::core::Vec3d aStar{ cellInv(0,0), cellInv(0,1), cellInv(0,2) };
  tadah::core::Vec3d bStar{ cellInv(1,0), cellInv(1,1), cellInv(1,2) };
  tadah::core::Vec3d cStar{ cellInv(2,0), cellInv(2,1), cellInv(2,2) };

  double l1 = aStar.norm();  // Magnitude of reciprocal vector a*
  double l2 = bStar.norm();
  double l3 = cStar.norm();

  double f1 = (l1 > 1e-12) ? 1.0 / l1 : 1.0;  // ~ direct space length of v1
  double f2 = (l2 > 1e-12) ? 1.0 / l2 : 1.0;
  double f3 = (l3 > 1e-12) ? 1.0 / l3 : 1.0;

  // Overestimate how many images we might need
  int trialN1 = (int)std::ceil(1.0 + cutoff * 2.0 / f1);
  int trialN2 = (int)std::ceil(1.0 + cutoff * 2.0 / f2);
  int trialN3 = (int)std::ceil(1.0 + cutoff * 2.0 / f3);


  // We'll track the min/max n1, n2, n3 for shifts < cutoff
  int n1_min = 0, n1_max = 0;
  int n2_min = 0, n2_max = 0;
  int n3_min = 0, n3_max = 0;
  bool firstValid = true;

  tadah::core::Vec3d row0{ st.cell(0,0), st.cell(0,1), st.cell(0,2) };
  tadah::core::Vec3d row1{ st.cell(1,0), st.cell(1,1), st.cell(1,2) };
  tadah::core::Vec3d row2{ st.cell(2,0), st.cell(2,1), st.cell(2,2) };

  // Loop over trial shifts
  double cutoff_sq = cutoff * cutoff;
  for (int n1 = -trialN1; n1 <= trialN1; ++n1) {
    for (int n2 = -trialN2; n2 <= trialN2; ++n2) {
      for (int n3 = -trialN3; n3 <= trialN3; ++n3) {

        tadah::core::Vec3d disp
          = double(n1)*row0 + double(n2)*row1 + double(n3)*row2;

        double distSq = disp[0] * disp[0]
          + disp[1] * disp[1]
          + disp[2] * disp[2];
        // double distSq = disp.squaredNorm();
        if (distSq < cutoff_sq) {
          // update bounding box
          if (firstValid) {
            n1_min = n1_max = n1;
            n2_min = n2_max = n2;
            n3_min = n3_max = n3;
            firstValid = false;
          } else {
            if (n1 < n1_min) n1_min = n1;
            if (n1 > n1_max) n1_max = n1;
            if (n2 < n2_min) n2_min = n2;
            if (n2 > n2_max) n2_max = n2;
            if (n3 < n3_min) n3_min = n3;
            if (n3 > n3_max) n3_max = n3;
          }
        }
      }
    }
  }

  // convert bounding box → final N
  N[0] = std::max(std::abs(n1_min), std::abs(n1_max));
  N[1] = std::max(std::abs(n2_min), std::abs(n2_max));
  N[2] = std::max(std::abs(n3_min), std::abs(n3_max));

  N[0] = std::max(N[0], 1);
  N[1] = std::max(N[1], 1);
  N[2] = std::max(N[2], 1);
}

// ---------------------------------------------------------------------------
// Binned approach - similarly remove local copies, store just indices & shifts
void NNFinder::calc_binned(const Structure &st,  std::shared_ptr<NNTable> tbl, double cutoff) const
{
  tbl->shift.resize(st.natoms());
  tbl->idx.resize(st.natoms());
  for (size_t i = 0; i < st.natoms(); i++) {
    tbl->shift[i].reserve(100);
    tbl->idx[i].reserve(100);
  }

  tadah::core::Matrix3d cellInv = st.cell.inverse();

  auto rowLength = [&](int row){
    double vx = st.cell(row, 0);
    double vy = st.cell(row, 1);
    double vz = st.cell(row, 2);
    return std::sqrt(vx*vx + vy*vy + vz*vz);
  };

  double cellLenA = rowLength(0);
  double cellLenB = rowLength(1);
  double cellLenC = rowLength(2);

  int nBinsA = std::max(1, (int)std::floor(cellLenA / cutoff));
  int nBinsB = std::max(1, (int)std::floor(cellLenB / cutoff));
  int nBinsC = std::max(1, (int)std::floor(cellLenC / cutoff));

  struct BinCell {
    std::vector<size_t> atomIndices;
  };
  std::vector<BinCell> bins(nBinsA * nBinsB * nBinsC);

  auto binIndex = [&](int ia, int ib, int ic){
    auto wrap = [&](int k, int n){ return ( (k % n) + n ) % n; };
    ia = wrap(ia, nBinsA);
    ib = wrap(ib, nBinsB);
    ic = wrap(ic, nBinsC);
    return size_t(ia + nBinsA * (ib + nBinsB * ic));
  };

  for (size_t i = 0; i < st.natoms(); i++) {
    const auto &atm = st(i);

    double fx = cellInv(0,0)*atm.position[0] 
      + cellInv(0,1)*atm.position[1] 
      + cellInv(0,2)*atm.position[2];

    double fy = cellInv(1,0)*atm.position[0] 
      + cellInv(1,1)*atm.position[1] 
      + cellInv(1,2)*atm.position[2];

    double fz = cellInv(2,0)*atm.position[0] 
      + cellInv(2,1)*atm.position[1] 
      + cellInv(2,2)*atm.position[2];

    // wrap fractional coords into [0,1)
    fx -= std::floor(fx);
    fy -= std::floor(fy);
    fz -= std::floor(fz);

    int ia = (int)std::floor(fx*nBinsA);
    int ib = (int)std::floor(fy*nBinsB);
    int ic = (int)std::floor(fz*nBinsC);
    bins[ binIndex(ia, ib, ic) ].atomIndices.push_back(i);
  }

  // search
  double cutoff_sq = cutoff * cutoff;
  for (int ia = 0; ia < nBinsA; ia++) {
    for (int ib = 0; ib < nBinsB; ib++) {
      for (int ic = 0; ic < nBinsC; ic++) {
        size_t b0 = binIndex(ia, ib, ic);
        auto &vec0 = bins[b0].atomIndices;

        for (int ja = ia - 1; ja <= ia + 1; ja++) {
          for (int jb = ib - 1; jb <= ib + 1; jb++) {
            for (int jc = ic - 1; jc <= ic + 1; jc++) {
              size_t b1 = binIndex(ja, jb, jc);
              auto &vec1 = bins[b1].atomIndices;

              int dA = ja - ia;
              int dB = jb - ib;
              int dC = jc - ic;

              for (size_t idxA : vec0) {
                for (size_t idxB : vec1) {
                  if ((b0 == b1) && (idxB <= idxA)) continue;
                  const Atom &a1 = st(idxA);
                  const Atom &a2 = st(idxB);

                  double fracX = double(dA)/double(nBinsA);
                  double fracY = double(dB)/double(nBinsB);
                  double fracZ = double(dC)/double(nBinsC);

                  tadah::core::Vec3d shiftDisp(
                      fracX * st.cell(0,0) + fracY * st.cell(1,0) + fracZ * st.cell(2,0),
                      fracX * st.cell(0,1) + fracY * st.cell(1,1) + fracZ * st.cell(2,1),
                      fracX * st.cell(0,2) + fracY * st.cell(1,2) + fracZ * st.cell(2,2)
                      );

                  double dx = a1.position[0] - (a2.position[0] + shiftDisp[0]);
                  double dy = a1.position[1] - (a2.position[1] + shiftDisp[1]);
                  double dz = a1.position[2] - (a2.position[2] + shiftDisp[2]);
                  double dist2 = dx*dx + dy*dy + dz*dz;

                  if (dist2 < cutoff_sq) {
                    // forward
                    tbl->idx[idxA].push_back(idxB);
                    tbl->shift[idxA].push_back(tadah::core::Vec3d(dA, dB, dC));

                    // reverse
                    tbl->idx[idxB].push_back(idxA);
                    tbl->shift[idxB].push_back(tadah::core::Vec3d(-dA, -dB, -dC));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // shrink
  for (size_t i = 0; i < st.natoms(); i++) {
    tbl->shift[i].shrink_to_fit();
    tbl->idx[i].shrink_to_fit();
  }
}

void NNFinder::deduplicate(const Structure &st,  std::shared_ptr<NNTable> tbl ) const
{
  // We'll do one pass per atom.
  // For each atom i, we:
  // 1) gather neighbor info,
  // 2) compute real-space neighbor positions (or displacements),
  // 3) sort by position,
  // 4) merge duplicates.

  const size_t natoms = st.natoms();

  // Pre-extract cell for speed (row-based)
  double c00 = st.cell(0,0), c01 = st.cell(0,1), c02 = st.cell(0,2);
  double c10 = st.cell(1,0), c11 = st.cell(1,1), c12 = st.cell(1,2);
  double c20 = st.cell(2,0), c21 = st.cell(2,1), c22 = st.cell(2,2);

  // We'll define a small struct to hold neighbor data
  struct NbrInfo {
    size_t nbrIdx;               // neighbor atom index
    tadah::core::Vec3d shift;    // the integer shift stored
    tadah::core::Vec3d pos;      // real-space position of the neighbor
  };

  // A small function to compare real-space positions (pos.x, pos.y, pos.z)
  auto eps = 1e-6;
  auto lessPos = [&](const NbrInfo &A, const NbrInfo &B) {
    // compare x
    if (std::fabs(A.pos[0] - B.pos[0]) > eps) 
      return (A.pos[0] < B.pos[0]);
    // compare y
    if (std::fabs(A.pos[1] - B.pos[1]) > eps) 
      return (A.pos[1] < B.pos[1]);
    // compare z
    return (A.pos[2] < B.pos[2] - eps);
  };

  // A function to see if two positions are "the same" within tolerance
  auto samePos = [&](const tadah::core::Vec3d &p1, const tadah::core::Vec3d &p2){
    return ( std::fabs(p1[0] - p2[0]) < eps
        && std::fabs(p1[1] - p2[1]) < eps
        && std::fabs(p1[2] - p2[2]) < eps );
  };

  // For each atom i, deduplicate
  for (size_t i = 0; i < natoms; i++)
  {
    const size_t nnCount = tbl->idx[i].size();
    if (nnCount <= 1) {
      continue;  // no duplicates possible if 0 or 1 neighbor
    }

    // Gather neighbors in a vector
    std::vector<NbrInfo> nvec;
    nvec.reserve(nnCount);

    // Build (nbrIdx, shift, realPos)
    for (size_t n = 0; n < nnCount; n++) {
      size_t idxB = tbl->idx[i][n];
      auto &sh  = tbl->shift[i][n];  // e.g. integer triple (n1, n2, n3)

      // compute the real displacement from shift
      double dxS = sh[0]*c00 + sh[1]*c10 + sh[2]*c20;
      double dyS = sh[0]*c01 + sh[1]*c11 + sh[2]*c21;
      double dzS = sh[0]*c02 + sh[1]*c12 + sh[2]*c22;

      // neighbor's real position
      double xN = st(idxB).position[0] + dxS;
      double yN = st(idxB).position[1] + dyS;
      double zN = st(idxB).position[2] + dzS;

      nvec.push_back( NbrInfo{ idxB, sh, tadah::core::Vec3d(xN, yN, zN) } );
    }

    // Sort by real-space pos
    std::sort(nvec.begin(), nvec.end(), lessPos);

    // Merge duplicates
    std::vector<size_t> newIdx;
    std::vector<tadah::core::Vec3d> newShift;
    newIdx.reserve(nnCount);
    newShift.reserve(nnCount);

    for (size_t k = 0; k < nvec.size(); k++)
    {
      if (k > 0) {
        // compare with prev
        if (samePos(nvec[k].pos, nvec[k-1].pos)) {
          // skip => it's a duplicate
          continue;
        }
      }
      // keep
      newIdx.push_back(nvec[k].nbrIdx);
      newShift.push_back(nvec[k].shift);
    }

    // Replace the old arrays
    tbl->idx[i]   = std::move(newIdx);
    tbl->shift[i] = std::move(newShift);
    tbl->idx[i].shrink_to_fit();
    tbl->shift[i].shrink_to_fit();
  }
}

// ---------------------------------------------------------------------------
void NNFinder::calc(Structure &st, double cutoff) const
{
  st.nn = build(st, cutoff);
}
void NNFinder::calc(Structure &st, const core::Context &context) const
{
  double cutoff = get_cutoff(context);
  st.nn = build(st, cutoff);
}

// ---------------------------------------------------------------------------
// Parallel loop over structure database
void NNFinder::calc(StructureDB &stdb, const core::Context &context) const
{
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for(size_t i = 0; i < stdb.size(); i++){
    calc(stdb(i), context);
  }
}
void NNFinder::calc(StructureDB &stdb, double cutoff) const
{
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for(size_t i = 0; i < stdb.size(); i++){
    calc(stdb(i), cutoff);
  }
}

NNTablePtr NNFinder::build(Structure &st, const core::Context &context) const 
{
  double cutoff = get_cutoff(context);
  return build(st, cutoff);
}
NNFinder::NNTablePtr NNFinder::build(Structure& st, double cutoff) const
{
    auto tbl = std::make_shared<NNTable>();

    tbl->idx.resize(st.natoms());
    tbl->shift.resize(st.natoms());

    if(!check_box(st, cutoff)) {
      calc_naive(st,tbl, cutoff);
      if (st.natoms() == 1) {
        deduplicate(st,tbl);
      }
    } else {
      calc_binned(st,tbl, cutoff);
    }

    return tbl;
}
NNFinder::NNTablePtr
NNFinder::build(const Structure& st, double cutoff) const
{
    auto tbl = std::make_shared<NNTable>();

    tbl->idx.resize(st.natoms());
    tbl->shift.resize(st.natoms());

    if(!check_box(st, cutoff)) {
        calc_naive(st, tbl, cutoff);
        if (st.natoms() == 1)
            deduplicate(st, tbl);
    } else
        calc_binned(st, tbl, cutoff);

    return tbl;
}

NNFinder::NNTablePtr
NNFinder::build(const Structure& st,
                const core::Context& ctx) const
{
    return build(st, get_cutoff(ctx));
}

} // namespace mlip
} // namespace tadah
