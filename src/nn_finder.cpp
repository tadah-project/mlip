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

// Constructor
NNFinder::NNFinder(tadah::core::Context &context)
{
  cutoff     = context.get<double>("RCUTMAX");
  cutoff_sq  = cutoff * cutoff;
}

// ---------------------------------------------------------------------------
bool NNFinder::check_box(Structure &st)
{
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
void NNFinder::num_shifts(Structure &st, int N[3]) {
  tadah::core::Matrix3d cell_inv = st.cell.inverse();

  double l1 = cell_inv.col(0).norm();
  double l2 = cell_inv.col(1).norm();
  double l3 = cell_inv.col(2).norm();

  double f1 = (l1 > 0) ? 1.0/l1 : 1.0;
  double f2 = (l2 > 0) ? 1.0/l2 : 1.0;
  double f3 = (l3 > 0) ? 1.0/l3 : 1.0;

  int b1 = std::max(int(f1/cutoff),1);
  int b2 = std::max(int(f2/cutoff),1);
  int b3 = std::max(int(f3/cutoff),1);

  N[0] = (int)std::round(0.5 + cutoff*b1/f1);
  N[1] = (int)std::round(0.5 + cutoff*b2/f2);
  N[2] = (int)std::round(0.5 + cutoff*b3/f3);
}

// ---------------------------------------------------------------------------
// Naive approach - store only neighbor indices and shifts, no local Atom copies
void NNFinder::calc_naive(Structure &st)
{
  st.near_neigh_shift.resize(st.natoms());
  st.near_neigh_idx.resize(st.natoms());
  for (size_t i = 0; i < st.natoms(); i++) {
    st.near_neigh_shift[i].reserve(100);
    st.near_neigh_idx[i].reserve(100);
  }

  // Compute shift bounds
  int N[3];
  num_shifts(st, N);

  // Precompute shifts (both real displacement and integer triple if needed)
  std::vector<tadah::core::Vec3d> shifts;
  std::vector<tadah::core::Vec3d> shiftIdx;
  shifts.reserve((2*N[0]+1)*(2*N[1]+1)*(2*N[2]+1));
  shiftIdx.reserve((2*N[0]+1)*(2*N[1]+1)*(2*N[2]+1));

  for (int n1 = -N[0]; n1 <= N[0]; n1++) {
    for (int n2 = -N[1]; n2 <= N[1]; n2++) {
      for (int n3 = -N[2]; n3 <= N[2]; n3++) {
        tadah::core::Vec3d disp;
        disp[0] = st.cell(0,0)*n1 + st.cell(1,0)*n2 + st.cell(2,0)*n3;
        disp[1] = st.cell(0,1)*n1 + st.cell(1,1)*n2 + st.cell(2,1)*n3;
        disp[2] = st.cell(0,2)*n1 + st.cell(1,2)*n2 + st.cell(2,2)*n3;
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
          st.near_neigh_idx[a1].push_back(a2);
          st.near_neigh_shift[a1].push_back(dispIdxVal);

          // reverse
          st.near_neigh_idx[a2].push_back(a1);
          st.near_neigh_shift[a2].push_back(-dispIdxVal);
        }
      }
    }
  }

  // shrink neighbor arrays
  for (size_t i = 0; i < natoms; i++) {
    st.near_neigh_shift[i].shrink_to_fit();
    st.near_neigh_idx[i].shrink_to_fit();
  }
}

// ---------------------------------------------------------------------------
// Binned approach - similarly remove local copies, store just indices & shifts
void NNFinder::calc_binned(Structure &st)
{

  st.near_neigh_shift.resize(st.natoms());
  st.near_neigh_idx.resize(st.natoms());
  for (size_t i = 0; i < st.natoms(); i++) {
    st.near_neigh_shift[i].reserve(100);
    st.near_neigh_idx[i].reserve(100);
  }

  // invert cell
  double invC[9];
  inverse_3x3_direct(st.cell.data(), invC);

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

  // fill bins
  for (size_t i = 0; i < st.natoms(); i++) {
    const auto &atm = st(i);
    double fx = invC[0]*atm.position[0] + invC[3]*atm.position[1] + invC[6]*atm.position[2];
    double fy = invC[1]*atm.position[0] + invC[4]*atm.position[1] + invC[7]*atm.position[2];
    double fz = invC[2]*atm.position[0] + invC[5]*atm.position[1] + invC[8]*atm.position[2];

    fx -= std::floor(fx);
    fy -= std::floor(fy);
    fz -= std::floor(fz);

    int ia = (int)std::floor(fx*nBinsA);
    int ib = (int)std::floor(fy*nBinsB);
    int ic = (int)std::floor(fz*nBinsC);
    bins[ binIndex(ia, ib, ic) ].atomIndices.push_back(i);
  }

  double cutSQ = cutoff_sq;

  // search
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
                  // avoid double counting
                  if ((b0 == b1) && (idxB <= idxA)) continue;

                  const Atom &a1 = st(idxA);
                  const Atom &a2 = st(idxB);

                  double fracX = double(dA)/double(nBinsA);
                  double fracY = double(dB)/double(nBinsB);
                  double fracZ = double(dC)/double(nBinsC);

                  // Real shift
                  tadah::core::Vec3d shiftDisp(
                    st.cell(0,0)*fracX + st.cell(0,1)*fracY + st.cell(0,2)*fracZ,
                    st.cell(1,0)*fracX + st.cell(1,1)*fracY + st.cell(1,2)*fracZ,
                    st.cell(2,0)*fracX + st.cell(2,1)*fracY + st.cell(2,2)*fracZ
                  );

                  double dx = a1.position[0] - (a2.position[0] + shiftDisp[0]);
                  double dy = a1.position[1] - (a2.position[1] + shiftDisp[1]);
                  double dz = a1.position[2] - (a2.position[2] + shiftDisp[2]);
                  double dist2 = dx*dx + dy*dy + dz*dz;

                  if (dist2 < cutSQ) {
                    // forward
                    st.near_neigh_idx[idxA].push_back(idxB);
                    st.near_neigh_shift[idxA].push_back(tadah::core::Vec3d(dA, dB, dC));

                    // reverse
                    st.near_neigh_idx[idxB].push_back(idxA);
                    st.near_neigh_shift[idxB].push_back(tadah::core::Vec3d(-dA, -dB, -dC));
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
    st.near_neigh_shift[i].shrink_to_fit();
    st.near_neigh_idx[i].shrink_to_fit();
  }
}

// ---------------------------------------------------------------------------
// Master calc that chooses naive or binned
void NNFinder::calc(Structure &st)
{
  if(!check_box(st)) {
    calc_naive(st);
  } else {
    calc_binned(st);
  }
}

// ---------------------------------------------------------------------------
// Parallel loop over structure database
void NNFinder::calc(StructureDB &stdb)
{
  // auto t0 = std::chrono::steady_clock::now();
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for(size_t i = 0; i < stdb.size(); i++){
    calc(stdb(i));
  }
  // auto t1 = std::chrono::steady_clock::now();
  // double seconds = std::chrono::duration<double>(t1 - t0).count();
  // std::cout << "calc(StructureDB &stdb) for-loop took "
  //           << seconds << " seconds\n";
}

} // namespace mlip
} // namespace tadah
