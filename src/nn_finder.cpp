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
/* ======================================================================
   Naïve O(N² × #images) neighbour search that keeps every periodic copy
   lying inside the cut-off sphere.  Returned table is symmetric (i→j and
   j→i present) and shift = (n1,n2,n3) is the integer lattice vector.
   ======================================================================*/
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

  deduplicate(st,tbl);
}

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
static inline bool is_tilted(const core::Matrix3d& C, double eps = 1e-12)
{
    return std::fabs(C(0,1)) > eps || std::fabs(C(0,2)) > eps ||
           std::fabs(C(1,0)) > eps || std::fabs(C(1,2)) > eps ||
           std::fabs(C(2,0)) > eps || std::fabs(C(2,1)) > eps;
}
// ======================================================================
// Binned neighbour search – works for arbitrary triclinic cells.
// ======================================================================
void NNFinder::calc_binned(const Structure&         st,
                               std::shared_ptr<NNTable> tbl,
                               double                   cutoff) const
{
    const std::size_t N = st.natoms();
    tbl->idx.resize(N);
    tbl->shift.resize(N);

    /* ------------------------------------------------------------ */
    /* 1. decide bin counts in fractional space                     */
    auto row_length = [&](int r){
        double x=st.cell(r,0), y=st.cell(r,1), z=st.cell(r,2);
        return std::sqrt(x*x+y*y+z*z); };

    const double aL = row_length(0);
    const double bL = row_length(1);
    const double cL = row_length(2);
    double Lmax = std::max({aL,bL,cL});

    if ( is_tilted(st.cell) && (Lmax / cutoff) > 1.0 )
      Lmax *= 0.5;                 // => bin-width doubled, nx,ny,nz halved

    const double df = cutoff / Lmax;              // fractional bin width
    const int nx = std::max(1, int(std::floor(1.0 / df)));
    const int ny = nx;
    const int nz = nx;                            // cubic bins in f-space

    struct Bin { std::vector<std::size_t> atoms; };
    std::vector<Bin> bins(std::size_t(nx*ny*nz));

    auto wrap=[](int k,int n){ return ((k%n)+n)%n; };
    auto bidx=[&](int ix,int iy,int iz){
        return std::size_t(wrap(ix,nx)+nx*(wrap(iy,ny)+ny*wrap(iz,nz))); };

    /* inverse cell & row vectors */
    core::Matrix3d Cinv = st.cell.inverse();
    core::Vec3d a(st.cell.row(0)), b(st.cell.row(1)), c(st.cell.row(2));

    /* ------------------------------------------------------------ */
    /* 2. put atoms into bins, store fractional coords              */
    struct Frac{ double x,y,z; };
    std::vector<Frac> fcoord(N);

    for(std::size_t i=0;i<N;++i){
        const auto& r = st(i).position;
        double fx=Cinv(0,0)*r[0]+Cinv(0,1)*r[1]+Cinv(0,2)*r[2];
        double fy=Cinv(1,0)*r[0]+Cinv(1,1)*r[1]+Cinv(1,2)*r[2];
        double fz=Cinv(2,0)*r[0]+Cinv(2,1)*r[1]+Cinv(2,2)*r[2];
        fx-=std::floor(fx); fy-=std::floor(fy); fz-=std::floor(fz);
        fcoord[i]={fx,fy,fz};
        bins[bidx(int(fx*nx),int(fy*ny),int(fz*nz))].atoms.push_back(i);
    }

    /* ------------------------------------------------------------ */
    /* 3. exact lattice-vector span from naive helper               */
    int Ns[3];  num_shifts(st,Ns,cutoff);

    auto bspan=[&](int nBin,int Ns){ return std::max(1,(Ns+nBin-1)/nBin); };
    const int sx=bspan(nx,Ns[0]), sy=bspan(ny,Ns[1]), sz=bspan(nz,Ns[2]);

    const double rc2 = cutoff*cutoff;
    for(std::size_t i=0;i<N;++i){
        tbl->idx[i].reserve(32); tbl->shift[i].reserve(32); }

    /* ------------------------------------------------------------ */
    /* 4. neighbour search                                          */
    for(int ix=0;ix<nx;++ix)
      for(int iy=0;iy<ny;++iy)
        for(int iz=0;iz<nz;++iz)
        {
            const auto& bin0 = bins[bidx(ix,iy,iz)].atoms;

            for(int dx=-sx;dx<=sx;++dx)
              for(int dy=-sy;dy<=sy;++dy)
                for(int dz=-sz;dz<=sz;++dz)
                {
                    const auto& bin1 = bins[bidx(ix+dx,iy+dy,iz+dz)].atoms;
                    for(std::size_t p:bin0)
                      for(std::size_t q:bin1)
                      {
                          if(dx==0 && dy==0 && dz==0 && q<=p) continue;

                          core::Vec3d shift(dx,dy,dz);
                          core::Vec3d del =
                               shift[0]*a + shift[1]*b + shift[2]*c +
                               st(q).position - st(p).position;

                          if(del*del < rc2){
                              tbl->idx  [p].push_back(q);
                              tbl->shift[p].push_back( shift);
                              tbl->idx  [q].push_back(p);
                              tbl->shift[q].push_back(-shift);
                          }
                      }
                }
        }

    /* ------------------------------------------------------------
       5. de-duplicate                                              */
    struct Rec{ std::size_t j; core::Vec3d s; };
    auto less=[](const Rec&a,const Rec&b){
        if(a.j!=b.j) return a.j<b.j;
        if(a.s[0]!=b.s[0]) return a.s[0]<b.s[0];
        if(a.s[1]!=b.s[1]) return a.s[1]<b.s[1];
        return a.s[2]<b.s[2]; };

    for(std::size_t i=0;i<N;++i){
        std::vector<Rec> rec;
        rec.reserve(tbl->idx[i].size());
        for(std::size_t k=0;k<tbl->idx[i].size();++k)
            rec.push_back({tbl->idx[i][k], tbl->shift[i][k]});
        std::sort(rec.begin(),rec.end(),less);
        rec.erase(std::unique(rec.begin(),rec.end(),
                    [](const Rec&a,const Rec&b){
                        return a.j==b.j && a.s==b.s;}),rec.end());
        tbl->idx[i].resize(rec.size());
        tbl->shift[i].resize(rec.size());
        for(std::size_t k=0;k<rec.size();++k){
            tbl->idx[i][k]=rec[k].j; tbl->shift[i][k]=rec[k].s;}
        tbl->idx[i].shrink_to_fit(); tbl->shift[i].shrink_to_fit();
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
