#include "nn_finder.h"
#include <limits>

NNFinder::NNFinder(Config &config):
  cutoff_sq(pow(config.get<double>("RCUTMAX"),2)),
  cutoff(config.get<double>("RCUTMAX"))
{}

void NNFinder::calc(Structure &st) {

  int N[3];
  num_shifts(st, N);

  Matrix shiftedpos(st.natoms(),3);
  // for convenience only:
  std::vector<std::vector<Atom>> &nnatoms = st.near_neigh_atoms;
  std::vector<std::vector<Vec3d>> &nnshift = st.near_neigh_shift;
  std::vector<std::vector<size_t>> &nnidx = st.near_neigh_idx;
  nnatoms.resize(st.natoms());
  nnshift.resize(st.natoms());
  nnidx.resize(st.natoms());

  Vec3d displacement;
  Vec3d delij;
  Vec3d shift;
  Atom atom1;
  Atom atom2;
  double min_double = std::numeric_limits<double>::min();
  for (int n1=-N[0]; n1<=N[0]; n1++) {
    for (int n2=-N[1]; n2<=N[1]; n2++) {
      for (int n3=-N[2]; n3<=N[2]; n3++) {

        shift(n1,n2,n3);
        displacement[0] = st.cell(0,0)*n1 + st.cell(1,0)*n2 +  st.cell(2,0)*n3 ;
        displacement[1] = st.cell(0,1)*n1 + st.cell(1,1)*n2 +  st.cell(2,1)*n3 ;
        displacement[2] = st.cell(0,2)*n1 + st.cell(1,2)*n2 +  st.cell(2,2)*n3 ;

        for (size_t a=0; a<st.natoms(); ++a) {
          shiftedpos(a,0) = st(a).position[0] + displacement[0];
          shiftedpos(a,1) = st(a).position[1] + displacement[1];
          shiftedpos(a,2) = st(a).position[2] + displacement[2];
        }

        // calculate all neighbours of a1 for this shift
        size_t start = n1==0 && n2==0 && n3==0 ? 1 : 0;
        for (size_t a1=0; a1<st.natoms(); ++a1) {
          for (size_t a2=a1+start; a2<st.natoms(); ++a2) {
            delij[0] = st(a1).position[0] - shiftedpos(a2,0);
            delij[1] = st(a1).position[1] - shiftedpos(a2,1);
            delij[2] = st(a1).position[2] - shiftedpos(a2,2);
            double rij_sq = delij[0]*delij[0] + delij[1]*delij[1] + delij[2]*delij[2];

            if(rij_sq<cutoff_sq && rij_sq>min_double) {
              atom1 = st(a1);
              atom2 = st(a2);
              for (size_t i=0; i<3; ++i) {
                atom2.position[i] = shiftedpos(a2,i);
                atom1.position[i] = st(a1).position[i]-displacement[i];
              }
              nnatoms[a1].push_back(atom2);
              nnidx[a1].push_back(a2);
              nnshift[a1].push_back(shift);

              nnatoms[a2].push_back(atom1);
              nnidx[a2].push_back(a1);
              nnshift[a2].push_back(-shift);
            }
          }
        }
      }
    }
  }
}
void NNFinder::calc(StructureDB &stdb) {
  #pragma omp parallel for
  for (size_t s=0; s<stdb.size(); ++s) {
    calc(stdb(s));
  }
}
bool NNFinder::check_box(Structure &st) {
  double f = 1.05;   // extra safety measure
  for (size_t i=0; i<3; ++i)
    if (st.cell.row(i)*st.cell.row(i) < f*cutoff_sq)
      return false;
  return true;
}
void NNFinder::num_shifts(Structure &st, int N[3]) {
  //Matrix3d cell_inv = st.cell.completeOrthogonalDecomposition().pseudoInverse();
  Matrix3d cell_inv = st.cell.inverse();

  double l1 = cell_inv.col(0).norm();
  double l2 = cell_inv.col(1).norm();
  double l3 = cell_inv.col(2).norm();

  double f1 = l1 > 0 ? 1.0/l1 : 1.0;
  double f2 = l2 > 0 ? 1.0/l2 : 1.0;
  double f3 = l3 > 0 ? 1.0/l3 : 1.0;

  int b1 = std::max(int(f1/cutoff),1);
  int b2 = std::max(int(f2/cutoff),1);
  int b3 = std::max(int(f3/cutoff),1);

  N[0] = (int)std::round(0.5+cutoff*b1/(f1));
  N[1] = (int)std::round(0.5+cutoff*b2/(f2));
  N[2] = (int)std::round(0.5+cutoff*b3/(f3));

}
