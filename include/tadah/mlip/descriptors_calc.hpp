#ifndef DESCRIPTORS_CALC_HPP
#define DESCRIPTORS_CALC_HPP

#include <algorithm>
#include <cstddef>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/detail/nn_guard.h>
#include <tadah/mlip/intercepts.h>
#include <tadah/mlip/parallel.h>
#include <tadah/core/intercept_layout.h>
#include <tadah/core/periodic_table.h>
#include <tadah/models/descriptors/dimer_element_pairs.h>

#include <cstdio>

namespace tadah {
namespace mlip {

template <typename D2, typename DM, typename C2, typename CM>
template <typename T1, typename T2>
DescriptorsCalc<D2,DM,C2,CM>::DescriptorsCalc(tadah::core::Context &c, T1 &t1, T2 &t2):
  context(c),
  d2(t1),
  dm(t2)
{
  if (context.get<size_t>("DSIZE")==0)
    common_constructor();
  else {
    // one intercept column per ATOM species (BIAS true), as DC_Selector
    const size_t bias = tadah::core::n_intercept_columns(context);
    if (context.get<bool>("INIT2B")) {
      d2.set_fidx(bias);
    }
    if (context.get<bool>("INITMB")) {
      dm.set_fidx(bias+d2.size());
    }
  }
  init_intercepts();
  // calc_dimer evaluates every atom's descriptors as Z=1
  tadah::models::require_dimer_element_pairs(context, d2, dm);
}
template <typename D2, typename DM, typename C2, typename CM>
DescriptorsCalc<D2,DM,C2,CM>::DescriptorsCalc(tadah::core::Context &c):
  DescriptorsCalc(c,c,c)
{
  if (c.get<bool>("INIT2B")) {
    c2 = C2(c.get<double>("RCUT2BMAX"));
    if (!context.contains("RCTYPE2B")) {
      context.add("RCTYPE2B",c2.label());
      d2.set_fcut(&c2,false);
    }
  }
  else {
    c2 = C2(0);
    if (!context.contains("RCTYPE2B")) {
      d2.set_fcut(&c2,false);
    }
  }
  if (c.get<bool>("INITMB")) {
    cm = CM(c.get<double>("RCUTMBMAX"));
    if (!many_body_components() && !context.contains("RCTYPEMB")) {
      context.add("RCTYPEMB",cm.label());
      dm.set_fcut(&cm,false);
    }
  }
  else {
    cm = CM(0);
    if (!context.contains("RCTYPEMB")) {
      dm.set_fcut(&cm,false);
    }
  }
}

template <typename D2, typename DM, typename C2, typename CM>
template <typename T1, typename T2,typename T3, typename T4>
DescriptorsCalc<D2,DM,C2,CM>::DescriptorsCalc(tadah::core::Context &c, T1 &d2, T2 &dm, T3 &c2, T4 &cm):
  context(c),
  c2(c2),
  cm(cm),
  d2(d2),
  dm(dm)
{
  if (c.get<bool>("INIT2B")) {
    if (!context.contains("RCTYPE2B")) {
      context.add("RCTYPE2B",c2.label());
      d2.set_fcut(&c2,false);
    }
  }
  if (c.get<bool>("INITMB")) {
    if (!many_body_components() && !context.contains("RCTYPEMB")) {
      context.add("RCTYPEMB",cm.label());
      dm.set_fcut(&cm,false);
    }
  }

  if (context.get<size_t>("DSIZE")==0)
    common_constructor();
  init_intercepts();
  // calc_dimer evaluates every atom's descriptors as Z=1
  tadah::models::require_dimer_element_pairs(context, d2, dm);
}
template <typename D2, typename DM, typename C2, typename CM>
void DescriptorsCalc<D2,DM,C2,CM>::common_constructor() {
  size_t dsize=0;

  // Calculate total size of the descriptor.
  // Add relevant keys to the context.
  // Set first indices for all types: one intercept column per ATOM species
  // (BIAS true), then the two-body and the many-body descriptors.
  const size_t bias = tadah::core::n_intercept_columns(context);
  if (context.get<bool>("INIT2B")) {
    if (!context.contains("TYPE2B"))
      context.add("TYPE2B",d2.label());
    dsize+=d2.size();
    d2.set_fidx(bias);
  }

  if (context.get<bool>("INITMB")) {
    if (!context.contains("TYPEMB"))
      context.add("TYPEMB",dm.label());
    dsize+=dm.size();
    dm.set_fidx(bias+d2.size());
  }

  if (!dsize)
    throw std::runtime_error("The descriptor size is 0: the descriptors of "
                             "the configuration give no feature");

  dsize += bias;

  context.addInternal("DSIZE",dsize);
}
template <typename D2, typename DM, typename C2, typename CM>
void DescriptorsCalc<D2,DM,C2,CM>::init_intercepts() {
  n_intercept_ = tadah::core::n_intercept_columns(context);
  intercept_col_.fill(-1);
  if (n_intercept_)   // the ATOM symbols are looked up in the PeriodicTable
    intercept_col_ = tadah::core::intercept_columns_by_Z(context);
}
template <typename D2, typename DM, typename C2, typename CM>
size_t DescriptorsCalc<D2,DM,C2,CM>::intercept_column(const Atom &a,
                                                     const std::string &where) const {
  const int c = a.Z >= 0 && a.Z < static_cast<int>(intercept_col_.size())
                    ? intercept_col_[a.Z] : -1;
  if (c < 0) throw_element_not_in_atom(a, context, where);
  return static_cast<size_t>(c);
}
template <typename D2, typename DM, typename C2, typename CM>
void DescriptorsCalc<D2,DM,C2,CM>::require_intercept_columns(const Structure &st,
                                                            const std::string &where) const {
  if (!n_intercept_) return;
  for (const Atom &a : st.atoms) (void)intercept_column(a, where);
}
template <typename D2, typename DM, typename C2, typename CM>
double DescriptorsCalc<D2,DM,C2,CM>::neighbour_cutoff() const {
  double rc = 0.0;
  if (context.get<bool>("INIT2B") && d2.size()) rc = std::max(rc, d2.get_rcut());
  if (context.get<bool>("INITMB") && dm.size()) rc = std::max(rc, dm.get_rcut());
  return rc;
}

template <typename D2, typename DM, typename C2, typename CM>
void DescriptorsCalc<D2,DM,C2,CM>::calc_rho(const Structure &st, StDescriptors &st_d) const {
  if (dm.size())
    detail::require_nn_cutoff(st, dm.get_rcut(), "DescriptorsCalc::calc_rho");
  double rcut_mb_sq = pow(context.get<double>("RCUTMBMAX"),2);
  tadah::core::rhos_type &rhos = st_d.rhos;
  size_t s = dm.rhoi_size()+dm.rhoip_size();
  rhos.resize(s,st.natoms());
  rhos.set_zero();

  tadah::core::Vec3d delij;
  for (size_t i=0; i<st.natoms(); ++i) {
    const Atom &a1 = st(i);

    int Zi = a1.Z;
    for (size_t jj=0; jj<st.nn_size(i); ++jj) {
      const tadah::core::Vec3d &a2pos = st.nn_pos(i,jj);
      //delij = a1.position - a2pos;
      delij[0] = a1.position[0] - a2pos[0];
      delij[1] = a1.position[1] - a2pos[1];
      delij[2] = a1.position[2] - a2pos[2];

      //double rij_sq = delij * delij;
      double rij_sq = delij[0]*delij[0] + delij[1]*delij[1] + delij[2]*delij[2];
      if (rij_sq >= rcut_mb_sq) continue;
      size_t neighIdx = st.nn->idx[i][jj];
      int Zj = st(neighIdx).Z;
      double rij = sqrt(rij_sq);
      dm.calc_rho(Zi,Zj,rij,rij_sq,delij,st_d.get_rho(i));
    }
  }
}

template <typename D2, typename DM, typename C2, typename CM>
void DescriptorsCalc<D2,DM,C2,CM>::calc(const Structure &st, StDescriptors &st_d) const {

  detail::require_nn_cutoff(st, neighbour_cutoff(), "DescriptorsCalc");
  if (n_intercept_)
    require_intercept_columns(st, st.label.empty() ? std::string("a structure")
                                                   : "structure '" + st.label + "'");

  bool init2b = context.get<bool>("INIT2B");
  bool initmb = context.get<bool>("INITMB");

  size_t size2b=0;
  size_t sizemb=0;
  if (init2b) size2b=d2.size();
  if (initmb) sizemb=dm.size();

  // deal with the case initmb is set to true but dummy is used
  init2b = init2b && size2b;
  initmb = initmb && sizemb;

  if (initmb)
    calc_rho(st,st_d);

  double rcut_max_sq = pow(context.get<double>("RCUTMAX"),2);
  double rcut_2b_sq = 0.0;
  double rcut_mb_sq = 0.0;

  if (init2b) rcut_2b_sq = pow(d2.get_rcut(),2);
  if (initmb) rcut_mb_sq = pow(dm.get_rcut(),2);

  // zero all aeds and set each atom's intercept: 1.0 in the column of its
  // species (BIAS true)
  for (size_t i=0; i<st.natoms(); ++i) {
    tadah::core::aed_type &aed = st_d.get_aed(i);
    aed.set_zero();
    if (n_intercept_)
      aed(intercept_column(st(i), "a structure")) = 1.0;
  }

  // calculate many-body energy
  // do it before main loop so rho prime
  // can be calculated
  if (initmb) {
    for (size_t i=0; i<st.natoms(); ++i) {
      tadah::core::aed_type &aed = st_d.get_aed(i);
      tadah::core::rho_type& rhoi = st_d.get_rho(i);
      dm.calc_aed(rhoi,aed);
    }
  }

  bool use_force = context.get<bool>("FORCE");
  bool use_stress = context.get<bool>("STRESS");
  // the two-body rows of the force descriptors: after the intercept columns
  const size_t fidx2b = d2.get_fidx();

  tadah::core::Vec3d delij;
  for (size_t i=0; i<st.natoms(); ++i) {
    const Atom &a1 = st(i);
    tadah::core::aed_type &aed = st_d.get_aed(i);

    int Zi = a1.Z;
    for (size_t jj=0; jj<st.nn_size(i); ++jj) {
      const tadah::core::Vec3d &a2pos = st.nn_pos(i,jj);

      delij[0] = a1.position[0] - a2pos[0];
      delij[1] = a1.position[1] - a2pos[1];
      delij[2] = a1.position[2] - a2pos[2];

      double rij_sq = delij[0]*delij[0] + delij[1]*delij[1] + delij[2]*delij[2];

      if (rij_sq >= rcut_max_sq) continue;
      size_t neighIdx = st.nn->idx[i][jj];
      int Zj = st(neighIdx).Z;
      double rij = sqrt(rij_sq);
      double rij_inv = 1.0/rij;

      // CALCULATE TWO-BODY TERM
      if (use_force || use_stress) {
        tadah::core::fd_type &fd_ij = st_d.fd[i][jj];
        if (init2b && rij_sq < rcut_2b_sq) {
          d2.calc_all(Zi,Zj,rij,rij_sq,aed,fd_ij,0.5);
          // Two-body descriptor calculates x-direction only - fd_ij(n,0)
          // so we have to copy x-dir to y- and z-dir
          // and scale them by the unit directional vector delij/rij.
          for (size_t n=fidx2b; n<fidx2b+size2b; ++n) {
            fd_ij(n,0) *= rij_inv;
            fd_ij(n,1) = fd_ij(n,0)*delij[1];
            fd_ij(n,2) = fd_ij(n,0)*delij[2];
            fd_ij(n,0) *= delij[0];
          }
        }
        // CALCULATE MANY-BODY TERM
        if (initmb && rij_sq < rcut_mb_sq) {
          tadah::core::rho_type& rhoi = st_d.get_rho(i);
          dm.calc_dXijdri(Zi,Zj,rij,rij_sq,delij,rhoi,fd_ij);
        }
      }
      else {
        if (init2b && rij_sq < rcut_2b_sq ) {
          d2.calc_aed(Zi, Zj,rij,rij_sq,aed,0.5);
        }
      }

    }
  }
}
template <typename D2, typename DM, typename C2, typename CM>
void DescriptorsCalc<D2,DM,C2,CM>::calc_dimer(const Structure &st, StDescriptors &st_d)  const{

  double r_b = context.get<double>("DIMER",1);
  bool bond_bool = context.get<bool>("DIMER",2);

  // Here we assume that training data consists of 4 atoms
  // The cutoff distance specified in the config file works in the usual way
  // i.e. the maximum distance between two atoms 
  // I,J are molecules, i1,i2 atom in I, similarly for J
  //    I- - - - - - - - - - - -J
  // i1---i2- - - - - - - - -j1---j2
  //    |- - - -r_com- - - - - -|   
  //  |- - - - -r_max- - - - - - -|

  bool init2b = context.get<bool>("INIT2B");
  bool initmb = context.get<bool>("INITMB");
  size_t size2b=0;
  size_t sizemb=0;
  if (init2b) size2b=d2.size();
  if (initmb) sizemb=dm.size();

  // deal with the case initmb is set to true and dummy is used
  init2b = init2b && size2b;
  initmb = initmb && sizemb;

  //double rcut_max_sq = pow(context.get<double>("RCUTMAX"),2);
  double rcut_2b_sq = 0.0;
  double rcut_mb_sq = 0.0;

  if (init2b) rcut_2b_sq = pow(d2.get_rcut(),2);
  if (initmb) rcut_mb_sq = pow(dm.get_rcut(),2);

  // Max distance between CoM of two interacting molecules
  double rcut_com_sq = pow(context.get<double>("RCUTMAX")-r_b,2);

  // Not that this differ to how lammps implements this
  // TODO make it consistent between those two
  tadah::core::Mat6R3C delM;  //i1-j1,i2-j1,i1-j2,i2-j2,i1-i2,j1-j2
  double r_sq[6];
  double r[6];

  const std::vector<Atom> &atoms = st.atoms;
  if (n_intercept_)
    require_intercept_columns(st, st.label.empty() ? std::string("a structure")
                                                   : "structure '" + st.label + "'");

  // The descriptors evaluate every atom as Z=1 (require_dimer_element_pairs);
  // the intercept is that of each atom's own species, as in compute_dimers.
  int Zi = 1;
  int Zj = 1;

  // map of atom label and distances
  // idx[0] - distance label between 0 and 2 atom
  std::vector<std::pair<int,int>> idx(6);
  idx[0] = std::make_pair(0,2);
  idx[1] = std::make_pair(1,2);
  idx[2] = std::make_pair(0,3);
  idx[3] = std::make_pair(1,3);
  idx[4] = std::make_pair(0,1);
  idx[5] = std::make_pair(2,3);

  // zero all aeds+rho and set each atom's intercept (BIAS true)
  for (size_t i=0; i<4; ++i) {
    tadah::core::aed_type &aed = st_d.get_aed(i);
    aed.set_zero();
    if (n_intercept_)
      aed(intercept_column(atoms[i], "a structure")) = 1.0;
  }
  if (initmb) {
    size_t s = dm.rhoi_size()+dm.rhoip_size();
    st_d.rhos.resize(s,4);
    st_d.rhos.set_zero();
  }

  tadah::core::Vec3d xicom = 0.5*(atoms[0].position + atoms[1].position);
  tadah::core::Vec3d xjcom = 0.5*(atoms[2].position + atoms[3].position);

  tadah::core::Vec3d del_com = xicom-xjcom;;
  double r_com_sq = del_com * del_com;
  if (r_com_sq > rcut_com_sq) {
    return;
  }

  // compute all 6 distances (Mat6R3C::row() is a read-only copy; a row is
  // written with set_row())
  for (size_t n=0; n<6; ++n) {
    delM.set_row(n, atoms[idx[n].first].position -
                    atoms[idx[n].second].position);
    r_sq[n] =  delM.row(n) * delM.row(n);
    r[n] = sqrt(r_sq[n]);
  }

  // compute densities and 2b aed for every atom
  // if bond is not included use 0,1,2,3 distances,
  // otherwise use all
  size_t N = bond_bool ? 6 : 4;
  for (size_t n=0; n<N; ++n) {
    if (r_sq[n] <= rcut_mb_sq && initmb) {
      dm.calc_rho(Zi,Zj, r[n],r_sq[n],delM.row(n),st_d.get_rho(idx[n].first));
      dm.calc_rho(Zi,Zj, r[n],r_sq[n],-delM.row(n),st_d.get_rho(idx[n].second));
    }
    // Do not compute 2b term between bonded atoms
    if (r_sq[n] <= rcut_2b_sq && init2b) {
      d2.calc_aed(Zi, Zj,r[n],r_sq[n],st_d.get_aed(idx[n].first));
      d2.calc_aed(Zi, Zj,r[n],r_sq[n],st_d.get_aed(idx[n].second));
    }
  }

  // calculate many-body aed
  if (initmb) {
    for (size_t i=0; i<4; ++i) {
      tadah::core::aed_type &aed = st_d.get_aed(i);
      tadah::core::rho_type& rhoi = st_d.get_rho(i);
      dm.calc_aed(rhoi,aed);
    }
  }

}

template <typename D2, typename DM, typename C2, typename CM>
StDescriptors DescriptorsCalc<D2,DM,C2,CM>::calc(const Structure &st) const {
  StDescriptors st_d(st, context);
  if (context.get<bool>("DIMER",0))
    calc_dimer(st,st_d);
  else
    calc(st,st_d);
  return st_d;
}

template <typename D2, typename DM, typename C2, typename CM>
StDescriptorsDB DescriptorsCalc<D2,DM,C2,CM>::calc(const StructureDB &stdb) const {
  StDescriptorsDB st_desc_db(stdb, context);

  // Refuse a missing or short neighbour list, and an atom without an
  // intercept column, before the parallel loop, so the error names the first
  // such structure whatever the thread order.
  if (!context.get<bool>("DIMER",0)) {
    const double rc = neighbour_cutoff();
    for (size_t i = 0; i < stdb.size(); ++i)
      detail::require_nn_cutoff(stdb(i), rc, "DescriptorsCalc");
  }
  if (n_intercept_)
    for (size_t i = 0; i < stdb.size(); ++i)
      require_intercept_columns(stdb(i), "structure " + std::to_string(i + 1) +
                                             (stdb(i).label.empty() ? std::string()
                                              : " ('" + stdb(i).label + "')"));

  // Stage 2.3: shared-pool backend when installed, else the OpenMP loop.
  // Each i writes a disjoint st_desc_db(i) => order-independent.
  parallel_for(stdb.size(), [&](size_t i) {
    if (context.get<bool>("DIMER",0))
      calc_dimer(stdb(i),st_desc_db(i));
    else
      calc(stdb(i),st_desc_db(i));
  });
  return st_desc_db;
}
template <typename D2, typename DM, typename C2, typename CM>
ColumnLayout DescriptorsCalc<D2,DM,C2,CM>::layout(const tadah::core::Context &c) const {
  ColumnLayout L;
  std::size_t offset = 0;
  const std::size_t n_int = tadah::core::n_intercept_columns(c);
  if (n_int) {
    ColumnBlock b;
    b.descriptor_label = "bias";
    b.family           = "bias";
    b.col_offset       = offset;
    b.col_count        = n_int;   // one intercept column per ATOM species
    L.blocks.push_back(b);
    offset += n_int;
  }
  if (c.get<bool>("INIT2B") && d2.size() > 0) {
    ColumnBlock b;
    b.descriptor_label = d2.label();
    b.family           = "D2";
    b.col_offset       = offset;
    b.col_count        = d2.size();
    L.blocks.push_back(b);
    offset += d2.size();
  }
  if (c.get<bool>("INITMB") && dm.size() > 0) {
    ColumnBlock b;
    b.descriptor_label = dm.label();
    b.family           = "DM";
    b.col_offset       = offset;
    b.col_count        = dm.size();
    L.blocks.push_back(b);
    offset += dm.size();
  }
  return L;
}

} // namespace mlip
} // namespace tadah
#endif
