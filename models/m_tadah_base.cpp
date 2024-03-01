#include "m_tadah_base.h"

void M_Tadah_Base::
fpredict(const size_t a, force_type &v,
         const StDescriptors &std, const Structure &st)
{
  for (size_t jj=0; jj<st.nn_size(a); ++jj) {
    size_t j=st.near_neigh_idx[a][jj];
    const size_t aa = st.get_nn_iindex(a,j,jj);
    const fd_type &fdji = std.fd[j][aa];
    const fd_type &fdij = std.fd[a][jj];
    const aed_type2 &aedi = std.get_aed(a);
    const aed_type2 &aedj = std.get_aed(j);
    v += fpredict(fdij,aedi);
    v -= fpredict(fdji,aedj);
  }
}
force_type M_Tadah_Base::
fpredict(const size_t a, const StDescriptors &std, const Structure &st)
{
  force_type v;
  fpredict(a,v,std,st);
  return v;
}
double M_Tadah_Base::
epredict(const StDescriptors &std)
{
  double energy = 0;
  for (size_t i=0;i<std.naed();++i) {
    energy+=epredict(std.get_aed(i));
  }
  return energy;
}

Structure M_Tadah_Base::
predict(const Config &c, StDescriptors &std, const Structure &st)
{
  if(c.get<bool>("NORM") && !std.normalised
    && c("MODEL")[1].find("Linear") == std::string::npos)
    norm.normalise(std);
  Structure st_(st);
  st_.energy = epredict(std);

  if (c.get<bool>("FORCE") && !c.get<bool>("STRESS")) {
    for (size_t a=0; a<st.natoms(); ++a) {
      st_.atoms[a].force = fpredict(a,std,st);
    }
  }

  else if (c.get<bool>("STRESS") && !c.get<bool>("FORCE")) {
    st_.stress = spredict(std,st);
  }

  else if (c.get<bool>("STRESS") && c.get<bool>("FORCE")) {
    stress_force_predict(std, st_);
  }

  return st_;
}

StructureDB M_Tadah_Base::
predict(const Config &c, StDescriptorsDB &st_desc_db, const StructureDB &stdb)
{
  StructureDB stdb_;
  stdb_.structures.resize(stdb.size());
  #pragma omp parallel for
  for (size_t i=0; i<stdb.size(); ++i) {
    stdb_(i) = predict(c,st_desc_db(i), stdb(i));
  }
  return stdb_;
}

StructureDB M_Tadah_Base::
predict(Config &c, const StructureDB &stdb, DC_Base &dc)
{
  StructureDB stdb_;
  stdb_.structures.resize(stdb.size());
  #pragma omp parallel for
  for (size_t i=0; i<stdb.size(); ++i) {
    StDescriptors st_d = dc.calc(stdb(i));
    stdb_(i) = predict(c,st_d, stdb(i));
  }
  return stdb_;
}

stress_type M_Tadah_Base::
spredict(const StDescriptors &std, const Structure &st)
{
  stress_type s;
  s.set_zero();
  for (size_t a=0; a<st.natoms(); ++a) {
    s += spredict(a,std,st);
  }
  return s;
}
stress_type M_Tadah_Base::
spredict(const size_t a, const StDescriptors &std, const Structure &st)
{
  stress_type s;
  s.set_zero();
  spredict(a,s,std,st);
  return s;
}
void M_Tadah_Base::
spredict(const size_t a, stress_type &s,
         const StDescriptors &std, const Structure &st)
{
  const Vec3d &ri = st.atoms[a].position;
  const aed_type2 &aedi = std.get_aed(a);
  for (size_t jj=0; jj<st.nn_size(a); ++jj) {
    size_t j=st.near_neigh_idx[a][jj];
    const size_t aa = st.get_nn_iindex(a,j,jj);
    const Vec3d &rj = st.nn_pos(a,jj);
    const fd_type &fdij = std.fd[a][jj];
    const fd_type &fdji = std.fd[j][aa];
    const aed_type2 &aedj = std.get_aed(j);
    const force_type fij = fpredict(fdij,aedi);
    const force_type fji = fpredict(fdji,aedj);

    // S_xy = (\sum nn of i) r_i[x]f_i(y) + rj[x]f_j(y)
    // where f_i and f_j are the forces on the atom i and j respectively
    // resutling from the pairwise i-j interaction.
    for (size_t x=0; x<3; ++x) {
      for (size_t y=x; y<3; ++y) {
        s(x,y) -= 0.5*(ri[x]-rj[x])*(fij[y]-fji[y]);
      }
    }
  }
}

void M_Tadah_Base::
stress_force_predict(const StDescriptors &std, Structure &st_)
{
  stress_type s;
  s.set_zero();
  for (size_t a=0; a<st_.natoms(); ++a) {
    force_type v;
    const Vec3d &ri = st_.atoms[a].position;
    const aed_type2 &aedi = std.get_aed(a);
    for (size_t jj=0; jj<st_.nn_size(a); ++jj) {
      size_t j=st_.near_neigh_idx[a][jj];
      const size_t aa = st_.get_nn_iindex(a,j,jj);
      const Vec3d &rj = st_.nn_pos(a,jj);
      const fd_type &fdij = std.fd[a][jj];
      const fd_type &fdji = std.fd[j][aa];
      const aed_type2 &aedj = std.get_aed(j);
      const force_type fij = fpredict(fdij,aedi);
      const force_type fji = fpredict(fdji,aedj);

      v += fpredict(fdij,aedi);
      v -= fpredict(fdji,aedj);

      for (size_t x=0; x<3; ++x) {
        for (size_t y=x; y<3; ++y) {
          s(x,y) -= 0.5*(ri[x]-rj[x])*(fij[y]-fji[y]);
        }
      }
    }
    st_.atoms[a].force = v;
  }
  st_.stress = s;
}
