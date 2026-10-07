#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/parallel.h>
#include <tadah/models/functions/weights_fit.h>
namespace tadah {
namespace mlip {

void M_Tadah_Base::
require_weights_fit(const tadah::core::Context &c, std::size_t dsize) const
{
  tadah::models::require_weights_fit(
      c, dsize, function(),
      tadah::models::TrainedSizes{weights.size(), norm.mean.size(),
                                  norm.std_dev.size()});
}

void M_Tadah_Base::
fpredict(const size_t a, tadah::core::force_type &v,
         const StDescriptors &std, const Structure &stOrig)
{
  for (size_t jj=0; jj<stOrig.nn_size(a); ++jj) {
    size_t j=stOrig.nn->idx[a][jj];
    const size_t aa = stOrig.get_nn_iindex(a,j,jj);
    const tadah::core::fd_type &fdji = std.fd[j][aa];
    const tadah::core::fd_type &fdij = std.fd[a][jj];
    const tadah::core::aed_type &aedi = std.get_aed(a);
    const tadah::core::aed_type &aedj = std.get_aed(j);
    v += fpredict(fdij,aedi);
    v -= fpredict(fdji,aedj);
  }
}
tadah::core::force_type M_Tadah_Base::
fpredict(const size_t a, const StDescriptors &std, const Structure &stOrig)
{
  tadah::core::force_type v;
  fpredict(a,v,std,stOrig);
  return v;
}
double M_Tadah_Base::
epredict(const StDescriptors &std)
{
  double energy = 0;
  for (size_t i=0;i<std.naed();++i) {
    double Ei =epredict(std.get_aed(i));
    energy+=Ei;
  }
  return energy;
}

Structure M_Tadah_Base::
predict(const tadah::core::Context &c, StDescriptors &std, const Structure &stOrig)
{
  require_weights_fit(c, std.dim());

  if(c.get<bool>("NORM") && !std.normalised
    && c("MODEL")[1].find("Linear") == std::string::npos)
    norm.normalise(std);

  Structure stPred;

  stPred.energy = epredict(std);
  if (c.get<bool>("FORCE") && !c.get<bool>("STRESS")) {
    stPred.atoms.resize(stOrig.natoms());
    for (size_t a=0; a<stOrig.natoms(); ++a) {
      stPred.atoms[a].force = fpredict(a,std,stOrig);
    }
  }

  else if (c.get<bool>("STRESS") && !c.get<bool>("FORCE")) {
    stPred.stress = spredict(std,stOrig);
  }

  else if (c.get<bool>("STRESS") && c.get<bool>("FORCE")) {
    stPred.atoms.resize(stOrig.natoms());
    stress_force_predict(std, stPred, stOrig);
  }
  // spredict and stress_force_predict fill the upper triangle (y >= x);
  // the stress is symmetric.
  if (c.get<bool>("STRESS")) {
    for (size_t x=0; x<3; ++x)
      for (size_t y=x+1; y<3; ++y)
        stPred.stress(y,x) = stPred.stress(x,y);
  }

  return stPred;
}

StructureDB M_Tadah_Base::
predict(const tadah::core::Context &c, StDescriptorsDB &st_desc_db, const StructureDB &stdb)
{
  StructureDB stdb_;
  stdb_.structures.resize(stdb.size());
  // Stage 2.3: shared-pool backend when installed, else the OpenMP loop.
  // Each i writes a disjoint stdb_(i) => order-independent.
  parallel_for(stdb.size(), [&](size_t i) {
    stdb_(i) = predict(c,st_desc_db(i), stdb(i));
  });
  return stdb_;
}

StructureDB M_Tadah_Base::
predict(tadah::core::Context &c, const StructureDB &stdb, DC_Base &dc)
{
  StructureDB stdb_;
  stdb_.structures.resize(stdb.size());
  // Stage 2.3: shared-pool backend when installed, else the OpenMP loop.
  // Each i writes a disjoint stdb_(i) => order-independent.
  parallel_for(stdb.size(), [&](size_t i) {
    stdb_(i) = predict(c, stdb(i), dc);
  });
  return stdb_;
}
Structure M_Tadah_Base::
predict(tadah::core::Context &c, const Structure &st, DC_Base &dc)
{
    StDescriptors st_d = dc.calc(st);
    return predict(c,st_d, st);
}

tadah::core::stress_type M_Tadah_Base::
spredict(const StDescriptors &std, const Structure &stOrig)
{
  tadah::core::stress_type s;
  s.set_zero();
  for (size_t a=0; a<stOrig.natoms(); ++a) {
    s += spredict(a,std,stOrig);
  }
  return s;
}
tadah::core::stress_type M_Tadah_Base::
spredict(const size_t a, const StDescriptors &std, const Structure &stOrig)
{
  tadah::core::stress_type s;
  s.set_zero();
  spredict(a,s,std,stOrig);
  return s;
}
void M_Tadah_Base::
spredict(const size_t a, tadah::core::stress_type &s,
         const StDescriptors &std, const Structure &stOrig)
{
  double V_inv = 1/stOrig.get_volume();
  const tadah::core::Vec3d &ri = stOrig.atoms[a].position;
  const tadah::core::aed_type &aedi = std.get_aed(a);
  for (size_t jj=0; jj<stOrig.nn_size(a); ++jj) {
    size_t j=stOrig.nn->idx[a][jj];
    const size_t aa = stOrig.get_nn_iindex(a,j,jj);
    const tadah::core::Vec3d &rj = stOrig.nn_pos(a,jj);
    const tadah::core::fd_type &fdij = std.fd[a][jj];
    const tadah::core::fd_type &fdji = std.fd[j][aa];
    const tadah::core::aed_type &aedj = std.get_aed(j);
    const tadah::core::force_type fij = fpredict(fdij,aedi);
    const tadah::core::force_type fji = fpredict(fdji,aedj);

    // S_xy = (\sum nn of i) r_i[x]f_i(y) + rj[x]f_j(y)
    // where f_i and f_j are the forces on the atom i and j respectively
    // resulting from the pairwise i-j interaction.
    for (size_t x=0; x<3; ++x) {
      for (size_t y=x; y<3; ++y) {
        s(x,y) -= V_inv*0.5*(ri[x]-rj[x])*(fij[y]-fji[y]);
      }
    }
  }
}

void M_Tadah_Base::
predict_dE_dW(const StDescriptors &std,
              const Structure &stOrig,
              tadah::core::t_type &out)
{
  const size_t n_desc = std.dim();
  if (out.size() != n_desc) {
    out = tadah::core::t_type(n_desc);
  } else {
    for (size_t c = 0; c < n_desc; ++c) out(c) = 0.0;
  }
  for (size_t a = 0; a < stOrig.natoms(); ++a) {
    const tadah::core::aed_type &aedi = std.get_aed(a);
    for (size_t c = 0; c < n_desc; ++c) {
      out(c) += aedi(c);
    }
  }
}

void M_Tadah_Base::
vjp_force(const StDescriptors &std,
          const Structure &stOrig,
          const std::vector<tadah::core::force_type> &lambda,
          tadah::core::t_type &out)
{
  const size_t n_desc = std.dim();
  if (out.size() != n_desc) {
    out = tadah::core::t_type(n_desc);
  } else {
    for (size_t c = 0; c < n_desc; ++c) out(c) = 0.0;
  }
  for (size_t a = 0; a < stOrig.natoms(); ++a) {
    const tadah::core::force_type &lam_a = lambda[a];
    for (size_t jj = 0; jj < stOrig.nn_size(a); ++jj) {
      size_t j = stOrig.nn->idx[a][jj];
      const size_t aa = stOrig.get_nn_iindex(a, j, jj);
      const tadah::core::fd_type &fdij = std.fd[a][jj];
      const tadah::core::fd_type &fdji = std.fd[j][aa];
      for (size_t k = 0; k < 3; ++k) {
        const double lk = lam_a[k];
        for (size_t c = 0; c < n_desc; ++c) {
          out(c) += lk * (-fdij(c, k) + fdji(c, k));
        }
      }
    }
  }
}

void M_Tadah_Base::
stress_force_predict(const StDescriptors &std, Structure &stPred, const Structure &stOrig)
{
  tadah::core::stress_type s;
  s.set_zero();
  double V_inv = 1/stOrig.get_volume();
  for (size_t a=0; a<stOrig.natoms(); ++a) {
    tadah::core::force_type v;
    const tadah::core::Vec3d &ri = stOrig.atoms[a].position;
    const tadah::core::aed_type &aedi = std.get_aed(a);
    for (size_t jj=0; jj<stOrig.nn_size(a); ++jj) {
      size_t j=stOrig.nn->idx[a][jj];
      const size_t aa = stOrig.get_nn_iindex(a,j,jj);
      const tadah::core::Vec3d &rj = stOrig.nn_pos(a,jj);
      const tadah::core::fd_type &fdij = std.fd[a][jj];
      const tadah::core::fd_type &fdji = std.fd[j][aa];
      const tadah::core::aed_type &aedj = std.get_aed(j);
      const tadah::core::force_type fij = fpredict(fdij,aedi);
      const tadah::core::force_type fji = fpredict(fdji,aedj);

      v += fij;
      v -= fji;

      for (size_t x=0; x<3; ++x) {
        for (size_t y=x; y<3; ++y) {
          s(x,y) -= V_inv*0.5*(ri[x]-rj[x])*(fij[y]-fji[y]);
        }
      }
    }
    stPred.atoms[a].force = v;
  }
  stPred.stress = s;
}
} // namespace mlip
} // namespace tadah
