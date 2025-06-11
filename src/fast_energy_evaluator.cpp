#include <tadah/mlip/fast_energy_evaluator.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/core/small_vec.h>

#include <memory>



using namespace tadah;
using core::Vec3d;

double mlip::FastEnergyEvaluator::operator()(const Structure&        st,
    const core::Context&    ctx,
    const models::DC_Selector& dc,
    const models::M_Predict&  model) const
{
  // Structure& st = const_cast<Structure&>(stConst);   // NNFinder needs non-const

  /* ---- runtime flags exactly as in DescriptorsCalc ---------------- */
  bool init2b = ctx.get<bool>("INIT2B");
  bool initmb = ctx.get<bool>("INITMB");
  std::size_t size2b = init2b ? ctx.get<std::size_t>("SIZE2B") : 0;
  std::size_t sizemb = initmb ? ctx.get<std::size_t>("SIZEMB") : 0;
  init2b = init2b && size2b;
  initmb = initmb && sizemb;

  const double rcut          = ctx.get<double>("RCUTMAX");
  const double rcut_sq       = rcut * rcut;
  const double rcut2b_sq     = init2b ? std::pow(dc.d2b->get_rcut(),2) : 0.0;
  const double rcutmb_sq     = initmb ? std::pow(dc.dmb->get_rcut(),2) : 0.0;
  const int    D             = int(dc.descriptor_size());
  const int    bias          = ctx.get<bool>("BIAS") ? 1 : 0;

  /* ---- canonical neighbour list ----------------------------------- */
  mlip::NNFinder nnf;
  auto nn = nnf.build(st, rcut);
  const auto& tbl = *nn;

  /* cell rows for shift → Cartesian */
  const auto& C = st.cell;
  Vec3d a(C(0,0), C(0,1), C(0,2));
  Vec3d b(C(1,0), C(1,1), C(1,2));
  Vec3d c(C(2,0), C(2,1), C(2,2));

  double Etot = 0.0;
#pragma omp parallel reduction(+:Etot)
  {
    core::SmallVec<>  aedBuf(D);
    /* 2.  build a non-owning aed_type view of that memory */
    core::aed_type    aed(aedBuf.data(), D, 1, /*allocated*/ false);
    core::rho_type rho;
    if(initmb) dc.dmb->init_rhoi(rho);

#pragma omp for schedule(static)
    for(std::size_t i = 0; i < st.natoms(); ++i)
    {
      aed.set_zero();
      if(bias) aed(0) = 1.0;
      if(initmb) rho.set_zero();

      const int Zi        = st.atoms[i].Z;
      const Vec3d& ri     = st.atoms[i].position;
      const std::size_t nn_i = tbl.nn_size(i);

      for(std::size_t k = 0; k < nn_i; ++k)
      {
        const std::size_t j = tbl.idx[i][k];
        const Vec3d& sft    = tbl.shift[i][k];

        Vec3d del = st.atoms[j].position + sft[0]*a + sft[1]*b + sft[2]*c - ri;
        const double rsq = del*del;
        if(rsq > rcut_sq) continue;          // safety

        const double r = std::sqrt(rsq);
        const int Zj   = st.atoms[j].Z;

        if(init2b && rsq <= rcut2b_sq)
          dc.d2b->calc_aed(Zi, Zj, r, rsq, aed, 0.5);   // ½ factor!

        if(initmb && rsq <= rcutmb_sq)
          dc.dmb->calc_rho(Zi, Zj, r, rsq, del, rho);   // scale = 1
      }

      if(initmb)
        dc.dmb->calc_aed(rho, aed);

      Etot += model.epredict(aed);
    }
  }

  // st.nn.reset();     // drop temporary list
  return Etot;
}
