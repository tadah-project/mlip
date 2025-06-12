#include <vector>
#include <cmath>
#include <tadah/core/core_types.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/fast_cell_frac.h>


#include <tadah/mlip/fast_energy_evaluator.h>
#include <tadah/core/small_vec.h>

using namespace tadah;
using core::Vec3d;

/* ------------------------------------------------------------------ */
double mlip::FastEnergyEvaluator::
operator()( const Structure&            st,
            const core::Context&        ctx,
            const models::DC_Selector&  dc,
            const models::M_Predict&    model ) const
{
    /* ---------- runtime flags exactly like DescriptorsCalc --------- */
    bool       init2b   = ctx.get<bool>("INIT2B");
    bool       initmb   = ctx.get<bool>("INITMB");
    std::size_t size2b  = init2b ? ctx.get<std::size_t>("SIZE2B") : 0;
    std::size_t sizemb  = initmb ? ctx.get<std::size_t>("SIZEMB") : 0;

    init2b = init2b && size2b;
    initmb = initmb && sizemb;

    const double rcut       = ctx.get<double>("RCUTMAX");
    const double rc_sq      = rcut * rcut;
    const double rc2b_sq    = init2b ? std::pow(dc.d2b->get_rcut(),2) : 0.0;
    const double rcmb_sq    = initmb ? std::pow(dc.dmb->get_rcut(),2) : 0.0;
    const int    D          = int(dc.descriptor_size());
    const bool   bias  = ctx.get<bool>("BIAS");

        /* ----------------------------------------------------------------
       Decide search backend                                           */
    bool tilted = std::fabs(st.cell(0,1))+std::fabs(st.cell(0,2))+
                  std::fabs(st.cell(1,0))+std::fabs(st.cell(1,2))+
                  std::fabs(st.cell(2,0))+std::fabs(st.cell(2,1)) > 1e-12;
    bool very_small = !mlip::NNFinder().check_box(st, rcut);

    const bool use_fcl = !tilted && !very_small;   // orthorhombic + box>rcut


    double Etot = 0.0;

#pragma omp parallel reduction(+:Etot)
    {
        core::SmallVec<>  buf(D);
        core::aed_type    aed(buf.data(), D, 1, /*allocated*/ false);
        core::rho_type    rho;

        if(initmb) dc.dmb->init_rhoi(rho);

        /* ------------------------------------------------------------
           choose neighbour iterator                                   */
        if(use_fcl)
        {   /* ---------- fast fractional cell list ------------------ */
          mlip::FastCellFrac cl(st, rcut);
#pragma omp for schedule(static)
          for(std::size_t i=0;i<st.natoms();++i)
          {
            aed.set_zero(); if(bias) aed(0)=1.0;
            if(initmb) rho.set_zero();
            int Zi = st.atoms[i].Z;

            cl.for_neigh(i, rc_sq, [&](std::size_t j,const Vec3d& del)
                {
                double rsq = del*del;
                if(rsq >= rc_sq) return;           // safety
                double r   = std::sqrt(rsq);
                int    Zj  = st.atoms[j].Z;
                if(init2b && rsq<rc2b_sq)
                  dc.d2b->calc_aed(Zi,Zj,r,rsq,aed,0.5);
                if(initmb && rsq<rcmb_sq)
                  dc.dmb->calc_rho(Zi,Zj,r,rsq,del,rho);
                });

            if(initmb) dc.dmb->calc_aed(rho,aed);
            Etot += model.epredict(aed);
          }
        }
        else
        {
          /* pre-compute cell rows for shift→Cartesian */
          const auto& C = st.cell;
          Vec3d a(C(0,0), C(0,1), C(0,2));
          Vec3d b(C(1,0), C(1,1), C(1,2));
          Vec3d c(C(2,0), C(2,1), C(2,2));

          /* ---------- build neighbour table with new NNFinder ------------ */
          mlip::NNFinder nnf;
          auto           nn  = nnf.build(st, rcut);
          const auto&    tbl = *nn;

#pragma omp for schedule(static)
          for(std::size_t i = 0; i < st.natoms(); ++i)
          {
            aed.set_zero();
            if(bias) aed(0) = 1.0;
            if(initmb)    rho.set_zero();

            const int Zi = st.atoms[i].Z;

            const std::size_t nn_i = tbl.nn_size(i);
            for(std::size_t k = 0; k < nn_i; ++k)
            {
              const std::size_t j = tbl.idx[i][k];
              const Vec3d& s     = tbl.shift[i][k];

              Vec3d del = st.atoms[i].position -
                (st.atoms[j].position + s[0]*a + s[1]*b + s[2]*c);
              const double rsq = del*del;
              if(rsq >= rc_sq) continue;           // safety

              const double r  = std::sqrt(rsq);
              const int    Zj = st.atoms[j].Z;

              if(init2b && rsq < rc2b_sq)
                dc.d2b->calc_aed(Zi,Zj,r,rsq,aed,0.5);

              if(initmb && rsq < rcmb_sq) {
                dc.dmb->calc_rho(Zi,Zj,r,rsq,del,rho);
              }
            }

            if(initmb) dc.dmb->calc_aed(rho,aed);

            double Ei = model.epredict(aed);
            Etot += Ei;
          }
        }
    }
    return Etot;
}
