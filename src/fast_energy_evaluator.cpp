#pragma once
#include <vector>
#include <cmath>
#include <tadah/core/core_types.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/nn_finder.h>

namespace tadah::mlip
{
/* cell-list in fractional space (fx,fy,fz) ∈ [0,1)³
 * – works for orthorhombic and triclinic cells                               */
class FastCellFrac
{
    const Structure& st_;
    double cutoff_;
    int nx_, ny_, nz_;                         // bins per axis
    std::vector<int> head_, next_;

    core::Matrix3d Cinv_;                      // inverse cell
    core::Vec3d    a_, b_, c_;                 // cell rows

    struct Frac { double x,y,z; };
    std::vector<Frac> f_;                      // wrapped fractional coords

    static double row_len(const core::Matrix3d& C,int r)
    { return std::sqrt(C(r,0)*C(r,0)+C(r,1)*C(r,1)+C(r,2)*C(r,2)); }

    inline int bin(int ix,int iy,int iz) const
    {
        auto w=[&](int k,int n){ return ((k%n)+n)%n; };
        return w(ix,nx_) + nx_*( w(iy,ny_) + ny_*w(iz,nz_) );
    }

public:
    FastCellFrac(const Structure& st,double cutoff) :
        st_(st), cutoff_(cutoff)
    {
        /* ---- bin counts so that bin width ≤ rcut ------------------- */
        double Lmax = std::max({row_len(st.cell,0),
                                row_len(st.cell,1),
                                row_len(st.cell,2)});
        double df = cutoff / Lmax;
        nx_ = std::max(1, int(std::floor(1.0/df)));
        ny_ = nx_;  nz_ = nx_;

        std::size_t N = st.natoms();
        head_.assign(std::size_t(nx_*ny_*nz_), -1);
        next_.assign(N,-1);

        Cinv_ = st.cell.inverse();
        a_ = st.cell.row(0);  b_ = st.cell.row(1);  c_ = st.cell.row(2);

        f_.resize(N);
        for(std::size_t i=0;i<N;++i)
        {
            auto frac=[&](int r,const core::Vec3d& r0){
                return Cinv_(r,0)*r0[0]+Cinv_(r,1)*r0[1]+Cinv_(r,2)*r0[2];};
            const auto& r = st(i).position;
            double fx=frac(0,r), fy=frac(1,r), fz=frac(2,r);
            fx-=std::floor(fx); fy-=std::floor(fy); fz-=std::floor(fz);
            f_[i]={fx,fy,fz};
            int ix=int(fx*nx_), iy=int(fy*ny_), iz=int(fz*nz_);
            int b = bin(ix,iy,iz);
            next_[i]=head_[b]; head_[b]=int(i);
        }
    }

    template<class F>
    void for_neigh(std::size_t i,double rc_sq,const F& fn) const
    {
        /* image span identical to NNFinder::num_shifts --------------- */
        int Ns[3];  { NNFinder tmp; tmp.num_shifts(st_,Ns,cutoff_); }
        auto bs=[&](int n,int Ns){ return std::max(1,(Ns+n-1)/n); };
        int sx=bs(nx_,Ns[0]), sy=bs(ny_,Ns[1]), sz=bs(nz_,Ns[2]);

        int ix0=int(f_[i].x*nx_), iy0=int(f_[i].y*ny_), iz0=int(f_[i].z*nz_);
        const auto& ri=st_.atoms[i].position;

        for(int dx=-sx;dx<=sx;++dx)
         for(int dy=-sy;dy<=sy;++dy)
          for(int dz=-sz;dz<=sz;++dz)
          {
              int b = bin(ix0+dx, iy0+dy, iz0+dz);
              for(int j=head_[b]; j!=-1; j=next_[j])
              {
                  if(std::size_t(j)==i) continue;
                  core::Vec3d del =
                        st_.atoms[j].position + dx*a_ + dy*b_ + dz*c_ - ri;
                  if(del*del < rc_sq) fn(std::size_t(j),del);
              }
          }
    }
};
} // namespace tadah::mlip



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


#pragma omp for schedule(static)
          for(std::size_t i = 0; i < st.natoms(); ++i)
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
            aed.set_zero();
            if(bias) aed(0) = 1.0;
            if(initmb)    rho.set_zero();

            const int Zi = st.atoms[i].Z;

            const std::size_t nn_i = tbl.nn_size(i);
            for(std::size_t k = 0; k < nn_i; ++k)
            {
              const std::size_t j = tbl.idx[i][k];
              const Vec3d& s     = tbl.shift[i][k];

              Vec3d del = st.atoms[j].position + s[0]*a + s[1]*b + s[2]*c
                - st.atoms[i].position;
              const double rsq = del*del;
              if(rsq >= rc_sq) continue;           // safety

              const double r  = std::sqrt(rsq);
              const int    Zj = st.atoms[j].Z;

              if(init2b && rsq < rc2b_sq)
                dc.d2b->calc_aed(Zi,Zj,r,rsq,aed,0.5);

              if(initmb && rsq < rcmb_sq)
                dc.dmb->calc_rho(Zi,Zj,r,rsq,del,rho);
            }

            if(initmb)
              dc.dmb->calc_aed(rho,aed);

            double Ei = model.epredict(aed);
            Etot += Ei;
          }
        }
    }
    return Etot;
}
