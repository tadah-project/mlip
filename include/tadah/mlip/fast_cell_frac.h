#pragma once
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

        // Rows-as-lattice-vectors convention: r = cell^T * frac, so frac = (cell^T)^{-1} * r.
        Cinv_ = st.cell.transpose().inverse();
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
                  core::Vec3d del = ri -
                        (st_.atoms[j].position + dx*a_ + dy*b_ + dz*c_);
                  if(del*del < rc_sq) fn(std::size_t(j),del);
              }
          }
    }
};
} // namespace tadah::mlip

