#ifndef DESCRIPTORS_CALC_HPP
#define DESCRIPTORS_CALC_HPP

// This file is includeed back to descriptors_calc.h
// So There is no need to include headers
// Useful for debugging though...
#include "descriptors_calc.h"

#include <cstdio>
template <typename D2, typename D3, typename DM, typename C2, typename C3, typename CM>
template <typename T1, typename T2, typename T3>
DescriptorsCalc<D2,D3,DM,C2,C3,CM>::DescriptorsCalc(Config &c, T1 &t1, T2 &t2, T3 &t3):
    config(c),
    d2(t1),
    d3(t2),
    dm(t3)
{
    if (!config.exist("DSIZE"))
        common_constructor();
    else {
        size_t bias=0;
        if (config.get<bool>("BIAS"))
            bias++;
        if (config.get<bool>("INIT2B")) {
            d2.fidx=bias;
        }
        if (config.get<bool>("INITMB")) {
            dm.fidx=bias+d2.size();
        }
    }
}
template <typename D2, typename D3, typename DM, typename C2, typename C3, typename CM>
DescriptorsCalc<D2,D3,DM,C2,C3,CM>::DescriptorsCalc(Config &c):
    DescriptorsCalc(c,c,c,c)
{
    if (c.get<bool>("INIT2B")) {
        c2 = C2(c.get<double>("RCUT2B"));
        if (!config.exist("RCTYPE2B"))
            config.add("RCTYPE2B",c2.label());
    }
    else {
        c2 = C2(0);
    }
    if (c.get<bool>("INIT3B")) {
        c3 = C3(c.get<double>("RCUT3B"));
        if (!config.exist("RCTYPE3B"))
            config.add("RCTYPE3B",c3.label());
    }
    else {
        c3 = C3(0);
    }
    if (c.get<bool>("INITMB")) {
        cm = CM(c.get<double>("RCUTMB"));
        if (!config.exist("RCTYPEMB"))
            config.add("RCTYPEMB",cm.label());
    }
    else {
        cm = CM(0);
    }
}

template <typename D2, typename D3, typename DM, typename C2, typename C3, typename CM>
        template <typename T1, typename T2, typename T3,typename T4, typename T5, typename T6>
DescriptorsCalc<D2,D3,DM,C2,C3,CM>::DescriptorsCalc(Config &c, T1 &d2, T2 &d3, T3 &dm, T4 &c2, T5 &c3, T6 &cm):
    config(c),
    c2(c2),
    c3(c3),
    cm(cm),
    d2(d2),
    d3(d3),
    dm(dm)
{
    if (c.get<bool>("INIT2B")) {
        if (!config.exist("RCTYPE2B"))
            config.add("RCTYPE2B",c2.label());
    }
    if (c.get<bool>("INIT3B")) {
        if (!config.exist("RCTYPE3B"))
            config.add("RCTYPE3B",c3.label());
    }
    if (c.get<bool>("INITMB")) {
        if (!config.exist("RCTYPEMB"))
            config.add("RCTYPEMB",cm.label());
    }

    if (!config.exist("DSIZE"))
        common_constructor();
}
template <typename D2, typename D3, typename DM, typename C2, typename C3, typename CM>
void DescriptorsCalc<D2,D3,DM,C2,C3,CM>::common_constructor() {
    size_t dsize=0;

    // Calculate total size of the descriptor.
    // Add relevant keys to the config.
    // Set first indices for all types
    size_t bias=0;
    if (config.get<bool>("BIAS"))
        bias++;
    if (config.get<bool>("INIT2B")) {
        config.add("SIZE2B",d2.size());
        if (!config.exist("TYPE2B"))
            config.add("TYPE2B",d2.label());
        dsize+=d2.size();
        d2.fidx=bias;
    }
    else {
        config.add("SIZE2B",0);
    }

    if (config.get<bool>("INIT3B")) {
        config.add("SIZE3B",d3.size());
        if (!config.exist("TYPE3B"))
            config.add("TYPE3B",d3.label());
        dsize+=d3.size();
    }
    else {
        config.add("SIZE3B",0);
    }

    if (config.get<bool>("INITMB")) {
        config.add("SIZEMB",dm.size());
        if (!config.exist("TYPEMB"))
            config.add("TYPEMB",dm.label());
        dsize+=dm.size();
        dm.fidx=bias+d2.size();
    }
    else {
        config.add("SIZEMB",0);
    }

    if (!dsize)
        throw std::runtime_error("The descriptor size is 0, check your config.");

    if (config.get<bool>("BIAS"))
        dsize++;

    config.add("DSIZE",dsize);
}
template <typename D2, typename D3, typename DM, typename C2, typename C3, typename CM>
void DescriptorsCalc<D2,D3,DM,C2,C3,CM>::calc_rho(const Structure &st, StDescriptors &st_d) {
    double rcut_mb_sq = pow(config.get<double>("RCUTMB"),2);
    rhos_type &rhos = st_d.rhos;
    size_t s = dm.rhoi_size()+dm.rhoip_size();
    rhos.resize(s,st.natoms());
    rhos.set_zero();

  Vec3d delij;
    for (size_t i=0; i<st.natoms(); ++i) {
        const Atom &a1 = st(i);

        for (size_t jj=0; jj<st.nn_size(i); ++jj) {
            const Vec3d &a2pos = st.nn_pos(i,jj);
            //delij = a1.position - a2pos;
            delij[0] = a1.position[0] - a2pos[0];
            delij[1] = a1.position[1] - a2pos[1];
            delij[2] = a1.position[2] - a2pos[2];

            //double rij_sq = delij * delij;
            double rij_sq = delij[0]*delij[0] + delij[1]*delij[1] + delij[2]*delij[2];
            if (rij_sq > rcut_mb_sq) continue;
            int Zj = st.near_neigh_atoms[i][jj].Z;
            double rij = sqrt(rij_sq);
            double fc_ij = Zj*cm.calc(rij);
            dm.calc_rho(rij,rij_sq,fc_ij,delij,st_d.get_rho(i));
        }
    }
}

template <typename D2, typename D3, typename DM, typename C2, typename C3, typename CM>
void DescriptorsCalc<D2,D3,DM,C2,C3,CM>::calc(const Structure &st, StDescriptors &st_d) {

    bool init2b = config.get<bool>("INIT2B");
    bool initmb = config.get<bool>("INITMB");

    size_t size2b=0;
    size_t sizemb=0;
    if (init2b) size2b=config.get<size_t>("SIZE2B");
    if (initmb) sizemb=config.get<size_t>("SIZEMB");

    // deal with the case initmb is set to true but dummy is used
    init2b = init2b && size2b;
    initmb = initmb && sizemb;

    size_t bias=0;
    if (config.get<bool>("BIAS"))
        bias++;

    if (initmb)
        calc_rho(st,st_d);

    double rcut_max_sq = pow(config.get<double>("RCUTMAX"),2);
    double rcut_2b_sq = 0.0;
    double rcut_mb_sq = 0.0;

    if (init2b) rcut_2b_sq = pow(config.get<double>("RCUT2B"),2);
    if (initmb) rcut_mb_sq = pow(config.get<double>("RCUTMB"),2);

    // zero all aeds and set bias
    for (size_t i=0; i<st.natoms(); ++i) {
        aed_type2 &aed = st_d.get_aed(i);
        aed.set_zero();
        aed(0)=static_cast<double>(bias);   // set bias
    }

    // calculate many-body energy
    // do it before main loop so rho prime
    // can be calculated
    if (initmb) {
        for (size_t i=0; i<st.natoms(); ++i) {
            aed_type2 &aed = st_d.get_aed(i);
            rho_type& rhoi = st_d.get_rho(i);
            dm.calc_aed(rhoi,aed);
        }
    }

  bool use_force = config.get<bool>("FORCE");
  bool use_stress = config.get<bool>("STRESS");

  Vec3d delij;
    for (size_t i=0; i<st.natoms(); ++i) {
        const Atom &a1 = st(i);
        aed_type2 &aed = st_d.get_aed(i);

        for (size_t jj=0; jj<st.nn_size(i); ++jj) {
            const Vec3d &a2pos = st.nn_pos(i,jj);

            delij[0] = a1.position[0] - a2pos[0];
            delij[1] = a1.position[1] - a2pos[1];
            delij[2] = a1.position[2] - a2pos[2];

            double rij_sq = delij[0]*delij[0] + delij[1]*delij[1] + delij[2]*delij[2];

            if (rij_sq > rcut_max_sq) continue;
            int Zj = st.near_neigh_atoms[i][jj].Z;
            double rij = sqrt(rij_sq);
            double rij_inv = 1.0/rij;

            // CALCULATE TWO-BODY TERM
            if (use_force || use_stress) {
                fd_type &fd_ij = st_d.fd[i][jj];
                if (rij_sq <= rcut_2b_sq && init2b) {
                    double fc_ij = Zj*c2.calc(rij);
                    double fcp_ij = Zj*c2.calc_prime(rij);
                    d2.calc_all(rij,rij_sq,fc_ij,fcp_ij,aed,fd_ij);
                    // Two-body descriptor calculates x-direction only - fd_ij(n,0)
                    // so we have to copy x-dir to y- and z-dir
                    // and scale them by the unit directional vector delij/rij.
                    for (size_t n=bias; n<size2b+bias; ++n) {
                        fd_ij(n,0) *= 0.5*rij_inv;
                        fd_ij(n,1) = fd_ij(n,0)*delij[1];
                        fd_ij(n,2) = fd_ij(n,0)*delij[2];
                        fd_ij(n,0) *= delij[0];
                    }
                }
                // CALCULATE MANY-BODY TERM
                if (rij_sq <= rcut_mb_sq && initmb) {
                    double fc_ij = Zj*cm.calc(rij);
                    double fcp_ij = Zj*cm.calc_prime(rij);
                    rho_type& rhoi = st_d.get_rho(i);
                    int mode = dm.calc_dXijdri(rij,rij_sq,delij,
                            fc_ij,fcp_ij,rhoi,fd_ij);
                    if (mode==0) {
                        // some dm compute x-dir only, similarly to d2 above
                        for (size_t n=size2b+bias; n<size2b+sizemb+bias; ++n) {
                            fd_ij(n,0) *= rij_inv;
                            fd_ij(n,1) = fd_ij(n,0);
                            fd_ij(n,2) = fd_ij(n,0);
                            fd_ij(n,0) *= delij[0];
                            fd_ij(n,1) *= delij[1];
                            fd_ij(n,2) *= delij[2];
                        }
                    }
                }
            }
            else {
                if (rij_sq <= rcut_2b_sq && init2b) {
                    double fc_ij = Zj*c2.calc(rij);
                    d2.calc_aed(rij,rij_sq,fc_ij,aed);
                }
            }

        }
    }
    if (init2b) {
        for (size_t n=0; n<st.natoms(); ++n) {
            for(size_t s=bias; s<bias+d2.size(); ++s) {
                st_d.get_aed(n)(s) *= 0.5;
            }
        }
    }
}
template <typename D2, typename D3, typename DM, typename C2, typename C3, typename CM>
void DescriptorsCalc<D2,D3,DM,C2,C3,CM>::calc_dimer(const Structure &st, StDescriptors &st_d) {

    double r_b = config.get<double>("DIMER",1);
    bool bond_bool = config.get<bool>("DIMER",2);

    // Here we assume that training data consists of 4 atoms
    // The cutoff distance specified in the config file works in the usual way
    // i.e. the maximum distance between two atoms 
    // I,J are molecules, i1,i2 atom in I, similarly for J
    //    I- - - - - - - - - - - -J
    // i1---i2- - - - - - - - -j1---j2
    //    |- - - -r_com- - - - - -|   
    //  |- - - - -r_max- - - - - - -|

    bool init2b = config.get<bool>("INIT2B");
    bool initmb = config.get<bool>("INITMB");
    size_t size2b=0;
    size_t sizemb=0;
    if (init2b) size2b=config.get<size_t>("SIZE2B");
    if (initmb) sizemb=config.get<size_t>("SIZEMB");

    // deal with the case initmb is set to true and dummy is used
    init2b = init2b && size2b;
    initmb = initmb && sizemb;

    //double rcut_max_sq = pow(config.get<double>("RCUTMAX"),2);
    double rcut_2b_sq = 0.0;
    double rcut_mb_sq = 0.0;

    if (init2b) rcut_2b_sq = pow(config.get<double>("RCUT2B"),2);
    if (initmb) rcut_mb_sq = pow(config.get<double>("RCUTMB"),2);

    // Max distance between CoM of two interacting molecules
    double rcut_com_sq = pow(config.get<double>("RCUTMAX")-r_b,2);

    // Not that this differ to how lammps implements this
    // TODO make it consistent between those two
    Mat6R3C delM;  //i1-j1,i2-j1,i1-j2,i2-j2,i1-i2,j1-j2
    double r_sq[6];
    double r[6];

    const std::vector<Atom> &atoms = st.atoms;

    size_t bias=0;
    if (config.get<bool>("BIAS"))
        bias++;

    // TODO weighting factors
    // For now assume all are the same type
    //int Zj = st.near_neigh_atoms[0][0].Z;
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

    // zero all aeds+rho and set bias
    for (size_t i=0; i<4; ++i) {
        aed_type2 &aed = st_d.get_aed(i);
        aed.set_zero();
        aed(0)=static_cast<double>(bias);   // set bias
    }
    if (initmb) {
        size_t s = dm.rhoi_size()+dm.rhoip_size();
        st_d.rhos.resize(s,4);
        st_d.rhos.set_zero();
    }

    Vec3d xicom = 0.5*(atoms[0].position + atoms[1].position);
    Vec3d xjcom = 0.5*(atoms[2].position + atoms[3].position);

    Vec3d del_com = xicom-xjcom;;
    double r_com_sq = del_com * del_com;
    if (r_com_sq > rcut_com_sq) {
        return;
    }

    // compute all 6 distances
    for (size_t n=0; n<6; ++n) {
        delM.row(n) = atoms[idx[n].first].position - atoms[idx[n].second].position;
        r_sq[n] =  delM.row(n) * delM.row(n);
        r[n] = sqrt(r_sq[n]);
    }

    // compute densities and 2b aed for every atom
    // if bond is not included use 0,1,2,3 distances,
    // otherwise use all
    size_t N = bond_bool ? 6 : 4;
    for (size_t n=0; n<N; ++n) {
        if (r_sq[n] <= rcut_mb_sq && initmb) {
            double fcmb = Zj*cm.calc(r[n]);
            dm.calc_rho(r[n],r_sq[n],fcmb,delM.row(n),st_d.get_rho(idx[n].first));
            dm.calc_rho(r[n],r_sq[n],fcmb,-delM.row(n),st_d.get_rho(idx[n].second));
        }
        // Do not compute 2b term between bonded atoms
        if (r_sq[n] <= rcut_2b_sq && init2b) {
            double fc2b = Zj*c2.calc(r[n]);
            d2.calc_aed(r[n],r_sq[n],fc2b,st_d.get_aed(idx[n].first));
            d2.calc_aed(r[n],r_sq[n],fc2b,st_d.get_aed(idx[n].second));
        }
    }

    // calculate many-body aed
    if (initmb) {
        for (size_t i=0; i<4; ++i) {
            aed_type2 &aed = st_d.get_aed(i);
            rho_type& rhoi = st_d.get_rho(i);
            dm.calc_aed(rhoi,aed);
        }
    }

    //if (init2b) {
    //    for (size_t n=0; n<st.natoms(); ++n) {
    //        for(size_t s=bias; s<bias+d2.size(); ++s) {
    //            aeds[n](s) *= 0.5;
    //        }
    //    }
    //}
}

template <typename D2, typename D3, typename DM, typename C2, typename C3, typename CM>
StDescriptors DescriptorsCalc<D2,D3,DM,C2,C3,CM>::calc(const Structure &st) {
    StDescriptors st_d(st, config);
    if (config.get<bool>("DIMER",0))
        calc_dimer(st,st_d);
    else
        calc(st,st_d);
    return st_d;
}

template <typename D2, typename D3, typename DM, typename C2, typename C3, typename CM>
StDescriptorsDB DescriptorsCalc<D2,D3,DM,C2,C3,CM>::calc(const StructureDB &stdb) {
    StDescriptorsDB st_desc_db(stdb, config);

#pragma omp parallel for
    for(size_t i=0; i<stdb.size(); ++i)
    if (config.get<bool>("DIMER",0))
        calc_dimer(stdb(i),st_desc_db(i));
    else
        calc(stdb(i),st_desc_db(i));
    return st_desc_db;
}
#endif
