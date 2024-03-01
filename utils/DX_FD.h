#ifndef TADAH_DX_FD_H
#define TADAH_DX_FD_H
#include "../../CORE/config/config.h"
#include "../../CORE/core_types.h"
#include "../../MODELS/descriptors/d_all.h"
#include "../../MODELS/cutoffs/cut_all.h"
#include "../../LIBS/Eigen/Dense"
#include <vector>

typedef Eigen::VectorXd rhoi_type;
class DX_FD {
    public:
        DX_FD() {};
        template <typename D2, typename C>
            void test_d2() {
                double rcut = 6.0;
                Config config;
                config.remove("INIT2B");
                config.remove("VERBOSE");
                config.remove("BIAS");
                config.add("VERBOSE", 1);
                config.add("INIT2B","true");
                config.add("BIAS","false");
                config.add("RCUT2B",rcut);
                config.add("CGRID2B","-1 5 1.1 5.0");
                //std::cout << config << std::endl;
                C cutoff(rcut);
                D2 d2(config);
                test_d2(&d2,&cutoff);
            }
        void test_d2(D2_Base *d2, Cut_Base *cutoff, double h=1e-4) {

            // Setup
            aed_type2 aed(d2->size());
            aed_type2 aedp(d2->size());
            aed_type2 aedn(d2->size());

            // Initial configuration
            Atom a1("a1",0.1,7.2,1.3,0,0,0);
            Atom a2("a2",2.2,4.1,1.8,0,0,0);
            double r12 = (a1.position-a2.position).norm();
            double fc12 = cutoff->calc(r12);
            double fcp12 = cutoff->calc_prime(r12);  // cutoff prime

            // "Positive" displacement
            double r12p = r12+h;
            double fc12p = cutoff->calc(r12p);

            // "Negative" displacement
            double r12n = r12-h;
            double fc12n = cutoff->calc(r12n);

            d2->calc_aed(r12, r12*r12,fc12,aed);
            d2->calc_aed(r12p, r12p*r12p,fc12p,aedp);
            d2->calc_aed(r12n, r12n*r12n,fc12n,aedn);

            fd_type fd(d2->size(),3);    // true fd
            fd_type fd1a(d2->size(),3);   // finite difference fd(approx)

            d2->calc_dXijdri(r12,r12*r12,fc12,fcp12,fd);
            d2->calc_fd_approx(r12,fc12p,fc12n,aedp,aedn,fd1a,h);

            std::cout << "\nfd:\n" <<  fd << std::endl;
            std::cout << "\nfda:\n" <<  fd1a << std::endl;
            std::cout << "\nfd-fd1a:\n" <<  fd-fd1a << std::endl;
        }
        template <typename DM, typename C>
            void test_dm() {
                double rcut = 5.3;
                Config config;
                config.remove("INITMB");
                config.add("INITMB","true");
                config.remove("BIAS");
                config.add("BIAS","false");
                config.add("RCUTMB",rcut);
                config.add("AGRIDMB",3);
                config.add("SGRIDMB","-2 3 0.01 0.4");
                config.add("CGRIDMB","-1 2 0.0 3.0");
                config.add("SETFL","/home/s1351949/PROJECTS/tadah/workspace/Ta/Ta1_Ravelo_2013.eam.alloy");
                config.remove("VERBOSE");
                config.add("VERBOSE", 0);
                //std::cout << config << std::endl;
                C cutoff(rcut);
                DM dm(config);
                test_dm(&dm,&cutoff);
            }
        void test_dm(DM_Base *dm, Cut_Base *cutoff, double h=1e-4) {

            // Initial configuration
            vec3d r1(0.0,0.1,0.0);
            vec3d r2(1.1,0.3,1.1);
            vec3d r3(3.1,4.2,0.8);
            vec3d r4(2.1,0.0,3.8);

            std::vector<vec3d> rs = {r1,r2,r3,r4};
            std::cout << "  True FD: " << std::endl;
            std::vector<fd_type> fdi = calc_true_fd_dm(dm,cutoff,rs);
            std::cout << "  Finite difference FD: " << std::endl;
            std::vector<fd_type> fda = calc_apporx_fd_dm(dm,cutoff,rs,h);

            std::cout << "  Difference True-Approximation: " << std::endl;
            for (size_t i=0; i<fdi.size(); ++i) {
                std::cout << "fd" << i << ":\n" <<  fdi[i]-fda[i] << std::endl;
            }


        }
        std::vector<fd_type> calc_true_fd_dm(DM_Base *dm, Cut_Base *cutoff, std::vector<vec3d> &rs) {
            vec3d
            // Copy vector to matrix
            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> rsm(rs.size(),3);   // [N,3]
            for (size_t i=0; i<rs.size(); ++i ) {
                for (size_t j=0; j<3; ++j ) {
                    rsm(i,j) = rs[i][j];
                }
            }

            Eigen::Matrix<vec3d,Eigen::Dynamic,Eigen::Dynamic> vecij(rs.size(),rs.size()); // [N,N]
            std::vector<aed_type> aeds(rs.size(), aed_type(dm->size()));
            std::vector<rhoi_type> rhois(rs.size());
            Eigen::Matrix<fd_type,Eigen::Dynamic,Eigen::Dynamic> fdij(rs.size(),rs.size()); // [N,N]
            std::vector<fd_type> fdi(rs.size()); // [N,1]


            // Initialize all data structures
            for (size_t i=0; i<rs.size(); ++i ) {
                dm->init_rhoi(rhois[i]);
                aeds[i].set_zero();
                fdi[i] = fd_type(dm->size(),3);
                fdi[i].set_zero();
                for (size_t j=i+1; j<rs.size(); ++j ) {
                    fdij(i,j) = fd_type(dm->size(),3);
                    fdij(i,j).set_zero();
                }
            }

            for (size_t i=0; i<rs.size(); ++i ) {
                for (size_t j=0; j<rs.size(); ++j ) {
                    vecij(i,j) = rsm.row(i) - rsm.row(j);
                }
            }

            for (size_t i=0; i<rs.size(); ++i ) {

                // Calculate densities
                for (size_t j=i+1; j<rs.size(); ++j ) {
                    vec3d &vij = vecij(i,j);
                    vec3d &vji = vecij(j,i);
                    double rij = vij.norm();
                    double fcij = cutoff->calc(rij);
                    dm->calc_rho(rij,rij*rij,fcij,vij,rhois[i]);
                    dm->calc_rho(rij,rij*rij,fcij,vji,rhois[j]);
                }

                // Calculate energy and derivative of the embedding
                // function at the atom
                dm->calc_aed(rhois[i], aeds[i]);

            }

            for (size_t i=0; i<rs.size(); ++i ) {

                for (size_t j=i+1; j<rs.size(); ++j ) {
                    vec3d &vij = vecij(i,j);
                    double rij = vij.norm();
                    double fcij = cutoff->calc(rij);
                    double fcpij = cutoff->calc_prime(rij);
                    int mode = dm->calc_dXijdri_dXjidri(rij,rij*rij,vij,fcij,fcpij,rhois[i],rhois[j],fdij(i,j));
                    if (mode==0) {
                        for (size_t k=0; k<3; ++k ) {
                            fdi[i].col(k) += vij(k)*fdij(i,j).col(0)/rij;
                            fdi[j].col(k) -= vij(k)*fdij(i,j).col(0)/rij;
                        }
                    }
                    else {
                        fdi[i] += fdij(i,j);
                        fdi[j] -= fdij(i,j);
                    }
                }
                std::cout << "fd" << i << ":\n" <<  fdi[i] << std::endl;
            }

            return fdi;
        }
        std::vector<fd_type> calc_apporx_fd_dm(DM_Base *dm, Cut_Base *cutoff, std::vector<vec3d> &rs, double h) {
            // Copy vector to matrix
            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> rsp(rs.size(),3);   // [N,3]
            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> rsn(rs.size(),3);   // [N,3]

            Eigen::Matrix<vec3d,Eigen::Dynamic,Eigen::Dynamic> vecijp(rs.size(),rs.size()); // [N,N]
            Eigen::Matrix<vec3d,Eigen::Dynamic,Eigen::Dynamic> vecijn(rs.size(),rs.size()); // [N,N]
            std::vector<aed_type> aedp(rs.size(), aed_type(dm->size()));
            std::vector<aed_type> aedn(rs.size(), aed_type(dm->size()));
            std::vector<rhoi_type> rhoip(rs.size());
            std::vector<rhoi_type> rhoin(rs.size());
            std::vector<fd_type> fdi(rs.size()); // [N,1]

            // reset all positions
            for (size_t i=0; i<rs.size(); ++i ) {
                fdi[i] = fd_type(dm->size(),3);
                fdi[i].set_zero();
                dm->init_rhoi(rhoip[i]);
                dm->init_rhoi(rhoin[i]);
                aedn[i].set_zero();
                for (size_t k=0; k<3; ++k ) {
                    rsp(i,k) = rs[i][k];
                    rsn(i,k) = rs[i][k];
                }
            }



            for (size_t ii=0; ii<rs.size(); ++ii ) {
                for (size_t k=0; k<3; ++k ) {

                    // displace atom ii by h in +/-k direction
                    rsp(ii,k) = rs[ii][k] + h;
                    rsn(ii,k) = rs[ii][k] - h;

                    // reset arrays for new k
                    for (size_t i=0; i<rs.size(); ++i ) {
                        aedp[i].set_zero();
                        aedn[i].set_zero();
                        rhoip[i].set_zero();
                        rhoin[i].set_zero);
                    }

                    for (size_t i=0; i<rs.size(); ++i ) {
                        for (size_t j=i+1; j<rs.size(); ++j ) {
                            vecijp(i,j) = rsp.row(i) - rsp.row(j);
                            vecijn(i,j) = rsn.row(i) - rsn.row(j);
                            vec3d vijp = vecijp(i,j);
                            vec3d vijn = vecijn(i,j);
                            vec3d vjip = -vijp;
                            vec3d vjin = -vijn;
                            double rijp = vijp.norm();
                            double rijn = vijn.norm();
                            double fcijp = cutoff->calc(rijp);
                            double fcijn = cutoff->calc(rijn);
                            dm->calc_rho(rijp,rijp*rijp,fcijp,vijp,rhoip[i]);
                            dm->calc_rho(rijp,rijp*rijp,fcijp,vjip,rhoip[j]);
                            dm->calc_rho(rijn,rijn*rijn,fcijn,vijn,rhoin[i]);
                            dm->calc_rho(rijn,rijn*rijn,fcijn,vjin,rhoin[j]);
                        }

                        dm->calc_aed(rhoip[i], aedp[i]);
                        dm->calc_aed(rhoin[i], aedn[i]);
                        fdi[ii].col(k) += (aedp[i]-aedn[i])/(2*h);
                    }
                    // reset back position for next k
                    rsp(ii,k) = rs[ii][k];
                    rsn(ii,k) = rs[ii][k];
                }

                std::cout << "fd" << ii << ":\n" <<  fdi[ii] << std::endl;
            }
            return fdi;
        }


};
class Lattice {
    public:
        double a; // lattice parameter
        double b; // lattice parameter
        double c; // lattice parameter
        double alpha; // Angle
        double betta; // Angle
        double gamma; // Angle
};
#endif

