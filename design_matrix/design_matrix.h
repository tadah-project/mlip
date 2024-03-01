#ifndef DESIGN_MATRIX_H
#define DESIGN_MATRIX_H

#include <stdexcept>
#include "../../CORE/config/config.h"
#include "../st_descriptors_db.h"
#include "../structure_db.h"
#include "../descriptors_calc.h"
#include "../normaliser.h"


/** Design Matrix \f$ \Phi \f$ and the corresponding target vector \f$ \mathrm{T} \f$
 *
 * An aggregation matrix composed of \phi_i matrices
 *
 *     Phi = | \phi_1 |
 *           | \phi_2 |
 *           | ...    |
 *           | \phi_N |
 *
 * where N corresponds to the number of structures in a provided StructureDB.
 *
 * The ordering in \f$ \phi_i \f$
 *
 *     | energy of the 1st structure
 *     | force on the 1st atom in the x-dir
 *     | force on the 1st atom in the y-dir
 *     | force on the 1st atom in the z-dir
 *     | force on the 2st atom in the x-dir
 *     | ...
 *     | force on the n-th atom in the z-dir
 *     | stress xx
 *     | stress xy
 *     | stress xz
 *     | stress yy
 *     | stress yz
 *     | stress zz
 *
 * We need to know: \ref STRESS, \ref FORCE
 *
 * Scalling factors \ref FWEIGHT, \ref  SWEIGHT
 *
 * Default scalling:
 *  - Forces are scaled by 1/N_i/3 where N_i is a number of atoms in the i-th structure.
 *  - Stresses are scaled by 1/6.
 *  - Energies are scaled by 1/N_i
 *
 * @tparam F Function_Base child -> BF_Base or Kern_Base
 *
 */
template <typename F>
class DesignMatrix {

    public:

        F f;
        phi_type Phi;
        t_type T;
        t_type Tlabels; // 0-Energy, 1-Force, 2-Stress
        bool scale=true;    // Control escale,fscale,sscale

        double e_std_dev=1;
        double f_std_dev=1;
        double s_std_dev[6] = {1,1,1,1,1,1};

        /** \brief This constructor fully initialise this object
         *
         * This class is used to build a Design Matrix.
         *
         * Usage example:
         * \code{.cpp}
         * Config config("Config");
         * BF_Linear bf(config);
         * DesignMatrix<LinearKernel> desmat(bf, config);
         *
         * \endcode
         */
        DesignMatrix(F &f, Config &c):
            f(f),
            config(c),
            force(config.get<bool>("FORCE")),
            stress(config.get<bool>("STRESS")),
            verbose(config.get<int>("VERBOSE")),
            eweightglob(config.get<double>("EWEIGHT")),
            fweightglob(config.get<double>("FWEIGHT")),
            sweightglob(config.get<double>("SWEIGHT"))
    {
        if (config.exist("ESTDEV"))
            e_std_dev = config.get<double>("ESTDEV");
        if (config.exist("FSTDEV"))
            f_std_dev = config.get<double>("FSTDEV");
        if (config.exist("SSTDEV"))
            config.get<double[6]>("SSTDEV", s_std_dev);
    }


        /** \brief Build design matrix from already calculated StDescriptorsDB
         *
         * Here we simply build matrix from already calculated descriptors.
         * The vector of targets  **T** is build from StructureDB.
         * D2, D3 and DM calculators are not used.
         */
        void build(StDescriptorsDB &st_desc_db, const StructureDB &stdb) {
            resize(stdb);
            compute_stdevs(stdb);
            fill_T(stdb);
            std::vector<size_t> rows(stdb.size());
            size_t row=0;
            for (size_t s=0; s<stdb.size(); ++s) {
                rows[s] = row;
                row++;
                if (force) {
                    for (size_t a=0; a<stdb(s).natoms(); ++a) {
                        row+=3;
                    }
                }
                if (stress) {
                    row+=6;
                }
            }
            // loop over all structures and build
#pragma omp parallel for
            for (size_t s=0; s<stdb.size(); ++s) {
                build(rows[s], stdb(s),st_desc_db(s));
            }
        };

        /** \brief Calculate descriptors and build design matrix. */
        template <typename DC>
        void build(const StructureDB &stdb, Normaliser &norm,
                DC &dc) {
            //DescriptorsCalc<D2,D3,DM,C2,C3,CM> dc(config);
            resize(stdb);   // call after dc set DSIZE key
            compute_stdevs(stdb);
            fill_T(stdb);
            // for opm we need to find first rows for each structure

            std::vector<size_t> rows(stdb.size());
            size_t row=0;
            for (size_t s=0; s<stdb.size(); ++s) {
                rows[s] = row;
                row++;
                if (force) {
                    for (size_t a=0; a<stdb(s).natoms(); ++a) {
                        row+=3;
                    }
                }
                if (stress) {
                    row+=6;
                }
            }
#pragma omp parallel for
            for (size_t s=0; s<stdb.size(); ++s) {
                StDescriptors st_d = dc.calc(stdb(s));

                if(config.get<bool>("NORM"))
                    norm.normalise(st_d);

                build(rows[s], stdb(s), st_d);
            }

        }

    private:
        Config &config;
        size_t rows=0;
        size_t cols=0;
        bool force;
        bool stress;
        int verbose;
        double eweightglob;
        double fweightglob;
        double sweightglob;

        // resize Phi and T and set all elements to zero
        void resize(const StructureDB &stdb) {
            rows=0;
            cols = f.get_phi_cols(config);

            for (size_t s=0; s<stdb.size(); ++s) {
                rows++;
                if (stress) rows+=6;
                if (force) rows+=stdb(s).natoms()*3;
            }

            T.resize(rows);
            Tlabels.resize(rows);
            Phi.resize(rows,cols);

            if (verbose) {
                std::cout << "Phi rows: "<< Phi.rows() << std::endl;
                std::cout << "Phi cols: "<< Phi.cols() << std::endl;
            }

            T.set_zero();
            Phi.set_zero();

        }

        void compute_stdevs(const StructureDB &stdb) {
            t_type evec(stdb.size());
            Matrix svec(stdb.size(),6);
            size_t num_forces=0;
            for (size_t s=0; s<stdb.size(); ++s) {
                evec(s) = stdb(s).energy/stdb(s).natoms();
                if (stress) {
                    size_t j=0;
                    for (size_t x=0; x<3; ++x) {
                        for (size_t y=x; y<3; ++y) {
                            svec(s,j++)=stdb(s).stress(x,y);
                        }
                    }
                }
                num_forces +=3*stdb(s).natoms();
            }

            //e_std_dev = std::sqrt((evec - evec.mean()).square().sum()/(evec.size()-1));
            e_std_dev = evec.std_dev(evec.mean(), evec.size()-1);
            if (verbose) std::cout << "Energy standard deviation (eV/atom): " << e_std_dev << std::endl;

            if (stress) {
                for (size_t j=0; j<6; ++j) {
                    s_std_dev[j] = svec.col(j).std_dev(svec.col(j).mean(), svec.col(j).size()-1);
                    if (verbose) std::cout << "Stress standard deviation (eV): " << s_std_dev[j] << std::endl;
                }
            }


            if (force) {
                size_t j=0;
                t_type fvec(num_forces);
                for (size_t s=0; s<stdb.size(); ++s) {
                    for (const Atom &a : stdb(s).atoms) {
                        fvec(j++) = a.force(0);
                        fvec(j++) = a.force(1);
                        fvec(j++) = a.force(2);
                    }
                }
                //fvec /= e_std_dev;
                //svec /= e_std_dev;
                // e_std_dev has units of energy
                // f_std_dev has units of inverse distance
                f_std_dev = fvec.std_dev(fvec.mean(),fvec.size()-1);
                if (verbose) std::cout << "Force standard deviation (1/A): " << f_std_dev << std::endl;
            }

            config.add("ESTDEV",e_std_dev);
            config.add("FSTDEV",f_std_dev);
            for (size_t j=0; j<6; ++j)
                config.add("SSTDEV",s_std_dev[j]);
        }
        void fill_T(const StructureDB &stdb) {
            size_t j=0;
            for (size_t s=0; s<stdb.size(); ++s) {


                double escale = 1;

                if (scale) escale = stdb(s).eweight*eweightglob/stdb(s).natoms();///e_std_dev;
                Tlabels(j) = 0;
                T(j++) = stdb(s).energy*escale;

                if (force) {
                    double fscale = 1;
                    if (scale) fscale = stdb(s).fweight*fweightglob/stdb(s).natoms()/3.0;///e_std_dev/f_std_dev;
                    for (const Atom &a : stdb(s).atoms) {
                        Tlabels(j) = 1;
                        T(j++) = a.force(0)*fscale;
                        Tlabels(j) = 1;
                        T(j++) = a.force(1)*fscale;
                        Tlabels(j) = 1;
                        T(j++) = a.force(2)*fscale;
                    }
                }
                if (stress) {
                    double sscale_arr[6] {1,1,1,1,1,1};
                    if (scale)
                        for(size_t xy=0;xy<6;++xy)
                            sscale_arr[xy] = stdb(s).sweight*sweightglob/6.0;///e_std_dev/s_std_dev[xy];
                    size_t xy=0;
                    for (size_t x=0; x<3; ++x) {
                        for (size_t y=x; y<3; ++y) {
                            Tlabels(j) = 2;
                            T(j++)=stdb(s).stress(x,y)*sscale_arr[xy++];
                        }
                    }
                }
            }
        }
        void build(size_t &row, const Structure &st, const StDescriptors &st_d) {

            double escale = 1;

            if (scale) escale = st.eweight*eweightglob/st.natoms();
            f.calc_phi_energy_row(Phi,row,escale,st,st_d);
            if (force) {
                double fscale = 1;
                if (scale) fscale = st.fweight*fweightglob/st.natoms()/3.0;///f_std_dev;
                f.calc_phi_force_rows(Phi,row,fscale,st,st_d);
            }
            if (stress) {
                double sscale_arr[6] {1,1,1,1,1,1};
                if (scale)
                    for(size_t xy=0;xy<6;++xy)
                        sscale_arr[xy] = st.sweight*sweightglob/6.0;///s_std_dev[xy];
                f.calc_phi_stress_rows(Phi,row,sscale_arr,st,st_d);
            }
        }
};
#endif
