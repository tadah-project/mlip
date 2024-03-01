#ifndef NORMALISER_H
#define NORMALISER_H

#include "st_descriptors_db.h"
#include "../CORE/normaliser_core.h"
#include "../CORE/config/config.h"
#include "../CORE/core_types.h"

#include <limits>
#include <stdexcept>

class Normaliser: public Normaliser_Core {
    public:
        using Normaliser_Core::normalise;

        Normaliser ():
            Normaliser_Core()
    {};
        Normaliser (Config &c):
            Normaliser_Core(c)
    {};
        Normaliser(Config &c,StDescriptorsDB &st_desc_db):
            Normaliser_Core(c)
        {
            learn(st_desc_db);

            // normalise
            normalise(st_desc_db);
        }
        void learn(StDescriptorsDB &st_desc_db) {
            // find total number of aeds
            size_t n=0; // total number of aeds, one aed per atom
            for (size_t s=0; s<st_desc_db.size(); ++s) {
                n+= st_desc_db(s).naed();
            }

            // prep containers
            size_t dim = st_desc_db(0).dim();
            mean = v_type(dim);
            std_dev = v_type(dim);

            //std::cout << "COLS NORM: " << nn << std::endl;
            // compute mean and st_dev
            for (size_t d=0; d<dim; ++d) {
                t_type v(n);
                size_t b=0;
                for (size_t s=0; s<st_desc_db.size(); ++s) {
                    //for (const auto &aed:st_desc_db(s).aed) {
                    for (size_t a=0; a<st_desc_db(s).naed(); ++a) {
                        v(b++) = st_desc_db(s).get_aed(a)(d);
                    }
                }
                //std_dev[d] = std::sqrt((v.array() - v.mean()).square().sum()/(v.size()));
                //std_dev[d] = v.std_dev(v.mean());

                std_dev[d] = v.std_dev(v.mean(), v.size());
                mean[d] = v.mean();
            }


            // Filter out std dev=0 -> set them to inf
            // such that division by stdev gives 0.
            // Do it for all but bias
            size_t b=0;
            if (bias)
                b++;
            for (size_t i=b; i<std_dev.size(); ++i) {
                if (std_dev[i]==0) {
                    std_dev[i] = std::numeric_limits<double>::max();
                    //std_dev[i] = std::numeric_limits<double>::infinity();
                }
            }

            if (verbose) std::cout << std::endl << "NORMALISER STDEV  : " << std_dev << std::endl;
            if (verbose) std::cout << std::endl << "NORMALISER MEAN   :" << mean << std::endl;

        }


        void normalise(StDescriptors &st_d) {
            if (st_d.normalised)
                throw std::runtime_error("StDescriptors object is already normalised\n");

            //for (auto &aed: st_d.aed) normalise(aed);
            for (size_t i=0; i<st_d.naed(); ++i) normalise(st_d.get_aed(i));
            for (auto &fdv: st_d.fd)
                for (auto &fd: fdv)
                    normalise(fd);
            st_d.normalised=true;
        }

        void normalise(StDescriptorsDB &st_desc_db) {
            for (size_t s=0; s<st_desc_db.size(); ++s)
                normalise(st_desc_db(s));
        }

};
#endif
