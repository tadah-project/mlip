#ifndef BASIS_H
#define BASIS_H

#include "../../CORE/config/config.h"
#include "../../CORE/core_types.h"
#include "../st_descriptors_db.h"
#include "../st_descriptors.h"

#include <numeric>
#include <stdexcept>
#include <vector>

template <typename K>
class Basis {
    private:
        Config &config;
        int verbose;
    public:
       Matrix b;
       t_type T;    // Vectors corresponding to basis vectors
        Basis(Config &c):
            config(c),
            verbose(c.get<int>("VERBOSE"))

    {}

        void set_basis(Matrix &b_) {
            b=b_;
        }
        void build_random_basis(size_t s, StDescriptorsDB &st_desc_db) {

            // generate indices
            std::vector<std::tuple<size_t,size_t>> indices;
            size_t counter=0;
            for( size_t st = 0; st < st_desc_db.size(); st++ ) {
                for( size_t a = 0; a < st_desc_db(st).naed() ; a++ ) {
                    indices.push_back(std::tuple<size_t,size_t>(st,a));
                    counter++;
                }
            }

            if (counter < s) {
                throw std::runtime_error("The number of requestd basis vectors is\n \
                        larger than the amount of available AEDs\n");
            }

            std::random_shuffle(indices.begin(), indices.end());

            b.resize(st_desc_db(0).dim(),s);
            b.set_zero();
            // set first basis function as "bias vector"
            b(0,0)=1;
            for (size_t i=1; i<s; ++i) {
                size_t st = std::get<0>(indices[i]);
                size_t a = std::get<1>(indices[i]);
                const aed_type2 &aed = st_desc_db(st).get_aed(a);
                for (size_t j=0; j<aed.size(); ++j) {
                    b(j,i)=aed[j];
                }
            }
        }
        void prep_basis_for_krr(StDescriptorsDB &st_desc_db,
                StructureDB &stdb) {

            size_t s = config.get<size_t>("SBASIS");

            if (stdb.size() != st_desc_db.size()) {
                throw std::runtime_error("The size of StructureDB is different \n \
                        from StDescriptorsDB. This is likely to be a bug.\n");
            }

            // generate indices
            std::vector<size_t> indices;
            for( size_t st = 0; st < stdb.size(); st++ ) {
                indices.push_back(st);
            }

            if (indices.size() < s) {
                throw std::runtime_error("The number of requestd basis vectors is\n \
                        larger than the amount of available AEDs\n");
            }

            std::random_shuffle(indices.begin(), indices.end());

            b.resize(st_desc_db(0).dim(),s);
            T.resize(s);
            b.set_zero();
            b(0,0)=1;
            // Here we add all atomic energy descriptors into single
            // descriptor which represents total energy of this configuration
            for (size_t i=1; i<s; ++i) {
                const size_t st = indices[i];
                T(i)=stdb(st).energy/st_desc_db(st).naed();
                for( size_t a=0; a<st_desc_db(st).naed(); a++ ) {
                    const aed_type2 &aed = st_desc_db(st).get_aed(a);
                    for (size_t j=0; j<aed.size(); ++j) {
                        b(j,i)+=aed[j]/st_desc_db(st).naed();
                    }
                }
            }
        }
};
#endif
