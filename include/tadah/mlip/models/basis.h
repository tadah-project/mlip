#ifndef BASIS_H
#define BASIS_H

#include <tadah/mlip/st_descriptors_db.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/core/core_types.h>
#include <tadah/core/context.h>
#include <tadah/core/intercept_layout.h>

#include <algorithm>
#include <string>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace tadah {
namespace mlip {

template <typename K>
class Basis {
private:
  const tadah::core::Context &context;
public:
  tadah::core::Matrix b;
  tadah::core::t_type T;    // Vectors corresponding to basis vectors
  Basis(const tadah::core::Context &c):
    context(c)
  {}

  void set_basis(tadah::core::Matrix &b_) {
    b=b_;
  }
  /** The basis starts with the intercept directions e_0 .. e_{n-1}: one per
   *  intercept column (ATOM species) with BIAS true; e_0 with BIAS false,
   *  as before per-species intercepts. Throws when SBASIS @p s cannot hold
   *  them. */
  size_t n_intercept_vectors(size_t s) const {
    const size_t n_int = tadah::core::n_intercept_columns(context);
    const size_t n = n_int ? n_int : 1;
    if (s < n)
      throw std::runtime_error("SBASIS " + std::to_string(s) +
          " cannot hold the " + std::to_string(n) + " intercept basis "
          "vector" + (n > 1 ? "s" : "") + " (one per ATOM species with BIAS "
          "true): SBASIS must be at least " + std::to_string(n));
    return n;
  }
  void build_random_basis(size_t s, StDescriptorsDB &st_desc_db) {
    const size_t n0 = n_intercept_vectors(s);

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
      throw std::runtime_error("The number of requested basis vectors is\n \
larger than the amount of available AEDs\n");
    }

    std::random_device rd;
    std::default_random_engine rng(rd()); // Initialize random engine
    std::shuffle(indices.begin(), indices.end(),rng);

    b.resize(st_desc_db(0).dim(),s);
    b.set_zero();
    // the first basis vectors are the intercept ("bias") vectors
    for (size_t k=0; k<n0; ++k) b(k,k)=1;
    for (size_t i=n0; i<s; ++i) {
      size_t st = std::get<0>(indices[i]);
      size_t a = std::get<1>(indices[i]);
      const tadah::core::aed_type &aed = st_desc_db(st).get_aed(a);
      for (size_t j=0; j<aed.size(); ++j) {
        b(j,i)=aed[j];
      }
    }
  }
  void prep_basis_for_krr(StDescriptorsDB &st_desc_db,
                          const StructureDB &stdb) {

    size_t s = context.get<size_t>("SBASIS");
    const size_t n0 = n_intercept_vectors(s);

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
      throw std::runtime_error("The number of requested basis vectors is\n \
larger than the amount of available AEDs\n");
    }

    std::random_device rd;
    std::default_random_engine rng(rd()); // Initialize random engine
    std::shuffle(indices.begin(), indices.end(),rng);

    b.resize(st_desc_db(0).dim(),s);
    T.resize(s);
    b.set_zero();
    // the first basis vectors are the intercept ("bias") vectors, target 0
    for (size_t k=0; k<n0; ++k) {
      b(k,k)=1;
      T(k)=0;
    }
    // Here we add all atomic energy descriptors into single
    // descriptor which represents total energy of this configuration
    for (size_t i=n0; i<s; ++i) {
      const size_t st = indices[i];
      T(i)=stdb(st).energy/st_desc_db(st).naed();
      for( size_t a=0; a<st_desc_db(st).naed(); a++ ) {
        const tadah::core::aed_type &aed = st_desc_db(st).get_aed(a);
        for (size_t j=0; j<aed.size(); ++j) {
          b(j,i)+=aed[j]/st_desc_db(st).naed();
        }
      }
    }
  }
};
} // namespace mlip
} // namespace tadah
#endif
