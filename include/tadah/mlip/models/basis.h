#ifndef BASIS_H
#define BASIS_H

#include <tadah/mlip/st_descriptors_db.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/core/core_types.h>
#include <tadah/core/context.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace tadah {
namespace mlip {

/** \brief The basis vectors of a non-linear M_KRR model.
 *
 *  build_random_basis() (MODEL M_KRR <kernel> 1) and prep_basis_for_krr()
 *  (MODEL M_KRR <kernel> 2) build SBASIS basis vectors, the columns of b:
 *  column 0 is the bias vector e0 = (1, 0, ..., 0), and columns 1 ..
 *  SBASIS-1 are SBASIS-1 candidates drawn without replacement from the
 *  training set: its atomic energy descriptors (AEDs), or for
 *  prep_basis_for_krr the mean AED of each structure.
 *
 *  The draw is deterministic and portable (draw_candidates()): the same
 *  training set (in the same order) and SBASIS give the same basis on
 *  every run and platform. So `tadah train` writes the same BASIS every
 *  time, and the loss of `tadah hpo` at a given point does not depend on
 *  the draw.
 */
template <typename K>
class Basis {
private:
  const tadah::core::Context &context;
public:
  tadah::core::Matrix b;
  tadah::core::t_type T;    // Vectors corresponding to basis vectors

  /** The seed of the draw: the default seed of std::mt19937_64 (5489),
   *  whose output sequence the C++ standard fixes. */
  static constexpr std::uint64_t kSeed = std::mt19937_64::default_seed;

  Basis(const tadah::core::Context &c):
    context(c)
  {}

  void set_basis(tadah::core::Matrix &b_) {
    b=b_;
  }

  /** \brief An integer in [0, n), n >= 1, drawn without bias from the raw
   *  64-bit output of \p rng: r is rejected while r < 2^64 mod n, so the
   *  accepted range holds a whole number of copies of [0, n). */
  static std::uint64_t draw_below(std::mt19937_64 &rng, std::uint64_t n) {
    const std::uint64_t threshold = (std::uint64_t(0) - n) % n;
    for (;;) {
      const std::uint64_t r = rng();
      if (r >= threshold) return r % n;
    }
  }

  /** \brief The positions, in 0 .. n-1, of the k candidates (k <= n) a
   *  basis draws, in the order they fill columns 1 .. k: the first k
   *  steps of a Fisher-Yates shuffle of 0 .. n-1, step i swapping
   *  position i with i + draw_below(rng, n - i), with rng a
   *  std::mt19937_64 seeded with kSeed. No std::random_device, and no
   *  std::shuffle or std:: distribution (their results depend on the
   *  standard library). */
  static std::vector<std::size_t> draw_candidates(std::size_t n, std::size_t k) {
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) idx[i] = i;
    std::mt19937_64 rng(kSeed);
    for (std::size_t i = 0; i < k; ++i) {
      const std::size_t j = i + static_cast<std::size_t>(draw_below(rng, n - i));
      std::swap(idx[i], idx[j]);
    }
    idx.resize(k);
    return idx;
  }

  /** \brief SBASIS s with n candidates: s >= 2 (the bias vector and at
   *  least one drawn vector) and s - 1 <= n. Throws naming SBASIS. */
  static void check_sbasis(std::size_t s, std::size_t n, const std::string &what) {
    if (s < 2)
      throw std::runtime_error(
          "SBASIS " + std::to_string(s) + ": a non-linear M_KRR model needs "
          "at least 2 basis vectors (column 0 is the bias vector e0, the "
          "others are drawn from the " + what + "). Set SBASIS to 2 or more.");
    if (s - 1 > n)
      throw std::runtime_error(
          "SBASIS " + std::to_string(s) + " draws " + std::to_string(s - 1) +
          " basis vectors from the " + what + " (column 0 is the bias vector "
          "e0), but the training set has " + std::to_string(n) + ": set "
          "SBASIS to at most " + std::to_string(n + 1) + ".");
  }

  /** \brief Column 0 the bias vector e0, columns 1 .. s-1 the AEDs at the
   *  positions draw_candidates(n, s-1) of the list of all n AEDs of
   *  st_desc_db (structure by structure, atom by atom). */
  void build_random_basis(size_t s, StDescriptorsDB &st_desc_db) {

    // the AEDs, structure by structure, atom by atom
    std::vector<std::pair<size_t,size_t>> indices;
    for( size_t st = 0; st < st_desc_db.size(); st++ ) {
      for( size_t a = 0; a < st_desc_db(st).naed() ; a++ ) {
        indices.emplace_back(st,a);
      }
    }

    check_sbasis(s, indices.size(), "training AEDs");
    const std::vector<std::size_t> drawn = draw_candidates(indices.size(), s - 1);

    const size_t dim = st_desc_db(indices[drawn[0]].first).dim();
    b.resize(dim,s);
    b.set_zero();
    // set first basis function as "bias vector"
    b(0,0)=1;
    for (size_t i=1; i<s; ++i) {
      const size_t st = indices[drawn[i-1]].first;
      const size_t a = indices[drawn[i-1]].second;
      const tadah::core::aed_type &aed = st_desc_db(st).get_aed(a);
      for (size_t j=0; j<aed.size(); ++j) {
        b(j,i)=aed[j];
      }
    }
  }

  /** \brief Column 0 the bias vector e0 (target 0), columns 1 .. s-1 the
   *  mean AEDs of the structures at the positions draw_candidates(n,
   *  s-1) of the n structures, with their energies per atom in T. */
  void prep_basis_for_krr(StDescriptorsDB &st_desc_db,
                          const StructureDB &stdb) {

    size_t s = context.get<size_t>("SBASIS");

    if (stdb.size() != st_desc_db.size()) {
      throw std::runtime_error("The size of StructureDB is different \n \
from StDescriptorsDB. This is likely to be a bug.\n");
    }

    check_sbasis(s, stdb.size(), "training structures");
    const std::vector<std::size_t> drawn = draw_candidates(stdb.size(), s - 1);

    b.resize(st_desc_db(drawn[0]).dim(),s);
    T.resize(s);
    T.set_zero();
    b.set_zero();
    b(0,0)=1;
    // Here we add all atomic energy descriptors into single
    // descriptor which represents total energy of this configuration
    for (size_t i=1; i<s; ++i) {
      const size_t st = drawn[i-1];
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
