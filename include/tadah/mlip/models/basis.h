#ifndef BASIS_H
#define BASIS_H

#include <tadah/mlip/st_descriptors_db.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/core/core_types.h>
#include <tadah/core/context.h>
#include <tadah/core/intercept_layout.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace tadah {
namespace mlip {

/** \brief The number n0 of intercept ("bias") vectors e_0 .. e_{n0-1}
 *  that the basis of a non-linear M_KRR model of @p ctx starts with: one
 *  per intercept column (ATOM species) with BIAS true, e_0 alone with BIAS
 *  false. */
inline std::size_t n_intercept_basis_vectors(const tadah::core::Context &ctx) {
  const std::size_t n_int = tadah::core::n_intercept_columns(ctx);
  return n_int ? n_int : 1;
}

/** \brief The intercept vectors e_0 .. e_{n0-1} of length @p dim (e_k is 1
 *  in component k, 0 elsewhere) as the columns of a dim x n0 matrix: the
 *  first n0 columns of every basis Basis builds. */
inline tadah::core::Matrix intercept_basis_vectors(std::size_t n0,
                                                   std::size_t dim) {
  if (dim < n0)
    throw std::logic_error(
        "intercept_basis_vectors: " + std::to_string(n0) + " intercept "
        "vectors need a descriptor of at least " + std::to_string(n0) +
        " components, not " + std::to_string(dim));
  tadah::core::Matrix b(dim, n0);
  b.set_zero();
  for (std::size_t k = 0; k < n0; ++k) b(k, k) = 1;
  return b;
}

/** \brief What columns 0 .. n0-1 of a basis hold, for messages: "column 0
 *  is the bias vector e0", or "columns 0 to 1 are the 2 intercept vectors
 *  e_0 .. e_1, one per ATOM species with BIAS true". */
inline std::string intercept_basis_vectors_text(std::size_t n0) {
  return n0 == 1
      ? "column 0 is the bias vector e0"
      : "columns 0 to " + std::to_string(n0 - 1) + " are the " +
            std::to_string(n0) + " intercept vectors e_0 .. e_" +
            std::to_string(n0 - 1) + ", one per ATOM species with BIAS true";
}

/** \brief The fewest basis vectors a non-linear M_KRR model takes, n0 + 1
 *  (the n0 intercept vectors and one drawn from the @p what, e.g. "training
 *  AEDs"): "" when SBASIS @p s has them, else why not ("SBASIS 2: a
 *  non-linear M_KRR model needs at least 3 basis vectors (...). Set SBASIS
 *  to 3 or more."). Basis::check_sbasis and, before training,
 *  model_weight_count (weight_provisioning.h) refuse it. */
inline std::string sbasis_lower_bound_problem(std::size_t s, std::size_t n0,
                                              const std::string &what) {
  if (s >= n0 + 1) return "";
  return "SBASIS " + std::to_string(s) + ": a non-linear M_KRR model needs "
         "at least " + std::to_string(n0 + 1) + " basis vectors (" +
         intercept_basis_vectors_text(n0) + ", the others are drawn from the " +
         what + "). Set SBASIS to " + std::to_string(n0 + 1) + " or more.";
}

/** \brief The basis vectors of a non-linear M_KRR model.
 *
 *  build_random_basis() (MODEL M_KRR <kernel> 1) and prep_basis_for_krr()
 *  (MODEL M_KRR <kernel> 2) build SBASIS basis vectors, the columns of b:
 *  columns 0 .. n0-1 are the intercept ("bias") vectors e_0 .. e_{n0-1}
 *  (e_k = 1 in component k, 0 elsewhere; n_intercept_vectors(): one per
 *  ATOM species with BIAS true, e_0 alone with BIAS false), and columns
 *  n0 .. SBASIS-1 are SBASIS-n0 candidates drawn without replacement from
 *  the training set: its atomic energy descriptors (AEDs), or for
 *  prep_basis_for_krr the mean AED of each structure. With one species,
 *  or BIAS false, column 0 is e0 and columns 1 .. SBASIS-1 are drawn.
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
  /** The number of intercept vectors the basis starts with, e_0 ..
   *  e_{n-1}: one per intercept column (ATOM species) with BIAS true; e_0
   *  with BIAS false, as before per-species intercepts. */
  size_t n_intercept_vectors() const {
    return n_intercept_basis_vectors(context);
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

  /** \brief SBASIS s with n candidates and n0 intercept vectors
   *  (n_intercept_vectors()): s >= n0 + 1 (the intercept vectors and at
   *  least one drawn vector) and s - n0 <= n. Throws naming SBASIS. */
  static void check_sbasis(std::size_t s, std::size_t n0, std::size_t n,
                           const std::string &what) {
    const std::string held = intercept_basis_vectors_text(n0);
    const std::string too_few = sbasis_lower_bound_problem(s, n0, what);
    if (!too_few.empty()) throw std::runtime_error(too_few);
    if (s - n0 > n)
      throw std::runtime_error(
          "SBASIS " + std::to_string(s) + " draws " + std::to_string(s - n0) +
          " basis vectors from the " + what + " (" + held + "), but the "
          "training set has " + std::to_string(n) + ": set SBASIS to at "
          "most " + std::to_string(n + n0) + ".");
  }

  /** \brief Columns 0 .. n0-1 the intercept vectors (n0 =
   *  n_intercept_vectors()), columns n0 .. s-1 the AEDs at the positions
   *  draw_candidates(n, s-n0) of the list of all n AEDs of st_desc_db
   *  (structure by structure, atom by atom). */
  void build_random_basis(size_t s, StDescriptorsDB &st_desc_db) {
    const size_t n0 = n_intercept_vectors();

    // the AEDs, structure by structure, atom by atom
    std::vector<std::pair<size_t,size_t>> indices;
    for( size_t st = 0; st < st_desc_db.size(); st++ ) {
      for( size_t a = 0; a < st_desc_db(st).naed() ; a++ ) {
        indices.emplace_back(st,a);
      }
    }

    check_sbasis(s, n0, indices.size(), "training AEDs");
    const std::vector<std::size_t> drawn = draw_candidates(indices.size(), s - n0);

    const size_t dim = st_desc_db(indices[drawn[0]].first).dim();
    b.resize(dim,s);
    b.set_zero();
    // the first basis vectors are the intercept ("bias") vectors
    for (size_t k=0; k<n0; ++k) b(k,k)=1;
    for (size_t i=n0; i<s; ++i) {
      const size_t st = indices[drawn[i-n0]].first;
      const size_t a = indices[drawn[i-n0]].second;
      const tadah::core::aed_type &aed = st_desc_db(st).get_aed(a);
      for (size_t j=0; j<aed.size(); ++j) {
        b(j,i)=aed[j];
      }
    }
  }

  /** \brief Columns 0 .. n0-1 the intercept vectors (target 0; n0 =
   *  n_intercept_vectors()), columns n0 .. s-1 the mean AEDs of the
   *  structures at the positions draw_candidates(n, s-n0) of the n
   *  structures, with their energies per atom in T. */
  void prep_basis_for_krr(StDescriptorsDB &st_desc_db,
                          const StructureDB &stdb) {

    size_t s = context.get<size_t>("SBASIS");
    const size_t n0 = n_intercept_vectors();

    if (stdb.size() != st_desc_db.size()) {
      throw std::runtime_error("The size of StructureDB is different \n \
from StDescriptorsDB. This is likely to be a bug.\n");
    }

    check_sbasis(s, n0, stdb.size(), "training structures");
    const std::vector<std::size_t> drawn = draw_candidates(stdb.size(), s - n0);

    b.resize(st_desc_db(drawn[0]).dim(),s);
    T.resize(s);
    T.set_zero();
    b.set_zero();
    // the first basis vectors are the intercept ("bias") vectors, target 0
    for (size_t k=0; k<n0; ++k) {
      b(k,k)=1;
      T(k)=0;
    }
    // Here we add all atomic energy descriptors into single
    // descriptor which represents total energy of this configuration
    for (size_t i=n0; i<s; ++i) {
      const size_t st = drawn[i-n0];
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
