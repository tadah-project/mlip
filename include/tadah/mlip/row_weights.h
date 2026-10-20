#ifndef TADAH_MLIP_ROW_WEIGHTS_H
#define TADAH_MLIP_ROW_WEIGHTS_H

// What a fit minimises, in one place: the weight of every row of its
// design matrix, its observations, its effective parameters and the
// absolute error scales its noise estimates are given in.
//
// A fit minimises
//
//     sum_i W_i (y_i - phi_i . w)^2 + lambda |w|^2_pen,   W_i = p_i m_i / sigma_i^2,
//
// over its observations i: the energy of each structure (ENERGY true), each
// Cartesian force component (FORCE true) and the six independent entries of
// each stress tensor (STRESS true), the off-diagonal ones (xy, xz, yz) with
// m_i = 2 so that the sum is the squared Frobenius norm of the residual
// tensor, m_i = 1 for every other row. p_i is the priority of the
// observation: the structure's eweight, fweight or sweight (its data file)
// times the WDBFILE value of its dataset. sigma_i is its error scale:
//
//     energy  N^q ESIGMA / 1000 eV, N the structure's atoms, q = 1 (ENORM
//             ATOM) or 1/2 (ENORM SQRT): the residual of the energy per
//             atom (or per sqrt(atom)) over ESIGMA in meV;
//     force   sqrt(a^2 + (b M)^2) eV/A, FSIGMA a b, M the root mean square
//             of the structure's own reference force components;
//     stress  sqrt(a^2 + (b M)^2) eV/A^3, SSIGMA a b with a converted from
//             GPa (kGPaInEvPerA3), M the root mean square of the nine
//             entries of the structure's own reference stress.
//
// A row is multiplied by its factor sqrt(W_i) (DesignMatrix). A row whose
// design-matrix row is zero in every column (a force that is zero by
// symmetry; zero to rounding, models::zero_rows()) carries no information
// and is not an observation, nor is a row of priority 0. Every caller -- DesignMatrix (both builds), the intercept
// check, the weight covariance and its noise, predict --error and the
// analytic gradients of tadah hpo -- takes the rule from here.

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/mlip/structure_db.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

/** 1 GPa in eV/A^3 (the unit of the stresses of the data). */
constexpr double kGPaInEvPerA3 = 6.241509074e-3;

/** The error scales and switches of a fit, as a Context gives them (its
 *  values, or the blueprint defaults): no check (require_fit_settings()
 *  checks them for a fit). */
struct ErrorScales {
  bool energy = true;      ///< ENERGY
  bool force = false;      ///< FORCE
  bool stress = false;     ///< STRESS
  bool sqrt_norm = false;  ///< ENORM SQRT (q = 1/2); ATOM otherwise (q = 1)
  double esigma = 1e-3;    ///< ESIGMA / 1000, eV (per atom, or sqrt(atom))
  double fsigma_a = 0.1;   ///< FSIGMA a, eV/A
  double fsigma_b = 0.0;   ///< FSIGMA b
  double ssigma_a = 0.5 * kGPaInEvPerA3;  ///< SSIGMA a, eV/A^3
  double ssigma_b = 0.0;   ///< SSIGMA b

  /** The scales of @p c (Context::get: a key it does not store reads its
   *  blueprint default, which the Context then holds). */
  static ErrorScales of(const core::Context &c);

  /** N^q for a structure of @p natoms atoms: N under ENORM ATOM, sqrt(N)
   *  under ENORM SQRT. */
  double energy_norm(std::size_t natoms) const;

  /** The absolute error scale of row type @p type (0 energy, 1 force, 2
   *  stress), in the units of the data: ESIGMA/1000 eV (per atom or
   *  sqrt(atom)), FSIGMA a eV/A, SSIGMA a eV/A^3 (the noise of WNOISE,
   *  noise_of_fitted_types()). */
  double absolute(std::size_t type) const;
};

/** The row factors of a fit of the structures @p stdb with the settings of
 *  a Context: the priority, error scale and factor of each of their rows,
 *  computed as the file comment says, in this order: energy sigma =
 *  energy_norm(N) * (ESIGMA / 1000.0), factor = sqrt(p) / sigma; force M =
 *  sqrt(sum F^2 / (3 N)) over the atoms in order, x, y, z, sigma =
 *  sqrt(a*a + (b*M)*(b*M)), factor sqrt(p) / sigma; stress a = SSIGMA a *
 *  kGPaInEvPerA3, M = sqrt(sum_xy S^2 / 9) (x, then y), sigma as for
 *  forces, factor sqrt(p * m) / sigma. A factor is 0 for a priority of 0
 *  and, for the energy, with ENERGY false. The datasets are the ranges of
 *  stdb.dbidx (one when it has none), each with its WDBFILE value (1
 *  without WDBFILE).
 *
 *  Nothing is checked here: a fit calls require_fit_settings() and
 *  require_valid_weights() first, which refuse what the factors cannot
 *  take (an error scale <= 0, a WDBFILE that does not have one value per
 *  dataset, a negative priority). */
class RowWeights {
public:
  RowWeights(const core::Context &c, const StructureDB &stdb);

  const ErrorScales &scales() const { return scales_; }
  /** The datasets (the ranges of stdb.dbidx; 1 without them). */
  std::size_t datasets() const { return n_datasets_; }
  /** The dataset of structure @p s (0-based). */
  std::size_t dataset_of(std::size_t s) const { return dataset_[s]; }
  /** WDBFILE of the dataset of structure @p s (1 without WDBFILE). */
  double dataset_priority(std::size_t s) const { return dprio_[dataset_[s]]; }

  /** eweight x WDBFILE; 0 with ENERGY false. */
  double energy_priority(std::size_t s) const;
  double force_priority(std::size_t s) const;   ///< fweight x WDBFILE
  double stress_priority(std::size_t s) const;  ///< sweight x WDBFILE

  double energy_sigma(std::size_t s) const;  ///< eV
  double force_sigma(std::size_t s) const;   ///< eV/A
  double stress_sigma(std::size_t s) const;  ///< eV/A^3

  double energy_factor(std::size_t s) const;
  double force_factor(std::size_t s) const;
  /** The factor of stress entry (@p x, @p y) of structure @p s, x <= y. */
  double stress_factor(std::size_t s, int x, int y) const;

private:
  ErrorScales scales_;
  const StructureDB &stdb_;
  std::size_t n_datasets_ = 1;
  std::vector<std::size_t> dataset_;
  std::vector<double> dprio_;
  std::vector<double> fsigma_, ssigma_;
};

/** The WDBFILE values of @p c, one per dataset of @p stdb (1 each without
 *  WDBFILE). @throws std::runtime_error when WDBFILE has another number of
 *  values than there are datasets, or a value that is not a finite number
 *  >= 0, naming the key, the dataset and its file (DBFILE). */
std::vector<double> dataset_priorities(const core::Context &c,
                                       const StructureDB &stdb);

/** Refuses the settings of a fit of @p c that a fit cannot take, before
 *  it reads anything else (M_BLR and M_KRR train, tadah train, tadah hpo
 *  when a run starts):
 *   - a retired key the Context records (core::require_no_retired_keys,
 *     with its conversion);
 *   - ESIGMA not a finite number > 0; FSIGMA or SSIGMA whose absolute part
 *     a is not a finite number > 0 or whose relative part b is not a finite
 *     number >= 0; an ENORM other than ATOM or SQRT;
 *   - an error scale other than its blueprint default, compared by value
 *     (FSIGMA 0.1 is FSIGMA 0.1 0), for a quantity that is off: FSIGMA with
 *     FORCE false, SSIGMA with STRESS false, ESIGMA and ENORM SQRT with
 *     ENERGY false (it would change nothing);
 *   - ENERGY, FORCE and STRESS all false (no observation).
 *  @p who begins each message ("tadah train", "M_BLR::train").
 *  @throws std::runtime_error */
void require_fit_settings(const core::Context &c, const std::string &who);

/** Makes @p c, the Context of a potential a fit writes, hold a line of
 *  each of ESIGMA, FSIGMA, SSIGMA, ENORM and ENERGY: the fit's value, or
 *  the blueprint default, which the fit used; ENORM in upper case. A
 *  retrain from the potential reads them (M_BLR, M_KRR get_param_file). */
void write_fit_scales(core::Context &c);

/** The INFO line of a fit: "Fit error scales: ESIGMA 1 meV/atom (default),
 *  ENORM ATOM (default), ENERGY true (default), FSIGMA 0.2 0.05 eV/Å, SSIGMA
 *  0.5 0 GPa (default; STRESS false)." -- every error scale, ENORM and
 *  ENERGY with its value, whether it is the default (compared by value) and
 *  the switch that is off for a scale it makes inert. ESIGMA is in meV per
 *  sqrt(atom) under ENORM SQRT. */
std::string fit_scales_info(const core::Context &c);

/** The observations of a design matrix (DesignMatrix): a data row of
 *  factor > 0 that is not zero in every column (models::zero_rows(): no
 *  value above 1e-10 of the largest magnitude of its column, so the
 *  rounding of sums that cancel, as in the force rows of an atom at a
 *  centre of symmetry, does not make a row an observation). */
struct ObservationCount {
  std::size_t total = 0;
  std::array<std::size_t, 3> by_type{{0, 0, 0}};  ///< E, F, S
  /** Whether each type has an observation (the types a fit fitted). */
  std::array<bool, 3> fitted{{false, false, false}};
};

/** The observations among the first @p data_rows rows of @p Phi, whose
 *  factors are @p factors and types @p labels (0, 1, 2); @p Phi holds the
 *  rows without the factors (the zero rows are those of the design
 *  matrix, whatever the priorities). */
ObservationCount count_observations(const core::phi_type &Phi,
                                    std::size_t data_rows,
                                    const core::t_type &factors,
                                    const core::t_type &labels);

/** The effective parameters of a fit, per quantity and per dataset. */
struct EffectiveParameters {
  /** gamma_q of the energies, the forces and the stresses. */
  std::array<double, 3> by_type{{0.0, 0.0, 0.0}};
  /** gamma of each type of each dataset. */
  std::vector<std::array<double, 3>> by_dataset;
  /** Their sum, sum_kept s_i^2/(s_i^2 + lambda) for a uniform ridge. */
  double total = 0.0;
  /** The prior's share, lambda tr(H^+ I_pen) over the retained
   *  directions (each direction's lambda |P v_i|^2 over its eigenvalue of
   *  H): the retained directions less @c total, without the rounding of
   *  that difference; 0 at lambda 0. */
  double prior = 0.0;
  /** The directions the solve keeps (the rank of H on them). */
  std::size_t retained = 0;
};

/** The effective parameters gamma_q = tr(H^+ H_q) of the weighted design
 *  matrix @p Phi (its rows the observations with their factors, no ridge
 *  rows), H = Phi^T Phi + @p lambda I_pen and H_q the part of Phi^T Phi of
 *  the rows of type q (@p labels: 0, 1, 2) and dataset d (@p row_dataset,
 *  < @p n_datasets). I_pen is 1 on the diagonal but for the columns
 *  @p unpenalised (the intercept columns of BIAS true). H^+ is the inverse
 *  over the directions the solve of OALGO @p oalgo keeps:
 *   - OALGO 3 (SVD of Phi, a uniform ridge): the right singular vectors of
 *     Phi whose singular value is above @p rcond times the largest (all of
 *     them when rcond <= 0); gamma_q = sum_kept |Phi_q v_i|^2 /
 *     (s_i^2 + lambda);
 *   - OALGO 1 (DGELSD of Phi and the ridge rows): the eigenvectors of H
 *     whose sqrt(eigenvalue) is above @p rcond times the largest (machine
 *     precision for rcond < 0);
 *   - OALGO 2 and 4: every eigenvector of H with a positive eigenvalue
 *     (above n eps times the largest).
 *  @throws std::logic_error for OALGO 3 with lambda != 0 and unpenalised
 *  columns (the fit refuses it), and for rows, labels and datasets of
 *  other sizes. */
EffectiveParameters effective_parameters(
    const core::phi_type &Phi, const core::t_type &labels,
    const std::vector<std::size_t> &row_dataset, std::size_t n_datasets,
    double lambda, const std::vector<std::size_t> &unpenalised, int oalgo,
    double rcond);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_ROW_WEIGHTS_H
