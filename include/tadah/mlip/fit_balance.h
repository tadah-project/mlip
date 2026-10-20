#ifndef TADAH_MLIP_FIT_BALANCE_H
#define TADAH_MLIP_FIT_BALANCE_H

// The balance table of a fit: how the sum a fit minimises (row_weights.h),
//
//     sum_i W_i r_i^2 + lambda |w|^2_pen,   W_i = p_i m_i / sigma_i^2,
//
// is shared among its quantities (energies, forces, stresses) and its
// datasets, and how many of the coefficients each of them determines.
// One row per quantity the fit observes (ENERGY, FORCE, STRESS true) over
// every dataset, then, with several datasets, one per dataset, and the
// prior's row:
//
//   n        the observations (count_observations(): rows of priority > 0
//            that are not zero in every column; six per stress tensor);
//   sigma    the error scale as its key gives it (ESIGMA; FSIGMA, SSIGMA
//            a b);
//   RMSE     the root mean square residual in the unit of the quantity's
//            metric: the energy per atom in meV/atom, the force components
//            in eV/Å, the nine entries of each stress tensor in GPa (an
//            off-diagonal entry counted twice, m_i = 2);
//   r/sigma  the root mean square of residual / error scale over the same
//            entries (sigma_i of each row: N^q ESIGMA for the energy of a
//            structure of N atoms, the structure's own force or stress
//            scale);
//   share    sum_i W_i r_i^2 of the rows over the minimised sum; the prior's
//            row has lambda |w|^2_pen over it, so the column sums to 1;
//   gamma    the effective parameters, tr(H^+ H_q) over the directions the
//            solve keeps (effective_parameters()); the prior's row has the
//            rest of the rank, lambda tr(H^+ I_pen), so the column sums to
//            the rank, sum_kept s_i^2 / (s_i^2 + lambda) without the prior
//            for a uniform ridge.
//
// Computing it is one O(n p^2) pass over the n rows and p coefficients of
// the fit (an SVD of its weighted rows), once per potential written: tadah
// train after its fit, tadah hpo for each new best potential it publishes.
// The table is printed (print_fit_balance) and kept in the potential's
// training record (add_fit_balance), from which it is read back
// (read_fit_balance).

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/mlip/training_record.h>

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

/** One row of the table: quantity @c quantity of dataset @c dataset. */
struct FitBalanceRow {
  std::size_t quantity = 0;  ///< 0 energy, 1 force, 2 stress
  std::size_t dataset = 0;   ///< 0: every dataset; k: the k-th (1-based)
  std::size_t n = 0;         ///< observations
  /** RMSE in the metric's unit (meV/atom, eV/Å, GPa); NaN without an
   *  observation. */
  double rmse = std::numeric_limits<double>::quiet_NaN();
  /** RMS of residual / error scale; NaN without an observation. */
  double rms_z = std::numeric_limits<double>::quiet_NaN();
  double share = 0.0;  ///< of the minimised sum
  double gamma = 0.0;  ///< effective parameters
};

/** The balance table of a fit, or why it has none. */
struct FitBalance {
  /** Why the fit has no table ("" when it has one): a model without a fit
   *  of its own, M_KRR implementation 2, ... */
  std::string none;
  double lambda = 0.0;       ///< LAMBDA as the configuration gives it
  double lambda_used = 0.0;  ///< the solve's (LAMBDA -1: ALPHA/BETA)
  std::size_t coefficients = 0;  ///< the coefficients the solve fitted
  std::size_t pinned = 0;        ///< those FIXINDEX pins (not fitted)
  std::size_t retained = 0;      ///< the rank: the directions kept
  std::size_t datasets = 1;
  /** The error scale of each quantity as its key gives it: "1", "0.1 0",
   *  "0.5 0" (ESIGMA, FSIGMA, SSIGMA). */
  std::array<std::string, 3> sigma;
  /** Per quantity observed (energy, force, stress, in that order): its row
   *  of every dataset, then, with several datasets, one per dataset. */
  std::vector<FitBalanceRow> rows;
  double prior_share = 0.0;  ///< lambda |w|^2_pen over the minimised sum
  double prior_gamma = 0.0;  ///< lambda tr(H^+ I_pen): the rank less the data's

  bool computed() const { return none.empty(); }
  /** The row of @p quantity and @p dataset (0: every dataset), or nullptr. */
  const FitBalanceRow *row(std::size_t quantity, std::size_t dataset) const;
};

/** A table of none: "Fit balance: none (<why>)". */
FitBalance no_fit_balance(const std::string &why);

/** The scales of one row of a design matrix, besides its factor. */
struct FitRowScale {
  double sigma = 1.0;  ///< its error scale in the unit of the data
  double m = 1.0;      ///< 2 for an off-diagonal stress entry, else 1
  /** A residual of the row times this is in the metric's unit: 1000/N for
   *  the energy of a structure of N atoms (eV -> meV/atom), 1 for a force
   *  (eV/Å), 1/kGPaInEvPerA3 for a stress (eV/Å^3 -> GPa). */
  double to_metric = 1.0;
};

/** How the fit solved for its coefficients, as the configuration @p c of
 *  its model holds it after the fit (LinearRegressor::train). */
struct FitSolve {
  double lambda = 0.0;       ///< LAMBDA
  double lambda_used = 0.0;  ///< LAMBDA, or ALPHA/BETA of LAMBDA -1
  int oalgo = 3;
  double rcond = 1e-8;       ///< OALGO's second value, 1e-8 without it
  std::vector<std::size_t> pinned;      ///< FIXINDEX columns, 0-based
  std::vector<std::size_t> intercepts;  ///< the columns no ridge penalises

  /** The solve of a fit of @p cols columns whose intercept columns
   *  (Function_Base::intercept_columns()) are @p intercepts. */
  static FitSolve of(const core::Context &c, std::size_t cols,
                     const std::vector<std::size_t> &intercepts);
};

/** The balance table of a fit (see the file comment).
 *  @param Phi the rows of the design matrix without their factors; its
 *         first @p data_rows are the rows of the structures.
 *  @param T their reference values (unweighted).
 *  @param factors, labels the factor sqrt(W_i) and the type (0, 1, 2) of
 *         each row.
 *  @param row_dataset, n_datasets the dataset of each row (0-based).
 *  @param scale the error scale, multiplicity and metric factor of each row.
 *  @param w the coefficients of the fit, one per column (FIXINDEX ones
 *         included).
 *  @param solve LAMBDA, OALGO and the columns of the solve.
 *  @param c the configuration of the fit: ENERGY, FORCE, STRESS (the
 *         quantities with a row) and the error scale keys (sigma).
 *  @throws std::logic_error for arrays of other sizes.
 *  @throws std::runtime_error when the SVD does not converge. */
FitBalance fit_balance(const core::phi_type &Phi, std::size_t data_rows,
                       const core::t_type &T, const core::t_type &factors,
                       const core::t_type &labels,
                       const std::vector<std::size_t> &row_dataset,
                       std::size_t n_datasets,
                       const std::vector<FitRowScale> &scale,
                       const core::t_type &w, const FitSolve &solve,
                       const core::Context &c);

/** The lines of the table as printed and recorded: "Fit balance: LAMBDA
 *  <LAMBDA>[ (evidence approximation: <lambda>)]; <p> coefficients[
 *  (FIXINDEX pins <k> more)], rank <r>", the column header, one line per
 *  row, then the prior's; or the one line "Fit balance: none (<why>)".
 *  The columns, two blanks apart after an indent of two, are as wide as
 *  their widest entry; a number has 4 significant digits from 0.001 to
 *  9999 and 3 in exponent form otherwise ("-" without an observation). */
std::vector<std::string> fit_balance_lines(const FitBalance &b);

/** One line per column, "  <column>  <what it is>", printed after the
 *  first table of a run (print_fit_balance). */
std::vector<std::string> fit_balance_legend();

/** Appends fit_balance_lines(@p b) to @p record: the table of the fit that
 *  wrote the potential, a comment of its training record. */
void add_fit_balance(TrainingRecord &record, const FitBalance &b);

/** The lines of the table in @p record, from its "Fit balance:" line to the
 *  prior's row (or that line alone for none); empty without one. */
std::vector<std::string> fit_balance_record_lines(const TrainingRecord &record);

/** The table of @p record (fit_balance_record_lines() read back, its
 *  numbers as printed); std::nullopt when it has none.
 *  @throws std::runtime_error naming the line of a table that cannot be
 *  read (a row of an unknown quantity, a number that is not one, a table
 *  without its prior's row). */
std::optional<FitBalance> read_fit_balance(const TrainingRecord &record);

/** Writes @p lines, each followed by '\n', and with @p legend the lines of
 *  fit_balance_legend() after them. */
void print_fit_balance(std::ostream &os, const std::vector<std::string> &lines,
                       bool legend);

/** True the first time it is called in the process: whether the table about
 *  to be printed is the first of the run, which gets the legend. */
bool first_fit_balance_legend();

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_FIT_BALANCE_H
