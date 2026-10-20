#ifndef TADAH_MLIP_FIT_BALANCE_H
#define TADAH_MLIP_FIT_BALANCE_H

// The balance table of a fit: how the sum a fit minimises (row_weights.h),
//
//     sum_i W_i r_i^2 + lambda |w|^2_pen,   W_i = p_i m_i / sigma_i^2,
//
// over its observations i is shared among its quantities (energies,
// forces, stresses) and its datasets, and how many of the coefficients
// each of them determines. After a first line (LAMBDA, ENORM SQRT, the
// coefficients and the rank) and the error scales of the quantities the
// fit observes (ENERGY, FORCE, STRESS true), one row per quantity over
// every dataset (the entries of every dataset pooled), then, with several
// datasets, one per dataset, and the prior's row. The rows are of the
// training data as the fit read it (after EFILTER, FFILTER and
// ZERO_COM_FORCE); a dataset of WDBFILE 0 has rows without entries:
//
//   n        the observations (count_observations(): rows of priority > 0
//            that are not zero in every column; at most six per stress
//            tensor, at most three per atom);
//   RMSE     the root mean square residual in the unit of the quantity's
//            metric over every row of priority > 0, those zero by
//            symmetry included: the energy per atom in meV/atom, the force
//            components in eV/Å, the nine entries of each stress tensor in
//            GPa (an off-diagonal entry counted twice, m_i = 2): the
//            row of a dataset is the metric of tadah predict -A of it
//            with the same filters; NaN ("-") without such an entry;
//   r/sigma  the root mean square of residual / error scale over the same
//            entries (sigma_i of each row: N^q ESIGMA for the energy of a
//            structure of N atoms, q 1 or 1/2 under ENORM SQRT, the
//            structure's own force or stress scale);
//   share    sum_i W_i r_i^2 of the row's observations over the minimised
//            sum; the prior's row has lambda |w|^2_pen over it, so the
//            rows of every dataset and the prior's sum to 1;
//   gamma    the effective parameters, tr(H^+ H_q) over the directions the
//            solve keeps (effective_parameters(): every direction under
//            OALGO 2 and 4); the prior's row has the rest of the rank,
//            lambda tr(H^+ I_pen), so the rows of every dataset and the
//            prior's sum to the rank, sum_kept s_i^2 / (s_i^2 + lambda)
//            without the prior for a uniform ridge.
//
// The rows of a dataset sum to its row of every dataset (n, share, gamma).
// Computing the table is one O(n p^2) pass over the n rows and p
// coefficients of the fit, which reads the fit's design matrix in blocks:
// its memory is of the order of (p + 1024 + 3 D) p for D datasets, not a
// copy of the rows. It is computed once per potential written: tadah train
// after its fit, tadah hpo for each evaluation that may become its new
// best. The table is
// printed (print_fit_balance) and kept in the potential's training record
// (add_fit_balance), from which it is read back (read_fit_balance).

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
   *  of its own, M_KRR implementation 2, a FIXINDEX that pins every
   *  coefficient, ... */
  std::string none;
  double lambda = 0.0;       ///< LAMBDA as the configuration gives it
  double lambda_used = 0.0;  ///< the solve's (LAMBDA -1: ALPHA/BETA)
  bool enorm_sqrt = false;   ///< ENORM SQRT: energies per sqrt(atom)
  std::size_t coefficients = 0;  ///< the coefficients the solve fitted
  std::size_t pinned = 0;        ///< those FIXINDEX pins (not fitted)
  std::size_t retained = 0;      ///< the rank: the directions kept
  std::size_t datasets = 1;      ///< the datasets of the fit
  /** The error scale of each quantity the fit observes as the table
   *  prints it, to 4 significant digits: "1", "0.1 0", "0.5 0" (ESIGMA;
   *  FSIGMA, SSIGMA a b); "" for a quantity without a row. */
  std::array<std::string, 3> sigma;
  /** Per quantity observed (energy, force, stress, in that order): its row
   *  of every dataset, then, with several datasets, one per dataset. */
  std::vector<FitBalanceRow> rows;
  double prior_share = 0.0;  ///< lambda |w|^2_pen over the minimised sum
  double prior_gamma = 0.0;  ///< lambda tr(H^+ I_pen): the rank less the data's

  /** Whether the fit has a table (@c none is empty). */
  bool computed() const { return none.empty(); }
  /** The row of @p quantity and @p dataset (0: every dataset), or nullptr. */
  const FitBalanceRow *row(std::size_t quantity, std::size_t dataset) const;
};

/** A table of none: "Fit balance: none (<why>)". */
FitBalance no_fit_balance(const std::string &why);

/** The table of a fit whose FIXINDEX pins every coefficient, which solves
 *  for none: none ("FIXINDEX pins every coefficient: none was fitted"),
 *  that of tadah train and of tadah hpo, which skips such a fit. */
FitBalance all_pinned_fit_balance();

/** The widest line of a table of fewer than 10^7 observations per row:
 *  78 characters, 80 in a training record after its "# " (the column n is
 *  as wide as its largest count, a row one character wider per digit
 *  beyond seven). */
constexpr std::size_t kFitBalanceWidth = 78;

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
  int oalgo = 3;             ///< OALGO
  double rcond = 1e-8;       ///< OALGO's second value, 1e-8 without it
  std::vector<std::size_t> pinned;      ///< FIXINDEX columns, 0-based
  std::vector<std::size_t> intercepts;  ///< the columns no ridge penalises

  /** The solve of a fit of @p cols columns whose intercept columns
   *  (Function_Base::intercept_columns()) are @p intercepts. */
  static FitSolve of(const core::Context &c, std::size_t cols,
                     const std::vector<std::size_t> &intercepts);
};

/** The balance table of a fit (see the file comment); none, saying so,
 *  when FIXINDEX pins every column (all_pinned_fit_balance()). It reads
 *  @p Phi where it is, in blocks of rows (effective_parameters() of
 *  WeightedRows), and allocates no copy of it.
 *  @param Phi the rows of the design matrix without their factors.
 *  @param data_rows the first rows of @p Phi, those of the structures.
 *  @param T their reference values (unweighted).
 *  @param factors the factor sqrt(W_i) of each row.
 *  @param labels the type of each row (0 energy, 1 force, 2 stress).
 *  @param row_dataset the dataset of each row (0-based).
 *  @param n_datasets the datasets.
 *  @param scale the error scale, multiplicity and metric factor of each row.
 *  @param w the coefficients of the fit, one per column (FIXINDEX ones
 *         included).
 *  @param solve LAMBDA, OALGO and the columns of the solve.
 *  @param c the configuration of the fit: ENERGY, FORCE, STRESS (the
 *         quantities with a row) and the error scale keys (sigma).
 *  @throws std::logic_error for arrays of other sizes.
 *  @throws std::runtime_error when the QR factorisation or the SVD fails.
 *  @throws std::bad_alloc without the memory of its blocks. */
FitBalance fit_balance(const core::phi_type &Phi, std::size_t data_rows,
                       const core::t_type &T, const core::t_type &factors,
                       const core::t_type &labels,
                       const std::vector<std::size_t> &row_dataset,
                       std::size_t n_datasets,
                       const std::vector<FitRowScale> &scale,
                       const core::t_type &w, const FitSolve &solve,
                       const core::Context &c);

/** The lines of the table as printed and recorded:
 *   - "Fit balance: LAMBDA <LAMBDA>[ (evidence approximation: <lambda>)][;
 *     ENORM SQRT]; <p> coefficients[ (FIXINDEX pins <k> more)], rank <r>";
 *   - "  error scales: ESIGMA <a> meV/atom (meV per sqrt(atom) under ENORM
 *     SQRT); FSIGMA <a> <b> eV/Å; SSIGMA <a> <b> GPa", the quantities with
 *     a row, each value to 4 significant digits;
 *   - the column header, one line per row, then the prior's;
 * or "Fit balance: none (<why>)". Those three first lines are wrapped to
 * kFitBalanceWidth after a "; " (within a clause longer than a line, at a
 * blank), each line after the first of one starting with four blanks. The
 * columns, two blanks apart after an indent of two, are as wide as their
 * widest entry, numbers to the right; a number has 4 significant digits,
 * its trailing zeros kept, from 0.001 to 9999 and 3 in exponent form
 * otherwise ("0" for 0, "-" without an observation), so each line is at
 * most kFitBalanceWidth wide for fewer than 10^7 observations. */
std::vector<std::string> fit_balance_lines(const FitBalance &b);

/** One line per column, "  <column>  <what it is>", and one on what sums
 *  to what, printed after the first table of a run (print_fit_balance). */
std::vector<std::string> fit_balance_legend();

/** Appends fit_balance_lines(@p b) to @p record: the table of the fit that
 *  wrote the potential, a comment of its training record. */
void add_fit_balance(TrainingRecord &record, const FitBalance &b);

/** The lines of the table in @p record, from its "Fit balance:" line to the
 *  prior's row (for none, that line and the lines that continue it); empty
 *  without one. */
std::vector<std::string> fit_balance_record_lines(const TrainingRecord &record);

/** @p record without the lines of its table (fit_balance_record_lines()):
 *  the record of a potential whose coefficients are no longer those of the
 *  fit (tadah refit --edit). */
TrainingRecord without_fit_balance(const TrainingRecord &record);

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
