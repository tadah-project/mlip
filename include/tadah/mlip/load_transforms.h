#ifndef TADAH_MLIP_LOAD_TRANSFORMS_H
#define TADAH_MLIP_LOAD_TRANSFORMS_H

// Load-time transforms applied to a freshly loaded StructureDB.
//
// Train ↔ predict ↔ HPO are a closed loop: LSCALE / ESHIFT applied at training
// are stamped into pot.tadah and re-applied at predict time so a raw DFT
// dataset round-trips. Train → LAMMPS is one-way: the LAMMPS plugin does NOT
// re-apply LSCALE/ESHIFT to its input positions or output energies. The user
// is expected to provide LAMMPS with positions at the trained scale (typically
// the experimental scale, when LSCALE was set), and the model produces forces
// and energies in that same scale. pot.tadah carries LSCALE/ESHIFT for
// documentation and so `tadah predict` can round-trip them; LammpsTadah's
// build_sub_objects() does not reference these keys.

#include <tadah/mlip/structure_db.h>
#include <tadah/core/context.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

/**
 * Report of which load-time transforms were applied to a StructureDB.
 *
 * Filled in by apply_load_transforms() and used by callers to:
 *   - log a "LOAD TRANSFORMS" section in pot.tadah / hpo_run.log
 *   - stamp LSCALE / ESHIFT into the pot.tadah Context for round-trip prediction
 */
struct LoadTransformReport {
  /** Whether anything was actually applied (false => the report is informational only). */
  bool   applied = false;

  /** Length rescale factor s. 1.0 = identity. */
  double lscale  = 1.0;

  /** Per-element reference energies (Z -> energy/atom in eV) that were SUBTRACTED. */
  std::map<int, double> eshift;

  /** Where ESHIFT came from. */
  enum class EshiftSource { None, User, Atom, DBAtom } eshift_source = EshiftSource::None;

  /** True iff the user supplied ESHIFT and the dataset also contained isolated-atom configs. */
  bool   single_atom_warning = false;

  /** Number of configurations dropped by EFILTER (per-atom energy threshold). */
  std::size_t n_dropped_energy = 0;

  /** Number of configurations dropped by FFILTER (per-atom force threshold). */
  std::size_t n_dropped_force = 0;

  /** Per-dataset weight multipliers actually applied (composed WDBFILE * WDBFILE_AUTO). */
  std::vector<double> wdbfile_applied;

  /** Per-dataset N_i used by WDBFILE_AUTO (post outlier-filter counts). */
  std::vector<std::size_t> wdbfile_counts;

  /** Boltzmann reweighting temperature (Kelvin); 0 = disabled. */
  double eweight_temp = 0.0;

  /** True iff per-config COM-force zeroing was applied. */
  bool   zero_com_force = false;

  /** Equilibrium-volume diagnostic: the configuration with minimum E/N after all transforms. */
  std::string eq_label;
  double      eq_volume_per_atom   = 0.0;
  double      eq_energy_per_atom   = 0.0;
};

/* --------------------------------------------------------------------------
 * Top-level entry point
 * -------------------------------------------------------------------------- */

/**
 * Apply all configured load-time transforms to @p stdb in place.
 *
 * Reads keys from @p context (LSCALE, ESHIFT, ESHIFT_ATOM, ESHIFT_DBATOM,
 * EFILTER, FFILTER, WDBFILE, WDBFILE_AUTO, EWEIGHT_TEMP, ZERO_COM_FORCE)
 * and emits a report describing what was actually done.
 *
 * Order of operations (matters because some transforms feed others):
 *  1. Outlier filter (EFILTER, FFILTER) - drop configs first.
 *  2. Length rescale (LSCALE).
 *  3. ESHIFT determination (precedence: explicit > ATOM > DBATOM, WARN if multiple set).
 *  4. ESHIFT application: E -= sum_Z N_Z * eshift[Z].
 *  5. Boltzmann eweight (EWEIGHT_TEMP).
 *  6. COM force zeroing (ZERO_COM_FORCE).
 *  7. Per-dataset weights: w_final[i] = WDBFILE[i] * N_i^(-WDBFILE_AUTO).
 *  8. Equilibrium-volume diagnostic.
 *
 * A structure with eweight 0 has no energy of its own (its energy line,
 * which the dataset format needs, holds a placeholder): steps 1, 3, 5 and 8 leave its energy out (it is
 * neither filtered by energy nor reweighted). One with fweight 0 has no
 * forces: FFILTER leaves it out. Refused before any transform changes a
 * weight: a WDBFILE value below 0 (naming the dataset), WDBFILE_AUTO
 * outside [0, 1] and EWEIGHT_TEMP <= 0.
 */
LoadTransformReport
apply_load_transforms(StructureDB &stdb, const tadah::core::Context &context);

/* --------------------------------------------------------------------------
 * Building blocks (also reused by DataBalanceEngine)
 * -------------------------------------------------------------------------- */

/**
 * Apply per-element energy shifts: for every structure,
 *     E -= sum_Z N_Z * eshift_per_Z[Z].
 *
 * This is the unified Tadah! convention: eshift[Z] is the per-atom reference
 * energy of species Z to be subtracted (so that, e.g., feeding the isolated-atom
 * energy as eshift removes that baseline from cohesive-energy reference data).
 *
 * Species in the structure that are not present in @p eshift_per_Z contribute
 * zero (i.e. their energy is not shifted).
 */
void apply_eshift(StructureDB &stdb,
                  const std::map<int, double> &eshift_per_Z);

/**
 * Derive ESHIFT[Z] from isolated-atom configurations (natoms == 1).
 *
 * - Group all single-atom configs with an energy (eweight != 0) by Z;
 *   eshift[Z] = mean energy.
 * - If multiple configs with the same Z disagree by more than
 *   @p spread_warn_threshold_eV, log INFO with the spread.
 * - Species in @p stdb that have no such isolated-atom config get
 *   eshift[Z] = 0 and a WARNING is logged.
 *
 * Used to implement the ESHIFT_ATOM key.
 */
std::map<int, double>
derive_eshift_from_isolated_atoms(const StructureDB &stdb,
                                  double spread_warn_threshold_eV = 1e-3);

/**
 * Derive ESHIFT[Z] by least-squares atomic-energy fit over the database.
 *
 * Solves min ||y - M beta||^2 where y[i] = total energy of config i and
 * M[i, k] = count of species k in config i, over the configs with an energy
 * (eweight != 0). Returns eshift[Z(k)] = beta_k. A species that only
 * configs of eweight 0 contain has no energy to fit: its shift is 0, with
 * a WARNING naming it, and beta runs over the other species. When the
 * compositions of the configs with an energy do not determine beta (a
 * singular normal matrix), every shift is 0, with a WARNING.
 *
 * Used to implement the ESHIFT_DBATOM key.
 */
std::map<int, double>
derive_eshift_by_lsq_fit(const StructureDB &stdb);

/**
 * Drop configurations whose per-atom energy or peak per-atom force violates
 * the supplied thresholds. Returns {n_dropped_energy, n_dropped_force}.
 * The energy of a config with eweight 0 and the forces of one with fweight
 * 0 are placeholders: the energy (force) thresholds do not read them.
 *
 * Pass std::nullopt for any threshold to disable that filter.
 */
struct OutlierCounts { std::size_t energy = 0; std::size_t force = 0; };
OutlierCounts apply_outlier_filter(StructureDB &stdb,
                                   std::optional<double> emin_per_atom,
                                   std::optional<double> emax_per_atom,
                                   std::optional<double> fmax_per_atom);

/**
 * Multiply eweight, fweight, sweight of every config in dataset i by w[i].
 * Datasets are identified via StructureDB::dbidx. @p w must have the same
 * length as the number of datasets, and each value be a finite number >= 0
 * (std::runtime_error otherwise).
 */
void apply_per_db_weights(StructureDB &stdb,
                          const std::vector<double> &w);

/**
 * Compute auto-balance per-dataset factors: factor[i] = N_i^(-alpha), where
 * N_i is the number of configurations in dataset i. Returns a vector with one
 * value per dataset (indexed via StructureDB::dbidx).
 *
 * Throws std::invalid_argument if alpha < 0 or alpha > 1.
 */
std::vector<double>
auto_balance_db_weights(const StructureDB &stdb, double alpha);

/**
 * Apply Boltzmann eweight reweighting:
 *     eweight *= exp(-(E/N - E_min)/(kB * T_kelvin))
 * where E_min is the minimum per-atom energy of the configs with an energy
 * (eweight != 0); a config with eweight 0 keeps it.
 *
 * Throws std::invalid_argument if T_kelvin is not > 0.
 */
void apply_boltzmann_eweight(StructureDB &stdb, double T_kelvin);

/**
 * For each configuration, subtract the mean per-atom force so that the sum of
 * forces over the configuration is exactly zero.
 */
void apply_com_force_zero(StructureDB &stdb);

/**
 * Stamp the resolved LSCALE / ESHIFT from @p report into a Context that will be
 * written into pot.tadah, so prediction can round-trip them: the model's
 * length scale and energy reference, key lines of the potential. The other
 * transform keys (EFILTER, FFILTER, WDBFILE, WDBFILE_AUTO, EWEIGHT_TEMP,
 * ZERO_COM_FORCE, and the ESHIFT_ATOM / ESHIFT_DBATOM toggles) prepared the
 * training data: the potential's writer puts them into its training record
 * (dataset_preparation_keys, training_record.h), and tadah predict ignores
 * them in a potential, applying to the data it predicts only the EFILTER,
 * FFILTER and ZERO_COM_FORCE predict is given (PredictEngine::run).
 */
void stamp_into_param_file(tadah::core::Context &param_file,
                           const LoadTransformReport &report);

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_LOAD_TRANSFORMS_H
