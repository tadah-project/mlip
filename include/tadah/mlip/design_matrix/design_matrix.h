#ifndef DESIGN_MATRIX_H
#define DESIGN_MATRIX_H

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <tadah/mlip/data_weights.h>
#include <tadah/mlip/st_descriptors_db.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/column_block.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/parallel.h>
#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/mlip/memory/imlip_workspace_manager.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/memory/design_matrix_workspace.h>

#include <stdexcept>
#include <string>
#include <utility>
namespace tadah {
namespace mlip {


class DesignMatrixBase {
public:
  static int phi_rows_num(const tadah::core::Context &c, int nstruct_tot, int natoms_tot) {
    bool force = c.get<bool>("FORCE");
    bool stress = c.get<bool>("STRESS");
    int rows = nstruct_tot;
    if (stress) rows += 6*nstruct_tot;
    if (force) rows += 3*natoms_tot;
    return rows;
  }
};


/** Design tadah::core::Matrix \f$ \Phi \f$ and the corresponding target vector \f$ \mathrm{T} \f$
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
 *     | force on the 2nd atom in the x-dir
 *     | ...
 *     | force on the n-th atom in the z-dir
 *     | stress xx
 *     | stress xy
 *     | stress xz
 *     | stress yy
 *     | stress yz
 *     | stress zz
 *
 * It reads \ref FORCE and \ref STRESS (which rows there are) and, for a
 * fit (apply_weights), \ref EWEIGHT, \ref FWEIGHT and \ref SWEIGHT.
 *
 * Row factors (compute_wfactors): a fit multiplies each row and its target
 * by the square root of the product of the structure's weight and the
 * global weight: sqrt(eweight x EWEIGHT) for the energy row (the total
 * energy of the structure, in eV), sqrt(fweight x FWEIGHT) for each of its
 * 3N force rows and sqrt(sweight x SWEIGHT) for each of its 6 stress rows
 * (xx, xy, xz, yy, yz, zz). No factor depends on the number of atoms: the
 * least-squares sum runs over the squared residuals of the total energies,
 * of the force components and of the six stress entries, each times its
 * weights. A row of weight 0 is a zero row: no observation.
 *
 * @tparam F tadah::models::Function_Base child -> tadah::models::BF_Base or tadah::models::Kern_Base
 *
 */
template <typename F>
class DesignMatrix : public DesignMatrixBase {

public:

  F f;
  tadah::mlip::memory::DesignMatrixWorkspace *ws = nullptr;
  /** Whether the rows are weighted by sqrt(weight * EWEIGHT/FWEIGHT/SWEIGHT)
   *  (a fit); false builds the rows as they are, for a prediction
   *  (M_BLR/M_KRR predict with error estimates). */
  bool apply_weights = true;
  /** Whether build() appends the ridge-augmentation rows, one per column
   *  after the rows of the structures (zero; LinearRegressor writes the
   *  LAMBDA ridge of OALGO 1 and 2 into them): a fit. false builds the
   *  rows of the structures only, for a prediction (M_BLR/M_KRR predict
   *  with errors). */
  bool augment = true;
  /** Whether build(stdb, norm, dc, wcopy) normalises the descriptors with
   *  @p norm when NORM is true: a fit, and the prediction of a model whose
   *  weights act on normalised descriptors. false for the prediction of a
   *  linear NORM model, whose weights act on the raw descriptors
   *  (weights_on_raw_descriptors). */
  bool normalise_descriptors = true;
  bool w_copy_=false;  // keep a copy of Phi and T without the row factors

  /** The observations of the last build: the rows of the structures with a
   *  non-zero weight (sqrt(weight x EWEIGHT/FWEIGHT/SWEIGHT) > 0), without
   *  the augmentation rows; every row of the structures when apply_weights
   *  is false. The number of observations an OALGO 3 fit estimates its
   *  noise variance with (LinearRegressor::train). */
  size_t n_observations() const { return n_obs_; }
  /** Whether the last build had rows of each type (E, F, S: Tlabels 0, 1,
   *  2) with a non-zero weight: the types the fit was fitted to. */
  const std::array<bool, 3> &fitted_types() const { return fitted_types_; }

  /** \brief The design-matrix column layout of the last build():
   *  f.phi_layout() of the descriptor layout (see column_block.h). For a
   *  linear function it is the descriptor layout (bias, D2 and DM
   *  blocks); a non-linear function owns one block over all columns.
   *
   *  Set by build(stdb, norm, dc, wcopy): empty when dc reports no
   *  layout (DC_Base's default). Empty until that build runs; the
   *  StDescriptorsDB overload leaves it as it is. HPO's per-iteration
   *  health monitor does not read it: it computes the same layout,
   *  f.phi_layout(dc.layout(ctx), ctx), for the weights of every
   *  evaluation's model (M_BLR and M_KRR build a design matrix of their
   *  own, and a FIXINDEX that pins every weight builds none).
   */
  ColumnLayout layout_;
  const ColumnLayout& layout() const { return layout_; }

  /** The standard deviations of the reference energies per atom, forces
   *  and stresses of the last build with them (compute_stdevs(), logged at
   *  INFO); 1 before. */
  double e_std_dev=1;
  double f_std_dev=1;
  double s_std_dev[6] = {1,1,1,1,1,1};

  /** \brief This constructor fully initialise this object
     *
     * This class is used to build a Design tadah::core::Matrix.
     *
     * Usage example:
     * \code{.cpp}
     * tadah::core::Context context("tadah::core::Context");
     * tadah::models::BF_Linear bf(context);
     * DesignMatrix<LinearKernel> desmat(bf, context);
     *
     * \endcode
     */
  DesignMatrix(F &f, const tadah::core::Context &c, tadah::mlip::memory::IMLIPWorkspaceManager& workspaceManager)
    : f(f),
    workspaceManager_(&workspaceManager),
    ownWorkspaceManager(false),
    context(c),
    force(context.get<bool>("FORCE")),
    stress(context.get<bool>("STRESS")),
    eweightglob(context.get<double>("EWEIGHT")),
    fweightglob(context.get<double>("FWEIGHT")),
    sweightglob(context.get<double>("SWEIGHT"))
  {
  }

  // Constructor without workspaceManager_ parameter (delegating constructor)
  DesignMatrix(F &f, const tadah::core::Context &c)
  : DesignMatrix(f, c, *new tadah::mlip::memory::MLIPWorkspaceManager())
  {
    ownWorkspaceManager = true;  // Set ownership flag
  }

  // Destructor
  ~DesignMatrix() {
    if (ownWorkspaceManager && workspaceManager_ != nullptr) {
      delete workspaceManager_;
      workspaceManager_ = nullptr;
    }
  }

  /** \brief Build design matrix from already calculated StDescriptorsDB
     *
     * Here we simply build matrix from already calculated descriptors.
     * The vector of targets  **T** is build from StructureDB.
     * D2, D3 and DM calculators are not used.
     */
  void build(StDescriptorsDB &st_desc_db, const StructureDB &stdb, bool wcopy) {
    if (apply_weights) require_valid_weights(stdb, context);
    w_copy_ = wcopy;
    calc_mn(stdb);
    ws = workspaceManager_->getDesignMatrixWorkspace(rows,cols, wcopy);
    ws->Phi.set_zero();
    ws->T.set_zero();
    ws->Tlabels.set_zero();
    compute_stdevs(stdb);
    fill_T(stdb);
    compute_wfactors(stdb); // call after fill_T
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
    // loop over all structures and build. Stage 2.3: routes onto the shared
    // ScriptExecutor pool when HPO installs a backend, else the historical
    // OpenMP loop. Each s writes a disjoint Phi row-block => order-independent.
    parallel_for(stdb.size(), [&](size_t s) {
      build(rows[s], stdb(s), st_desc_db(s));
    });
    // The row factors, as the build from a DescriptorsCalc applies them.
    if (w_copy_) {
      copyAndScale();
    }
    else {
      scale();
    }
  };

  /** \brief Calculate descriptors and build design matrix. */
  template <typename DC>
  void build(const StructureDB &stdb, Normaliser &norm,
             DC &dc, bool wcopy) {
    //DescriptorsCalc<D2,D3,DM,C2,C3,CM> dc(context);
    if (apply_weights) require_valid_weights(stdb, context);
    w_copy_ = wcopy;
    calc_mn(stdb);
    ws = workspaceManager_->getDesignMatrixWorkspace(rows,cols, wcopy);
    ws->Phi.set_zero();
    ws->T.set_zero();
    ws->Tlabels.set_zero();

    // Two checks, each failing loud:
    //  1. the descriptor layout (bias + D2 + DM blocks) against DSIZE, the
    //     descriptor size every descriptor vector is built with: a
    //     disagreement is a stale descriptor / DSIZE;
    //  2. the function's layout of its design matrix (phi_layout) against
    //     `cols`, the width calc_mn() took from f.get_phi_cols(context).
    //     A linear function's columns are the descriptor's; a kernel's are
    //     its basis vectors, and BF_Polynomial2's the D(D+1)/2 products of
    //     descriptor components (column_block.h).
    // (Before, the descriptor layout was compared with `cols`, which only a
    // linear function's width equals: every non-linear model was refused.)
    //
    // DC_Base's default layout() returns an empty ColumnLayout; in that
    // case layout_ stays empty and both checks are skipped (legacy callers
    // stay byte-identical). Concrete DescriptorsCalc<...>::layout()
    // overrides return a populated layout.
    layout_ = ColumnLayout{};
    const ColumnLayout descriptor_layout = dc.layout(context);
    if (!descriptor_layout.blocks.empty()) {
      const size_t dsize = context.get<size_t>("DSIZE");
      std::string err = descriptor_layout.validate(dsize, "DSIZE");
      if (!err.empty()) {
        throw std::runtime_error(
            "DesignMatrix::build: " + err +
            " — the descriptor layout disagrees with the descriptor size.");
      }
      ColumnLayout phi_layout = f.phi_layout(descriptor_layout, context);
      err = phi_layout.validate(cols);
      if (!err.empty()) {
        throw std::runtime_error(
            "DesignMatrix::build: " + err + " — the column layout of " +
            f.get_label() + " disagrees with its design-matrix width.");
      }
      layout_ = std::move(phi_layout);
    }

    compute_stdevs(stdb);
    fill_T(stdb);
    compute_wfactors(stdb); // call after fill_T

    // for openmp we need to find first rows for each structure
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
    // Stage 2.3: shared-pool backend when installed, else the OpenMP loop.
    // Each s writes a disjoint Phi row-block (rows[s]) => order-independent.
    const bool normalise_rows = normalise_descriptors && context.get<bool>("NORM");
    parallel_for(stdb.size(), [&](size_t s) {
      StDescriptors st_d = dc.calc(stdb(s));

      if (normalise_rows)  // TODO move to scale()
        norm.normalise(st_d);

      build(rows[s], stdb(s), st_d);
    });
    if (w_copy_) {
      copyAndScale();
    }
    else {
      scale();
    }

  }
  void build(size_t &row, const Structure &st, const StDescriptors &st_d) {

    f.calc_phi_energy_row(ws->Phi,row,st,st_d);
    if (force) {
      f.calc_phi_force_rows(ws->Phi,row,st,st_d);
    }
    if (stress) {
      f.calc_phi_stress_rows(ws->Phi,row,st,st_d);
    }
  }
  void fill_T(const StructureDB &stdb, size_t start=0) {
    size_t j=start;
    for (size_t s=0; s<stdb.size(); ++s) {

      ws->Tlabels(j) = 0;
      ws->T(j++) = stdb(s).energy;

      if (force) {
        for (const Atom &a : stdb(s).atoms) {
          ws->Tlabels(j) = 1;
          ws->T(j++) = a.force(0);
          ws->Tlabels(j) = 1;
          ws->T(j++) = a.force(1);
          ws->Tlabels(j) = 1;
          ws->T(j++) = a.force(2);
        }
      }
      if (stress) {
        for (size_t x=0; x<3; ++x) {
          for (size_t y=x; y<3; ++y) {
            ws->Tlabels(j) = 2;
            ws->T(j++)=stdb(s).stress(x,y);
          }
        }
      }
    }
  }

  tadah::core::phi_type &getPhi() { return ws->Phi; }
  tadah::core::t_type &getT() { return ws->T; }
  tadah::core::t_type &getTlabels() { return ws->Tlabels; }
  bool hasCopy() { return w_copy_; }

  /** The values the weights @p w predict for the structures of the last
   *  build, from its rows without the row factors: the energy of each
   *  structure, and with FORCE (STRESS) its forces (its stress tensor). The
   *  rows are those of the copy a build with wcopy true keeps (or of the
   *  matrix of a build with apply_weights false); the ridge rows after the
   *  data rows are not read.
   *  @param stdb The structures of the last build: the same number, and
   *  each with the same atoms (their species and positions, in order) and
   *  cell, bit for bit (structure_identity()); their labels (energies,
   *  forces, stresses, weights) are not read.
   *  @param w The weights, one per column.
   *  @throws std::logic_error when the last build kept no rows without the
   *  factors, or @p w has another size than the columns.
   *  @throws std::runtime_error when @p stdb has other structures or rows
   *  than the last build, or a structure whose atoms or cell are not those
   *  of the last build's structure at its place. */
  StructureDB predict_built(const StructureDB &stdb,
                            const tadah::core::t_type &w) const {
    if (!ws || (!w_copy_ && apply_weights))
      throw std::logic_error(
          "DesignMatrix::predict_built: the last build kept no rows without "
          "the row factors (wcopy false)");
    const tadah::core::phi_type &P = w_copy_ ? ws->Phi_cpy : ws->Phi;
    if (w.size() != P.cols())
      throw std::logic_error("DesignMatrix::predict_built: " +
                             std::to_string(w.size()) + " weights for " +
                             std::to_string(P.cols()) + " columns");
    size_t rows_of_stdb = 0;
    for (size_t s = 0; s < stdb.size(); ++s)
      rows_of_stdb += 1 + (force ? 3 * stdb(s).natoms() : 0) + (stress ? 6 : 0);
    if (stdb.size() != built_structures_ || rows_of_stdb != data_rows_)
      throw std::runtime_error(
          "the StructureDB has " + std::to_string(stdb.size()) +
          " structures (" + std::to_string(rows_of_stdb) + " rows), the last "
          "fit " + std::to_string(built_structures_) + " structures (" +
          std::to_string(data_rows_) + " rows): it predicts the structures "
          "it was fitted to");
    for (size_t s = 0; s < stdb.size(); ++s)
      if (structure_identity(stdb(s)) != built_identity_[s])
        throw std::runtime_error(
            structure_in_data(stdb, s) + " is not structure " +
            std::to_string(s + 1) + " of the last fit: its species, cell or "
            "atom positions differ, and the rows give the values of the "
            "fit's structure; it predicts the structures it was fitted to "
            "(predict(Context, StructureDB, DescriptorsCalc) predicts any "
            "structures)");
    StructureDB out;
    out.dbidx = stdb.dbidx;
    out.structures.resize(stdb.size());
    auto row = [&](size_t i) {
      double v = 0.0;
      for (size_t j = 0; j < P.cols(); ++j) v += P(i, j) * w(j);
      return v;
    };
    size_t i = 0;
    for (size_t s = 0; s < stdb.size(); ++s) {
      out(s).label = stdb(s).label;
      out(s).energy = row(i++);
      if (force) {
        out(s).atoms.resize(stdb(s).natoms());
        for (size_t a = 0; a < stdb(s).natoms(); ++a)
          for (size_t k = 0; k < 3; ++k) out(s).atoms[a].force[k] = row(i++);
      }
      if (stress) {
        for (size_t x = 0; x < 3; ++x)
          for (size_t y = x; y < 3; ++y) {
            out(s).stress(x, y) = row(i++);
            if (x != y) out(s).stress(y, x) = out(s).stress(x, y);
          }
      }
    }
    return out;
  }

  void resize(size_t rows, size_t cols, bool wcopy=false) {
    ws = workspaceManager_->getDesignMatrixWorkspace(rows,cols, wcopy);
    set_zero();
  }
  void set_zero() {
    ws->Phi.set_zero();
    ws->T.set_zero();
    ws->Tlabels.set_zero();
  }
  void compute_wfactors(const StructureDB &stdb) {
    n_obs_ = 0;
    fitted_types_ = {false, false, false};
    if (!apply_weights) {  // a prediction: the rows as they are
      for (std::size_t i = 0; i < ws->wfactors.size(); ++i) ws->wfactors(i) = 1.0;
      n_obs_ = rows - (augment ? cols : 0);
      fitted_types_ = {n_obs_ > 0, force && n_obs_ > 0, stress && n_obs_ > 0};
      return;
    }
    size_t j=0;
    // One row of type `type` and factor `w`; the observations are the rows
    // of non-zero weight (a weight-0 row is a zero row of the fit).
    auto row = [&](double w, size_t type) {
      ws->wfactors(j++) = w;
      if (w > 0.0) {
        ++n_obs_;
        fitted_types_[type] = true;
      }
    };
    for (size_t s=0; s<stdb.size(); ++s) {
      row(std::sqrt(stdb(s).eweight*eweightglob), 0);
      if (force) {
        double fscale = std::sqrt(stdb(s).fweight*fweightglob);
        for (size_t a=0; a<stdb(s).natoms(); ++a) {
          row(fscale, 1);
          row(fscale, 1);
          row(fscale, 1);
        }
      }
      if (stress) {
        for (size_t x=0; x<3; ++x) {
          for (size_t y=x; y<3; ++y) {
            row(std::sqrt(stdb(s).sweight*sweightglob), 2);
          }
        }
      }
    }
  }

private:
  tadah::mlip::memory::IMLIPWorkspaceManager* workspaceManager_ = nullptr;
  bool ownWorkspaceManager = false;
  const tadah::core::Context & context;
  size_t rows = 0;
  size_t cols = 0;
  bool force;
  bool stress;
  double eweightglob;
  double fweightglob;
  double sweightglob;
  size_t n_obs_ = 0;
  std::array<bool, 3> fitted_types_{{false, false, false}};
  size_t built_structures_ = 0;  // the structures of the last build
  size_t data_rows_ = 0;         // its rows without the ridge rows
  // structure_identity() of each structure of the last build that kept its
  // rows without the row factors (predict_built()); empty otherwise.
  std::vector<std::uint64_t> built_identity_;

  /* What identifies a structure's rows besides the model: its number of
   * atoms, its cell, and the species and position of each atom in order,
   * bit for bit (an FNV-1a hash of their bytes). The labels (energy,
   * forces, stress, weights) are not part of it. */
  static std::uint64_t structure_identity(const Structure &st) {
    std::uint64_t h = 14695981039346656037ULL;
    auto mix = [&h](const void *p, std::size_t n) {
      const unsigned char *b = static_cast<const unsigned char *>(p);
      for (std::size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
      }
    };
    auto mix_double = [&mix](double v) {
      std::uint64_t u;
      std::memcpy(&u, &v, sizeof u);
      mix(&u, sizeof u);
    };
    const std::uint64_t n = st.natoms();
    mix(&n, sizeof n);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) mix_double(st.cell(i, j));
    for (const Atom &a : st.atoms) {
      const std::int64_t Z = a.Z;
      mix(&Z, sizeof Z);
      for (int k = 0; k < 3; ++k) mix_double(a.position[k]);
    }
    return h;
  }

  // resize Phi and T and set all elements to zero
  void calc_mn(const StructureDB &stdb) {
    rows=0;
    cols = f.get_phi_cols(context);

    for (size_t s=0; s<stdb.size(); ++s) {
      rows++;
      if (stress) rows+=6;
      if (force) rows+=stdb(s).natoms()*3;
    }
    built_structures_ = stdb.size();
    data_rows_ = rows;
    built_identity_.clear();
    if (w_copy_ || !apply_weights) {  // rows predict_built() can read
      built_identity_.reserve(stdb.size());
      for (size_t s = 0; s < stdb.size(); ++s)
        built_identity_.push_back(structure_identity(stdb(s)));
    }


    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "DesignMatrix: rows: " + std::to_string(rows) + " cols: " + std::to_string(cols));

    // Add space for augmented matrix (a fit)
    if (augment) {
      rows += cols;
      tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "DesignMatrix (including augmented matrix): rows: " + std::to_string(rows) + " cols: " + std::to_string(cols));
    }
  }

  // The standard deviations of the reference energies per atom, forces and
  // stresses of the structures that have them: a structure with eweight 0
  // (fweight 0, sweight 0) has no energy (forces, stress) of its own, only a
  // placeholder. With fewer than two values a deviation is not computed
  // (it stays as it was) and not logged.
  void compute_stdevs(const StructureDB &stdb) {
    size_t n_e = 0, n_s = 0, num_forces = 0;
    for (size_t s=0; s<stdb.size(); ++s) {
      if (stdb(s).eweight != 0.0) ++n_e;
      if (stdb(s).sweight != 0.0) ++n_s;
      if (stdb(s).fweight != 0.0) num_forces += 3*stdb(s).natoms();
    }
    tadah::core::t_type evec(n_e);
    tadah::core::Matrix svec(n_s,6);
    size_t ie = 0, is = 0;
    for (size_t s=0; s<stdb.size(); ++s) {
      if (stdb(s).eweight != 0.0)
        evec(ie++) = stdb(s).energy/stdb(s).natoms();
      if (stress && stdb(s).sweight != 0.0) {
        size_t j=0;
        for (size_t x=0; x<3; ++x) {
          for (size_t y=x; y<3; ++y) {
            svec(is,j++)=stdb(s).stress(x,y);
          }
        }
        ++is;
      }
    }

    if (n_e > 1) {
      e_std_dev = evec.std_dev(evec.mean(), evec.size()-1);
      tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Energy standard deviation (eV/atom): ", e_std_dev);
    }

    if (stress && n_s > 1) {
      std::string s_std_dev_str;
      for (size_t j=0; j<6; ++j) {
        s_std_dev[j] = svec.col(j).std_dev(svec.col(j).mean(), svec.col(j).size()-1);
        s_std_dev_str += std::to_string(s_std_dev[j]) + " ";
      }
      tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Stress standard deviation (eV/A^3): ", s_std_dev_str);
    }

    if (force && num_forces > 1) {
      size_t j=0;
      tadah::core::t_type fvec(num_forces);
      for (size_t s=0; s<stdb.size(); ++s) {
        if (stdb(s).fweight == 0.0) continue;
        for (const Atom &a : stdb(s).atoms) {
          fvec(j++) = a.force(0);
          fvec(j++) = a.force(1);
          fvec(j++) = a.force(2);
        }
      }
      f_std_dev = fvec.std_dev(fvec.mean(),fvec.size()-1);
      tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO, "Force standard deviation (ev/A): ", f_std_dev);
    }

    // context.addInternal("ESTDEV",e_std_dev);
    // context.addInternal("FSTDEV",f_std_dev);
    // for (size_t j=0; j<6; ++j)
    //   context.addInternal("SSTDEV",s_std_dev[j]);
  }


  void copyAndScale()
  {
    // first copy scaled Phi to Phi_cpy and T to T_cpy
    // followed by swapping pointers
    for (std::size_t j = 0; j < ws->Phi.cols(); ++j) {
      for (std::size_t i = 0; i < ws->Phi.rows(); ++i) {
        ws->Phi_cpy(i,j) = ws->Phi(i,j)*ws->wfactors(i);
      }
    }
    for (std::size_t i = 0; i < ws->Phi.rows(); ++i) {
      ws->T_cpy(i) = ws->T(i)*ws->wfactors(i);
    }
    std::swap(ws->Phi, ws->Phi_cpy);
    std::swap(ws->T, ws->T_cpy);
  }
  void scale()
  {
    for (std::size_t j = 0; j < ws->Phi.cols(); ++j) {
      for (std::size_t i = 0; i < ws->Phi.rows(); ++i) {
        ws->Phi(i,j) *= ws->wfactors(i);
      }
    }
    for (std::size_t i = 0; i < ws->Phi.rows(); ++i) {
      ws->T(i) *= ws->wfactors(i);
    }
  }
};
} // namespace mlip
} // namespace tadah
#endif
