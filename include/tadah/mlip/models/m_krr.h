#ifndef M_KRR_H
#define M_KRR_H

#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/raw_descriptor_weights.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/design_matrix.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/models/m_krr_train.h>
#include <tadah/core/context.h>
#include <tadah/models/descriptors/dm/embf0.h>
#include <tadah/models/descriptors/dm/rhoform.h>
#include <tadah/mlip/models/basis.h>
#include <tadah/core/intercept_layout.h>
#include <tadah/mlip/intercepts.h>
#include <tadah/mlip/row_weights.h>
#include <tadah/mlip/memory/imlip_workspace_manager.h>
#include <tadah/mlip/training_datasets.h>
#include <tadah/mlip/data_weights.h>
#include <tadah/core/number_format.h>

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <iostream>
namespace tadah {
namespace mlip {

/**
 * @class M_KRR
 * @brief Kernel Ridge Regression (KRR) on a set of basis vectors.
 *
 * `MODEL M_KRR <kernel> [1|2]`: the third token selects the
 * implementation (1, the default, or 2).
 *
 * **Implementation 1** fits energies, and forces and stresses when FORCE and
 * STRESS are true. With Kern_Linear the model is linear in the descriptor
 * (one weight per descriptor component, as M_BLR BF_Linear). With a
 * non-linear kernel the atomic energy is
 * \f$ E_i = \sum_{j} w_j K(\mathbf{b}_j, \mathbf{x}_i) \f$ over SBASIS
 * basis vectors \f$ \mathbf{b}_j \f$: columns 0 .. n0-1 are the
 * intercept vectors e_0 .. e_{n0-1} (n0 = N, one per ATOM species, with
 * BIAS true; e0 alone, n0 = 1, otherwise) and the other SBASIS-n0 are
 * atomic energy descriptors of the training set drawn by a fixed, portable
 * draw (Basis), so training is reproducible.
 * The design-matrix column j is K(b_j, x) summed over the atoms (and its
 * derivatives for the force and stress rows), and the weights solve the
 * least-squares problem of those columns (OALGO, LAMBDA). The potential
 * records SBASIS and BASIS with the weights.
 *
 * **Implementation 2** fits energies only, by solving
 * \f$ (K + \lambda \sigma^2 I) w = T \f$ on SBASIS basis vectors (the n0
 * intercept vectors, with target 0, and the mean descriptors of drawn
 * structures, with their energies per atom), \f$ \sigma \f$ = ESIGMA in
 * eV: the residuals of the energies per atom over ESIGMA, as every fit
 * weighs them (row_weights.h). No priority of the data enters this solve:
 * a fit that asks for forces, stresses, priorities or ENORM SQRT is
 * refused (refuse_what_train2_ignores()). A non-linear kernel's weights
 * are w, and its potential records SBASIS and BASIS. Kern_Linear's are the
 * descriptor weights of its basis vectors, \f$ \sum_j w_j \mathbf{b}_j \f$
 * (converted to the raw descriptors with NORM, as implementation 1's): its
 * potential has DSIZE WEIGHTS and no BASIS, as implementation 1's.
 *
 * **Configuration options**:
 * - **MPARAMS**: the kernel's parameters: sigma for Kern_RBF
 *   (\f$ \gamma = 1/\sigma^2 \f$), d gamma c for Kern_Polynomial, gamma c
 *   for Kern_Sigmoid; Kern_Linear, Kern_Quadratic and Kern_LQ take none.
 * - **SBASIS**: the number of basis vectors of a non-linear kernel, at least
 *   n0 + 1 and at most the number of training AEDs (structures for
 *   implementation 2) plus n0.
 * - **LAMBDA**: Controls regularization. Use `0` for ordinary least squares,
 *   positive values for manual setting, or `-1` for automatic tuning
 *   (OALGO 3). The kernel's intercept columns
 *   (Function_Base::intercept_columns()) are not penalised.
 *
 * @tparam K DM_Kern_Base child, Kernel function
 */
template
<class K=DM_Function_Base&>
class M_KRR: public M_Tadah_Base,
  public tadah::models::M_KRR_Train<K>
{

public:

  /** 
     * @brief Initializes for training or prediction using a configuration.
     * 
   * **Example**:
   * \code{.cpp}
   * tadah::core::Context context("tadah::core::Context");
   * M_KRR<tadah::models::Kern_Linear> krr(context);
   * \endcode
   *
     * @param c Contexturation object.
     */
  M_KRR(tadah::core::Context &c):
    tadah::models::M_KRR_Train<K>(c),
    basis(c),
    desmat(kernel,c)
  {
    norm = Normaliser(c);
  }

  /** 
   * @brief Initializes for training or prediction using a kernel and configuration.
   * 
   * @param kernel Kernel function.
   * @param c Contexturation object.
   */
  M_KRR(K &kernel, tadah::core::Context &c):
    tadah::models::M_KRR_Train<K>(kernel,c),
    basis(c),
    desmat(kernel,c)
  {
    norm = Normaliser(c);
  }

  M_KRR(K &kernel, tadah::core::Context &c, tadah::mlip::memory::IMLIPWorkspaceManager& workspaceManager):
    tadah::models::M_KRR_Train<K>(kernel,c,workspaceManager),
    basis(c),
    desmat(kernel,c)
  {
    norm = Normaliser(c);
  }
  double epredict(const tadah::core::aed_type &aed) const {
    return kernel.epredict(weights,aed);
  };

  double fpredict(const tadah::core::fd_type &fdij, const tadah::core::aed_type &aedi, const size_t k) const {
    return kernel.fpredict(weights,fdij,aedi,k);
  }

  tadah::core::force_type fpredict(const tadah::core::fd_type &fdij, const tadah::core::aed_type &aedi) const {
    return kernel.fpredict(weights,fdij,aedi);
  }

  void train(StDescriptorsDB &st_desc_db, const StructureDB &stdb) {
    // The settings of the fit first (retired keys, error scales), then the
    // priorities of the data: the checks below read them.
    require_fit_settings(context, "M_KRR::train");
    require_valid_weights(stdb, context);
    check_bias_constraints(context);

    // Precomputed descriptors run implementation 1, whatever MODEL's third
    // value: implementation 2 fits the mean descriptors of whole structures,
    // which train(StructureDB, DescriptorsCalc) computes.
    if (context.size("MODEL")==3 && context.template get<int>("MODEL",2)!=1)
      throw std::runtime_error(
          "MODEL M_KRR " + kernel.get_label() + " " +
          context.template get<std::string>("MODEL",2) + ": the fit of "
          "precomputed descriptors (train(StDescriptorsDB, StructureDB)) runs "
          "implementation 1 only (MODEL M_KRR " + kernel.get_label() + " 1); "
          "implementation 2 fits the mean descriptors of whole structures, "
          "which train(StructureDB, DescriptorsCalc) computes.");

    // Precomputed descriptors: no basis is drawn on this path, so a
    // non-linear kernel would fit no column (or the columns of a basis
    // left over from a potential).
    if (kernel.get_label()!="Kern_Linear")
      throw std::runtime_error(
          "M_KRR " + kernel.get_label() + ": training from precomputed "
          "descriptors (train(StDescriptorsDB, StructureDB)) draws no basis "
          "vectors; train a non-linear kernel with train(StructureDB, "
          "DescriptorsCalc), which draws them.");
    require_intercepts_determined(stdb, context, kernel);

    if(context.template get<bool>("NORM"))
      norm = Normaliser(context,st_desc_db);

    // The rows without the row factors are kept for predict(StructureDB&).
    bool wcopy=true;
    desmat.build(st_desc_db,stdb,wcopy);
    train(desmat);
  }
  void train(const StructureDB &stdb, DC_Base &dc) {
    // The settings of the fit first (retired keys, error scales), whatever
    // the implementation.
    require_fit_settings(context, "M_KRR::train");
    check_bias_constraints(context);
    int modelN;
    if (context.size("MODEL")==3) {
      modelN=context.template get<int>("MODEL",2);
    } else {
      // use default model
      modelN=1;
      context.add("MODEL", modelN);
    }
    require_krr_implementation(context);
    if (modelN==1)
      train1(stdb,dc);
    else
      train2(stdb,dc);
  }
  void train1(const StructureDB &stdb, DC_Base &dc) {
    // The weights first: the intercept check below reads them.
    require_valid_weights(stdb, context);
    // KRR on the design matrix: for a non-linear kernel the columns are
    // its basis vectors, drawn here from the (normalised) training AEDs
    if(context.template get<bool>("NORM") || kernel.get_label()!="Kern_Linear") {
      // either build basis or prep normaliser
      std::string force=context.template get<std::string>("FORCE");
      std::string stress=context.template get<std::string>("STRESS");

      context.remove("FORCE");
      context.remove("STRESS");
      context.add("FORCE", "false");
      context.add("STRESS", "false");

      StDescriptorsDB st_desc_db_temp = dc.calc(stdb);

      if(context.template get<bool>("NORM")) {
        norm = Normaliser(context);
        norm.learn(st_desc_db_temp);
        norm.normalise(st_desc_db_temp);
      }

      context.remove("FORCE");
      context.remove("STRESS");
      context.add("FORCE", force);
      context.add("STRESS", stress);

      if (kernel.get_label()!="Kern_Linear") {
        basis.build_random_basis(context.template get<size_t>("SBASIS"),st_desc_db_temp);
        desmat.f.set_basis(basis.b);
        kernel.set_basis(basis.b);
      }
    }
    // The kernel's intercept columns (with BIAS true those of the intercept
    // basis vectors, drawn above, or Kern_Linear's) must be determined by
    // the energy-weighted compositions.
    require_intercepts_determined(stdb, context, kernel);
    // The rows without the row factors are kept for predict(StructureDB&).
    bool wcopy=true;
    desmat.build(stdb,norm,dc,wcopy);
    train(desmat);
  }

  /** Refuses what implementation 2 would ignore, before it fits
   *  (refuse_what_krr_implementation2_ignores of its context: FORCE,
   *  STRESS, the weights of the configuration, and a structure whose
   *  eweight is not 1).
   *  @throws std::runtime_error naming the key and its values (or the
   *  structure and its eweight) and `MODEL M_KRR <kernel> 1`. */
  void refuse_what_train2_ignores(const StructureDB &stdb) const {
    refuse_what_krr_implementation2_ignores(context, stdb);
  }

  void train2(const StructureDB &stdb, DC_Base &dc) {
    require_valid_weights(stdb, context);
    refuse_what_train2_ignores(stdb);
    fit_rows_ = false;  // implementation 2 fits no design matrix

    // NEW (BASIC) IMPLEMENTATION OF KRR //
    std::string force=context.template get<std::string>("FORCE");
    std::string stress=context.template get<std::string>("STRESS");
    context.remove("FORCE");
    context.remove("STRESS");
    context.add("FORCE", "false");
    context.add("STRESS", "false");
    StDescriptorsDB st_desc_db_temp = dc.calc(stdb);
    if(context.template get<bool>("NORM")) {
      norm = Normaliser(context);
      norm.learn(st_desc_db_temp);
      norm.normalise(st_desc_db_temp);
    }
    context.remove("FORCE");
    context.remove("STRESS");
    context.add("FORCE", force);
    context.add("STRESS", stress);
    basis.prep_basis_for_krr(st_desc_db_temp,stdb);
    // A non-linear kernel predicts with the basis vectors; Kern_Linear's
    // weights are the descriptor weights they make (M_KRR_Train::train2),
    // so it keeps no basis.
    if (kernel.get_label()!="Kern_Linear")
      kernel.set_basis(basis.b);
    // The energies per atom over ESIGMA, as every fit weighs them
    // (ENORM ATOM: implementation 2 refuses SQRT).
    tadah::models::M_KRR_Train<K>::train2(basis.b, basis.T,
                                          ErrorScales::of(context).esigma);
    // With NORM the basis vectors are normalised descriptors: Kern_Linear's
    // weights go to the raw descriptors, as implementation 1's do
    // (train(desmat)), since its prediction does not normalise.
    if (weights_on_raw_descriptors(context, kernel))
      to_raw_descriptors(*this, norm,
                         tadah::core::n_intercept_columns(context));
    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO,
                                        fit_scales_info(context));
  }

  /** Every prediction from descriptors (also through predict(c, stdb, dc)
   *  and predict(c, st, dc)) comes here: the potential's fit to these
   *  descriptors (WEIGHTS, the basis vectors' length, NMEAN/NSTDEV of a
   *  non-linear NORM model: models::require_weights_fit) is checked first,
   *  before they are normalised. */
  Structure predict(const tadah::core::Context &c, StDescriptors &std, const Structure &st) {
    require_weights_fit(c, std.dim());
    if(context.template get<bool>("NORM") && !std.normalised &&
       !weights_on_raw_descriptors(context, kernel))
      norm.normalise(std);
    return M_Tadah_Base::predict(c,std,st);
  }

  StructureDB predict(tadah::core::Context &c, const StructureDB &stdb, DC_Base &dc) {
    return M_Tadah_Base::predict(c,stdb,dc);
  }

  const tadah::models::Function_Base &function() const override {
    return kernel;
  }

tadah::core::Context get_param_file() {
    tadah::core::Context c = context;
    // DM_REAM potentials always carry explicit EMBF0 and RHOFORM lines (also
    // for a DM_mJoin parent, whose children read copies of the context); any
    // other potential never does. RHOFORM is never supplied: a DM_REAM
    // context without it throws (the DM_REAM ctor refused it already).
    tadah::models::normalise_embf0_key(c);
    tadah::models::normalise_rhoform_key(c);
    // The model's settings and its fit, not the training data, as M_BLR's:
    // no DBFILE; ALPHA and BETA (the evidence approximation's starting
    // guesses, or the values a LAMBDA -1 fit found) stay, and so do FORCE
    // and STRESS: the potential records what its fit used (tadah predict's
    // -F/-S and pair_style tadah do not read them).
    remove_training_datasets(c);

    c.remove("MODEL");
    c.add("MODEL", label);
    c.add("MODEL", kernel.get_label());
    int modelN=context.template get<int>("MODEL",2);
    c.add("MODEL", modelN);

    // The error scales and switches of the fit, the defaults included: a
    // retrain from the potential reads them.
    write_fit_scales(c);
    if (c.contains("WEIGHTS")) {
      c.remove("WEIGHTS");
    }
    for (size_t i=0;i<weights.size();++i) {
      c.add("WEIGHTS", weights(i));
    }

    // NMEAN and NSTDEV of this fit only: a training configuration may carry
    // those of another potential (tadah train -c pot.tadah).
    if (c.has_stored("NMEAN")) c.remove("NMEAN");
    if (c.has_stored("NSTDEV")) c.remove("NSTDEV");
    if(context.template get<bool>("NORM")) {
      for (size_t i=0;i<norm.mean.size();++i) {
        c.add("NMEAN", norm.mean[i]);
      }
      for (size_t i=0;i<norm.std_dev.size();++i) {
        c.add("NSTDEV", norm.std_dev[i]);
      }
    }
    if (kernel.get_label()!="Kern_Linear") {
      // dump basis to the context file file
      // make sure keys are not accidentally assigned
      if (c.contains("SBASIS"))
        c.remove("SBASIS");
      if (c.contains("BASIS"))
        c.remove("BASIS");
      // The kernel's basis: the one training drew (set_basis) or the one
      // a potential loaded (BASIS), the vectors the weights belong to.
      const tadah::core::Matrix &b =
          dynamic_cast<const tadah::models::Kern_Base&>(kernel).get_basis();
      c.add("SBASIS", b.cols());
      for (size_t i=0;i<b.cols();++i) {
        for (size_t j=0;j<b.rows();++j) {
          c.add("BASIS", b(j,i));
        }
      }
    }
    return c;
  }
  /** The values the last fit of this model predicts for the structures
   *  it was fitted to, @p stdb, from the rows of its design matrix without
   *  the row factors, as M_BLR::predict(StructureDB&).
   *  @throws std::runtime_error for a model without a fit of a design
   *  matrix of its own (one built from a potential's WEIGHTS, or a fit of
   *  implementation 2, which builds none), and for a StructureDB with other
   *  structures than the fit's. */
  StructureDB predict(StructureDB &stdb) {
    if (!fit_rows_)
      throw std::runtime_error(
          "M_KRR::predict(StructureDB&) predicts the structures of the "
          "model's last fit from its design matrix: this model has no fit "
          "of its own (it was built from a potential's WEIGHTS, or fitted "
          "by implementation 2, which builds no design matrix); "
          "predict(Context, StructureDB, DescriptorsCalc) predicts any "
          "structures.");
    try {
      return desmat.predict_built(stdb, fit_weights_);
    } catch (const std::runtime_error &e) {
      throw std::runtime_error(std::string("M_KRR::predict(StructureDB&): ") +
                               e.what());
    }
  }

private:
  std::string label="M_KRR";
  Basis<K> basis;
  DesignMatrix<K> desmat;
  /** The weights of the last fit of the design matrix as it solved them
   *  (before Kern_Linear's NORM weights are converted to the raw
   *  descriptors): predict(StructureDB&) multiplies its rows by them. */
  tadah::core::t_type fit_weights_;
  /** Whether this object fitted a design matrix (predict(StructureDB&)). */
  bool fit_rows_ = false;

  template <typename D>
  void train(D &desmat) {
    // The solve overwrites the weighted rows it is given (see M_BLR).
    tadah::core::phi_type &Phi = desmat.getPhi();
    tadah::core::t_type T = desmat.getT();
    fit_rows_ = false;
    tadah::models::M_KRR_Train<K>::train(Phi, T, desmat.n_observations());
    fit_weights_ = weights;
    fit_rows_ = desmat.hasCopy();

    if (weights_on_raw_descriptors(context, kernel))
      to_raw_descriptors(*this, norm,
                         tadah::core::n_intercept_columns(context));
    tadah::core::Logger::instance().log(tadah::core::LogLevel::INFO,
                                        fit_scales_info(context));
  }

  // Do we want to confuse user with those and make them public?
  // Either way they must be stated as below to silence clang warning
  using tadah::models::M_KRR_Train<K>::predict;
  using tadah::models::M_KRR_Train<K>::train;
  using tadah::models::M_KRR_Train<K>::trained;
  using tadah::models::M_KRR_Train<K>::weights;
  using tadah::models::M_KRR_Train<K>::context;
  using tadah::models::M_KRR_Train<K>::kernel;
};
} // namespace mlip
} // namespace tadah
#endif
