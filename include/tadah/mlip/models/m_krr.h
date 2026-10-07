#ifndef M_KRR_H
#define M_KRR_H

#include <tadah/mlip/models/m_tadah_base.h>
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
#include <tadah/mlip/memory/imlip_workspace_manager.h>

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
 * basis vectors \f$ \mathbf{b}_j \f$: column 0 is the bias vector e0 and
 * the other SBASIS-1 are atomic energy descriptors of the training set
 * drawn by a fixed, portable draw (Basis), so training is reproducible.
 * The design-matrix column j is K(b_j, x) summed over the atoms (and its
 * derivatives for the force and stress rows), and the weights solve the
 * least-squares problem of those columns (OALGO, LAMBDA). The potential
 * records SBASIS and BASIS with the weights.
 *
 * **Implementation 2** fits energies only, by solving
 * \f$ (K + \lambda I) w = T \f$ on SBASIS basis vectors (e0 and the mean
 * descriptors of drawn structures).
 *
 * **Configuration options**:
 * - **MPARAMS**: the kernel's parameters: sigma for Kern_RBF
 *   (\f$ \gamma = 1/\sigma^2 \f$), d gamma c for Kern_Polynomial, gamma c
 *   for Kern_Sigmoid; Kern_Linear, Kern_Quadratic and Kern_LQ take none.
 * - **SBASIS**: the number of basis vectors of a non-linear kernel, at least
 *   2 and at most the number of training AEDs (structures for
 *   implementation 2) plus 1.
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
    check_bias_constraints(context);
    require_intercepts_determined(stdb, context);

    // Precomputed descriptors: no basis is drawn on this path, so a
    // non-linear kernel would fit no column (or the columns of a basis
    // left over from a potential).
    if (kernel.get_label()!="Kern_Linear")
      throw std::runtime_error(
          "M_KRR " + kernel.get_label() + ": training from precomputed "
          "descriptors (train(StDescriptorsDB, StructureDB)) draws no basis "
          "vectors; train a non-linear kernel with train(StructureDB, "
          "DescriptorsCalc), which draws them.");

    if(context.template get<bool>("NORM"))
      norm = Normaliser(context,st_desc_db);

    bool wcopy=false;
    desmat.build(st_desc_db,stdb,wcopy);
    train(desmat);
  }
  void train(const StructureDB &stdb, DC_Base &dc) {
    check_bias_constraints(context);
    require_intercepts_determined(stdb, context);
    int modelN;
    if (context.size("MODEL")==3) {
      modelN=context.template get<int>("MODEL",2);
    } else {
      // use default model
      modelN=1;
      context.add("MODEL", modelN);
    }
    if (modelN==1)
      train1(stdb,dc);
    else if (modelN==2)
      train2(stdb,dc);
    else
      throw
    std::runtime_error(
      "This KRR implementation does exist: "\
      + std::to_string(modelN)+"\n");
  }
  void train1(const StructureDB &stdb, DC_Base &dc) {
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
    bool wcopy=false;
    desmat.build(stdb,norm,dc,wcopy);
    train(desmat);
  }

  void train2(const StructureDB &stdb, DC_Base &dc) {

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
    kernel.set_basis(basis.b);
    tadah::models::M_KRR_Train<K>::train2(basis.b, basis.T);
  }

  /** Every prediction from descriptors (also through predict(c, stdb, dc)
   *  and predict(c, st, dc)) comes here: the basis vectors are checked
   *  against the length of these descriptors first. */
  Structure predict(const tadah::core::Context &c, StDescriptors &std, const Structure &st) {
    require_weights_fit(c, std.dim());
    check_basis_length(std.dim());
    if(context.template get<bool>("NORM") && !std.normalised && kernel.get_label()!="Kern_Linear")
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
    c.remove("ALPHA");
    c.remove("BETA");
    c.remove("DBFILE");
    // FORCE and STRESS stay, as in M_BLR: the potential records what its
    // fit used (tadah predict's -F/-S and pair_style tadah do not read them).

    c.remove("MODEL");
    c.add("MODEL", label);
    c.add("MODEL", kernel.get_label());
    int modelN=context.template get<int>("MODEL",2);
    c.add("MODEL", modelN);

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
  StructureDB predict(tadah::core::Context context_pred, StructureDB &stdb, DC_Base &dc,
                      tadah::core::aed_type &predicted_error) {

    require_weights_fit(context_pred, context_pred.get<size_t>("DSIZE"));
    tadah::models::LinearRegressor::read_sigma(context_pred,Sigma);
    DesignMatrix<K> dm(kernel,context_pred);
    dm.apply_weights = false;  // predict the values, not the weighted rows
    bool wcopy=false;
    dm.build(stdb,norm,dc,wcopy);

    // compute error
    predicted_error = T_MDMT_diag(dm.getPhi(), Sigma);
    double pmean = sqrt(predicted_error.mean());

    // compute energy, forces and stresses
    tadah::core::aed_type Tpred = T_dgemv(dm.getPhi(), weights);

    // Construct StructureDB object with predicted values
    StructureDB stdb_;
    stdb_.structures.resize(stdb.size());
    size_t i=0;
    for (size_t s=0; s<stdb.size(); ++s) {
      stdb_(s).atoms.resize(stdb(s).natoms());

      predicted_error(i) = (sqrt(predicted_error(i))-pmean);
      stdb_(s).energy = Tpred(i++);
      if (context_pred.get<bool>("FORCE")) {
        for (size_t a=0; a<stdb(s).natoms(); ++a) {
          for (size_t k=0; k<3; ++k) {
            predicted_error(i) = (sqrt(predicted_error(i))-pmean);
            stdb_(s).atoms[a].force[k] = Tpred(i++);
          }
        }
      }
      if (context_pred.get<bool>("STRESS")) {
        for (size_t x=0; x<3; ++x) {
          for (size_t y=x; y<3; ++y) {
            predicted_error(i) = (sqrt(predicted_error(i))-pmean);
            stdb_(s).stress(x,y) = Tpred(i++);
            if (x!=y)
              stdb_(s).stress(y,x) = stdb_(s).stress(x,y);
          }
        }
      }
    }
    return stdb_;
  }
  StructureDB predict(StructureDB &stdb) {
    if(!trained) throw std::runtime_error("This object is not trained!\n\
Hint: check different predict() methods.");

    tadah::core::phi_type &Phi = desmat.getPhi();

    // compute energy, forces and stresses
    tadah::core::aed_type Tpred = T_dgemv(Phi, weights);

    // double eweightglob=context.template get<double>("EWEIGHT");
    // double fweightglob=context.template get<double>("FWEIGHT");
    // double sweightglob=context.template get<double>("SWEIGHT");

    // Construct StructureDB object with predicted values
    StructureDB stdb_;
    stdb_.structures.resize(stdb.size());
    size_t s=0;
    size_t i=0;
    while (i<Phi.rows()) {

      stdb_(s).energy = Tpred(i++);
      if (context.template get<bool>("FORCE")) {
        stdb_(s).atoms.resize(stdb(s).natoms());
        for (size_t a=0; a<stdb(s).natoms(); ++a) {
          for (size_t k=0; k<3; ++k) {
            stdb_(s).atoms[a].force[k] = Tpred(i++);
          }
        }
      }
      if (context.template get<bool>("STRESS")) {
        for (size_t x=0; x<3; ++x) {
          for (size_t y=x; y<3; ++y) {
            stdb_(s).stress(x,y) = Tpred(i++);
            if (x!=y)
              stdb_(s).stress(y,x) = stdb_(s).stress(x,y);
          }
        }
      }
      s++;
    }
    return stdb_;
  }

private:
  std::string label="M_KRR";
  Basis<K> basis;
  DesignMatrix<K> desmat;

  // Convert weights from normalised feature space to the equivalent weights
  // for raw (unnormalised) descriptors, as M_BLR does: the constant the
  // normalisation introduces (-sum mean[i] kw[i]) goes into the intercept
  // columns, one per ATOM species (each atom has 1.0 in exactly one of
  // them, so the same correction lands on every species' intercept).
  // Requires BIAS true: there is no other column to absorb it.
  tadah::core::t_type convert_to_nweights(const tadah::core::t_type &weights) const {
    if(kernel.get_label()!="Kern_Linear") {
      throw std::runtime_error("Cannot convert weights to nweights for\n\
non linear kernel\n");
    }
    const size_t nb = tadah::core::n_intercept_columns(context);
    if (nb == 0)
      throw std::runtime_error("NORM true with Kern_Linear requires BIAS true: "
          "the mean correction of the normalised weights goes into the "
          "intercept columns");
    // normalise weights such that when predict is called
    // we can supply it with a non-normalised descriptor
    tadah::core::t_type kw(weights.rows());
    for (size_t b=0; b<nb; ++b) kw(b) = weights(b);
    for (size_t i=nb; i<weights.size(); ++i) {

      if (norm.std_dev[i] > std::numeric_limits<double>::min())
        kw(i) = weights(i) / norm.std_dev[i];
      else
        kw(i) = weights(i);

      for (size_t b=0; b<nb; ++b) kw(b) -= norm.mean[i]*kw(i);

    }
    return kw;
  }
  // The opposite of convert_to_nweights()
  tadah::core::t_type convert_to_weights(const tadah::core::t_type &kw) const {
    if(kernel.get_label()!="Kern_Linear") {
      throw std::runtime_error("Cannot convert nweights to weights for\n\
non linear kernel\n");
    }
    const size_t nb = tadah::core::n_intercept_columns(context);
    if (nb == 0)
      throw std::runtime_error("NORM true with Kern_Linear requires BIAS true: "
          "the mean correction of the normalised weights goes into the "
          "intercept columns");
    // convert normalised weights back to "normal"
    tadah::core::t_type w(kw.rows());
    for (size_t b=0; b<nb; ++b) w(b) = kw(b);
    for (size_t i=nb; i<kw.size(); ++i) {
      if (norm.std_dev[i] > std::numeric_limits<double>::min())
        w(i) = kw(i) * norm.std_dev[i];
      else
        w(i) = kw(i);

      for (size_t b=0; b<nb; ++b) w(b) += kw(i)*norm.mean[i];
    }
    return w;
  }

  template <typename D>
  void train(D &desmat) {
    // TODO see comments in M_BLR
    tadah::core::phi_type Phi = desmat.getPhi();
    tadah::core::t_type T = desmat.getT();
    tadah::models::M_KRR_Train<K>::train(Phi,T);

    if (context.template get<bool>("NORM") &&
      kernel.get_label()=="Kern_Linear") {
      weights = convert_to_nweights(weights);
    }
  }

  // Do we want to confuse user with those and make them public?
  // Either way they must be stated as below to silence clang warning
  using tadah::models::M_KRR_Train<K>::predict;
  using tadah::models::M_KRR_Train<K>::train;
  using tadah::models::M_KRR_Train<K>::trained;
  using tadah::models::M_KRR_Train<K>::weights;
  using tadah::models::M_KRR_Train<K>::Sigma;
  using tadah::models::M_KRR_Train<K>::context;
  using tadah::models::M_KRR_Train<K>::kernel;

  /** A potential's basis vectors (BASIS, read by the kernel) must have the
   *  length of the descriptors they are compared with, @p dsize (DSIZE).
   *  It is checked where the two meet: here, from predict(c, std, st),
   *  and in the design matrix (DM_Kern_Base::phi_layout), so a model may
   *  be built before its DescriptorsCalc. */
  void check_basis_length(size_t dsize) const {
    if (kernel.get_label()=="Kern_Linear") return;
    const auto *kb = dynamic_cast<const tadah::models::Kern_Base*>(&kernel);
    if (kb == nullptr || kb->get_basis().cols() == 0) return;
    const size_t len = kb->get_basis().rows();
    if (len != dsize)
      throw std::runtime_error(
          "M_KRR " + kernel.get_label() + ": BASIS holds SBASIS " +
          std::to_string(kb->get_basis().cols()) + " basis vectors of " +
          std::to_string(len) + " values each, but the descriptor of the "
          "potential has " + std::to_string(dsize) + " components (DSIZE): "
          "BASIS must have SBASIS x " + std::to_string(dsize) + " values.");
  }
};
} // namespace mlip
} // namespace tadah
#endif
