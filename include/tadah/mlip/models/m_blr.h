#ifndef M_BLR_H
#define M_BLR_H

#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/design_matrix.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/models/m_blr_train.h>
#include <tadah/core/context.h>
#include <tadah/mlip/memory/IMLIPWorkspaceManager.h>

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <iostream>
namespace tadah {
namespace mlip {

/**
 * @class M_BLR
 * @brief Bayesian Linear Regression (BLR).
 *
 * This class implements Bayesian Linear Regression, a statistical method to make predictions using linear models with both linear and nonlinear features.
 *
 * **Model Supported Training Modes**:
 * - **LINEAR**: Uses Ordinary Least Squares or Ridge Regression for linear relationships.
 * - **NONLINEAR**: Utilizes basis functions to handle nonlinear input spaces, transforming input descriptors into higher-dimensional feature spaces. For example, polynomial transformations.
 *
 * **Prediction**:
 * - Computes predictions as a weighted sum of basis functions applied to input vectors.
 *
 * **Training**:
 * - Employs regularized least squares, allowing for optional regularization through the \f$\lambda\f$ parameter.
 * - Ordinary Least Squares (OLS) is a special case when \f$\lambda = 0\f$.
 *
 * **Contexturation Options**:
 * - **LAMBDA**: Set to `0` for OLS, a positive value for specified regularization, or `-1` for automatic tuning using evidence approximation.
 *
 * @tparam BF DM_BF_Base child, Basis function
 */
template
<class BF=DM_Function_Base&>
class M_BLR: public M_Tadah_Base, public tadah::models::M_BLR_Train<BF> {

public:

  using tadah::models::M_BLR_Train<BF>::context;
  using tadah::models::M_BLR_Train<BF>::bf;

  /** 
   * @brief Initializes for training or prediction using a contexturation.
   *
   * **Example**:
   * \code{.cpp}
   * tadah::core::Context context("tadah::core::Context");
   * M_BLR<tadah::models::BF_Linear> blr(context);
   * \endcode
   * 
   * @param c Contexturation object.
   */
  M_BLR(tadah::core::Context &c):
    tadah::models::M_BLR_Train<BF>(c),
    desmat(tadah::models::M_BLR_Train<BF>::bf,c)
  {
    norm = Normaliser(c);
  }

  /** 
   * @brief Initializes for training or prediction using a basis function and contexturation.
   * 
   * @param bf Basis function.
   * @param c Contexturation object.
   */
  M_BLR(BF &bf, tadah::core::Context &c):
    tadah::models::M_BLR_Train<BF>(bf,c),
    desmat(bf,c)
  {
    norm = Normaliser(c);
  }

  M_BLR(BF &bf, tadah::core::Context &c, tadah::mlip::memory::IMLIPWorkspaceManager& workspaceManager):
    tadah::models::M_BLR_Train<BF>(bf,c,workspaceManager),
    desmat(bf,c,workspaceManager)
  {
    norm = Normaliser(c);
  }

  double epredict(const tadah::core::aed_type &aed) const{
    return bf.epredict(weights,aed);
  };

  double fpredict(const tadah::core::fd_type &fdij, const tadah::core::aed_type &aedi, const size_t k) const{
    return bf.fpredict(weights,fdij,aedi,k);
  }

  tadah::core::force_type fpredict(const tadah::core::fd_type &fdij, const tadah::core::aed_type &aedi) const{
    return bf.fpredict(weights,fdij,aedi);
  }

  void train(StDescriptorsDB &st_desc_db, const StructureDB &stdb) {

    if(context.template get<bool>("NORM"))
      norm = Normaliser(context,st_desc_db);

    bool wcopy=false;
    desmat.build(st_desc_db,stdb, wcopy);
    train(desmat);
  }

  void train(StructureDB &stdb, DC_Base &dc) {

    if(context.template get<bool>("NORM")) {

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
        // norm.normalise(st_desc_db_temp);
      }

      context.remove("FORCE");
      context.remove("STRESS");
      context.add("FORCE", force);
      context.add("STRESS", stress);
    }

    bool wcopy=false;
    desmat.build(stdb,norm,dc,wcopy);
    train(desmat);
  }

  Structure predict(const tadah::core::Context &c, StDescriptors &std, const Structure &st) {
    if(context.template get<bool>("NORM") && !std.normalised && bf.get_label()!="BF_Linear")
      norm.normalise(std);
    return M_Tadah_Base::predict(c,std,st);
  }

  StructureDB predict(tadah::core::Context &c, const StructureDB &stdb, DC_Base &dc) {
    return M_Tadah_Base::predict(c,stdb,dc);
  }

  void set_context(const tadah::core::Context &c) {
    context = c;
    // desmat.set_context(c);
    // norm.set_context(c);
  }

  tadah::core::Context get_param_file() {
    tadah::core::Context c = context;
    //c.remove("ALPHA");
    //c.remove("BETA");
    c.remove("DBFILE");
    c.remove("FORCE");
    c.remove("STRESS");

    c.remove("MODEL");
    c.add("MODEL", label);
    c.add("MODEL", bf.get_label());

    if (!c.contains("WEIGHTS")) {
      for (size_t i=0;i<weights.size();++i) {
        c.add("WEIGHTS", weights(i));
      }
    }

    if(context.template get<bool>("NORM")) {
      for (size_t i=0;i<norm.mean.size();++i) {
        c.add("NMEAN", norm.mean[i]);
      }
      for (size_t i=0;i<norm.std_dev.size();++i) {
        c.add("NSTDEV", norm.std_dev[i]);
      }
    }
    return c;
  }
  StructureDB predict(tadah::core::Context context_pred, StructureDB &stdb, DC_Base &dc,
                      tadah::core::aed_type &predicted_error) {

    tadah::models::LinearRegressor::read_sigma(context_pred,Sigma);
    DesignMatrix<BF> dm(bf,context_pred);
    // dm.scale=false; // do not scale energy, forces and stresses
    bool wcopy=false;
    dm.build(stdb,norm,dc,wcopy);

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
  std::string label="M_BLR";
  DesignMatrix<BF> desmat;

  // normalise weights such that when predict is called
  // we can supply it with a non-normalised descriptor
  tadah::core::t_type convert_to_nweights(const tadah::core::t_type &weights) const {
    if(bf.get_label()!="BF_Linear") {
      throw std::runtime_error("Cannot convert weights to nweights for\n\
non linear basis function\n");
    }
    tadah::core::t_type nw(weights.rows());
    nw(0) = weights(0);
    for (size_t i=1; i<weights.size(); ++i) {

      if (norm.std_dev[i] > std::numeric_limits<double>::min())
        nw(i) = weights(i) / norm.std_dev[i];
      else
        nw(i) = weights(i);

      nw(0) -= norm.mean[i]*nw(i);

    }
    return nw;
  }
  // The opposite of convert_to_nweights()
  tadah::core::t_type convert_to_weights(const tadah::core::t_type &nw) const {
    if(bf.get_label()!="BF_Linear") {
      throw std::runtime_error("Cannot convert nweights to weights for\n\
non linear basis function\n");
    }
    // convert normalised weights back to "normal"
    tadah::core::t_type w(nw.rows());
    w(0) = nw(0);
    for (size_t i=1; i<nw.size(); ++i) {
      if (norm.std_dev[i] > std::numeric_limits<double>::min())
        w(i) = nw(i) * norm.std_dev[i];
      else
        w(i) = nw(i);

      w(0) += nw(i)*norm.mean[i];
    }
    return w;
  }

  template <typename D>
  void train(D &desmat) {
    // TODO 
    // the train method destroys the Phi matrix
    // In consequence, we cannot use it for quick prediction
    // The simple solution, for now, is to make a copy of the Phi matrix
    //tadah::core::phi_type &Phi = desmat.Phi;
    tadah::core::phi_type Phi = desmat.getPhi();
    tadah::core::t_type T = desmat.getT();
    //tadah::core::t_type &T = desmat.T;
    tadah::models::M_BLR_Train<BF>::train(Phi,T);

    if (context.template get<bool>("NORM") &&
      bf.get_label()=="BF_Linear") {
      weights = convert_to_nweights(weights);
    }
  }

  // Do we want to confuse user with those and make them public?
  // Either way they must be stated as below to silence clang warning
  using tadah::models::M_BLR_Train<BF>::predict;
  using tadah::models::M_BLR_Train<BF>::train;
  using tadah::models::M_BLR_Train<BF>::trained;
  using tadah::models::M_BLR_Train<BF>::weights;
  using tadah::models::M_BLR_Train<BF>::Sigma;
};
} // namespace mlip
} // namespace tadah
#endif
