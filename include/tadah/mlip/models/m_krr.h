#ifndef M_KRR_H
#define M_KRR_H

#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/design_matrix.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/models/m_krr_train.h>
#include <tadah/core/config.h>
#include <tadah/mlip/models/basis.h>
#include <tadah/mlip/memory/IMLIPWorkspaceManager.h>

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <iostream>
namespace tadah {
namespace mlip {

/**
 * @class M_KRR
 * @brief Kernel Ridge Regression (KRR) with Empirical Kernel Map (EKM).
 *
 * This class performs Kernel Ridge Regression using an Empirical Kernel Map 
 * to efficiently handle high-dimensional data. EKM is used to map objects into a kernel feature space, where linear methods can be applied.
 *
 * **Empirical Kernel Map (EKM)**:
 * - Maps sample objects into finite-dimensional vectors in the kernel feature space.
 * - Requires a kernel and basis samples to project new samples into the space defined by these bases.
 * - Facilitates the kernelization of algorithms traditionally operating on vectors.
 * - Supports basis sample selection through methods like random sampling or finding linearly independent subsets.
 *
 * **Usage**:
 * - **Kernels**: Use kernel functions (e.g., linear, RBF) to transform data for effective modeling in higher-dimensional spaces.
 *
 * **Configuration Options**:
 * - **LAMBDA**: Controls regularization. Use `0` for ordinary least squares, positive values for manual setting, or `-1` for automatic tuning.
 * - **SBASIS**: Defines the number of basis functions for nonlinear kernels.
 *
 *
 * @tparam K DM_Kern_Base child, Kernel function
 */
template
<class K=DM_Function_Base&>
class M_KRR: public M_Tadah_Base,
  public M_KRR_Train<K>
{

public:

  /** 
     * @brief Initializes for training or prediction using a configuration.
     * 
   * **Example**:
   * \code{.cpp}
   * tadah::core::Config config("tadah::core::Config");
   * M_KRR<Kern_Linear> krr(config);
   * \endcode
   *
     * @param c Configuration object.
     */
  M_KRR(tadah::core::Config &c):
    M_KRR_Train<K>(c),
    basis(c),
    desmat(kernel,c)
  {
    norm = Normaliser(c);
  }

  /** 
   * @brief Initializes for training or prediction using a kernel and configuration.
   * 
   * @param kernel Kernel function.
   * @param c Configuration object.
   */
  M_KRR(K &kernel, tadah::core::Config &c):
    M_KRR_Train<K>(kernel,c),
    basis(c),
    desmat(kernel,c)
  {
    norm = Normaliser(c);
  }

  M_KRR(K &kernel, tadah::core::Config &c, tadah::mlip::memory::IMLIPWorkspaceManager& workspaceManager):
    M_KRR_Train<K>(kernel,c,workspaceManager),
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

    if(config.template get<bool>("NORM"))
      norm = Normaliser(config,st_desc_db);

    bool wcopy=false;
    desmat.build(st_desc_db,stdb,wcopy);
    train(desmat);
  }

  void train(StructureDB &stdb, DC_Base &dc) {
    int modelN;
    try {
      modelN=config.template get<int>("MODEL",2);
    }
    catch(std::runtime_error &e) {
      // use default model
      modelN=1;
      config.add("MODEL", modelN);
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
  void train1(StructureDB &stdb, DC_Base &dc) {
    // KRR implementation using EKM
    if(config.template get<bool>("NORM") || kernel.get_label()!="Kern_Linear") {
      // either build basis or prep normaliser
      std::string force=config.template get<std::string>("FORCE");
      std::string stress=config.template get<std::string>("STRESS");

      config.remove("FORCE");
      config.remove("STRESS");
      config.add("FORCE", "false");
      config.add("STRESS", "false");

      StDescriptorsDB st_desc_db_temp = dc.calc(stdb);

      if(config.template get<bool>("NORM")) {
        norm = Normaliser(config);
        norm.learn(st_desc_db_temp);
        norm.normalise(st_desc_db_temp);
      }

      config.remove("FORCE");
      config.remove("STRESS");
      config.add("FORCE", force);
      config.add("STRESS", stress);

      if (kernel.get_label()!="Kern_Linear") {
        basis.build_random_basis(config.template get<size_t>("SBASIS"),st_desc_db_temp);
        desmat.f.set_basis(basis.b);
        kernel.set_basis(basis.b);
        // to configure ekm, we need basis vectors
        ekm.configure(basis.b);
      }
    }
    bool wcopy=false;
    desmat.build(stdb,norm,dc,wcopy);
    train(desmat);
  }

  void train2(StructureDB &stdb, DC_Base &dc) {

    // NEW (BASIC) IMPLEMENTATION OF KRR //
    std::string force=config.template get<std::string>("FORCE");
    std::string stress=config.template get<std::string>("STRESS");
    config.remove("FORCE");
    config.remove("STRESS");
    config.add("FORCE", "false");
    config.add("STRESS", "false");
    StDescriptorsDB st_desc_db_temp = dc.calc(stdb);
    if(config.template get<bool>("NORM")) {
      norm = Normaliser(config);
      norm.learn(st_desc_db_temp);
      norm.normalise(st_desc_db_temp);
    }
    config.remove("FORCE");
    config.remove("STRESS");
    config.add("FORCE", force);
    config.add("STRESS", stress);
    basis.prep_basis_for_krr(st_desc_db_temp,stdb);
    kernel.set_basis(basis.b);
    M_KRR_Train<K>::train2(basis.b, basis.T);
  }

  Structure predict(const tadah::core::Config &c, StDescriptors &std, const Structure &st) {
    if(config.template get<bool>("NORM") && !std.normalised && kernel.get_label()!="Kern_Linear")
      norm.normalise(std);
    return M_Tadah_Base::predict(c,std,st);
  }

  StructureDB predict(tadah::core::Config &c, const StructureDB &stdb, DC_Base &dc) {
    return M_Tadah_Base::predict(c,stdb,dc);
  }

  tadah::core::Config get_param_file() {
    tadah::core::Config c = config;
    c.remove("ALPHA");
    c.remove("BETA");
    c.remove("DBFILE");
    c.remove("FORCE");
    c.remove("STRESS");
    c.remove("VERBOSE");
    c.add("VERBOSE", 0);

    c.remove("MODEL");
    c.add("MODEL", label);
    c.add("MODEL", kernel.get_label());
    int modelN=config.template get<int>("MODEL",2);
    c.add("MODEL", modelN);

    for (size_t i=0;i<weights.size();++i) {
      c.add("WEIGHTS", weights(i));
    }

    if(config.template get<bool>("NORM")) {
      for (size_t i=0;i<norm.mean.size();++i) {
        c.add("NMEAN", norm.mean[i]);
      }
      for (size_t i=0;i<norm.std_dev.size();++i) {
        c.add("NSTDEV", norm.std_dev[i]);
      }
    }
    if (kernel.get_label()!="Kern_Linear") {
      // dump basis to the config file file
      // make sure keys are not accidently assigned
      if (c.exist("SBASIS"))
        c.remove("SBASIS");
      if (c.exist("BASIS"))
        c.remove("BASIS");
      c.add("SBASIS", basis.b.cols());
      for (size_t i=0;i<basis.b.cols();++i) {
        for (size_t j=0;j<basis.b.rows();++j) {
          c.add("BASIS", basis.b(j,i));
        }
      }
    }
    c.clear_internal_keys();
    return c;
  }
  StructureDB predict(tadah::core::Config config_pred, StructureDB &stdb, DC_Base &dc,
                      tadah::core::aed_type &predicted_error) {

    LinearRegressor::read_sigma(config_pred,Sigma);
    DesignMatrix<K> dm(kernel,config_pred);
    // dm.scale=false; // do not scale energy, forces and stresses
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
      stdb_(s) = Structure(stdb(s));

      stdb_(s).energy = Tpred(i++);
      predicted_error(i) = (sqrt(predicted_error(i))-pmean);
      if (config_pred.get<bool>("FORCE")) {
        for (size_t a=0; a<stdb(s).natoms(); ++a) {
          for (size_t k=0; k<3; ++k) {
            stdb_(s).atoms[a].force[k] = Tpred(i++);
            predicted_error(i) = (sqrt(predicted_error(i))-pmean);
          }
        }
      }
      if (config_pred.get<bool>("STRESS")) {
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

    // double eweightglob=config.template get<double>("EWEIGHT");
    // double fweightglob=config.template get<double>("FWEIGHT");
    // double sweightglob=config.template get<double>("SWEIGHT");

    // Construct StructureDB object with predicted values
    StructureDB stdb_;
    stdb_.structures.resize(stdb.size());
    size_t s=0;
    size_t i=0;
    while (i<Phi.rows()) {

      stdb_(s).energy = Tpred(i++);
      if (config.template get<bool>("FORCE")) {
        stdb_(s).atoms.resize(stdb(s).natoms());
        for (size_t a=0; a<stdb(s).natoms(); ++a) {
          for (size_t k=0; k<3; ++k) {
            stdb_(s).atoms[a].force[k] = Tpred(i++);
          }
        }
      }
      if (config.template get<bool>("STRESS")) {
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

  tadah::core::t_type convert_to_nweights(const tadah::core::t_type &weights) const {
    if(kernel.get_label()!="Kern_Linear") {
      throw std::runtime_error("Cannot convert weights to nweights for\n\
non linear kernel\n");
    }
    tadah::core::t_type kw(weights.rows());
    if(config.template get<bool>("NORM") && kernel.get_label()=="Kern_Linear") {
      // normalise weights such that when predict is called
      // we can supply it with a non-normalised descriptor
      kw.resize(weights.rows());
      kw(0) = weights(0);
      for (size_t i=1; i<weights.size(); ++i) {

        if (norm.std_dev[i] > std::numeric_limits<double>::min())
          kw(i) = weights(i) / norm.std_dev[i];
        else
          kw(i) = weights(i);

        kw(0) -= norm.mean[i]*kw(i);

      }
    }
    return kw;
  }
  // The opposite of convert_to_nweights()
  tadah::core::t_type convert_to_weights(const tadah::core::t_type &kw) const {
    if(kernel.get_label()!="Kern_Linear") {
      throw std::runtime_error("Cannot convert nweights to weights for\n\
non linear kernel\n");
    }
    // convert normalised weights back to "normal"
    tadah::core::t_type w(kw.rows());
    w(0) = kw(0);
    for (size_t i=1; i<kw.size(); ++i) {
      if (norm.std_dev[i] > std::numeric_limits<double>::min())
        w(i) = kw(i) * norm.std_dev[i];
      else
        w(i) = kw(i);

      w(0) += kw(i)*norm.mean[i];
    }
    return w;
  }

  template <typename D>
  void train(D &desmat) {
    // TODO see comments in M_BLR
    tadah::core::phi_type Phi = desmat.getPhi();
    tadah::core::t_type T = desmat.getT();
    M_KRR_Train<K>::train(Phi,T);

    if (config.template get<bool>("NORM") &&
      kernel.get_label()=="Kern_Linear") {
      weights = convert_to_nweights(weights);
    }
  }

  // Do we want to confuse user with those and make them public?
  // Either way they must be stated as below to silence clang warning
  using M_KRR_Train<K>::predict;
  using M_KRR_Train<K>::train;
  using M_KRR_Train<K>::trained;
  using M_KRR_Train<K>::weights;
  using M_KRR_Train<K>::Sigma;
  using M_KRR_Train<K>::config;
  using M_KRR_Train<K>::kernel;
  using M_KRR_Train<K>::ekm;
};
} // namespace mlip
} // namespace tadah
#endif
