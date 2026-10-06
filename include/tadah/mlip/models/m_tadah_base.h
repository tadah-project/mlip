#ifndef M_TADAH_BASE_H
#define M_TADAH_BASE_H

#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/st_descriptors_db.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/descriptors_calc_base.h>
#include <tadah/core/core_types.h>
#include <tadah/models/m_core.h>
#include <tadah/models/m_predict.h>
namespace tadah {
namespace mlip {

/** This interface provides functionality required from all models.
 */
class M_Tadah_Base:
    public virtual tadah::models::M_Core,
    public virtual tadah::models::M_Predict
{

public:

  Normaliser norm;    // TODO?
  virtual ~M_Tadah_Base() {};

  /** \brief Predict total energy for a set of atoms. */
  using tadah::models::M_Predict::epredict;
  using tadah::models::M_Predict::fpredict;
  using tadah::models::M_Core::predict;
  double epredict(const StDescriptors &std);

  ///** \brief Predict force between a pair of atoms in a k-direction. */
  //virtual double fpredict(const tadah::core::fd_type &fdij, const tadah::core::aed_type &aedi, size_t k)=0;

  ///** \brief Predict force between a pair of atoms. */
  //virtual tadah::core::force_type fpredict(const tadah::core::fd_type &fdij,
  //        const tadah::core::aed_type &aedi)=0;

  /** \brief Predict total force on an atom a. */
  virtual void fpredict(const size_t a, tadah::core::force_type &v,
                        const StDescriptors &std, const Structure &stOrig);

  /** \brief Predict total force on an atom a. */
  virtual tadah::core::force_type fpredict(const size_t a, const StDescriptors &std,
                              const Structure &stOrig);

  /** \brief Predict energy, forces and stresses for the Structure stOrig. */
  virtual Structure predict(const tadah::core::Context &c,
                            /*not const*/ StDescriptors &std, const Structure &stOrig);

  /** \brief Predict energy, forces and stresses for a set of Structures.
         *
         * Use precalculated descriptors.
         */
  virtual StructureDB predict(const tadah::core::Context &c,
                              /*not const*/ StDescriptorsDB &st_desc_db,
                              const StructureDB &stdb);

  /** \brief Predict energy, forces and stresses for a set of Structures. */
  virtual StructureDB predict(tadah::core::Context &c, const StructureDB &stdb, DC_Base &dc);

  /** \brief Predict energy, forces and stresses for a set of Structures. */
  virtual Structure predict(tadah::core::Context &c, const Structure &st, DC_Base &dc);

  /** \brief Predict the stress (Cauchy, eV/A^3, positive under tension) of a Structure stOrig. */
  virtual tadah::core::stress_type spredict(const StDescriptors &std, const Structure &stOrig);

  /** \brief Predict virial stresses in all Structures. */
  virtual tadah::core::stress_type spredict(const size_t a, const StDescriptors &std,
                               const Structure &stOrig);

  /** \brief Predict both virial stresses and forces for a Structure */
  virtual void stress_force_predict(const StDescriptors &std, Structure &stPred, const Structure &stOrig);

  /** \brief Predict virial stress for atom a in a Structure stOrig. */
  virtual void spredict(const size_t a, tadah::core::stress_type &s,
                        const StDescriptors &std, const Structure &stOrig);

  /** \brief Compute ∂E/∂W for a Structure.
   *
   *  Linear-model contract (BF_Linear / Kern_Linear): the total energy is
   *  E = W · Σ_a aed_a, so ∂E/∂W = Σ_a aed_a (size = descriptor dim).
   *
   *  Nonlinear model classes (Kern_RBF, KAN, ...) must override.
   *
   *  Used by the typed PC_OBSERVABLE path to compose closed-form dO/dW
   *  contributions without finite differences in W-space.
   *
   *  @param std Precomputed descriptors for stOrig.
   *  @param stOrig The Structure.
   *  @param out Output vector; resized to descriptor dim, zero-initialised.
   */
  virtual void predict_dE_dW(const StDescriptors &std,
                             const Structure &stOrig,
                             tadah::core::t_type &out);

  /** \brief Compute the VJP ∂(λᵀ·F)/∂W for a Structure.
   *
   *  Linear-model contract (BF_Linear / Kern_Linear): walks every
   *  (atom, neighbour) pair the way fpredict does and contracts
   *      out[c] += Σ_a Σ_k λ_a[k] · (−fd_ij(c,k) + fd_ji(c,k)).
   *
   *  Nonlinear model classes must override with their own reverse-mode pass.
   *
   *  Used by the IFT-adjoint pipeline once an observable's λ has been
   *  obtained from the adjoint solve. The mixed partial ∂²E/∂x∂W is never
   *  materialised — only its contraction with λ.
   *
   *  @param std Precomputed descriptors for stOrig.
   *  @param stOrig The Structure.
   *  @param lambda Force-shaped multiplier, one Vec3d per atom.
   *  @param out Output vector; resized to descriptor dim, zero-initialised.
   */
  virtual void vjp_force(const StDescriptors &std,
                         const Structure &stOrig,
                         const std::vector<tadah::core::force_type> &lambda,
                         tadah::core::t_type &out);


  /** Return potential file. */
  virtual tadah::core::Context get_param_file()=0;

  // TRAINING

  /** This will fit a model without precalculated StDescriptorsDB object.
         *
         * Structure stdb object must have all nearest neighbours calculated
         * with NN_Finder.
         *
         * @param dc is a DescriptorCalc object
         */
  virtual void train(const StructureDB &, DC_Base &) {};


  /** This will fit a model with precalculated StDescriptorsDB object.
         *
         * Structure stdb object must have all nearest neighbours calculated
         * with NN_Finder.
         */
  virtual void train(StDescriptorsDB &, const StructureDB &) {};

  virtual StructureDB predict(tadah::core::Context context_pred, StructureDB &stdb, DC_Base &dc,
                              tadah::core::aed_type &predicted_error)=0;

  virtual StructureDB predict(StructureDB &stdb)=0;

};
//template<> inline Registry<M_Tadah_Base,DM_Function_Base&,tadah::core::Context&>::Map Registry<M_Tadah_Base,DM_Function_Base&,tadah::core::Context&>::registry{};
} // namespace mlip
} // namespace tadah
#endif
