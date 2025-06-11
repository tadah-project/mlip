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

  /** \brief Predict virial stresses in a Structure stOrig. */
  virtual tadah::core::stress_type spredict(const StDescriptors &std, const Structure &stOrig);

  /** \brief Predict virial stresses in all Structures. */
  virtual tadah::core::stress_type spredict(const size_t a, const StDescriptors &std,
                               const Structure &stOrig);

  /** \brief Predict both virial stresses and forces for a Structure */
  virtual void stress_force_predict(const StDescriptors &std, Structure &stPred, const Structure &stOrig);

  /** \brief Predict virial stress for atom a in a Structure stOrig. */
  virtual void spredict(const size_t a, tadah::core::stress_type &s,
                        const StDescriptors &std, const Structure &stOrig);


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
