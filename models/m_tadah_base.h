#ifndef M_TADAH_BASE_H
#define M_TADAH_BASE_H

#include "../structure.h"
#include "../structure_db.h"
#include "../st_descriptors.h"
#include "../st_descriptors_db.h"
#include "../normaliser.h"
#include "../descriptors_calc_base.h"
#include "../../CORE/core_types.h"
#include "../../MODELS/m_core.h"
#include "../../MODELS/m_predict.h"

/** This interface provides functionality required from all models.
 */
class M_Tadah_Base:
    public virtual M_Core,
    public virtual M_Predict
{

public:

  Normaliser norm;    // TODO?
  virtual ~M_Tadah_Base() {};

  /** \brief Predict total energy for a set of atoms. */
  using M_Predict::epredict;
  using M_Predict::fpredict;
  double epredict(const StDescriptors &std);

  ///** \brief Predict force between a pair of atoms in a k-direction. */
  //virtual double fpredict(const fd_type &fdij, const aed_type2 &aedi, size_t k)=0;

  ///** \brief Predict force between a pair of atoms. */
  //virtual force_type fpredict(const fd_type &fdij,
  //        const aed_type2 &aedi)=0;

  /** \brief Predict total force on an atom a. */
  virtual void fpredict(const size_t a, force_type &v,
                        const StDescriptors &std, const Structure &st);

  /** \brief Predict total force on an atom a. */
  virtual force_type fpredict(const size_t a, const StDescriptors &std,
                              const Structure &st);

  /** \brief Predict energy, forces and stresses for the Structure st. */
  virtual Structure predict(const Config &c,
                            /*not const*/ StDescriptors &std, const Structure &st);

  /** \brief Predict energy, forces and stresses for a set of Structures.
         *
         * Use precalculated descriptors.
         */
  virtual StructureDB predict(const Config &c,
                              /*not const*/ StDescriptorsDB &st_desc_db,
                              const StructureDB &stdb);

  /** \brief Predict energy, forces and stresses for a set of Structures. */
  virtual StructureDB predict(Config &c, const StructureDB &stdb, DC_Base &dc);

  /** \brief Predict virial stresses in a Structure st. */
  virtual stress_type spredict(const StDescriptors &std, const Structure &st);

  /** \brief Predict virial stresses in all Structures. */
  virtual stress_type spredict(const size_t a, const StDescriptors &std,
                               const Structure &st);

  /** \brief Predict both virial stresses and forces for a Structure */
  virtual void stress_force_predict(const StDescriptors &std, Structure &st);

  /** \brief Predict virial stress for atom a in a Structure st. */
  virtual void spredict(const size_t a, stress_type &s,
                        const StDescriptors &std, const Structure &st);


  /** Return potential file. */
  virtual Config get_param_file()=0;

  // TRAINING

  /** This will fit a model without precalculated StDescriptorsDB object.
         *
         * Structure stdb object must have all nearest neighbours calculated
         * with NN_Finder.
         *
         * @param dc is a DescriptorCalc object
         */
  virtual void train(StructureDB &, DC_Base &) {};


  /** This will fit a model with precalculated StDescriptorsDB object.
         *
         * Structure stdb object must have all nearest neighbours calculated
         * with NN_Finder.
         */
  virtual void train(StDescriptorsDB &, const StructureDB &) {};

  virtual StructureDB predict(Config config_pred, StructureDB &stdb, DC_Base &dc,
                              aed_type2 &predicted_error)=0;

  virtual StructureDB predict(StructureDB &stdb)=0;

};
//template<> inline Registry<M_Tadah_Base,DM_Function_Base&,Config&>::Map Registry<M_Tadah_Base,DM_Function_Base&,Config&>::registry{};
#endif
