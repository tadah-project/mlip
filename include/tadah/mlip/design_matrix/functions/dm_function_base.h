#ifndef DM_FUNCTION_H
#define DM_FUNCTION_H

#include <tadah/core/context.h>
#include <tadah/core/registry.h>
#include <tadah/core/core_types.h>
#include <tadah/models/functions/function_base.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/st_descriptors.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
namespace tadah {
namespace mlip {

/** Base class for Kernels and Basis Functions */
struct DM_Function_Base: public virtual tadah::models::Function_Base {

  // Derived classes must implement Derived() and Derived(tadah::core::Context)
  DM_Function_Base();
  DM_Function_Base(const tadah::core::Context &c);
  virtual ~DM_Function_Base();

  /** The design-matrix width: weights_count(DSIZE) of the basis function
   *  or kernel (models::Function_Base::weights_count). */
  virtual size_t get_phi_cols(const tadah::core::Context &)=0;
  virtual void calc_phi_energy_row(tadah::core::phi_type &, size_t &,
                                   const Structure &, const StDescriptors &)=0;
  virtual void calc_phi_force_rows(tadah::core::phi_type &, size_t &,
                                   const Structure &, const StDescriptors &)=0;
  virtual void calc_phi_stress_rows(tadah::core::phi_type &, size_t &,
                                   const Structure &, const StDescriptors &)=0;
};
//template<> inline context::Registry<DM_Function_Base>::Map context::Registry<DM_Function_Base>::registry{};
//template<> inline context::Registry<DM_Function_Base,tadah::core::Context&>::Map context::Registry<DM_Function_Base,tadah::core::Context&>::registry{};
} // namespace mlip
} // namespace tadah
#endif
