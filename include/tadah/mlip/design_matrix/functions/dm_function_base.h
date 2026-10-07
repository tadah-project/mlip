#ifndef DM_FUNCTION_H
#define DM_FUNCTION_H

#include <tadah/core/context.h>
#include <tadah/core/registry.h>
#include <tadah/core/core_types.h>
#include <tadah/models/functions/function_base.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/design_matrix/column_block.h>

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

  virtual size_t get_phi_cols(const tadah::core::Context &)=0;

  /** \brief The column layout of this function's design matrix, given the
   *  layout of the descriptor it is built from (DescriptorsCalc::layout(),
   *  DSIZE columns).
   *
   *  BF_Linear and Kern_Linear (one column per descriptor component)
   *  return the descriptor layout unchanged. A non-linear function owns
   *  one block over all of its get_phi_cols() columns: a kernel one
   *  "kernel" block of its basis vectors (it throws when it has no basis
   *  vectors, or when their length is not the descriptor's), and
   *  BF_Polynomial2 one "BF" block of D(D+1)/2 columns. Which columns are
   *  intercepts the function states with intercept_columns() (MODELS).
   *
   *  Pure, so that a new function cannot inherit a layout that is not
   *  its own. */
  virtual ColumnLayout phi_layout(const ColumnLayout &descriptor_layout,
                                  const tadah::core::Context &) const=0;
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
