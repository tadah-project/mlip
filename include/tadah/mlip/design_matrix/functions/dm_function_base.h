#ifndef DM_FUNCTION_H
#define DM_FUNCTION_H

#include <tadah/core/config.h>
#include <tadah/core/registry.h>
#include <tadah/core/core_types.h>
#include <tadah/models/functions/function_base.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/st_descriptors.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

/** Base class for Kernels and Basis Functions */
struct DM_Function_Base: public virtual Function_Base {

  // Derived classes must implement Derived() and Derived(Config)
  DM_Function_Base();
  DM_Function_Base(const Config &c);
  virtual ~DM_Function_Base();

  virtual size_t get_phi_cols(const Config &)=0;
  virtual void calc_phi_energy_row(phi_type &, size_t &,
                                   const double , const Structure &, const StDescriptors &)=0;
  virtual void calc_phi_force_rows(phi_type &, size_t &,
                                   const double , const Structure &, const StDescriptors &)=0;
  virtual void calc_phi_stress_rows(phi_type &, size_t &,
                                    const double[6], const Structure &, const StDescriptors &)=0;
};
//template<> inline CONFIG::Registry<DM_Function_Base>::Map CONFIG::Registry<DM_Function_Base>::registry{};
//template<> inline CONFIG::Registry<DM_Function_Base,Config&>::Map CONFIG::Registry<DM_Function_Base,Config&>::registry{};
#endif
