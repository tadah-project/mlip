#ifndef TADAH_MLIP_TRAINER_H
#define TADAH_MLIP_TRAINER_H
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/intercepts.h>
#include <tadah/mlip/design_matrix/design_matrix.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/core/context.h>
#include <tadah/models/dc_selector.h>
#include <tadah/mlip/memory/imlip_workspace_manager.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>

#include <iostream>
namespace tadah {
namespace mlip {

class Trainer {
  public:
    /** \brief The configuration @p c without what a fit writes: WEIGHTS,
     *  BASIS (the basis vectors of a non-linear kernel), NMEAN, NSTDEV and
     *  the weight covariance (WCOV, WNOISE, and the SIGMA of a potential
     *  of an earlier version of Tadah!). A training configuration
     *  may be a potential (`tadah train -c pot.tadah`, a `tadah hpo -c`
     *  seed); the fit replaces all of these, so a model built for training
     *  reads none of them. */
    static tadah::core::Context training_context(const tadah::core::Context &c) {
      tadah::core::Context t = c;
      for (const char *key : {"WEIGHTS", "BASIS", "NMEAN", "NSTDEV", "SIGMA",
                              "WCOV", "WNOISE"})
        if (t.has_stored(key)) t.remove(key);
      return t;
    }

    tadah::core::Context context;
    tadah::models::DC_Selector DCS; // const
    DescriptorsCalc<> dc; // modifies Context at init only
    NNFinder nnf; // const
    DM_Function_Base *fb = nullptr;
    M_Tadah_Base *model = nullptr;
    tadah::mlip::memory::IMLIPWorkspaceManager* workspaceManager_ = nullptr;

    ~Trainer() {
      if(model)
        delete model;
      if(fb)
        delete fb;
    }
    Trainer (tadah::core::Context &c, tadah::mlip::memory::IMLIPWorkspaceManager& workspaceManager):
      context(training_context(c)), // a copy, without the fit's arrays
      DCS(context), // const, initialized with context
      dc(context,*DCS.d2b,*DCS.dmb,
          *DCS.c2b,*DCS.cmb), // modifies context at init only
      fb(tadah::core::factory<DM_Function_Base,const tadah::core::Context&>(
            context.get<std::string>("MODEL",1),context)),
      model(tadah::core::factory<M_Tadah_Base,DM_Function_Base&,tadah::core::Context&,tadah::mlip::memory::IMLIPWorkspaceManager&>
          (context.get<std::string>("MODEL",0),*fb,context,workspaceManager)),  // MODIFIES CONTEXT ON TRAINING
      workspaceManager_(&workspaceManager)
  {
    // The descriptor of training (dc, DSIZE) is the descriptor of
    // pair_style tadah (DC_Selector): the trained weights fit both.
    require_descriptor_layout(DCS, context, "Trainer");
  }

    void train(StructureDB &stdb) {
      nnf.calc(stdb, context);
      model->train(stdb,dc);
    }

    void train(const StructureDB &stdb) {
      model->train(stdb,dc);
    }

    /** Set the model weights directly from FIXINDEX/FIXWEIGHT without any
     *  training data. Requires FIXINDEX to cover every model weight; see
     *  weight_provisioning.h for the contract. No fit computes a covariance
     *  of these weights: the model has none (an earlier fit's is removed),
     *  so the potential has no WCOV or WNOISE. */
    void provision_fixed_weights() {
      tadah::mlip::provision_fixed_weights(*model, context,
                                           fb->get_phi_cols(context));
    }

    tadah::core::Context get_param_file() {
      return model->get_param_file();
    }

};
} // namespace mlip
} // namespace tadah
#endif  // TADAH_MLIP_TRAINER_H
