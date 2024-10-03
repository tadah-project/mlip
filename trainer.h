#include <iostream>
#include "../CORE/config/config.h"
#include "../MODELS/dc_selector.h"
#include "descriptors_calc.h"
#include "design_matrix/functions/dm_function_base.h"
#include "models/m_tadah_base.h"
#include "nn_finder.h"

class Trainer {
  public:
    Config config;
    DC_Selector DCS;
    DescriptorsCalc<> dc;
    NNFinder nnf;
    DM_Function_Base *fb;
    M_Tadah_Base *model;

    ~Trainer() {
      if(model)
        delete model;
      if(fb)
        delete fb;
    }
    Trainer (Config &c):
      config(c),
      DCS(config),
      dc(config,*DCS.d2b,*DCS.d3b,*DCS.dmb,
          *DCS.c2b,*DCS.c3b,*DCS.cmb),
      nnf(config)
  {
    config.postprocess();
    config.check_for_training();

    fb = CONFIG::factory<DM_Function_Base,Config&>(
        config.get<std::string>("MODEL",1),config);

    model = CONFIG::factory<M_Tadah_Base,DM_Function_Base&,Config&>
      (config.get<std::string>("MODEL",0),*fb,config);
  }
    Trainer (std::string &c_file):
      config(c_file),
      DCS(config),
      dc(config,*DCS.d2b,*DCS.d3b,*DCS.dmb,
          *DCS.c2b,*DCS.c3b,*DCS.cmb),
      nnf(config)
  {
    config.postprocess();
    config.check_for_training();

    fb = CONFIG::factory<DM_Function_Base,Config&>(
        config.get<std::string>("MODEL",1),config);

    model = CONFIG::factory<M_Tadah_Base,DM_Function_Base&,Config&>
      (config.get<std::string>("MODEL",0),*fb,config);
  }

    void train(StructureDB &stdb) {
        nn.calc(stdb);
        model->train(stdb,dc);
    }

    Config get_param_file() {
      return model->get_param_file();
    }
};
