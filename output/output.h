#include <ios>
#include <iostream>
#include <iomanip>
#include <string>

#include "../../CORE/config/config.h"
#include "../structure_db.h"
#include "../../CORE/utils/utils.h"
#include "../../CORE/core_types.h"

class Output {
    private:
        Config &pot_config;
        bool pred_err_bool;
        //size_t sigfig;
        //size_t nw;
        size_t sigfig=5;  // set number of sig fig;
        size_t fmtw=sigfig+7;  // width of e.g. -1.12345e+12 -> sigfig+7
        size_t emptyspace=5;
        size_t nw=emptyspace+fmtw;   // width of numeric column
    public:
        Output (Config &c, bool peb):
            pot_config(c),
            pred_err_bool(peb)
    {}
        void print_train_unc(t_type &weights, t_type &unc) {
            std::string index_str="Index";

            size_t iwe=(size_t)unc.size() > max_number(index_str.size()) ?
                number_of_digits(unc.size()) : index_str.size();
            std::ofstream out_unc("weights.tadah");
            out_unc << std::scientific << std::setprecision(sigfig);
            out_unc << std::setw(iwe) << index_str
                << std::setw(nw) << "Mean weight"
                << std::setw(nw) << "Uncertainty" << std::endl;

            for(size_t i=0; i<unc.size(); ++i) {
                out_unc << std::setw(iwe) << i
                    << std::setw(nw) << weights(i)
                    << std::setw(nw) << unc(i) << std::endl;
            }

            out_unc.close();
        }
        void print_predict_all(StructureDB &stdb, StructureDB &stpred, aed_type2 & predicted_error) {
            std::ofstream out_error("error.pred");
            std::ofstream out_energy("energy.pred");
            std::ofstream out_force("forces.pred");
            std::ofstream out_stress("stress.pred");

            // Common Headers
            std::string index_str="Index";
            std::string perror_str="Pred Error";

            // energy.pred
            std::string tenergy_str="True Energy";
            std::string penergy_str="Pred Energy";

            // forces.pred
            std::string tforce_str="True Force";
            std::string pforce_str="Pred Force";

            // stress.pred
            std::string tstress_str="True Stress";
            std::string pstress_str="Pred Stress";

            // error.pred
            std::string err_e_str="Energy";
            std::string err_f_str="Force (max)";
            std::string err_s_str="Stress (max)";

            // Grow width of the index column if necessary
            // width of energy index column
            size_t iwe=stdb.size() > max_number(index_str.size()) ?
                number_of_digits(stdb.size()) : index_str.size();
            // width of force index column, 3 for xyz
            size_t iwf=stdb.calc_natoms()*3 > max_number(index_str.size()) ?
                number_of_digits(stdb.calc_natoms()*3) : index_str.size();
            // width of energy index column, 6 components of stress tensor
            size_t iws=stdb.size()*6 > max_number(index_str.size()) ?
                number_of_digits(stdb.size()*6) : index_str.size();


            out_error << std::scientific << std::setprecision(sigfig);
            out_energy << std::scientific << std::setprecision(sigfig);
            out_force << std::scientific << std::setprecision(sigfig);
            out_stress << std::scientific << std::setprecision(sigfig);


            out_energy << std::setw(iwe) << index_str
                << std::setw(nw) << tenergy_str
                << std::setw(nw) << penergy_str;
            out_force << std::setw(iwf) << index_str
                << std::setw(nw) << tforce_str
                << std::setw(nw) << pforce_str;
            out_stress << std::setw(iws) << index_str
                << std::setw(nw) << tstress_str
                << std::setw(nw) << pstress_str;
            if (pred_err_bool) {
                out_energy << std::setw(nw) << perror_str;
                out_force << std::setw(nw) << perror_str;
                out_stress << std::setw(nw) << perror_str;
                out_error << std::setw(iws) << index_str
                    << std::setw(nw) << err_e_str
                    << std::setw(nw) << err_f_str
                    << std::setw(nw) << err_s_str
                    << std::endl;
            }
            out_energy << std::endl;
            out_force << std::endl;
            out_stress << std::endl;
            size_t j=0;
            size_t fidx=0, sidx=0;
            bool newline=false;
            auto it_dbidx = stdb.dbidx.begin();
            it_dbidx++;
            for (size_t i=0; i<stdb.size(); ++i) {

                if (stdb.dbidx.size()>1 && i==*it_dbidx) {
                    newline=true;
                    it_dbidx++;
                }

                if (newline)
                    out_energy << std::endl << std::endl;

                out_energy << std::setw(iwe) << i << std::setw(nw)
                    << stdb(i).energy/stdb(i).natoms()
                    << std::setw(nw) << stpred(i).energy/stdb(i).natoms();
                if(pred_err_bool) {
                    out_error << std::setw(iwe) << i <<
                        std::setw(nw)  << predicted_error(j)/stdb(i).natoms();
                    out_energy << std::setw(nw) << predicted_error(j++)/stdb(i).natoms();
                }
                out_energy << std::endl;


                if (pot_config.get<bool>("FORCE")) {
                    double max_ferr=0.0;
                    if (newline)
                        out_force << std::endl << std::endl;
                    for (size_t a=0; a<stdb(i).natoms(); ++a) {
                        Vec3d &f = stdb(i).atoms[a].force;
                        Vec3d &fp = stpred(i).atoms[a].force;
                        for (size_t k=0; k<3; ++k) {
                            out_force << std::setw(iwf) << fidx++ << std::setw(nw)
                                << f[k] << std::setw(nw) << fp[k];
                            if(pred_err_bool) {
                                if (max_ferr<predicted_error(j))
                                    max_ferr = predicted_error(j);
                                out_force <<  std::setw(nw) << predicted_error(j++);
                            }
                            out_force << std::endl;
                        }
                    }
                    if(pred_err_bool)
                        out_error << std::setw(nw) << max_ferr;
                }
                if (pot_config.get<bool>("STRESS")) {
                    double max_serr=0.0;
                    if (newline)
                        out_stress << std::endl << std::endl;
                    stress_type &s = stdb(i).stress;
                    stress_type &sp = stpred(i).stress;
                    for (size_t x=0; x<3; ++x) {
                        for (size_t y=x; y<3; ++y) {
                            out_stress << std::setw(iws) << sidx++ << std::setw(nw) << s(x,y)
                                << std::setw(nw) << sp(x,y);
                            if(pred_err_bool) {
                                if (max_serr<predicted_error(j))
                                    max_serr = predicted_error(j);
                                out_stress << std::setw(nw) << predicted_error(j++);
                            }
                            out_stress << std::endl;
                        }
                    }
                    if(pred_err_bool)
                        out_error << std::setw(nw) << max_serr;
                }

                newline=false;
                if(pred_err_bool)
                    out_error << std::endl;
            }
            out_error.close();
            out_energy.close();
            out_force.close();
            out_stress.close();
        }
};
