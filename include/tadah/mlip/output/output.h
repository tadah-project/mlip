#ifndef OUTPUT_H
#define OUTPUT_H

#include <tadah/mlip/structure_db.h>
#include <tadah/core/context.h>
#include <tadah/core/utils.h>
#include <tadah/core/core_types.h>

#include <ios>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>

namespace tadah {
namespace mlip {

class Output {
  private:
    const tadah::core::Context &context;

    size_t sigfig = 5;       // set number of sig fig
    size_t fmtw = sigfig+7;  // width of e.g. -1.12345e+12 -> sigfig+7
    size_t emptyspace = 5;
    size_t nw = emptyspace+fmtw;   // width of numeric column

    std::string energyFile;
    std::string forceFile;
    std::string stressFile;
    std::string errorFile = "error.pred";

    struct OutputFiles {
      std::string energy;
      std::string force;
      std::string stress;
    };

    OutputFiles resolve_output_files() const {
      const bool wantForce  = context.get<bool>("FORCE");
      const bool wantStress = context.get<bool>("STRESS");
      const size_t nout     = context.size("OUTFILE");
      const size_t required = 1 + (wantForce ? 1 : 0) + (wantStress ? 1 : 0);

      // Accept:
      //   - exact size for requested outputs, or
      //   - default size 3 interpreted as [energy, force, stress]
      if (nout != required && nout != 3) {
        std::ostringstream oss;
        oss << "Invalid OUTFILE size " << nout
            << ". Expected either " << required
            << " for the requested outputs or 3 for the default "
            << "[energy, force, stress] layout.";
        throw std::runtime_error(oss.str());
      }

      OutputFiles files;
      files.energy = context.get<std::string>("OUTFILE", 0);

      if (wantForce && wantStress) {
        // [E,F,S]
        files.force  = context.get<std::string>("OUTFILE", 1);
        files.stress = context.get<std::string>("OUTFILE", 2);
      } else if (wantForce) {
        // [E,F] or [E,F,S]
        files.force = context.get<std::string>("OUTFILE", 1);
      } else if (wantStress) {
        // [E,S] or [E,F,S]
        files.stress = context.get<std::string>("OUTFILE", (nout == 2) ? 1 : 2);
      }

      return files;
    }

  public:
    Output(const tadah::core::Context &c) :
      context(c)
    {
      set_numeric(c.get<size_t>("NUMERIC"));
    }

    /** weights.tadah: the index, value and standard deviation of every
     *  weight (@p unc, the square roots of the diagonal of WCOV). */
    void print_train_unc(tadah::core::t_type &weights, tadah::core::t_type &unc) {
      std::string index_str = "Index";

      size_t iwe = (size_t)unc.size() > tadah::core::max_number(index_str.size()) ?
        tadah::core::number_of_digits(unc.size()) : index_str.size();

      std::ofstream out_unc("weights.tadah");
      out_unc << std::scientific << std::setprecision(sigfig);
      out_unc << std::setw(iwe) << index_str
              << std::setw(nw) << "Mean weight"
              << std::setw(nw) << "Std. dev." << std::endl;

      for (size_t i = 0; i < unc.size(); ++i) {
        out_unc << std::setw(iwe) << i
                << std::setw(nw) << weights(i)
                << std::setw(nw) << unc(i) << std::endl;
      }

      out_unc.close();
    }

    void print_predict_all(StructureDB &stdb, StructureDB &stpred,
                           tadah::core::aed_type &predicted_error) {
      const bool wantForce    = context.get<bool>("FORCE");
      const bool wantStress   = context.get<bool>("STRESS");
      const bool pred_err_bool = context.get<bool>("ERROR");

      OutputFiles files = resolve_output_files();
      energyFile = files.energy;
      forceFile  = files.force;
      stressFile = files.stress;

      std::string outfileInfo = "E: " + energyFile;
      if (wantForce) {
        outfileInfo += ", F: " + forceFile;
      }
      if (wantStress) {
        outfileInfo += ", S: " + stressFile;
      }
      core::Logger::instance().log(
        tadah::core::LogLevel::INFO,
        "Output file(s): " + outfileInfo
      );

      std::ofstream out_energy(energyFile);
      if (!out_energy) {
        throw std::runtime_error("Failed to open energy output file: " + energyFile);
      }

      std::ofstream out_force;
      if (wantForce) {
        out_force.open(forceFile);
        if (!out_force) {
          throw std::runtime_error("Failed to open force output file: " + forceFile);
        }
      }

      std::ofstream out_stress;
      if (wantStress) {
        out_stress.open(stressFile);
        if (!out_stress) {
          throw std::runtime_error("Failed to open stress output file: " + stressFile);
        }
      }

      std::ofstream out_error;
      if (pred_err_bool) {
        out_error.open(errorFile);
        if (!out_error) {
          throw std::runtime_error("Failed to open error output file: " + errorFile);
        }
      }

      // Common headers
      std::string index_str   = "Index";
      std::string perror_str  = "Pred Error";

      // energy.pred
      std::string tenergy_str = "True Energy";
      std::string penergy_str = "Pred Energy";

      // forces.pred
      std::string tforce_str  = "True Force";
      std::string pforce_str  = "Pred Force";

      // stress.pred
      std::string tstress_str = "True Stress";
      std::string pstress_str = "Pred Stress";

      // error.pred
      std::string err_e_str   = "Energy";
      std::string err_f_str   = "Force (max)";
      std::string err_s_str   = "Stress (max)";

      // Grow width of the index column if necessary
      size_t iwe = stdb.size() > tadah::core::max_number(index_str.size()) ?
        tadah::core::number_of_digits(stdb.size()) : index_str.size();

      size_t iwf = stdb.calc_natoms() * 3 > tadah::core::max_number(index_str.size()) ?
        tadah::core::number_of_digits(stdb.calc_natoms() * 3) : index_str.size();

      size_t iws = stdb.size() * 6 > tadah::core::max_number(index_str.size()) ?
        tadah::core::number_of_digits(stdb.size() * 6) : index_str.size();

      out_energy << std::scientific << std::setprecision(sigfig);
      if (wantForce) {
        out_force << std::scientific << std::setprecision(sigfig);
      }
      if (wantStress) {
        out_stress << std::scientific << std::setprecision(sigfig);
      }
      if (pred_err_bool) {
        out_error << std::scientific << std::setprecision(sigfig);
      }

      // Write headers
      out_energy << std::setw(iwe) << index_str
                 << std::setw(nw) << tenergy_str
                 << std::setw(nw) << penergy_str;
      if (pred_err_bool) {
        out_energy << std::setw(nw) << perror_str;
      }
      out_energy << std::endl;

      if (wantForce) {
        out_force << std::setw(iwf) << index_str
                  << std::setw(nw) << tforce_str
                  << std::setw(nw) << pforce_str;
        if (pred_err_bool) {
          out_force << std::setw(nw) << perror_str;
        }
        out_force << std::endl;
      }

      if (wantStress) {
        out_stress << std::setw(iws) << index_str
                   << std::setw(nw) << tstress_str
                   << std::setw(nw) << pstress_str;
        if (pred_err_bool) {
          out_stress << std::setw(nw) << perror_str;
        }
        out_stress << std::endl;
      }

      if (pred_err_bool) {
        out_error << std::setw(iwe) << index_str
                  << std::setw(nw) << err_e_str;
        if (wantForce) {
          out_error << std::setw(nw) << err_f_str;
        }
        if (wantStress) {
          out_error << std::setw(nw) << err_s_str;
        }
        out_error << std::endl;
      }

      size_t j = 0;
      size_t fidx = 0, sidx = 0;
      bool newline = false;

      auto it_dbidx = stdb.dbidx.begin();
      if (it_dbidx != stdb.dbidx.end()) {
        ++it_dbidx;
      }

      for (size_t i = 0; i < stdb.size(); ++i) {
        if (it_dbidx != stdb.dbidx.end() && i == *it_dbidx) {
          newline = true;
          ++it_dbidx;
        }

        if (newline) {
          out_energy << std::endl << std::endl;
          if (wantForce) {
            out_force << std::endl << std::endl;
          }
          if (wantStress) {
            out_stress << std::endl << std::endl;
          }
        }

        out_energy << std::setw(iwe) << i
                   << std::setw(nw) << stdb(i).energy / stdb(i).natoms()
                   << std::setw(nw) << stpred(i).energy / stdb(i).natoms();

        if (pred_err_bool) {
          out_error << std::setw(iwe) << i
                    << std::setw(nw) << predicted_error(j) / stdb(i).natoms();
          out_energy << std::setw(nw) << predicted_error(j++) / stdb(i).natoms();
        }
        out_energy << std::endl;

        if (wantForce) {
          double max_ferr = 0.0;

          for (size_t a = 0; a < stdb(i).natoms(); ++a) {
            tadah::core::Vec3d &f  = stdb(i).atoms[a].force;
            tadah::core::Vec3d &fp = stpred(i).atoms[a].force;

            for (size_t k = 0; k < 3; ++k) {
              out_force << std::setw(iwf) << fidx++
                        << std::setw(nw) << f[k]
                        << std::setw(nw) << fp[k];

              if (pred_err_bool) {
                if (max_ferr < predicted_error(j)) {
                  max_ferr = predicted_error(j);
                }
                out_force << std::setw(nw) << predicted_error(j++);
              }

              out_force << std::endl;
            }
          }

          if (pred_err_bool) {
            out_error << std::setw(nw) << max_ferr;
          }
        }

        if (wantStress) {
          double max_serr = 0.0;
          tadah::core::stress_type &s  = stdb(i).stress;
          tadah::core::stress_type &sp = stpred(i).stress;

          for (size_t x = 0; x < 3; ++x) {
            for (size_t y = x; y < 3; ++y) {
              out_stress << std::setw(iws) << sidx++
                         << std::setw(nw) << s(x, y)
                         << std::setw(nw) << sp(x, y);

              if (pred_err_bool) {
                if (max_serr < predicted_error(j)) {
                  max_serr = predicted_error(j);
                }
                out_stress << std::setw(nw) << predicted_error(j++);
              }

              out_stress << std::endl;
            }
          }

          if (pred_err_bool) {
            out_error << std::setw(nw) << max_serr;
          }
        }

        newline = false;

        if (pred_err_bool) {
          out_error << std::endl;
        }
      }

      if (pred_err_bool) {
        out_error.close();
      }
      out_energy.close();
      if (wantForce) {
        out_force.close();
      }
      if (wantStress) {
        out_stress.close();
      }
    }

    void set_numeric(size_t n) {
      sigfig = n;
      fmtw = sigfig + 7;
      emptyspace = 5;
      nw = emptyspace + fmtw;
    }
};

} // namespace mlip
} // namespace tadah

#endif // !OUTPUT_H

