#ifndef DESIGN_MATRIX_H
#define DESIGN_MATRIX_H

#include <cmath>
#include <tadah/mlip/st_descriptors_db.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/core/config.h>
#include <tadah/mlip/memory/IMLIPWorkspaceManager.h>
#include <tadah/mlip/memory/MLIPWorkspaceManager.h>
#include <tadah/mlip/memory/DesignMatrixWorkspace.h>

#include <stdexcept>


class DesignMatrixBase {
public:
  static int phi_rows_num(Config &c, int nstruct_tot, int natoms_tot) {
    bool force = c.get<bool>("FORCE");
    bool stress = c.get<bool>("STRESS");
    int rows = nstruct_tot;
    if (stress) rows += 6*nstruct_tot;
    if (force) rows += 3*natoms_tot;
    return rows;
  }
};


/** Design Matrix \f$ \Phi \f$ and the corresponding target vector \f$ \mathrm{T} \f$
 *
 * An aggregation matrix composed of \phi_i matrices
 *
 *     Phi = | \phi_1 |
 *           | \phi_2 |
 *           | ...    |
 *           | \phi_N |
 *
 * where N corresponds to the number of structures in a provided StructureDB.
 *
 * The ordering in \f$ \phi_i \f$
 *
 *     | energy of the 1st structure
 *     | force on the 1st atom in the x-dir
 *     | force on the 1st atom in the y-dir
 *     | force on the 1st atom in the z-dir
 *     | force on the 2st atom in the x-dir
 *     | ...
 *     | force on the n-th atom in the z-dir
 *     | stress xx
 *     | stress xy
 *     | stress xz
 *     | stress yy
 *     | stress yz
 *     | stress zz
 *
 * We need to know: \ref STRESS, \ref FORCE
 *
 * Scalling factors \ref FWEIGHT, \ref  SWEIGHT
 *
 * Default scalling:
 *  - Forces are scaled by 1/N_i/3 where N_i is a number of atoms in the i-th structure.
 *  - Stresses are scaled by 1/6.
 *  - Energies are scaled by 1/N_i
 *
 * @tparam F Function_Base child -> BF_Base or Kern_Base
 *
 */
template <typename F>
class DesignMatrix : public DesignMatrixBase {

public:

  F f;
  tadah::mlip::memory::DesignMatrixWorkspace *ws;
  bool w_copy_=false;  // keep copy of Phi and T witoout weights

  double e_std_dev=1;
  double f_std_dev=1;
  double s_std_dev[6] = {1,1,1,1,1,1};

  /** \brief This constructor fully initialise this object
     *
     * This class is used to build a Design Matrix.
     *
     * Usage example:
     * \code{.cpp}
     * Config config("Config");
     * BF_Linear bf(config);
     * DesignMatrix<LinearKernel> desmat(bf, config);
     *
     * \endcode
     */
  DesignMatrix(F &f, Config &c, tadah::mlip::memory::IMLIPWorkspaceManager& workspaceManager)
    : f(f),
    workspaceManager_(&workspaceManager),
    ownWorkspaceManager(false),
    config(c),
    force(config.get<bool>("FORCE")),
    stress(config.get<bool>("STRESS")),
    verbose(config.get<int>("VERBOSE")),
    eweightglob(config.get<double>("EWEIGHT")),
    fweightglob(config.get<double>("FWEIGHT")),
    sweightglob(config.get<double>("SWEIGHT"))
  {
    if (config.exist("ESTDEV"))
      e_std_dev = config.get<double>("ESTDEV");
    if (config.exist("FSTDEV"))
      f_std_dev = config.get<double>("FSTDEV");
    if (config.exist("SSTDEV"))
      config.get<double[6]>("SSTDEV", s_std_dev);
  }

  // Constructor without workspaceManager_ parameter (delegating constructor)
  DesignMatrix(F &f, Config &c)
  : DesignMatrix(f, c, *new tadah::mlip::memory::MLIPWorkspaceManager())
  {
    ownWorkspaceManager = true;  // Set ownership flag
  }

  // Destructor
  ~DesignMatrix() {
    if (ownWorkspaceManager && workspaceManager_ != nullptr) {
      delete workspaceManager_;
      workspaceManager_ = nullptr;
    }
  }

  /** \brief Build design matrix from already calculated StDescriptorsDB
     *
     * Here we simply build matrix from already calculated descriptors.
     * The vector of targets  **T** is build from StructureDB.
     * D2, D3 and DM calculators are not used.
     */
  void build(StDescriptorsDB &st_desc_db, const StructureDB &stdb, bool wcopy) {
    w_copy_ = wcopy;
    calc_mn(stdb);
    ws = workspaceManager_->getDesignMatrixWorkspace(rows,cols, wcopy);
    ws->Phi.set_zero();
    ws->T.set_zero();
    ws->Tlabels.set_zero();
    compute_stdevs(stdb);
    fill_T(stdb);
    compute_wfactors(stdb); // call after fill_T
    std::vector<size_t> rows(stdb.size());
    size_t row=0;
    for (size_t s=0; s<stdb.size(); ++s) {
      rows[s] = row;
      row++;
      if (force) {
        for (size_t a=0; a<stdb(s).natoms(); ++a) {
          row+=3;
        }
      }
      if (stress) {
        row+=6;
      }
    }
    // loop over all structures and build
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t s=0; s<stdb.size(); ++s) {
      build(rows[s], stdb(s),st_desc_db(s));
    }
  };

  /** \brief Calculate descriptors and build design matrix. */
  template <typename DC>
  void build(const StructureDB &stdb, Normaliser &norm,
             DC &dc, bool wcopy) {
    //DescriptorsCalc<D2,D3,DM,C2,C3,CM> dc(config);
    w_copy_ = wcopy;
    calc_mn(stdb);
    ws = workspaceManager_->getDesignMatrixWorkspace(rows,cols, wcopy);
    ws->Phi.set_zero();
    ws->T.set_zero();
    ws->Tlabels.set_zero();
    compute_stdevs(stdb);
    fill_T(stdb);
    compute_wfactors(stdb); // call after fill_T

    // for opm we need to find first rows for each structure
    std::vector<size_t> rows(stdb.size());
    size_t row=0;
    for (size_t s=0; s<stdb.size(); ++s) {
      rows[s] = row;
      row++;
      if (force) {
        for (size_t a=0; a<stdb(s).natoms(); ++a) {
          row+=3;
        }
      }
      if (stress) {
        row+=6;
      }
    }
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t s=0; s<stdb.size(); ++s) {
      StDescriptors st_d = dc.calc(stdb(s));

      if(config.get<bool>("NORM"))  // TODO move to scale()
        norm.normalise(st_d);

      build(rows[s], stdb(s), st_d);
    }
    if (w_copy_) {
      copyAndScale();
    }
    else {
      scale();
    }

  }
  void build(size_t &row, const Structure &st, const StDescriptors &st_d) {

    f.calc_phi_energy_row(ws->Phi,row,st,st_d);
    if (force) {
      f.calc_phi_force_rows(ws->Phi,row,st,st_d);
    }
    if (stress) {
      f.calc_phi_stress_rows(ws->Phi,row,st,st_d);
    }
  }
  void fill_T(const StructureDB &stdb, size_t start=0) {
    size_t j=start;
    for (size_t s=0; s<stdb.size(); ++s) {

      ws->Tlabels(j) = 0;
      ws->T(j++) = stdb(s).energy;

      if (force) {
        for (const Atom &a : stdb(s).atoms) {
          ws->Tlabels(j) = 1;
          ws->T(j++) = a.force(0);
          ws->Tlabels(j) = 1;
          ws->T(j++) = a.force(1);
          ws->Tlabels(j) = 1;
          ws->T(j++) = a.force(2);
        }
      }
      if (stress) {
        for (size_t x=0; x<3; ++x) {
          for (size_t y=x; y<3; ++y) {
            ws->Tlabels(j) = 2;
            ws->T(j++)=stdb(s).stress(x,y);
          }
        }
      }
    }
  }

  phi_type &getPhi() { return ws->Phi; }
  t_type &getT() { return ws->T; }
  t_type &getTlabels() { return ws->Tlabels; }
  bool hasCopy() { return w_copy_; }

private:
  tadah::mlip::memory::IMLIPWorkspaceManager* workspaceManager_ = nullptr;
  bool ownWorkspaceManager = false;
  Config & config;
  size_t rows = 0;
  size_t cols = 0;
  bool force;
  bool stress;
  int verbose;
  double eweightglob;
  double fweightglob;
  double sweightglob;

  // resize Phi and T and set all elements to zero
  void calc_mn(const StructureDB &stdb) {
    rows=0;
    cols = f.get_phi_cols(config);

    for (size_t s=0; s<stdb.size(); ++s) {
      rows++;
      if (stress) rows+=6;
      if (force) rows+=stdb(s).natoms()*3;
    }

    // Add space for augmented matrix
    rows += cols;

    if (verbose) {
      std::cout << "Phi rows: "<< rows << std::endl;
      std::cout << "Phi cols: "<< cols << std::endl;
    }
  }

  void compute_stdevs(const StructureDB &stdb) {
    t_type evec(stdb.size());
    Matrix svec(stdb.size(),6);
    size_t num_forces=0;
    for (size_t s=0; s<stdb.size(); ++s) {
      evec(s) = stdb(s).energy/stdb(s).natoms();
      if (stress) {
        size_t j=0;
        for (size_t x=0; x<3; ++x) {
          for (size_t y=x; y<3; ++y) {
            svec(s,j++)=stdb(s).stress(x,y);
          }
        }
      }
      num_forces +=3*stdb(s).natoms();
    }

    e_std_dev = evec.std_dev(evec.mean(), evec.size()-1);
    if (verbose) std::cout << "Energy standard deviation (eV/atom): " << e_std_dev << std::endl;

    if (stress) {
      for (size_t j=0; j<6; ++j) {
        s_std_dev[j] = svec.col(j).std_dev(svec.col(j).mean(), svec.col(j).size()-1);
        if (verbose) std::cout << "Stress standard deviation (eV/A^3): " << s_std_dev[j] << std::endl;
      }
    }

    if (force) {
      size_t j=0;
      t_type fvec(num_forces);
      for (size_t s=0; s<stdb.size(); ++s) {
        for (const Atom &a : stdb(s).atoms) {
          fvec(j++) = a.force(0);
          fvec(j++) = a.force(1);
          fvec(j++) = a.force(2);
        }
      }
      f_std_dev = fvec.std_dev(fvec.mean(),fvec.size()-1);
      if (verbose) std::cout << "Force standard deviation (ev/A): " << f_std_dev << std::endl;
    }

    config.add("ESTDEV",e_std_dev);
    config.add("FSTDEV",f_std_dev);
    for (size_t j=0; j<6; ++j)
      config.add("SSTDEV",s_std_dev[j]);
  }

  void compute_wfactors(const StructureDB &stdb) {
    size_t j=0;
    for (size_t s=0; s<stdb.size(); ++s) {
      ws->wfactors(j++) = std::sqrt(stdb(s).eweight*eweightglob);
      if (force) {
        double fscale = std::sqrt(stdb(s).fweight*fweightglob);
        for (size_t a=0; a<stdb(s).natoms(); ++a) {
          ws->wfactors(j++) = fscale;
          ws->wfactors(j++) = fscale;
          ws->wfactors(j++) = fscale;
        }
      }
      if (stress) {
        for (size_t x=0; x<3; ++x) {
          for (size_t y=x; y<3; ++y) {
            ws->wfactors(j++) = std::sqrt(stdb(s).sweight*sweightglob);
          }
        }
      }
    }
  }

  void copyAndScale()
  {
    // first copy scalled Phi to Phi_cpy and T to T_cpy
    // followed by swapping pointers
    for (std::size_t j = 0; j < ws->Phi.cols(); ++j) {
      for (std::size_t i = 0; i < ws->Phi.rows(); ++i) {
        ws->Phi_cpy(i,j) = ws->Phi(i,j)*ws->wfactors(i);
      }
    }
    for (std::size_t i = 0; i < ws->Phi.rows(); ++i) {
      ws->T_cpy(i) = ws->T(i)*ws->wfactors(i);
    }
    std::swap(ws->Phi, ws->Phi_cpy);
    std::swap(ws->T, ws->T_cpy);
  }
  void scale()
  {
    for (std::size_t j = 0; j < ws->Phi.cols(); ++j) {
      for (std::size_t i = 0; i < ws->Phi.rows(); ++i) {
        ws->Phi(i,j) *= ws->wfactors(i);
      }
    }
    for (std::size_t i = 0; i < ws->Phi.rows(); ++i) {
      ws->T(i) *= ws->wfactors(i);
    }
  }
};
#endif
