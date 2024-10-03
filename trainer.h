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
        nnf.calc(stdb);
        model->train(stdb,dc);
    }

    Config get_param_file() {
      return model->get_param_file();
    }

#ifdef TADAH_BUILD_MPI
    int train_worker(int &rows_available, MPI_Status &status, size_t &phi_row, 
        int &phi_cols1, DesignMatrix<DM_Function_Base&> &dm, MPI_Datatype &rowvecs, 
        int &worker, int &tag) {

      // release a worker
      if (status.MPI_TAG == TadahCLI::RELEASE_TAG) {
        int temp;
        MPI_Recv (&temp, 1, MPI_INT, worker, tag, MPI_COMM_WORLD, &status);
        if (rows_available!=0) { throw std::runtime_error("Attempting to release a worker... but the worker requires more data!!");}
        return 0;
      }
      else if (status.MPI_TAG == TadahCLI::WAIT_TAG) {
        int temp;
        MPI_Recv (&temp, 1, MPI_INT, worker, tag, MPI_COMM_WORLD, &status);
        // do nothing here; ask for more work in the next cycle
      }
      else if (status.MPI_TAG == TadahCLI::DATA_TAG) {
        // other worker is giving me some data
        int arr_size;
        MPI_Get_count(&status, MPI_DOUBLE, &arr_size);
        int rows_accepted = arr_size/phi_cols1;
        if (rows_available<rows_accepted) { throw std::runtime_error("Number of rows available is smaller than number of provided rows");}
        MPI_Recv (&dm.Phi.data()[phi_row], rows_available, rowvecs, worker, tag, MPI_COMM_WORLD, &status);
        MPI_Recv (&dm.T.data()[phi_row], rows_available, MPI_DOUBLE, worker, tag, MPI_COMM_WORLD, &status);
        MPI_Recv (&dm.Tlabels.data()[phi_row], rows_available, MPI_DOUBLE, worker, tag, MPI_COMM_WORLD, &status);
        rows_available -= rows_accepted;
        phi_row += rows_accepted;
      }
      else if (status.MPI_TAG == TadahCLI::WORK_TAG) {
        // otherwise get work package
        int fn_length;  // length of the filename char array
        int first;  // index of the first structure to read from the file
        int nstruc; // number of structures to be processed
        MPI_Get_count(&status, MPI_CHAR, &fn_length);

        char *fn  = (char *) malloc(fn_length+1);
        MPI_Recv (fn, fn_length, MPI_CHAR, 0, TadahCLI::WORK_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv (&first, 1, MPI_INT, 0, TadahCLI::WORK_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv (&nstruc, 1, MPI_INT, 0, TadahCLI::WORK_TAG, MPI_COMM_WORLD, &status);

        // do work
        StructureDB stdb;
        stdb.add(std::string(fn,fn_length),first,nstruc);
        nnf.calc(stdb);

        // compute number of rows needed for a given StructureDB
        int rows_needed = 0;
        for (size_t s=0; s<stdb.size(); ++s) {
          int natoms = stdb(s).natoms();
          rows_needed += DesignMatrixBase::phi_rows_num(config, 1, natoms);
        }

        if (rows_available<rows_needed) {
          // we do not have enough rows in the local phi matrix
          // so we create temp DM of required size
          DesignMatrix<DM_Function_Base&> temp_dm(*fb, config);
          temp_dm.Phi.resize(rows_needed,phi_cols1);
          temp_dm.T.resize(rows_needed);
          temp_dm.Tlabels.resize(rows_needed);

          // and compute all rows
          size_t temp_phi_row=0;
          temp_dm.fill_T(stdb);
          for (size_t s=0; s<stdb.size(); ++s) {
            StDescriptors st_d = dc.calc(stdb(s));
            temp_dm.build(temp_phi_row,stdb(s),st_d); // phi_row++
          }

          // first we try to fill remaining rows in the local phi matrix
          // copy top of temp_dm.Phi to the bottom of dm. Phi in reverse order
          if (rows_available>0) {
            for (rows_available; rows_available>0; rows_available--) {
              for (int c=0; c<phi_cols1; c++) {
                dm.Phi(phi_row,c) = temp_dm.Phi(rows_available-1,c); 
                dm.T(phi_row) = temp_dm.T(rows_available-1); 
                dm.Tlabels(phi_row) = temp_dm.Tlabels(rows_available-1); 
              }
              phi_row++;
              rows_needed--;
            }
          }

          // there are no more available rows
          // send remaining data to available processes
          while (rows_needed > 0) {
            // request host 
            MPI_Send (&rows_needed, 1, MPI_INT, 0, TadahCLI::DATA_TAG, MPI_COMM_WORLD);
            int rows_accepted; // number of accepted rows
            int dest; // receiving process
                      // host returns which dest can accept and how much
            MPI_Recv (&dest, 1, MPI_INT, 0, TadahCLI::DATA_TAG, MPI_COMM_WORLD, &status);

            MPI_Recv (&rows_accepted, 1, MPI_INT, 0, TadahCLI::DATA_TAG, MPI_COMM_WORLD, &status);
            // we send data to the host or a willing worker
            int start=temp_dm.Phi.rows()-rows_needed;

            // Define temp data type for temp Phi matrix 
            // Phi is stored in a column-major order
            MPI_Datatype trowvec, trowvecs;
            MPI_Type_vector( temp_dm.Phi.cols(), 1, temp_dm.Phi.rows(), MPI_DOUBLE, &trowvec); 
            MPI_Type_commit(&trowvec);
            MPI_Type_create_resized(trowvec, 0, 1*sizeof(double), &trowvecs);
            MPI_Type_commit(&trowvecs);

            // ready to send
            MPI_Send (&temp_dm.Phi.data()[start], rows_accepted, trowvecs, dest, TadahCLI::DATA_TAG, MPI_COMM_WORLD);
            MPI_Send (&temp_dm.T.data()[start], rows_accepted, MPI_DOUBLE, dest, TadahCLI::DATA_TAG, MPI_COMM_WORLD);
            MPI_Send (&temp_dm.Tlabels.data()[start], rows_accepted, MPI_DOUBLE, dest, TadahCLI::DATA_TAG, MPI_COMM_WORLD);
            rows_needed -= rows_accepted;
            MPI_Type_free(&trowvec);
            MPI_Type_free(&trowvecs);
          }
        }
        else {
          // just fill local phi array as it is large enough
          // fill_T must be called before phi_row is incremented
          dm.fill_T(stdb,phi_row);  // phi_row is not incremented by this method
          for (size_t s=0; s<stdb.size(); ++s) {
            StDescriptors st_d = dc.calc(stdb(s));
            dm.build(phi_row,stdb(s),st_d); // build() increments phi_row++
          }
          rows_available-=rows_needed;
        }

        if (fn)
          delete fn;
      }
      return 1;
    }
#endif
};
