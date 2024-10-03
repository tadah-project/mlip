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
    DesignMatrix<DM_Function_Base&> dm;

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
      nnf(config),
      fb(CONFIG::factory<DM_Function_Base,Config&>(
            config.get<std::string>("MODEL",1),config)),
      model(CONFIG::factory<M_Tadah_Base,DM_Function_Base&,Config&>
          (config.get<std::string>("MODEL",0),*fb,config)),
      dm(*fb, config)
  {
    config.postprocess();
    config.check_for_training();
  }

    void train(StructureDB &stdb) {
      nnf.calc(stdb);
      model->train(stdb,dc);
    }

    Config get_param_file() {
      return model->get_param_file();
    }

};
#ifdef TADAH_BUILD_MPI
extern "C" void blacs_get_(int*, int*, int*);
extern "C" void blacs_pinfo_(int*, int*);
extern "C" void blacs_gridinit_(int*, char*, int*, int*);
extern "C" void blacs_gridinfo_(int*, int*, int*, int*, int*);
extern "C" void descinit_(int*, int*, int*, int*, int*,
    int*, int*, int*, int*, int*);
extern "C" void pdpotrf_(char*, int*, double*, int*, int*, int*, int*);
extern "C" void blacs_gridexit_(int*);
extern "C" int numroc_(int*, int*, int*, int*, int*);
extern "C" void	pdgels_(char* trans, int* m, int* n, int* nrhs,
    double* a, int* ia, int* ja, int* desca, double* b, int* ib,
    int* jb, int* descb, double* work, int* lwork, int* info);
extern "C" void	pdgemr2d_(int *m, int *n, double *a, int *ia, int *ja, int *desca,
    double *b, int *ib, int *jb, int *descb, int *context);
extern "C" void pdgemv_(char* transa, int* m, int* n, double* alpha, double* a,
    int* ia, int* ja, int* desc_a, double* x, int* ix, int* jx, int* desc_x,
    int* incx, double* beta, double* y, int* iy, int* jy, int* desc_y, int* incy);	

class MPI_Trainer: public Trainer {
  public:
    int context1,      context2;
    int b_row1,        b_row2;
    int b_col1,        b_col2;
    int b_nrows1,      b_nrows2;    // Number of row procs
    int b_ncols1,      b_ncols2;    // Number of column procs
    int rnb1,          rnb2;   // Row block size
    int cnb1,          cnb2;   // Column block size
    int PHI_cols;
    int PHI_rows;
    int rows_available;
    int b_rank;
    int phi_rows1;
    int phi_cols1;
    int phi_rows2;
    int phi_cols2;
    int izero = 0;
    int ione = 1;
    char layout='R'; // Block cyclic, Row major processor mapping
    size_t phi_row = 0; // next row to be filled in the local phi array
    MPI_Datatype rowvec, rowvecs;

    MPI_Trainer(Config &c):
      Trainer(c)
  {}
    void init(int &rank, int &ncpu) {

      if (rank==0) {
        int nstruct_tot = StructureDB::count(config).first;
        int natoms_tot = StructureDB::count(config).second;
        PHI_cols = fb->get_phi_cols(config);
        PHI_rows = DesignMatrixBase::phi_rows_num(config, nstruct_tot, natoms_tot);
      }
      MPI_Bcast(&PHI_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&PHI_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

      // Initialize BLACS
      // We create two contexts.
      // context1 is used for the computation of phi matrices
      // context2 is used for distribution of local phi to "block cyclic phi"
      blacs_pinfo_(&b_rank, &ncpu) ; // BLACS rank and world size

      rnb1 = ceil(PHI_rows/ncpu);
      rnb2 = config.get<int>("MBLOCK");   // Row block size
      cnb1 = PHI_cols;
      cnb2 = config.get<int>("NBLOCK");   // Column block size

      b_ncols1 = 1;        b_ncols2 = 2;
      b_nrows1 = ncpu;     b_nrows2 = ncpu/b_ncols2;

      // make as sqaure grid as possible
      int sr = sqrt(ncpu);
      if (sr*sr==ncpu) {
        b_nrows2 = sr;
        b_ncols2 = sr;
      }
      else {
        // loop over all possible divisors
        int before, /*sqrt(ncpu),*/ after;
        for (int i = 1; i <= ncpu; ++i){
          if (ncpu % i == 0) {
            if (i>sqrt(ncpu)) {
              after=i; 
              break;
            }
            before=i;
          }
        }
        b_nrows2 = after;
        b_ncols2 = before;
      }

      assert(b_nrows2 * b_ncols2 == ncpu);
      assert(b_nrows1 * b_ncols1 == ncpu);


      // Create first context
      blacs_get_(&izero,&izero, &context1 ); // -> Create context1
      blacs_gridinit_(&context1, &layout, &b_nrows1, &b_ncols1 ); // context1 -> Initialize the grid
      blacs_gridinfo_(&context1, &b_nrows1, &b_ncols1, &b_row1, &b_col1 );

      // Create second context
      blacs_get_(&izero,&izero, &context2 ); // -> Create context2
      blacs_gridinit_(&context2, &layout, &b_nrows2, &b_ncols2 ); // context2 -> Initialize the grid
      blacs_gridinfo_(&context2, &b_nrows2, &b_ncols2, &b_row2, &b_col2 );

      // Compute the size of the local phi matrices
      phi_rows1 = numroc_( &PHI_rows, &rnb1, &b_row1, &izero, &b_nrows1 );
      phi_cols1 = numroc_( &PHI_cols, &cnb1, &b_col1, &izero, &b_ncols1 );
      phi_rows2 = numroc_( &PHI_rows, &rnb2, &b_row2, &izero, &b_nrows2 );
      phi_cols2 = numroc_( &PHI_cols, &cnb2, &b_col2, &izero, &b_ncols2 );

      // Define MPI datatype to send rows from the PHI matrix with column-major order
      // used only in context1
      MPI_Type_vector( phi_cols1, 1, phi_rows1, MPI_DOUBLE, &rowvec); 
      MPI_Type_commit(&rowvec);
      MPI_Type_create_resized(rowvec, 0, sizeof(double), &rowvecs);
      MPI_Type_commit(&rowvecs);

      // COUNTERS
      rows_available=phi_rows1;  // number of available rows in the local phi array

      // once we know the size of local phi, we can allocate memory to it
      // including host as well. The host will collect excees computations from
      // workers.
      //DesignMatrix<DM_Function_Base&> dm(*fb, config);
      dm.Phi.resize(phi_rows1,phi_cols1);
      dm.T.resize(phi_rows1);
      dm.Tlabels.resize(phi_rows1);

    }
    int calc_desc(MPI_Status &status, int &rows_available, size_t &phi_row) {

      // get work package
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
        temp_dm.Phi.resize(rows_needed,dm.Phi.cols());
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
            for (int c=0; c<dm.Phi.cols(); c++) {
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
      return 0;
    }
};

class MPI_Trainer_HOST {
  private:
    MPI_Trainer &tr;
    int &rank;
    int &ncpu;
    MPI_Status &status;
  public:
    std::vector<std::tuple<std::string,int,int>> wpckgs;
    MPI_Trainer_HOST(MPI_Trainer &MPIT, MPI_Status &status, int &rank, int &ncpu):
      tr(MPIT)
  {}

    void prep_wpckgs() {
      // HOST: prepare work packages
      // filename, first structure index, number of structures to read
      int nstruc = tr.config.get<int>("MPIWPCKG");
      for (const std::string &fn : tr.config("DBFILE")) {
        // get number of structures
        int dbsize = StructureDB::count(fn).first;
        int first=0;
        while(true) {
          if (nstruc < dbsize) {
            wpckgs.push_back(std::make_tuple(fn,first,nstruc));
            first += nstruc;
          } else {
            wpckgs.push_back(std::make_tuple(fn,first,dbsize));
            break;
          }
          dbsize-=nstruc;
        }
      }
    }

    void work_tag() {

          int rows_available;
          MPI_Recv (&rows_available, 1, MPI_INT, worker, tag, MPI_COMM_WORLD, &status);

          std::tuple<std::string,int,int> wpckg = wpckgs.back();
          wpckgs.pop_back();

          // send dataset filename
          const char *fn = std::get<0>(wpckg).c_str();
          int fn_length = std::get<0>(wpckg).length()+1;  // +1 for char
          MPI_Send (fn, fn_length, MPI_CHAR, worker, tag, MPI_COMM_WORLD);

          // send index of the first structure to load
          int first = std::get<1>(wpckg);
          MPI_Send (&first, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);

          // send number of structures to load
          int nstruc = std::get<2>(wpckg);
          MPI_Send (&nstruc, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
    }

    void data_tag() {
          int rows_needed;
          MPI_Recv (&rows_needed, 1, MPI_INT, worker, tag, MPI_COMM_WORLD, &status);
          if (tr.rows_available>0) {
            int rows_accepted = tr.rows_available < rows_needed ? tr.rows_available : rows_needed;
            MPI_Send (&tr.b_rank, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
            MPI_Send (&rows_accepted, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
            MPI_Recv (&tr.dm.Phi.data()[tr.phi_row], rows_accepted, tr.rowvecs, worker, tag, MPI_COMM_WORLD, &status);
            MPI_Recv (&tr.dm.T.data()[tr.phi_row], rows_accepted, MPI_DOUBLE, worker, tag, MPI_COMM_WORLD, &status);
            MPI_Recv (&tr.dm.Tlabels.data()[tr.phi_row], rows_accepted, MPI_DOUBLE, worker, tag, MPI_COMM_WORLD, &status);
            tr.rows_available -= rows_accepted;
            tr.phi_row += rows_accepted;
            if (tr.rows_available<0 ) { throw std::runtime_error(" HOST1: The number of rows in the local array is smaller than requested.");}
          }
          else {
            // Host is unable to fit data we have to ask workers for their storage availability
            // Locate a worker willing to accept at least some data.
            int worker2;
            MPI_Status status2;
            // find a worker capable of accepting data
            int w_rows_available;
            while (true) {
              MPI_Recv (&w_rows_available, 1, MPI_INT, MPI_ANY_SOURCE, TadahCLI::WORK_TAG, MPI_COMM_WORLD, &status2);
              worker2 = status2.MPI_SOURCE;
              if (worker==worker2) {throw std::runtime_error("worker and worker2 are the same."); }
              if (w_rows_available==0 ) {
                // give up on this worker
                MPI_Send (&worker2, 1, MPI_INT, worker2, TadahCLI::WAIT_TAG, MPI_COMM_WORLD);
              } 
              else {
                // found a worker
                break;
              }
            }
            int rows_accepted = w_rows_available < rows_needed ? w_rows_available : rows_needed;
            // tell worker about worker2
            MPI_Send (&worker2, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
            MPI_Send (&rows_accepted, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
          }
        }

    void run() {
      // work finised, collect remaining data and release all workers
      int count=1;  // count number of release workers, skip host
      while(true) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int worker = status.MPI_SOURCE;
        int tag = status.MPI_TAG;

        if (tag==TadahCLI::DATA_TAG) {
          int rows_needed;
          MPI_Recv (&rows_needed, 1, MPI_INT, worker, tag, MPI_COMM_WORLD, &status);
          if (tr.rows_available>0) {
            int rows_accepted = tr.rows_available < rows_needed ? tr.rows_available : rows_needed;
            MPI_Send (&tr.b_rank, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
            MPI_Send (&rows_accepted, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
            MPI_Recv (&tr.dm.Phi.data()[tr.phi_row], rows_accepted, tr.rowvecs, worker, tag, MPI_COMM_WORLD, &status);
            MPI_Recv (&tr.dm.T.data()[tr.phi_row], rows_accepted, MPI_DOUBLE, worker, tag, MPI_COMM_WORLD, &status);
            MPI_Recv (&tr.dm.Tlabels.data()[tr.phi_row], rows_accepted, MPI_DOUBLE, worker, tag, MPI_COMM_WORLD, &status);
            tr.rows_available -= rows_accepted;
            tr.phi_row += rows_accepted;
            if (tr.rows_available<0 ) { throw std::runtime_error(" HOST2: The number of rows in the local array is smaller than requested.");}
          }
          else {
            // host is unable to fit data we have to ask workers for their storage availability
            // find a worker to accept at least some data
            MPI_Status status2;
            int worker2;
            // find a worker capable of accepting data
            int w_rows_available;
            while (true) {
              MPI_Recv (&w_rows_available, 1, MPI_INT, MPI_ANY_SOURCE, TadahCLI::WORK_TAG, MPI_COMM_WORLD, &status2);
              worker2 = status2.MPI_SOURCE;
              if (worker==worker2) {throw std::runtime_error("worker and worker2 are the same."); }
              if (w_rows_available==0 ) {
                // give up on this worker and release him as there is no more work to be done
                MPI_Send (0, 0, MPI_INT, worker2, TadahCLI::RELEASE_TAG, MPI_COMM_WORLD);
                count++;
              } 
              else {
                // found a worker
                break;
              }
            }
            int rows_accepted = w_rows_available < rows_needed ? w_rows_available : rows_needed;
            MPI_Send (&worker2, 1, MPI_INT, worker, TadahCLI::DATA_TAG, MPI_COMM_WORLD);
            MPI_Send (&rows_accepted, 1, MPI_INT, worker, TadahCLI::DATA_TAG, MPI_COMM_WORLD);
          }
        }
        else {
          int rows_available;
          MPI_Recv (&rows_available, 1, MPI_INT, worker, TadahCLI::WORK_TAG, MPI_COMM_WORLD, &status);
          // there is no more work so release a worker
          if (rows_available==0) {
            MPI_Send (0, 0, MPI_INT, worker, TadahCLI::RELEASE_TAG, MPI_COMM_WORLD);
            count++;
          }
          else {
            MPI_Send (0, 0, MPI_INT, worker, TadahCLI::WAIT_TAG, MPI_COMM_WORLD);
          }
        }
        if (count==ncpu) {  break; }  // count starts from 1
      }
    }
};
#endif
