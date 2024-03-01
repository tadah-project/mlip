#ifndef M_KRR_H
#define M_KRR_H

#include "basis.h"
#include "m_tadah_base.h"
#include "../descriptors_calc.h"
#include "../design_matrix/design_matrix.h"
#include "../design_matrix/functions/dm_function_base.h"
#include "../../MODELS/m_krr_train.h"
#include "../../CORE/config/config.h"

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <iostream>

/** Kernel Ridge Regression.
 *
 * Usage:
 *
 *   M_KRR KERNEL ALGO
 *
 * Example:
 *
 *   M_KRR Kern_RBF 1
 *
 * KERNEL is any kernel defined in Kernels, for example:
 *
 *    Kern_Linear
 *    Kern_RBF
 *    ...
 *
 * ALGO specify which implementation to use:
 *
 *    ALGO = 1 (default) uses empirical kernel map to achieve sparse model
 *    ALGO = 2 standard with the the covariane matrix
 *
 * **Optional Config keys**:
 *
 *  - \ref LAMBDA = 0  -> use OLS
 *  - \ref LAMBDA > 0  -> manually set to given value
 *  - \ref LAMBDA = -1 -> use evidence approximation to find \f$\lambda\f$
 *  - \ref SBASIS N    -> Use N basis functions when nonlinear K is used
 *
 * @tparam K DM_Kern_Base child, Kernel function
 */
template
<class K=DM_Function_Base&>
class M_KRR: public M_Tadah_Base,
    public M_KRR_Train<K>
    //public M_KRR_Predict<K>
{

    public:

        /** This constructor will preapare this object for either training
         *  or prediction (if potential is provides as a Config)
         *
         * Usage example:
         *
         * \code{.cpp}
         * Config config("Config");
         * M_KRR<Kern_Linear> krr(config);
         * \endcode
         *
         */
        M_KRR(Config &c):
            M_KRR_Train<K>(c),
            basis(c),
            desmat(kernel,c)
    {
        norm = Normaliser(c);
    }

        /** This constructor will preapare this object for either training
         *  or prediction (if potential is provides as a Config)
         *
         * Usage example:

         * \code{.cpp}
         * Config config("Config");
         * Kern_Linear kernel(config);
         * M_KRR<> krr(kernel, config);
         * \endcode
         *
         */
        M_KRR(K &kernel, Config &c):
            M_KRR_Train<K>(kernel,c),
            basis(c),
            desmat(kernel,c)
    {
        norm = Normaliser(c);
    }

        double epredict(const aed_type2 &aed) {
            return kernel.epredict(weights,aed);
        };

        double fpredict(const fd_type &fdij, const aed_type2 &aedi, const size_t k) {
            return kernel.fpredict(weights,fdij,aedi,k);
        }

        force_type fpredict(const fd_type &fdij, const aed_type2 &aedi) {
            return kernel.fpredict(weights,fdij,aedi);
        }

        void train(StDescriptorsDB &st_desc_db, const StructureDB &stdb) {

            if(config.template get<bool>("NORM"))
                norm = Normaliser(config,st_desc_db);

            desmat.build(st_desc_db,stdb);
            train(desmat);
        }

        void train(StructureDB &stdb, DC_Base &dc) {
            int modelN;
            try {
                modelN=config.template get<int>("MODEL",2);
            }
            catch(std::runtime_error &e) {
                // use default model
                modelN=1;
                config.add("MODEL", modelN);
            }
            if (modelN==1)
                train1(stdb,dc);
            else if (modelN==2)
                train2(stdb,dc);
            else
                throw
                    std::runtime_error(
                        "This KRR implementation does exist: "\
                        + std::to_string(modelN)+"\n");
        }
        void train1(StructureDB &stdb, DC_Base &dc) {
            // KRR implementation using EKM
            if(config.template get<bool>("NORM") || kernel.get_label()!="Kern_Linear") {
                // either build basis or prep normaliser
                std::string force=config.template get<std::string>("FORCE");
                std::string stress=config.template get<std::string>("STRESS");

                config.remove("FORCE");
                config.remove("STRESS");
                config.add("FORCE", "false");
                config.add("STRESS", "false");

                StDescriptorsDB st_desc_db_temp = dc.calc(stdb);

                if(config.template get<bool>("NORM")) {
                    norm = Normaliser(config);
                    norm.learn(st_desc_db_temp);
                    norm.normalise(st_desc_db_temp);
                }

                config.remove("FORCE");
                config.remove("STRESS");
                config.add("FORCE", force);
                config.add("STRESS", stress);

                if (kernel.get_label()!="Kern_Linear") {
                    basis.build_random_basis(config.template get<size_t>("SBASIS"),st_desc_db_temp);
                    desmat.f.set_basis(basis.b);
                    kernel.set_basis(basis.b);
                    // to configure ekm, we need basis vectors
                    ekm.configure(basis.b);
                }
            }
            desmat.build(stdb,norm,dc);
            train(desmat);
        }

        void train2(StructureDB &stdb, DC_Base &dc) {

            // NEW (BASIC) IMPLEMENTATION OF KRR //
            std::string force=config.template get<std::string>("FORCE");
            std::string stress=config.template get<std::string>("STRESS");
            config.remove("FORCE");
            config.remove("STRESS");
            config.add("FORCE", "false");
            config.add("STRESS", "false");
            StDescriptorsDB st_desc_db_temp = dc.calc(stdb);
            if(config.template get<bool>("NORM")) {
                norm = Normaliser(config);
                norm.learn(st_desc_db_temp);
                norm.normalise(st_desc_db_temp);
            }
            config.remove("FORCE");
            config.remove("STRESS");
            config.add("FORCE", force);
            config.add("STRESS", stress);
            basis.prep_basis_for_krr(st_desc_db_temp,stdb);
            kernel.set_basis(basis.b);
            M_KRR_Train<K>::train2(basis.b, basis.T);
        }

        Structure predict(const Config &c, StDescriptors &std, const Structure &st) {
            if(config.template get<bool>("NORM") && !std.normalised && kernel.get_label()!="Kern_Linear")
                norm.normalise(std);
            return M_Tadah_Base::predict(c,std,st);
        }

        StructureDB predict(Config &c, const StructureDB &stdb, DC_Base &dc) {
            return M_Tadah_Base::predict(c,stdb,dc);
        }

        Config get_param_file() {
            Config c = config;
            c.remove("ALPHA");
            c.remove("BETA");
            c.remove("DBFILE");
            c.remove("FORCE");
            c.remove("STRESS");
            c.remove("VERBOSE");

            c.clear_internal_keys();
            c.remove("MODEL");
            c.add("MODEL", label);
            c.add("MODEL", kernel.get_label());
            int modelN=config.template get<int>("MODEL",2);
            c.add("MODEL", modelN);

            for (size_t i=0;i<weights.size();++i) {
                c.add("WEIGHTS", weights(i));
            }

            if(config.template get<bool>("NORM")) {
                for (size_t i=0;i<norm.mean.size();++i) {
                    c.add("NMEAN", norm.mean[i]);
                }
                for (size_t i=0;i<norm.std_dev.size();++i) {
                    c.add("NSTDEV", norm.std_dev[i]);
                }
            }
            if (kernel.get_label()!="Kern_Linear") {
                // dump basis to the config file file
                // make sure keys are not accidently assigned
                if (c.exist("SBASIS"))
                    c.remove("SBASIS");
                if (c.exist("BASIS"))
                    c.remove("BASIS");
                c.add("SBASIS", basis.b.cols());
                for (size_t i=0;i<basis.b.cols();++i) {
                    for (size_t j=0;j<basis.b.rows();++j) {
                        c.add("BASIS", basis.b(j,i));
                    }
                }
            }
            return c;
        }
        StructureDB predict(Config config_pred, StructureDB &stdb, DC_Base &dc,
                aed_type2 &predicted_error) {

            LinearRegressor::read_sigma(config_pred,Sigma);
            DesignMatrix<K> dm(kernel,config_pred);
            dm.scale=false; // do not scale energy, forces and stresses
            dm.build(stdb,norm,dc);

            // compute error
            predicted_error = T_MDMT_diag(dm.Phi, Sigma);
            double beta = config.template get<double>("BETA");
            predicted_error += 1.0/beta;

            // compute energy, forces and stresses
            aed_type2 Tpred = T_dgemv(dm.Phi, weights);

            // Construct StructureDB object with predicted values
            StructureDB stdb_;
            stdb_.structures.resize(stdb.size());
            size_t i=0;
            for (size_t s=0; s<stdb.size(); ++s) {
                stdb_(s) = Structure(stdb(s));

                predicted_error(i) = sqrt(predicted_error(i));

                stdb_(s).energy = Tpred(i++);
                if (config_pred.get<bool>("FORCE")) {
                    for (size_t a=0; a<stdb(s).natoms(); ++a) {
                        for (size_t k=0; k<3; ++k) {
                            predicted_error(i) = sqrt(predicted_error(i));
                            stdb_(s).atoms[a].force[k] = Tpred(i++);
                        }
                    }
                }
                if (config_pred.get<bool>("STRESS")) {
                    for (size_t x=0; x<3; ++x) {
                        for (size_t y=x; y<3; ++y) {
                            predicted_error(i) = sqrt(predicted_error(i));
                            stdb_(s).stress(x,y) = Tpred(i++);
                            if (x!=y)
                                stdb_(s).stress(y,x) = stdb_(s).stress(x,y);
                        }
                    }
                }
            }
            return stdb_;
        }
        StructureDB predict(StructureDB &stdb) {
            if(!trained) throw std::runtime_error("This object is not trained!\n\
                    Hint: check different predict() methods.");

            phi_type &Phi = desmat.Phi;

            // compute energy, forces and stresses
            aed_type2 Tpred = T_dgemv(Phi, weights);

            double eweightglob=config.template get<double>("EWEIGHT");
            double fweightglob=config.template get<double>("FWEIGHT");
            double sweightglob=config.template get<double>("SWEIGHT");

            // Construct StructureDB object with predicted values
            StructureDB stdb_;
            stdb_.structures.resize(stdb.size());
            size_t s=0;
            size_t i=0;
            while (i<Phi.rows()) {

                stdb_(s).energy = Tpred(i++)*stdb(s).natoms()/eweightglob/stdb(s).eweight;
                if (config.template get<bool>("FORCE")) {
                    stdb_(s).atoms.resize(stdb(s).natoms());
                    for (size_t a=0; a<stdb(s).natoms(); ++a) {
                        for (size_t k=0; k<3; ++k) {
                            stdb_(s).atoms[a].force[k] = Tpred(i++)/fweightglob/stdb(s).fweight;
                        }
                    }
                }
                if (config.template get<bool>("STRESS")) {
                    for (size_t x=0; x<3; ++x) {
                        for (size_t y=x; y<3; ++y) {
                            stdb_(s).stress(x,y) = Tpred(i++)/sweightglob/stdb(s).sweight;
                            if (x!=y)
                                stdb_(s).stress(y,x) = stdb_(s).stress(x,y);
                        }
                    }
                }
                s++;
            }
            return stdb_;
        }

    private:
        std::string label="M_KRR";
        Basis<K> basis;
        DesignMatrix<K> desmat;

        t_type convert_to_nweights(const t_type &weights) const {
            if(kernel.get_label()!="Kern_Linear") {
                throw std::runtime_error("Cannot convert weights to nweights for\n\
                        non linear kernel\n");
            }
            t_type kw(weights.rows());
            if(config.template get<bool>("NORM") && kernel.get_label()=="Kern_Linear") {
                // normalise weights such that when predict is called
                // we can supply it with a non-normalised descriptor
                kw.resize(weights.rows());
                kw(0) = weights(0);
                for (size_t i=1; i<weights.size(); ++i) {

                    if (norm.std_dev[i] > std::numeric_limits<double>::min())
                        kw(i) = weights(i) / norm.std_dev[i];
                    else
                        kw(i) = weights(i);

                    kw(0) -= norm.mean[i]*kw(i);

                }
            }
            return kw;
        }
        // The opposite of convert_to_nweights()
        t_type convert_to_weights(const t_type &kw) const {
            if(kernel.get_label()!="Kern_Linear") {
                throw std::runtime_error("Cannot convert nweights to weights for\n\
                        non linear kernel\n");
            }
            // convert normalised weights back to "normal"
            t_type w(kw.rows());
            w(0) = kw(0);
            for (size_t i=1; i<kw.size(); ++i) {
                if (norm.std_dev[i] > std::numeric_limits<double>::min())
                    w(i) = kw(i) * norm.std_dev[i];
                else
                    w(i) = kw(i);

                w(0) += kw(i)*norm.mean[i];
            }
            return w;
        }

        template <typename D>
            void train(D &desmat) {
                phi_type &Phi = desmat.Phi;
                t_type &T = desmat.T;
                M_KRR_Train<K>::train(Phi,T);

                if (config.template get<bool>("NORM") &&
                        kernel.get_label()=="Kern_Linear") {
                    weights = convert_to_nweights(weights);
                }
            }

        // Do we want to confuse user with those and make them public?
        // Either way they must be stated as below to silence clang warning
        using M_KRR_Train<K>::predict;
        using M_KRR_Train<K>::train;
        using M_KRR_Train<K>::trained;
        using M_KRR_Train<K>::weights;
        using M_KRR_Train<K>::Sigma;
        using M_KRR_Train<K>::config;
        using M_KRR_Train<K>::kernel;
        using M_KRR_Train<K>::ekm;
};
#endif
