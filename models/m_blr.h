/** Bayesian Linear Regression
 *
 * Model supported training modes:
 *
 *  - LINEAR:     Ordinary Least Squares or Ridge Regression (Regularised Least Squares)
 *
 *  - NONLINEAR:  Linear in parameters but nonlinear in an input space.
 *                Each row of a matrix \f$\Phi\f$ is a vector-valued function
 *                of the original input descriptor \f$ \mathbf{\phi(x)}^T \f$
 *                e.g. BF_Polynomial2:
 *                (1, x_1, x_2) -> (1, x_1, x_2, x_1x_2, x_1^2, x_2^2)
 *
 * **Prediction**:
 *
 * \f[
 * y(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{M-1} w_j \phi_j(\mathbf(x))
 * \f]
 *
 * where M is a number of parameters, **w** are machine learned weights
 * and **x** is a vector of input variables.

 *
 * **Training**:
 *
 * \f[
 * \mathbf{w} = (\lambda \mathbf{I} + \mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
 * \f]
 *
 * for \f$\lambda=0\f$ this reduces to Ordinary Last Squares (OLS) aka
 * Linear Least Squares with the normal equation:
 *
 * \f[
 * \mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
 * \f]
 *
 * This model is always a linear function of its parameters \f$w_i\f$.
 *
 * When used with linear basis function \f$\phi\f$ (here BF_Linear)
 * it is also linear wrt to the input variables (here descriptors).
 *
 * When nonlinear basis function \f$\phi\f$ is used
 * the function \f$y(\mathbf{x}, \mathbf{w})\f$ becomes nonlinear wrt to **x**
 * but still linear wrt to **w**.
 *
 * **Optional Config keys**:
 *
 *  - \ref LAMBDA = 0  -> use OLS
 *  - \ref LAMBDA > 0  -> manually set to given value
 *  - \ref LAMBDA = -1 -> use evidence approximation to find \f$\lambda\f$
 *
 * @tparam BF DM_BF_Base child, Basis function
 */
#ifndef M_BLR_H
#define M_BLR_H

#include "m_tadah_base.h"
#include "../descriptors_calc.h"
#include "../design_matrix/design_matrix.h"
#include "../design_matrix/functions/dm_function_base.h"
#include "../normaliser.h"
#include "../../MODELS/m_blr_train.h"
#include "../../CORE/config/config.h"

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <iostream>


template
<class BF=DM_Function_Base&>
class M_BLR: public M_Tadah_Base, public M_BLR_Train<BF> {

    public:

        /** This constructor will preapare this object for either training
         *  or prediction (if potential is provides as a Config)
         *
         * Usage example:
         *
         * \code{.cpp}
         * Config config("Config");
         * M_BLR<BF_Linear> blr(config);
         * \endcode
         *
         */
        using M_BLR_Train<BF>::config;
        using M_BLR_Train<BF>::bf;
        M_BLR(Config &c):
            M_BLR_Train<BF>(c),
            desmat(M_BLR_Train<BF>::bf,c)
    {
        norm = Normaliser(c);
    }

        /** This constructor will preapare this object for either training
         *  or prediction (if potential is provides as a Config)
         *
         * Usage example:

         * \code{.cpp}
         * Config config("Config");
         * BF_Linear bf(config);
         * M_BLR<> blr(bf, config);
         * \endcode
         *
         */
        M_BLR(BF &bf, Config &c):
            M_BLR_Train<BF>(bf,c),
            desmat(bf,c)
    {
        norm = Normaliser(c);
    }

        double epredict(const aed_type2 &aed) {
            return bf.epredict(weights,aed);
        };

        double fpredict(const fd_type &fdij, const aed_type2 &aedi, const size_t k) {
            return bf.fpredict(weights,fdij,aedi,k);
        }

        force_type fpredict(const fd_type &fdij, const aed_type2 &aedi) {
            return bf.fpredict(weights,fdij,aedi);
        }

        void train(StDescriptorsDB &st_desc_db, const StructureDB &stdb) {

            if(config.template get<bool>("NORM"))
                norm = Normaliser(config,st_desc_db);

            desmat.build(st_desc_db,stdb);
            train(desmat);
        }

        void train(StructureDB &stdb, DC_Base &dc) {

            if(config.template get<bool>("NORM")) {

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
                    // norm.normalise(st_desc_db_temp);
                }

                config.remove("FORCE");
                config.remove("STRESS");
                config.add("FORCE", force);
                config.add("STRESS", stress);
            }

            desmat.build(stdb,norm,dc);
            train(desmat);
        }

        Structure predict(const Config &c, StDescriptors &std, const Structure &st) {
            if(config.template get<bool>("NORM") && !std.normalised && bf.get_label()!="BF_Linear")
                norm.normalise(std);
            return M_Tadah_Base::predict(c,std,st);
        }

        StructureDB predict(Config &c, const StructureDB &stdb, DC_Base &dc) {
            return M_Tadah_Base::predict(c,stdb,dc);
        }

        Config get_param_file() {
            Config c = config;
            //c.remove("ALPHA");
            //c.remove("BETA");
            c.remove("DBFILE");
            c.remove("FORCE");
            c.remove("STRESS");
            c.remove("VERBOSE");

            c.clear_internal_keys();
            c.remove("MODEL");
            c.add("MODEL", label);
            c.add("MODEL", bf.get_label());

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
            return c;
        }
        StructureDB predict(Config config_pred, StructureDB &stdb, DC_Base &dc,
                aed_type2 &predicted_error) {

            LinearRegressor::read_sigma(config_pred,Sigma);
            DesignMatrix<BF> dm(bf,config_pred);
            dm.scale=false; // do not scale energy, forces and stresses
            dm.build(stdb,norm,dc);


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
        std::string label="M_BLR";
        DesignMatrix<BF> desmat;

        // normalise weights such that when predict is called
        // we can supply it with a non-normalised descriptor
        t_type convert_to_nweights(const t_type &weights) const {
            if(bf.get_label()!="BF_Linear") {
                throw std::runtime_error("Cannot convert weights to nweights for\n\
                        non linear basis function\n");
            }
            t_type nw(weights.rows());
            nw(0) = weights(0);
            for (size_t i=1; i<weights.size(); ++i) {

                if (norm.std_dev[i] > std::numeric_limits<double>::min())
                    nw(i) = weights(i) / norm.std_dev[i];
                else
                    nw(i) = weights(i);

                nw(0) -= norm.mean[i]*nw(i);

            }
            return nw;
        }
        // The opposite of convert_to_nweights()
        t_type convert_to_weights(const t_type &nw) const {
            if(bf.get_label()!="BF_Linear") {
                throw std::runtime_error("Cannot convert nweights to weights for\n\
                        non linear basis function\n");
            }
            // convert normalised weights back to "normal"
            t_type w(nw.rows());
            w(0) = nw(0);
            for (size_t i=1; i<nw.size(); ++i) {
                if (norm.std_dev[i] > std::numeric_limits<double>::min())
                    w(i) = nw(i) * norm.std_dev[i];
                else
                    w(i) = nw(i);

                w(0) += nw(i)*norm.mean[i];
            }
            return w;
        }

        template <typename D>
            void train(D &desmat) {
                phi_type &Phi = desmat.Phi;
                t_type &T = desmat.T;
                M_BLR_Train<BF>::train(Phi,T);

                if (config.template get<bool>("NORM") &&
                        bf.get_label()=="BF_Linear") {
                    weights = convert_to_nweights(weights);
                }
            }

        // Do we want to confuse user with those and make them public?
        // Either way they must be stated as below to silence clang warning
        using M_BLR_Train<BF>::predict;
        using M_BLR_Train<BF>::train;
        using M_BLR_Train<BF>::trained;
        using M_BLR_Train<BF>::weights;
        using M_BLR_Train<BF>::Sigma;
};
#endif
