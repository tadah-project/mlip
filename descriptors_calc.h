#ifndef DESCRIPTORS_CALC_H
#define DESCRIPTORS_CALC_H

#include "descriptors_calc_base.h"
#include "../CORE/config/config.h"
#include "structure.h"
#include "structure_db.h"
#include "st_descriptors.h"
#include "st_descriptors_db.h"
#include "../MODELS/descriptors/d2/d2_base.h"
#include "../MODELS/descriptors/d3/d3_base.h"
#include "../MODELS/descriptors/dm/dm_base.h"
#include "../MODELS/cutoffs/cutoffs.h"

/** \brief Descriptors calculator
 *
 * This object does not store any data.
 *
 * Usage example:
 * \code{.cpp}
 * # Build calculator using Config object.
 * DescriptorsCalc<D2_LJ, D3_Dummy, DM_Dummy> dc(config);
 *
 * # Calculate all descriptors in a Structure st
 * StDescriptors std = dc.calc(st);
 *
 * # Calculate all descriptors in a StructureDB st_db
 * StDescriptorsDB std_db = dc.calc(st_db);
 * \endcode
 *
 *  This object sets the following \ref INTERNAL_KEYS
 *  to the provided Config object:
 *
 *  \ref SIZE2B,\ref SIZE3B,\ref SIZEMB,\ref DSIZE
 *
 *  Required keys:
 *  \ref INIT2B,\ref INIT3B,\ref INITMB,
 *  \ref RCUT2B,\ref RCUT3B,\ref RCUTMB (for initialised ones)
 *
 *
 *  @tparam D2 D2_Base child, two-body type calculator.
 *  @tparam D3 D3_Base child, three-body type calculator.
 *  @tparam DM DM_Base child, many-body type calculator.
 *  @tparam C2 Cut_Base child, two-body cutoff function.
 *  @tparam C3 Cut_Base child, three-body cutoff function.
 *  @tparam CM Cut_Base child, many-body cutoff function.
 */
template <typename D2=D2_Base&, typename D3=D3_Base&, typename DM=DM_Base&, typename C2=Cut_Base&, typename C3=Cut_Base&, typename CM=Cut_Base&>
class DescriptorsCalc: public DC_Base {

    public:
        Config &config;
        /** Constructor to fully initialise this object.
         *
         *  REQUIRED CONFIG KEYS:
         *  \ref FORCE, \ref STRESS,
         *  at least one of: \ref RCUT2B, \ref RCUT3B,\ref RCUTMB
         *
         */
        DescriptorsCalc(Config &c);

        /** This constructor is equivalent to the main constructor above.
         *
         * The difference is technical in nature.
         * Here we use OOP with generic programming to call virtual functions
         * to calculate descriptors - this incurs small efficiency penalty
         * but makes selecting of descriptors somewhat easier at runtime.
         *
         * In contrast, the main constructor above is purely generic.
         *
         * The rule of thumb is: Use the main contructor unless it's a pain.
         *
         * Usage example:
         * \code{.cpp}
         * # Build calculator using Config object.
         * D2_Base *d2 = new D2_LJ(config);
         * D2_Base *d3 = new D3_Dummy(config);
         * D2_Base *dm = new DM_Dummy(config);
         * DescriptorsCalc<> dc(config, d2, d3, dm);
         *
         * # Calculate all descriptors in a Structure st
         * StDescriptors std = dc.calc(st);
         *
         * # Calculate all descriptors in a StructureDB st_db
         * StDescriptorsDB std_db = dc.calc(st_db);
         * \endcode
         *
         * \warning {
         *  The unfortunate consequence is that
         *  you might want to try this
         *  \code{.cpp}
         *  DescriptorsCalc<> dc(config);
         *  \endcode
         *  Basically you are trying to init abstract classes with config object.
         *  It won't compile!
         * }
         *
         */
        template <typename T1, typename T2, typename T3,typename T4, typename T5, typename T6>
        DescriptorsCalc(Config &c, T1 &d2, T2 &d3, T3 &dm, T4 &c2, T5 &c3, T6 &cm);

        /** Calculate all descriptors in a Structure st */
        StDescriptors calc(const Structure &st);

        /** Calculate all descriptors in a StructureDB st_db */
        StDescriptorsDB calc(const StructureDB &st_db);


        /** Calculate density vector for the structure */
        void calc_rho(const Structure &st, StDescriptors &std);

        C2 c2;
        C3 c3;
        CM cm;
        D2 d2;
        D3 d3;
        DM dm;
    private:
        template <typename T1, typename T2, typename T3>
        DescriptorsCalc(Config &c,T1 &t1,T2 &t2,T3 &t3);

        void calc(const Structure &st, StDescriptors &std);
        void calc_dimer(const Structure &st, StDescriptors &std);
        void common_constructor();

};

#include "descriptors_calc.hpp"
#endif
