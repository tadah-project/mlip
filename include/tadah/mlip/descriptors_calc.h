#ifndef DESCRIPTORS_CALC_H
#define DESCRIPTORS_CALC_H

#include <tadah/mlip/descriptors_calc_base.h>
#include <tadah/mlip/design_matrix/column_block.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/st_descriptors_db.h>
#include <tadah/core/context.h>
#include <tadah/models/descriptors/d2/d2_base.h>
#include <tadah/models/descriptors/dm/dm_base.h>
#include <tadah/models/cutoffs.h>

#include <array>
#include <cstddef>
#include <string>

namespace tadah {
namespace mlip {
/** \brief Descriptors calculator
 *
 * This object does not store any data.
 *
 * Usage example:
 * \code{.cpp}
 * # Build calculator using tadah::core::Context object.
 * DescriptorsCalc<D2_LJ, DM_Dummy> dc(context);
 *
 * # Calculate all descriptors in a Structure st
 * StDescriptors std = dc.calc(st);
 *
 * # Calculate all descriptors in a StructureDB st_db
 * StDescriptorsDB std_db = dc.calc(st_db);
 * \endcode
 *
 *  This object sets the following \ref INTERNAL_KEYS
 *  to the provided tadah::core::Context object:
 *
 *  \ref SIZE2B,\ref SIZE3B,\ref SIZEMB,\ref DSIZE
 *
 *  Descriptor layout of an atom: with \ref BIAS true, N = size(\ref ATOM)
 *  one-hot intercept columns in ATOM order (1.0 in the column of the
 *  atom's species, 0 in the others; core::intercept_columns_by_Z), then
 *  the two-body and the many-body descriptors. It is the layout of
 *  models::DC_Selector (pair_style tadah), so DSIZE equals its
 *  descriptor_size(). With BIAS true an atom of an element ATOM does not
 *  list is refused (std::runtime_error).
 *
 *  Required keys:
 *  \ref INIT2B,\ref INIT3B,\ref INITMB,
 *  \ref RCUT2B,\ref RCUT3B,\ref RCUTMB (for initialised ones)
 *
 *
 *  @tparam D2 D2_Base child, two-body type calculator.
 *  @tparam DM DM_Base child, many-body type calculator.
 *  @tparam C2 Cut_Base child, two-body cutoff function.
 *  @tparam CM Cut_Base child, many-body cutoff function.
 */
template <typename D2=tadah::models::D2_Base&, typename DM=tadah::models::DM_Base&, typename C2=tadah::models::Cut_Base&, typename CM=tadah::models::Cut_Base&>
class DescriptorsCalc: public DC_Base {

    public:
        tadah::core::Context &context;
        /** Constructor to fully initialise this object.
         *
         *  REQUIRED context KEYS:
         *  \ref FORCE, \ref STRESS,
         *  at least one of: \ref RCUT2B, \ref RCUT3B,\ref RCUTMB
         *
         */
        DescriptorsCalc(tadah::core::Context &c);

        /** This constructor is equivalent to the main constructor above.
         *
         * The difference is technical in nature.
         * Here we use OOP with generic programming to call virtual functions
         * to calculate descriptors - this incurs small efficiency penalty
         * but makes selecting of descriptors somewhat easier at runtime.
         *
         * In contrast, the main constructor above is purely generic.
         *
         * The rule of thumb is: Use the main constructor unless it's a pain.
         *
         * Usage example:
         * \code{.cpp}
         * # Build calculator using tadah::core::Context object.
         * D2_Base *d2 = new D2_LJ(context);
         * D2_Base *dm = new DM_Dummy(context);
         * DescriptorsCalc<> dc(context, d2, dm);
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
         *  DescriptorsCalc<> dc(context);
         *  \endcode
         *  Basically you are trying to init abstract classes with context object.
         *  It won't compile!
         * }
         *
         */
        template <typename T1, typename T2, typename T3,typename T4>
        DescriptorsCalc(tadah::core::Context &c, T1 &d2, T2 &dm, T3 &c2, T4 &cm);

        /** Calculate all descriptors in a Structure st */
        StDescriptors calc(const Structure &st) const;

        /** Calculate all descriptors in a StructureDB st_db */
        StDescriptorsDB calc(const StructureDB &st_db) const;

        /** Calculate density vector for the structure */
        void calc_rho(const Structure &st, StDescriptors &std) const;

        /** \brief Cutoff (Å) up to which this calculator reads a
         *  structure's neighbour list: the largest cutoff of its enabled
         *  two-body and many-body descriptors. The list (NNFinder, Structure::nn)
         *  must be built with at least this cutoff; calc() and calc_rho()
         *  refuse a shorter one with std::logic_error, since it would
         *  silently lack pairs. */
        double neighbour_cutoff() const;

        /** \brief Compute the design-matrix column layout for the
         *  descriptors held by this calculator.
         *
         *  Reads BIAS / ATOM / INIT2B / INITMB from the supplied context
         *  and reports per-block label, family, offset, and count in the
         *  same order they are placed in the design matrix:
         *
         *    [bias]  if BIAS=true            -> "bias", family "bias",
         *                                       N = size(ATOM) columns, one
         *                                       intercept per species
         *    [d2]    if INIT2B and d2.size>0 -> d2.label(), family "D2"
         *    [dm]    if INITMB and dm.size>0 -> dm.label(), family "DM"
         *
         *  Offsets match the `set_fidx()` calls done in
         *  `common_constructor()` (descriptors_calc.hpp). The total
         *  column count agrees with `context.get<size_t>("DSIZE")`.
         */
        ColumnLayout layout(const tadah::core::Context &c) const override;

        C2 c2;
        CM cm;
        D2 d2;
        DM dm;
    private:
        template <typename T1, typename T2>
        DescriptorsCalc(tadah::core::Context &c,T1 &t1,T2 &t2);

        void calc(const Structure &st, StDescriptors &std) const;
        void calc_dimer(const Structure &st, StDescriptors &std) const;
        void common_constructor();

        /* BIAS true: the number of intercept columns (N = size(ATOM)) and
         * the intercept column of each Z; set by every constructor. */
        std::size_t n_intercept_ = 0;
        std::array<int, 119> intercept_col_{};
        void init_intercepts();
        /* The intercept column of atom a (BIAS true); throws for an element
         * ATOM does not list, naming `where`. */
        std::size_t intercept_column(const Atom &a, const std::string &where) const;
        /* Refuses a structure with an atom of an element ATOM does not list
         * (BIAS true), naming `where`. */
        void require_intercept_columns(const Structure &st, const std::string &where) const;
};
} // namespace mlip
} // namespace tadah

#include "descriptors_calc.hpp"
#endif
