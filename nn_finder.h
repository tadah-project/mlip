#ifndef NN_FINDER_H
#define NN_FINDER_H

#include "../CORE/config/config.h"
#include "structure.h"
#include "structure_db.h"

/** Nearest Neighbour Finder
 *
 * Construct a full nearest neighbour list
 * for every atom in a structure.
 *
 * The lists are stored with a Structure object provided.
 *
 * The cutoff used is  max(\ref RCUT2B,\ref RCUT3B,\ref RCUTMB)
 */
class NNFinder {
    private:
        double cutoff_sq;
        double cutoff;
        /** Return false if cutoff is larger than
         * one of cell dimensions.
         */
        bool check_box(Structure &st);
        void num_shifts(Structure &st, int N[3]);
    public:
        /** Constructor to initalise this object
         *
         *  Required keys: at least one of:
         *  \ref RCUT2B,\ref RCUT3B,\ref RCUTMB
         */
        NNFinder(Config &config);

        /** \brief Find nearest neighbours for all atoms in a Structure */
        void calc(Structure &st);

        /** \brief Find nearest neighbours for all atoms in all Structure(s) */
        void calc(StructureDB &stdb);
};
#endif
