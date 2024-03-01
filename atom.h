#ifndef ATOM_H
#define ATOM_H

#include <string>
#include <iostream>
#include "../CORE/element.h"
#include "../CORE/core_types.h"

/**
 * Container to represent atom properties
 *
 * Usage example:
 *
 *     # create an empty atom object - all attributes are left uninitialised
 *     Atom atom;
 *
 *     # set atom symbol
 *     atom.symbol="Ti"
 *
 *     # set atom name, atomic number... see Element class for attributes
 *
 *     # set atom position to (1.1, 2.2, 3.3)
 *     atom.position(0) = 1.1;
 *     atom.position(1) = 2.2;
 *     atom.position(2) = 3.3;
 *
 *     # set force
 *     atom.force(0)= 0.1;
 *     atom.force(1)= -0.2;
 *     atom.force(2)= 0.3;
 *
 * Usage example:
 *
 *     # Use constructor to fully initialise this object
 *     # with the position and force as in the example above
 *     Element element = PeriodicTable().find_by_symbol("Ti");
 *     Atom atom(element, 1.1, 2.2, 3.3, 0.1, -0.2, 0.3);
 *
 * Usage example:
 *
 *     # Print atom object using streams:
 *     std::cout << atom;
 *
 *     # Print position only
 *     std::cout << atom.position
 *
 * @see Structure
 */
struct Atom: public Element {
    /**
     * Create an empty atom object. All class attributes are left uninitialised.
     */
    Atom();

    /** This constructor fully initialise this object
     *
     * @param[in] element   Chemical Element
     * @param[in] px,py,pz  Atomic coordinates
     * @param[in] fx,fy,fz  Force acting on the atom
     *
     */
    Atom(const Element &element,
            const double px, const double py, const double pz,
            const double fx, const double fy, const double fz);

    /** Hold position of the atom. */
    Vec3d position;

    /** Hold force on the atom. */
    Vec3d force;

    /** Print object summary to the stream */
    friend std::ostream& operator<<(std::ostream& os, const Atom& atom);

    /** Return true if both atoms are the same.
     *
     * This operator compares chemical type, position and force.
     */
    bool operator==(const Atom &) const;

};
#endif
