#include "catch2/catch.hpp"
#include <tadah/mlip/atom.h>
#include <string>


char symbol1[]="Ti";
int Z1 = 22;
Vec3d pos1(1.0,2.0,3.0);
Vec3d force1(4.0,5.0,6.0);
Element element1(symbol1,Z1);

char symbol2[]="Nb";
int Z2 = 41;
Vec3d pos2(-1.0,-2.0,-3.0);
Vec3d force2(-4.0,-5.0,-6.0);
Element element2(symbol2,Z2);

TEST_CASE( "Testing Atom class constructor", "[atom]" ) {


    // Trivial case 1/2
    Atom a(element1, 1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0);

    REQUIRE( pos1 == a.position );
    REQUIRE( force1 == a.force );
    REQUIRE( symbol1[0] == a.symbol[0] );
    REQUIRE( symbol1[1] == a.symbol[1] );
    REQUIRE( Z1 == a.Z );

    // Trivial case 2/2
    Atom b;
    b.position = pos1;
    b.force = force1;
    b.symbol[0] = symbol1[0];
    b.symbol[1] = symbol1[1];
    b.Z = Z1;

    REQUIRE( pos1 == b.position );
    REQUIRE( force1 == b.force );
    REQUIRE( symbol1[0] == b.symbol[0] );
    REQUIRE( symbol1[1] == b.symbol[1] );
    REQUIRE( Z1 == b.Z );

    REQUIRE( a == b );
}
TEST_CASE( "Testing Atom operator==", "[atom_operator==]" ) {
    Atom a(element1, 1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0);
    Atom b(element1, 1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0);
    REQUIRE(a==b);

    Atom c(element2, -1.0, -2.0, -3.0, 
            -4.0, -5.0, -6.0);
    REQUIRE(!(a==c));
}
TEST_CASE( "Testing Atom copy", "[atom_copy]" ) {
    Atom a(element1, 1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0);
    Atom b=a;
    REQUIRE(a==b);
}
