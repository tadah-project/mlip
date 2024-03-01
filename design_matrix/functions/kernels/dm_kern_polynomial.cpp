#include "dm_kern_polynomial.h"

Registry<DM_Function_Base>::Register<DM_Kern_Polynomial> DM_Kern_Polynomial_1( "Kern_Polynomial" );
Registry<DM_Function_Base,Config&>::Register<DM_Kern_Polynomial> DM_Kern_Polynomial_2( "Kern_Polynomial" );

DM_Kern_Polynomial::DM_Kern_Polynomial():
    Kern_Polynomial()
{}
DM_Kern_Polynomial::DM_Kern_Polynomial(const Config &c):
    Kern_Polynomial(c)
{}
