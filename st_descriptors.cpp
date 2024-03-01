#include "st_descriptors.h"

StDescriptors::StDescriptors(const Structure &s, const Config &c):
    // fully initialize aed
    aeds(c.get<size_t>("DSIZE"), s.natoms()),
    // partially init fd
    fd(c.get<bool>("FORCE")  || c.get<bool>("STRESS") ? s.natoms() : 0)
    // fully initialize sd
    //sd(c.get<bool>("STRESS") ? c.get<size_t>("DSIZE"),6 : 0,6)
{
    // fd: we still need to resize nn individually for each atom
    // and init fd_type with size
    if (c.get<bool>("FORCE") || c.get<bool>("STRESS")) {
        for (size_t i=0; i<fd.size(); ++i) {
            fd[i].resize(s.nn_size(i),fd_type(c.get<size_t>("DSIZE")));
            //for (auto &v:fd[i]) v.set_zero();
        }
    }

    //if (c.get<bool>("STRESS")) {
    //    sd.set_zero();
    //}

    // Partially init rho. We do not know the dimension of rhoi
    // as it is specific to DMB calculator
    // so DescriptorCalc will have to resize
    if (c.get<size_t>("SIZEMB"))
        rhos.resize(1,s.natoms());


}
StDescriptors::StDescriptors() {}

aed_type2 & StDescriptors::get_aed(const size_t i) {
    return aeds.col(i);
}

const aed_type2 &StDescriptors::get_aed(const size_t i) const {
    return aeds.col(i);
}
rho_type& StDescriptors::get_rho(const size_t i) {
    return rhos.col(i);
}
size_t StDescriptors::naed() const {
    return aeds.cols();
}
size_t StDescriptors::dim() const {
    return aeds.rows();
}
