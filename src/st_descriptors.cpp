#include <tadah/mlip/st_descriptors.h>
namespace tadah {
namespace mlip {

StDescriptors::StDescriptors(const Structure &s, const tadah::core::Context &c):
    // fully initialize aed
    aeds(c.get<size_t>("DSIZE"), s.natoms()),
    // partially init fd
    fd(c.get<bool>("FORCE")  || c.get<bool>("STRESS") ? s.natoms() : 0)
    // fully initialize sd
    //sd(c.get<bool>("STRESS") ? c.get<size_t>("DSIZE"),6 : 0,6)
{
    // fd: we still need to resize nn individually for each atom
    // and init tadah::core::fd_type with size
    if (c.get<bool>("FORCE") || c.get<bool>("STRESS")) {
        for (size_t i=0; i<fd.size(); ++i) {
            fd[i].resize(s.nn_size(i),tadah::core::fd_type(c.get<size_t>("DSIZE")));
            //for (auto &v:fd[i]) v.set_zero();
        }
    }

}
StDescriptors::StDescriptors() {}

tadah::core::aed_type & StDescriptors::get_aed(const size_t i) {
    return aeds.col(i);
}

const tadah::core::aed_type &StDescriptors::get_aed(const size_t i) const {
    return aeds.col(i);
}
tadah::core::rho_type& StDescriptors::get_rho(const size_t i) {
    return rhos.col(i);
}
size_t StDescriptors::naed() const {
    return aeds.cols();
}
size_t StDescriptors::dim() const {
    return aeds.rows();
}
} // namespace mlip
} // namespace tadah
