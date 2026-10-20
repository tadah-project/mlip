#include <tadah/mlip/data_weights.h>

#include <tadah/core/number_format.h>
#include <tadah/mlip/detail/cell_geometry.h>

#include <stdexcept>
#include <vector>

namespace tadah {
namespace mlip {

std::string structure_in_data(const StructureDB &stdb, std::size_t s,
                              const tadah::core::Context *context) {
  std::string name = "structure " + std::to_string(s + 1) + " of " +
                     std::to_string(stdb.size());
  if (s < stdb.size() && !stdb(s).label.empty())
    name += " ('" + stdb(s).label + "')";
  const std::size_t n_datasets =
      stdb.dbidx.size() > 1 ? stdb.dbidx.size() - 1 : 0;
  for (std::size_t d = 0; d < n_datasets; ++d) {
    if (s < stdb.dbidx[d] || s >= stdb.dbidx[d + 1]) continue;
    name += " of dataset " + std::to_string(d + 1);
    if (context && context->has_stored("DBFILE") &&
        context->size("DBFILE") == n_datasets)
      name += " (" + (*context)("DBFILE")[d] + ")";
    break;
  }
  return name;
}

void require_valid_weights(const StructureDB &stdb,
                           const tadah::core::Context &context) {
  auto valid = [](double w) { return detail::is_finite_bits(w) && w >= 0.0; };
  const char *what[3] = {"energy", "forces", "stress"};
  for (std::size_t s = 0; s < stdb.size(); ++s) {
    const double w[3] = {stdb(s).eweight, stdb(s).fweight, stdb(s).sweight};
    const char *name[3] = {"eweight", "fweight", "sweight"};
    for (int k = 0; k < 3; ++k) {
      if (valid(w[k])) continue;
      throw std::runtime_error(
          std::string(name[k]) + " " + tadah::core::format_roundtrip(w[k]) +
          " of " + structure_in_data(stdb, s, &context) +
          ": a weight of the training data is a finite number >= 0 (0 "
          "leaves the structure's " + what[k] + " out of the fit); no "
          "weights were fitted.");
    }
  }
  for (int k = 0; k < 3; ++k) {
    const char *key[3] = {"EWEIGHT", "FWEIGHT", "SWEIGHT"};
    const double w = context.get<double>(key[k]);
    if (valid(w)) continue;
    throw std::runtime_error(
        std::string(key[k]) + " " + tadah::core::format_roundtrip(w) +
        ": a weight is a finite number >= 0 (0 leaves the " + what[k] +
        " of every structure out of the fit); no weights were fitted.");
  }
}

std::string krr_implementation2_ignores(const std::string &kernel,
                                        const std::string &what,
                                        const std::string &fits,
                                        const std::string &instead) {
  return "MODEL M_KRR " + kernel + " 2 fits the energies per atom of whole "
         "structures, without forces, stresses or weights: " + what +
         " would be ignored. Implementation 1 (MODEL M_KRR " + kernel +
         " 1) " + fits + "; implementation 2 fits " + instead + ".";
}

void refuse_what_krr_implementation2_ignores(
    const tadah::core::Context &context, const StructureDB &stdb) {
  const std::string k = context.get<std::string>("MODEL", 1);
  auto given = [&](const std::string &key) {
    std::string t = key;
    for (const std::string &v : context(key)) t += " " + v;
    return t;
  };
  auto refuse = [&](const std::string &what, const std::string &fits,
                    const std::string &instead) {
    throw std::runtime_error(krr_implementation2_ignores(k, what, fits,
                                                         instead));
  };
  if (context.get<bool>("FORCE"))
    refuse(given("FORCE"), "fits the forces", "with FORCE false");
  if (context.get<bool>("STRESS"))
    refuse(given("STRESS"), "fits the stresses", "with STRESS false");
  for (const char *key : {"EWEIGHT", "FWEIGHT", "SWEIGHT"})
    if (context.get<double>(key) != 1.0)
      refuse(given(key), "applies it", std::string("with ") + key + " 1");
  if (context.has_stored("WDBFILE")) {
    std::vector<double> w(context.size("WDBFILE"));
    context.get("WDBFILE", w);
    for (double v : w)
      if (v != 1.0) refuse(given("WDBFILE"), "applies it", "without WDBFILE");
  }
  if (context.contains("WDBFILE_AUTO") &&
      context.get<double>("WDBFILE_AUTO") != 0.0)
    refuse(given("WDBFILE_AUTO"), "applies it", "with WDBFILE_AUTO 0");
  if (context.has_stored("EWEIGHT_TEMP"))
    refuse(given("EWEIGHT_TEMP"), "applies it", "without EWEIGHT_TEMP");
  for (std::size_t s = 0; s < stdb.size(); ++s)
    if (stdb(s).eweight != 1.0)
      refuse("eweight " + tadah::core::format_roundtrip(stdb(s).eweight) +
                 " of " + structure_in_data(stdb, s, &context),
             "weights each structure's energy (and leaves out one of "
             "eweight 0)",
             "when every structure has eweight 1");
}

}  // namespace mlip
}  // namespace tadah
