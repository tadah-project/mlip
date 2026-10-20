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

}  // namespace mlip
}  // namespace tadah
