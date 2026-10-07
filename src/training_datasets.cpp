#include <tadah/mlip/training_datasets.h>

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace tadah {
namespace mlip {

namespace {

/* Whether the DBFILE entries a and b name one file: the same file for two
 * existing paths, else the same absolute normal path. */
bool same_dataset(const std::string &a, const std::string &b) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const bool both_exist = fs::exists(a, ec) && fs::exists(b, ec);
  if (both_exist) {
    const bool eq = fs::equivalent(a, b, ec);
    if (!ec) return eq;
  }
  ec.clear();
  const fs::path pa = fs::absolute(a, ec).lexically_normal();
  if (ec) return a == b;
  const fs::path pb = fs::absolute(b, ec).lexically_normal();
  if (ec) return a == b;
  return pa == pb;
}

}  // namespace

void remove_training_datasets(tadah::core::Context &pot) {
  if (pot.has_stored("DBFILE")) pot.remove("DBFILE");
}

void require_distinct_training_datasets(const tadah::core::Context &ctx) {
  if (!ctx.has_stored("DBFILE")) return;
  const std::vector<std::string> &files = ctx("DBFILE");
  for (std::size_t j = 1; j < files.size(); ++j)
    for (std::size_t i = 0; i < j; ++i)
      if (same_dataset(files[i], files[j]))
        throw std::runtime_error(
            "DBFILE lists one dataset twice: entry " + std::to_string(i + 1) +
            " (" + files[i] + ") and entry " + std::to_string(j + 1) + " (" +
            files[j] + ") are one file, so each of its structures would be "
            "fitted twice. List each dataset once; to give a dataset more "
            "weight, use WDBFILE. An M_BLR potential written by Tadah! 1.5.x "
            "or earlier, or by a 1.6.0 development build, records the DBFILE "
            "it was trained on: remove that line from a potential used as the "
            "training configuration.");
}

}  // namespace mlip
}  // namespace tadah
