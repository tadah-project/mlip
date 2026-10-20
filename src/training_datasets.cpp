#include <tadah/mlip/training_datasets.h>

#include <cstddef>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
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

std::string potential_has_no_datasets_advice() {
  return "A potential has no DBFILE line (no potential tadah writes one; "
         "its training record, the comment block above its key lines, lists "
         "the datasets its weights were fitted to in a '# DBFILE' line when "
         "there were any: a refit has none, and a potential of an earlier "
         "version has no record): to train from one, add a DBFILE line to a "
         "copy of it; a potential of tadah refit or refit --edit also pins "
         "every weight with its FIXINDEX and FIXWEIGHT lines, the seed of a "
         "tadah hpo coefficient search, so delete them from the copy for "
         "the fit to change the weights (the Training page of the "
         "documentation, Retraining from a potential).";
}

void require_training_datasets(const tadah::core::Context &ctx) {
  if (ctx.has_stored("DBFILE")) return;
  throw std::runtime_error(
      "The training configuration has no DBFILE: tadah train fits the "
      "datasets it lists. " +
      potential_has_no_datasets_advice());
}

std::optional<std::pair<std::size_t, std::size_t>> repeated_dataset(
    const std::vector<std::string> &files) {
  for (std::size_t j = 1; j < files.size(); ++j)
    for (std::size_t i = 0; i < j; ++i)
      if (same_dataset(files[i], files[j])) return std::make_pair(i, j);
  return std::nullopt;
}

void require_distinct_training_datasets(const tadah::core::Context &ctx) {
  if (!ctx.has_stored("DBFILE")) return;
  const std::vector<std::string> &files = ctx("DBFILE");
  const auto twice = repeated_dataset(files);
  if (!twice) return;
  const auto [i, j] = *twice;
  throw std::runtime_error(
      "DBFILE lists one dataset twice: entry " + std::to_string(i + 1) +
      " (" + files[i] + ") and entry " + std::to_string(j + 1) + " (" +
      files[j] + ") are one file, so each of its structures would be "
      "fitted twice. List each dataset once; to give a dataset a higher "
      "priority, use WDBFILE. An M_BLR potential written by an earlier "
      "version of Tadah! records the DBFILE it was trained on: remove that "
      "line from a potential used as the training configuration.");
}

}  // namespace mlip
}  // namespace tadah
