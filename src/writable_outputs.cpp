#include <tadah/mlip/output/writable_outputs.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>

namespace tadah {
namespace mlip {

void require_writable_outputs(const std::vector<std::string> &paths,
                              const std::string &who) {
  namespace fs = std::filesystem;
  for (const auto &path : paths) {
    std::error_code ec;
    const bool existed = fs::exists(path, ec);
    {
      std::ofstream probe(path, std::ios::app);
      if (!probe.is_open()) {
        const fs::path parent = fs::path(path).parent_path();
        const std::string why =
            !parent.empty() && !fs::is_directory(parent, ec)
                ? "the directory '" + parent.string() + "' does not exist"
                : "it cannot be opened for writing";
        throw std::runtime_error(who + ": cannot write the output file '" + path +
                                 "': " + why + "; nothing was written");
      }
    }
    if (!existed) fs::remove(path, ec);
  }
}

}  // namespace mlip
}  // namespace tadah
