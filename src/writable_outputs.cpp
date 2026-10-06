#include <tadah/mlip/output/writable_outputs.h>

#include <filesystem>
#include <stdexcept>
#include <system_error>

#include <unistd.h>

namespace tadah {
namespace mlip {

namespace {
namespace fs = std::filesystem;

/* The file that writing to @p p writes: @p p, or, when it is a symbolic link,
 * the end of its chain of links (which need not exist yet). */
fs::path write_target(fs::path p) {
  for (int hops = 0; hops < 40; ++hops) {
    std::error_code ec;
    if (!fs::is_symlink(fs::symlink_status(p, ec)) || ec) return p;
    const fs::path next = fs::read_symlink(p, ec);
    if (ec) return p;
    p = next.is_absolute() ? next : p.parent_path() / next;
  }
  return p;  // a loop of links: fs::status() reports it below
}
}  // namespace

void require_writable_outputs(const std::vector<std::string> &paths,
                              const std::string &who) {
  for (const auto &path : paths) {
    const auto refuse = [&](const std::string &why) {
      return std::runtime_error(who + ": cannot write the output file '" + path +
                                "': " + why + "; nothing was written");
    };
    const fs::path target = write_target(path);
    std::error_code ec;
    const fs::file_status st = fs::status(target, ec);
    if (st.type() == fs::file_type::none) throw refuse(ec.message());
    if (fs::exists(st)) {
      if (fs::is_directory(st)) throw refuse("it is a directory");
      if (::access(target.c_str(), W_OK) != 0) throw refuse("it is not writable");
      continue;
    }
    // A new file: its directory must exist and take new files.
    const fs::path dir = target.parent_path().empty() ? fs::path(".") : target.parent_path();
    if (!fs::is_directory(dir, ec))
      throw refuse("the directory '" + dir.string() + "' does not exist");
    if (::access(dir.c_str(), W_OK | X_OK) != 0)
      throw refuse("the directory '" + dir.string() + "' is not writable");
  }
}

}  // namespace mlip
}  // namespace tadah
