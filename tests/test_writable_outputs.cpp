/* require_writable_outputs (output/writable_outputs.h): the check a data
 * command that writes several files makes before writing any. */
#include "tadah_test_skip.hpp"
#include "catch2/catch.hpp"

#include <tadah/mlip/output/writable_outputs.h>

#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

using Catch::Matchers::Contains;
using tadah::mlip::require_writable_outputs;

namespace {

namespace fs = std::filesystem;

/* A scratch directory, unique to this process and test case, removed on
 * exit. */
struct ScratchDir {
  fs::path dir;
  explicit ScratchDir(const std::string &stem)
      : dir(fs::temp_directory_path() /
            ("tadah_writable_outputs_" + stem + "_" + std::to_string(::getpid()))) {
    fs::remove_all(dir);
    fs::create_directories(dir);
  }
  ~ScratchDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  ScratchDir(const ScratchDir &) = delete;
  ScratchDir &operator=(const ScratchDir &) = delete;
};

std::string slurp(const fs::path &p) {
  std::ifstream is(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>()};
}

}  // namespace

/* Pins: the check changes nothing on disk:
 *  - an output that is a symbolic link whose target does not exist yet, in a
 *    directory that does, passes; the link stays a link and its target is
 *    not created (the command then writes through the link);
 *  - such a link into a missing directory is refused, naming the output and
 *    that directory;
 *  - an output that is a directory is refused, naming it;
 *  - a new file in an existing directory passes and is not created;
 *  - an existing file passes, unchanged.
 * Before the fix: the check created a missing output through its path and
 * removed it again, so the target of the link was created (empty) and the
 * link removed, and the data then went to a regular file in the link's
 * place; a link into a missing directory and a directory were refused with
 * 'it cannot be opened for writing'. */
TEST_CASE("require_writable_outputs changes nothing on disk",
          "[writable_outputs][data]") {
  ScratchDir tmp("disk");
  const fs::path target_dir = tmp.dir / "target";
  fs::create_directories(target_dir);
  const fs::path link = tmp.dir / "out.tadah";
  fs::create_symlink(target_dir / "out.tadah", link);
  const fs::path existing = tmp.dir / "existing.tadah";
  { std::ofstream(existing) << "S1\n"; }
  const fs::path fresh = tmp.dir / "fresh.tadah";

  REQUIRE_NOTHROW(require_writable_outputs(
      {link.string(), existing.string(), fresh.string()}, "tadah data split"));
  CHECK(fs::is_symlink(fs::symlink_status(link)));
  CHECK_FALSE(fs::exists(target_dir / "out.tadah"));
  CHECK(slurp(existing) == "S1\n");
  CHECK_FALSE(fs::exists(fs::symlink_status(fresh)));

  const fs::path into_missing = tmp.dir / "lost.tadah";
  fs::create_symlink(tmp.dir / "nodir" / "lost.tadah", into_missing);
  CHECK_THROWS_WITH(require_writable_outputs({into_missing.string()}, "tadah data split"),
                    Contains("lost.tadah") && Contains("nodir") &&
                        Contains("tadah data split"));
  CHECK(fs::is_symlink(fs::symlink_status(into_missing)));

  const fs::path dir = tmp.dir / "adir";
  fs::create_directories(dir);
  CHECK_THROWS_WITH(require_writable_outputs({dir.string()}, "tadah data split"),
                    Contains("adir") && Contains("is a directory"));
}

/* Pins: the check does not open an output that is a FIFO (named pipe): with
 * no reader, it returns at once in a child process (opening the FIFO for
 * writing would block until a reader came, and closing it would end that
 * reader's input before the data was written).
 * Before the fix: the check opened every existing output for appending, so
 * the child blocked and was killed after 10 s. */
TEST_CASE("require_writable_outputs does not open a FIFO",
          "[writable_outputs][data]") {
  ScratchDir tmp("fifo");
  const std::string fifo = (tmp.dir / "pipe").string();
  if (::mkfifo(fifo.c_str(), 0600) != 0) TADAH_SKIP("mkfifo is not available here");
  const pid_t pid = ::fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    try {
      require_writable_outputs({fifo}, "tadah data split");
    } catch (...) {
      ::_exit(2);
    }
    ::_exit(0);
  }
  int status = 0;
  pid_t done = 0;
  for (int k = 0; k < 100 && done == 0; ++k) {
    done = ::waitpid(pid, &status, WNOHANG);
    if (done == 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (done == 0) {
    ::kill(pid, SIGKILL);
    ::waitpid(pid, &status, 0);
  }
  INFO((done == 0 ? "the check blocked: it opened the FIFO" : "the check returned"));
  CHECK(done == pid);
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 0);
}
