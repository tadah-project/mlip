// TaskFactory and the keys of other commands in a configuration file.
//
// A key that only some commands read (an option of their command line such
// as ANALYTICS of tadah predict or INDEX of tadah data print, or a key of
// their configuration file such as REFIT_MODE of tadah refit) is no key of a
// model or of a training configuration. A configuration file given with -c
// to a command that does not read it refuses its line, naming the key, the
// line, the file and the commands that read it; the command that reads it
// takes it. A potential (a file with WEIGHTS) written by an earlier version
// may hold such a line: it is not read, with a notice, and the potential
// loads.

#include "catch2/catch.hpp"

#include <tadah/core/key_notice.h>
#include <tadah/core/logger.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/task_factory.h>

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using KV = std::unordered_map<std::string, std::vector<std::string>>;

/* A programmatic task input: a task name and its options. */
struct NamedInput : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  NamedInput(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* TaskFactory stores the blueprint default VERBOSE 1 into each task's
 * context, which sets the global log level: put it back afterwards. */
struct LogLevelGuard {
  tadah::core::LogLevel saved = tadah::core::Logger::instance().getLogLevel();
  ~LogLevelGuard() { tadah::core::Logger::instance().setLogLevel(saved); }
};

/* A scratch directory, removed on exit. */
struct Scratch {
  fs::path dir;
  Scratch() {
    dir = fs::temp_directory_path() /
          ("tadah_command_keys_" + std::to_string(::getpid()) + "_" +
           std::to_string(std::chrono::steady_clock::now()
                              .time_since_epoch()
                              .count()));
    fs::create_directories(dir);
  }
  ~Scratch() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string write(const std::string &name, const std::string &text) const {
    const std::string path = (dir / name).string();
    std::ofstream(path) << text;
    return path;
  }
};

/* The lines of a small training configuration before the line under
 * test, which is line 3. */
const char *const kHead = "RCUT2B 5.0\nATOM Ta\n";

/* What TaskFactory makes of `tadah <task> -c <config>`: the message of the
 * exception it throws, or "" and the task's context in @p ctx. */
std::string read_config(const std::string &task, const std::string &config,
                        tadah::core::Context *ctx = nullptr) {
  LogLevelGuard guard;
  try {
    auto tasks = tadah::mlip::TaskFactory().createTasksFromInput(
        NamedInput(task, KV{{"config", {config}}}));
    if (ctx && !tasks.empty()) *ctx = tasks.front().getContext();
    return "";
  } catch (const std::exception &e) {
    return e.what();
  }
}

}  // namespace

/* Pins: a configuration file of tadah train, tadah hpo or tadah analysis
 * check_bounds with a line of a key of another command is refused at that
 * line, naming the key, the line, the file, the command that reads the key
 * and the command that reads the file: "Key ANALYTICS on line 3 of
 * analytics.cfg is not accepted here: ANALYTICS is the option --analytics
 * (-A) of tadah predict, not a key of a configuration file of tadah
 * train", and likewise REFIT_MODE (a key of tadah refit) and INDEX (an
 * option of tadah analysis bfunc, ... and tadah data write).
 * Before the fix: each file was read and the key stored (tadah train wrote
 * it into the potential as a key line, where it changed nothing). */
TEST_CASE("TaskFactory refuses a key of another command in a configuration "
          "file",
          "[task_factory][command_keys]") {
  Scratch tmp;
  struct Case {
    const char *line;
    const char *key;
    const char *owner;
  };
  for (const Case &c : {Case{"ANALYTICS true", "ANALYTICS", "tadah predict"},
                        Case{"REFIT_MODE dense", "REFIT_MODE", "tadah refit"},
                        Case{"INDEX 1,3", "INDEX", "tadah data print"}}) {
    const std::string file =
        tmp.write(std::string(c.key) + ".cfg", std::string(kHead) + c.line + "\n");
    for (const auto &[task, command] :
         {std::pair<std::string, std::string>{"TRAIN", "tadah train"},
          {"HPO", "tadah hpo"},
          {"ANALYSIS.check_bounds", "tadah analysis check_bounds"}}) {
      DYNAMIC_SECTION(c.line << " in a configuration file of " << command) {
        const std::string what = read_config(task, file);
        INFO(what);
        CHECK(what.find("Key " + std::string(c.key) + " on line 3 of " +
                        file + " is not accepted here") != std::string::npos);
        CHECK(what.find(c.owner) != std::string::npos);
        CHECK(what.find("a configuration file of " + command) !=
              std::string::npos);
      }
    }
  }
}

/* Pins: the command that reads a key takes it in its configuration file:
 * tadah refit -c with REFIT_MODE dense reads it (the task's REFIT_MODE is
 * dense).
 * Before the fix: passes as well (a pin). */
TEST_CASE("TaskFactory takes a key of the command in its configuration file",
          "[task_factory][command_keys]") {
  Scratch tmp;
  const std::string file =
      tmp.write("refit.cfg", "EAM Ta.eam.alloy\nREFIT_MODE dense\n");
  tadah::core::Context ctx;
  const std::string what = read_config("REFIT", file, &ctx);
  INFO(what);
  REQUIRE(what.empty());
  CHECK(ctx.get<std::string>("REFIT_MODE") == "dense");
}

/* Pins: a potential (a file whose lines give WEIGHTS) with a line of a key
 * of another command, as tadah train of an earlier version wrote it, is
 * read by tadah train -c without that line, and one notice names the key,
 * its line, the file and the command that reads it ("Key ANALYTICS is
 * ignored: ANALYTICS true on line 4 of old_pot.tadah ... tadah predict
 * ..."); the potential's other keys are read.
 * Before the fix: ANALYTICS was stored and no notice given. */
TEST_CASE("TaskFactory reads a potential without its line of a key of "
          "another command, with a notice",
          "[task_factory][command_keys]") {
  Scratch tmp;
  const std::string file = tmp.write(
      "old_pot.tadah", std::string(kHead) + "WEIGHTS 1.0 2.0\nANALYTICS true\n");
  tadah::core::IgnoredKeyNoticeCapture cap;
  tadah::core::Context ctx;
  const std::string what = read_config("TRAIN", file, &ctx);
  INFO(what);
  REQUIRE(what.empty());
  CHECK_FALSE(ctx.has_stored("ANALYTICS"));
  CHECK(ctx.has_stored("WEIGHTS"));
  std::string notices;
  for (const std::string &n : cap.notices()) notices += n + "\n";
  INFO(notices);
  REQUIRE(cap.notices().size() == 1);
  CHECK(notices.find("ANALYTICS true on line 4 of " + file) !=
        std::string::npos);
  CHECK(notices.find("tadah predict") != std::string::npos);
}
