// TaskFactory and the keys of other commands in a configuration file.
//
// A key that only some commands read (an option of their command line such
// as ANALYTICS of tadah predict or INDEX of tadah data print, or a key of
// their configuration file such as REFIT_MODE of tadah refit) is no key of a
// model or of a training configuration. A configuration file given with -c
// to a command that does not read it refuses its line, naming the key, the
// line, the file and the commands that read it; the command that reads it
// takes it, unless the command's -c needs the key's option on the command
// line, whose value would replace the line: then the line is refused too.
// A potential (a file with WEIGHTS) written by an earlier version
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

/* The commands whose tables list OUTFILE, as a refusal names them. */
const char *const kOutfileCommands =
    "tadah analysis bfunc, tadah analysis cutoff, tadah analysis "
    "dataset_stats, tadah analysis descriptor, tadah analysis suggest_bounds, "
    "tadah data balance, tadah data convert, tadah data dedup, tadah data "
    "merge, tadah data sample, tadah data split, tadah data write, tadah "
    "predict, tadah properties pairwise, tadah refit and tadah train";

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

/* Pins: the refusal of a key of another command names the commands that
 * read it, not one that retires it, and a positional key as the argument
 * of its command: in a configuration file of tadah train, "RESCALE is the
 * option --rescale of tadah analysis bfunc", "THRESHOLD is the option
 * --threshold of tadah data dedup" (tadah data balance refuses both as
 * retired options) and "OPTION is the argument of tadah explain".
 * Before the fix: RESCALE was an option of tadah analysis bfunc and tadah
 * data balance, THRESHOLD of tadah data balance and tadah data dedup, and
 * OPTION "a key of tadah explain". */
TEST_CASE("TaskFactory names the commands that read a key of another "
          "command, not one that retires it, and a positional key as the "
          "argument of its command",
          "[task_factory][command_keys]") {
  Scratch tmp;
  struct Case {
    const char *line;
    const char *key;
    const char *role;
  };
  for (const Case &c :
       {Case{"RESCALE true", "RESCALE",
             "RESCALE is the option --rescale of tadah analysis bfunc, not "
             "a key of a configuration file of tadah train"},
        Case{"THRESHOLD 1e-4", "THRESHOLD",
             "THRESHOLD is the option --threshold of tadah data dedup, not "
             "a key of a configuration file of tadah train"},
        Case{"OPTION train", "OPTION",
             "OPTION is the argument of tadah explain, not a key of a "
             "configuration file of tadah train"}}) {
    DYNAMIC_SECTION(c.line) {
      const std::string file = tmp.write(std::string(c.key) + ".cfg",
                                         std::string(kHead) + c.line + "\n");
      const std::string what = read_config("TRAIN", file);
      INFO(what);
      CHECK(what.find("Key " + std::string(c.key) + " on line 3 of " + file +
                      " is not accepted here: " + c.role) !=
            std::string::npos);
      CHECK(what.find("tadah data balance") == std::string::npos);
    }
  }
}

/* Pins: OUTFILE is a key of the commands whose tables list it (tadah
 * train, refit, predict, properties pairwise and the data and analysis
 * commands that write a file), not of tadah hpo, which writes files of
 * its own (best_pot.tadah and the others), or of tadah analysis
 * check_bounds, which writes no file; both take the same training
 * configuration as tadah train: an OUTFILE line in a configuration file
 * of either is refused at its line, naming the commands that read the
 * key, the key's option and the command that reads the file.
 * Before the fix: OUTFILE was a key of every configuration file, and tadah
 * hpo -c and tadah analysis check_bounds -c read the line and wrote
 * nothing to the file it names. */
TEST_CASE("TaskFactory refuses OUTFILE in a configuration file of tadah hpo "
          "and tadah analysis check_bounds",
          "[task_factory][command_keys][outfile]") {
  Scratch tmp;
  const std::string file =
      tmp.write("outfile.cfg", std::string(kHead) + "OUTFILE pot2.tadah\n");
  for (const auto &[task, command] :
       {std::pair<std::string, std::string>{"HPO", "tadah hpo"},
        {"ANALYSIS.check_bounds", "tadah analysis check_bounds"}}) {
    DYNAMIC_SECTION("a configuration file of " << command) {
      const std::string what = read_config(task, file);
      CHECK(what ==
            "Key OUTFILE on line 3 of " + file + " is not accepted here: "
            "OUTFILE is the option --outfile (-o) of " +
            std::string(kOutfileCommands) +
            ", not a key of a configuration file of " + command);
    }
  }
}

/* Pins: a command whose -c (--config) needs an option on the command line
 * (the blueprint's cli_needs of its CONFIG key) refuses a line of that
 * option's key in its configuration file, at its line, naming the option,
 * the command and its -c: tadah analysis dataset_stats -c needs
 * --outfile (-o), and tadah analysis bfunc -c needs --range (-r) and
 * --type, so the command line always gives the key and its value replaces
 * the line.
 * Before the fix: the line was read and then replaced by the option's
 * value, without a word: tadah analysis dataset_stats -c cfg.tadah -o
 * stats.txt with an OUTFILE line in cfg.tadah wrote stats.txt only, and
 * tadah analysis bfunc took --range and --type over the RANGE and TYPE
 * lines; read_config here returned "" for each. */
TEST_CASE("TaskFactory refuses a line of an option that -c of the command "
          "needs",
          "[task_factory][command_keys][outfile]") {
  Scratch tmp;
  struct Case {
    const char *task;
    const char *command;
    const char *line;
    const char *key;
    const char *option;
  };
  for (const Case &c :
       {Case{"ANALYSIS.dataset_stats", "tadah analysis dataset_stats",
             "OUTFILE stats.txt", "OUTFILE", "--outfile (-o)"},
        Case{"ANALYSIS.bfunc", "tadah analysis bfunc", "RANGE 0.5 5 3",
             "RANGE", "--range (-r)"},
        Case{"ANALYSIS.bfunc", "tadah analysis bfunc", "TYPE mb N", "TYPE",
             "--type"}}) {
    DYNAMIC_SECTION(c.command << ": " << c.line) {
      const std::string file = tmp.write(
          std::string(c.key) + ".cfg", std::string(kHead) + c.line + "\n");
      const std::string what = read_config(c.task, file);
      CHECK(what == "Key " + std::string(c.key) + " on line 3 of " + file +
                        " is not accepted here: " + c.key +
                        " is the option " + c.option +
                        " that --config (-c) of " + c.command +
                        " needs on the command line, not a key of a "
                        "configuration file of " + c.command);
    }
  }
}

/* Pins: tadah analysis suggest_bounds -c and tadah analysis bfunc -c read
 * the OUTFILE line of their configuration file (the task's OUTFILE is
 * out.txt): their -c needs no --outfile.
 * Before the fix: passes as well (a pin). */
TEST_CASE("TaskFactory reads OUTFILE in a configuration file of tadah "
          "analysis suggest_bounds and tadah analysis bfunc",
          "[task_factory][command_keys][outfile]") {
  Scratch tmp;
  const std::string file =
      tmp.write("analysis.cfg", std::string(kHead) + "OUTFILE out.txt\n");
  for (const std::string task : {"ANALYSIS.suggest_bounds", "ANALYSIS.bfunc"}) {
    DYNAMIC_SECTION(task) {
      tadah::core::Context ctx;
      const std::string what = read_config(task, file, &ctx);
      INFO(what);
      REQUIRE(what.empty());
      CHECK(ctx.get<std::string>("OUTFILE") == "out.txt");
    }
  }
}

/* Pins: tadah train -c reads the OUTFILE line of its configuration file
 * (the task's OUTFILE is pot2.tadah), as each command whose table lists
 * the key does.
 * Before the fix: passes as well (a pin). */
TEST_CASE("TaskFactory reads OUTFILE in a configuration file of tadah train",
          "[task_factory][command_keys][outfile]") {
  Scratch tmp;
  const std::string file =
      tmp.write("train.cfg", std::string(kHead) + "OUTFILE pot2.tadah\n");
  tadah::core::Context ctx;
  const std::string what = read_config("TRAIN", file, &ctx);
  INFO(what);
  REQUIRE(what.empty());
  CHECK(ctx.get<std::string>("OUTFILE") == "pot2.tadah");
}

/* Pins: a potential (a file whose lines give WEIGHTS) with an OUTFILE
 * line, as tadah hpo of 1.5.0 wrote it into best_pot.tadah when its
 * configuration had an OUTFILE line, loads as the configuration file
 * of tadah hpo and of tadah train: the line is a setting of the run that
 * wrote the potential and is not read (tadah hpo has no OUTFILE, tadah
 * train its default pot.tadah), and no notice names it as a key of
 * another command.
 * Before the fix: passes as well (a pin). */
TEST_CASE("TaskFactory reads a potential with an OUTFILE line without the "
          "line",
          "[task_factory][command_keys][outfile]") {
  Scratch tmp;
  const std::string file = tmp.write(
      "pot.tadah", std::string(kHead) + "WEIGHTS 1.0 2.0\nOUTFILE old.tadah\n");
  for (const auto &[task, outfile] :
       {std::pair<std::string, std::string>{"HPO", ""},
        {"TRAIN", "pot.tadah"}}) {
    DYNAMIC_SECTION(task) {
      tadah::core::IgnoredKeyNoticeCapture cap;
      tadah::core::Context ctx;
      const std::string what = read_config(task, file, &ctx);
      INFO(what);
      REQUIRE(what.empty());
      CHECK(ctx.has_stored("WEIGHTS"));
      if (outfile.empty())
        CHECK_FALSE(ctx.has_stored("OUTFILE"));
      else
        CHECK(ctx.get<std::string>("OUTFILE") == outfile);
      for (const std::string &n : cap.notices())
        CHECK(n.find("OUTFILE") == std::string::npos);
    }
  }
}
