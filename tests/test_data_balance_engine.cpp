// test_data_balance_engine.cpp
//
// tadah data balance (DataBalanceEngine) shifts the energies of its input
// files by a reference energy per element (ESHIFT) and writes them, one
// output file per input file or, with MERGE, all to one; the priorities of
// the structures (eweight, fweight, sweight) are written as they are. Its
// rescaling of the priorities (RESCALE, with THRESHOLD, FORCE and STRESS),
// which replaced the priorities of the data by inverse magnitudes, is
// retired: a TASK data balance section or a command line that gives one
// of its options is refused with the reason, and so is a RESCALE line of
// the GLOBAL section of a task file with a TASK data balance section
// (blueprint: test_blueprint_retired.cpp).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/engines/data_balance_engine.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/task_factory.h>

#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>
#include <unistd.h>

namespace {

using Catch::Matchers::Contains;
using tadah::core::Context;
using tadah::core::Element;
using tadah::mlip::Atom;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Task;
using tadah::mlip::engines::DataBalanceEngine;
namespace fs = std::filesystem;

struct ScratchCwd {
  fs::path old_cwd, dir;
  ScratchCwd() {
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_data_balance_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    fs::current_path(dir);
  }
  ~ScratchCwd() {
    fs::current_path(old_cwd);
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

/* A file of @p n Ta structures of 2 atoms, energy @p e each, with the
 * priorities 0.5, 2 and 3. */
void write_file(const std::string &path, std::size_t n, double e) {
  StructureDB db;
  for (std::size_t s = 0; s < n; ++s) {
    Structure st;
    st.label = path + " " + std::to_string(s);
    st.cell.load(3.3, 0.0, 0.0, 0.0, 3.3, 0.0, 0.0, 0.0, 3.3);
    st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0, 0.1, 0.0, 0.0));
    st.add_atom(Atom(Element("Ta"), 1.65, 1.65, 1.65, -0.1, 0.0, 0.0));
    st.energy = e;
    st.eweight = 0.5;
    st.fweight = 2.0;
    st.sweight = 3.0;
    db.add(st);
  }
  db.dump_to_file(path, false);
}

Context balance(const std::vector<std::string> &in,
                const std::vector<std::string> &out, bool merge) {
  Context c;
  for (const auto &f : in) c.add("DBFILE", f);
  for (const auto &f : out) c.add("OUTFILE", f);
  c.add("ESHIFT", -11.0);
  c.add("MERGE", merge ? "true" : "false");
  c.add("APPEND", "false");
  c.add("NUMERIC", 18);
  return c;
}

std::string refusal(const Context &c) {
  try {
    Task t("data", c);
    DataBalanceEngine e;
    e.validate(t);
    e.run(t);
  } catch (const std::exception &ex) {
    return ex.what();
  }
  return "";
}

}  // namespace

/* Pins: with MERGE true, two input files (2 and 3 structures, energies -24
 * and -23 eV) and one output file, data balance writes the five structures
 * to it, each energy shifted by 2 x 11 eV (-2 and -1 eV), with the
 * priorities 0.5, 2 and 3 as they were; one output file per input file
 * without MERGE; a number of output files that does not fit is refused,
 * naming the counts; a run without ESHIFT is refused, naming --eshift.
 * Before the fix: MERGE true with two input files was refused "Number of
 * input and output files must match." (its help says it merges them); a
 * run without ESHIFT and without RESCALE was refused naming RESCALE. */
TEST_CASE("tadah data balance shifts energies; --merge writes several input "
          "files to one",
          "[data_balance]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  write_file("a.tadah", 2, -24.0);
  write_file("b.tadah", 3, -23.0);
  REQUIRE(refusal(balance({"a.tadah", "b.tadah"}, {"merged.tadah"}, true))
              .empty());
  StructureDB merged;
  merged.add("merged.tadah");
  REQUIRE(merged.size() == 5);
  for (std::size_t s = 0; s < 5; ++s) {
    INFO("structure " << s);
    CHECK(merged(s).energy == Approx(s < 2 ? -2.0 : -1.0).epsilon(1e-14));
    CHECK(merged(s).eweight == 0.5);
    CHECK(merged(s).fweight == 2.0);
    CHECK(merged(s).sweight == 3.0);
  }
  REQUIRE(refusal(balance({"a.tadah", "b.tadah"}, {"a2.tadah", "b2.tadah"},
                          false))
              .empty());
  StructureDB b2;
  b2.add("b2.tadah");
  CHECK(b2.size() == 3);
  CHECK_THAT(refusal(balance({"a.tadah", "b.tadah"}, {"x.tadah", "y.tadah"},
                             true)),
             Contains("--merge") && Contains("one output file"));
  CHECK_THAT(refusal(balance({"a.tadah", "b.tadah"}, {"x.tadah"}, false)),
             Contains("one output file per input file"));
  Context none = balance({"a.tadah"}, {"x.tadah"}, false);
  none.remove("ESHIFT");
  CHECK_THAT(refusal(none), Contains("--eshift"));
}

namespace {

using KV = std::unordered_map<std::string, std::vector<std::string>>;

/* A task input: a task name and its options, as the command line gives
 * them. */
struct Input : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  Input(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* What TaskFactory throws for @p input, "" when it makes the tasks. */
std::string factory_error(const Input &input) {
  try {
    (void)tadah::mlip::TaskFactory().createTasksFromInput(input);
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

}  // namespace

/* Pins: a TASK data balance section of a task file that gives RESCALE,
 * THRESHOLD, FORCE or STRESS, the options of the rescaling Tadah! 1.5.0
 * applied, is refused before any task runs, naming the key, its line of
 * the file, the command and why ("tadah data balance shifts energies
 * only"); a GLOBAL section's FORCE true (the setting of another task of
 * the file) is accepted; the command line's --rescale, --threshold,
 * --force and --stress are refused alike, naming the option.
 * Before the fix: the task file ran, the four keys ignored (the output
 * kept the priorities, where 1.5.0 rescaled them), and the command line
 * did not know the options. */
TEST_CASE("tadah data balance refuses the options of the retired rescaling",
          "[data_balance][retired]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  write_file("db4.dat", 2, -11.2);
  for (const std::string line :
       {"RESCALE true", "THRESHOLD 0.1", "FORCE true", "STRESS true"}) {
    const std::string key = line.substr(0, line.find(' '));
    INFO(key);
    {
      std::ofstream t("b4.task");
      t << "TASK data balance\n  DBFILE db4.dat\n  ESHIFT -8.0\n  " << line
        << "\n  OUTFILE balX.dat\n";
    }
    CHECK_THAT(factory_error(Input("", {{"task", {"b4.task"}}})),
               Contains(key) && Contains("line 4 of b4.task") &&
                   Contains("tadah data balance") &&
                   Contains("tadah data balance shifts energies only"));
    std::string option = key;
    for (char &ch : option) ch = static_cast<char>(std::tolower(ch));
    std::vector<std::string> value = {"true"};
    if (key == "THRESHOLD") value = {"0.1"};
    CHECK_THAT(factory_error(Input("DATA.balance",
                                   {{"dbfile", {"db4.dat"}},
                                    {"eshift", {"-8.0"}},
                                    {"outfile", {"balX.dat"}},
                                    {option, value}})),
               Contains("--" + option) &&
                   Contains("tadah data balance shifts energies only"));
  }
  {
    std::ofstream t("b5.task");
    t << "FORCE true\nTASK data balance\n  DBFILE db4.dat\n  ESHIFT -8.0\n"
         "  OUTFILE balX.dat\n";
  }
  CHECK(factory_error(Input("", {{"task", {"b5.task"}}})).empty());
}

/* Pins: the GLOBAL section of a task file sets the keys of every task of
 * the file, so a RESCALE line in it reaches a TASK data balance section,
 * as it did in Tadah! 1.5.0, which rescaled the priorities with it: the
 * file is refused before any task runs, whatever the value, naming the
 * line in the GLOBAL section, the task it reaches and why ("tadah data
 * balance shifts energies only", replaced by the priorities of the data).
 * When another TASK section of the file takes RESCALE (TASK analysis
 * bfunc, --rescale), the message names it and its line: the RESCALE line
 * meant for it belongs in that section. A GLOBAL RESCALE line without a
 * TASK data balance section is accepted, and so are GLOBAL lines of
 * THRESHOLD, FORCE and STRESS, which in 1.5.0 acted only with RESCALE
 * (their 1.5.0 output equals that without them). A RESCALE line of a file
 * that a TASK data balance or the GLOBAL section includes with CONFIG is
 * refused alike, naming its line of that file.
 * Before the fix: the GLOBAL line reached the task and was ignored (the
 * output kept the priorities, which 1.5.0 rescaled), and a TASK section
 * was refused only for its own lines, so a RESCALE line of a file it
 * included was ignored too. */
TEST_CASE("tadah data balance refuses a RESCALE line of the GLOBAL section "
          "of a task file",
          "[data_balance][retired]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  write_file("db4.dat", 2, -11.2);
  const std::string balance_task =
      "TASK data balance\n  DBFILE db4.dat\n  ESHIFT -8.0\n"
      "  OUTFILE balX.dat\n";
  auto run = [](const std::string &file, const std::string &text) {
    std::ofstream(file) << text;
    return factory_error(Input("", {{"task", {file}}}));
  };
  const std::string why = "tadah data balance shifts energies only";
  for (const std::string value : {"true", "false"}) {
    INFO("RESCALE " << value);
    const std::string err =
        run("g1.task", "RESCALE " + value + "\n" + balance_task);
    CHECK_THAT(err, Contains("RESCALE (line 1 of g1.task) in the GLOBAL "
                             "section reaches TASK data balance (the GLOBAL "
                             "section sets the keys of every task of the "
                             "file): --rescale of tadah data balance is "
                             "retired: ") &&
                        Contains("Replaced by: the priorities of the data") &&
                        Contains(why));
    CHECK(err.find("belongs in") == std::string::npos);
  }
  CHECK_THAT(run("g2.task", "# analysis and balance\nRESCALE true\n"
                            "TASK analysis bfunc\n" + balance_task),
             Contains("RESCALE (line 2 of g2.task) in the GLOBAL section "
                      "reaches TASK data balance") &&
                 Contains(why) &&
                 Contains(". A RESCALE line meant for TASK analysis bfunc "
                          "(line 3 of g2.task) belongs in that TASK "
                          "section."));
  CHECK(run("g3.task", "RESCALE true\nTASK analysis bfunc\n").empty());
  CHECK(run("g4.task", "FORCE true\nSTRESS true\nTHRESHOLD 0.1 0.1 0.1\n" +
                           balance_task)
            .empty());
  std::ofstream("inc.cfg") << "# shared\nRESCALE true\n";
  CHECK_THAT(run("g5.task", "CONFIG inc.cfg\n" + balance_task),
             Contains("RESCALE (line 2 of inc.cfg) in the GLOBAL section "
                      "reaches TASK data balance") &&
                 Contains(why));
  CHECK_THAT(run("g6.task", "TASK data balance\n  CONFIG inc.cfg\n"
                            "  DBFILE db4.dat\n  ESHIFT -8.0\n"
                            "  OUTFILE balX.dat\n"),
             Contains("RESCALE (line 2 of inc.cfg) in TASK data balance: "
                      "--rescale of tadah data balance is retired: ") &&
                 Contains(why));
  CHECK_FALSE(fs::exists("balX.dat"));
}
