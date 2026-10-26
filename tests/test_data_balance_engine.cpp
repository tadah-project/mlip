// test_data_balance_engine.cpp
//
// tadah data balance (DataBalanceEngine) shifts the energies of its input
// files by a reference energy per element (ESHIFT) and writes them, one
// output file per input file or, with MERGE, all to one; the priorities of
// the structures (eweight, fweight, sweight) are written as they are. Its
// rescaling of the priorities (RESCALE, with THRESHOLD, FORCE and STRESS),
// which replaced the priorities of the data by inverse magnitudes, is
// retired: a command line that gives one of its options is refused with
// the reason (blueprint: test_blueprint_retired.cpp).

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

/* Pins: the command line's --rescale, --threshold, --force and --stress,
 * the options of the rescaling Tadah! 1.5.0 applied, are refused by
 * tadah data balance, naming the option and why ("tadah data balance
 * shifts energies only").
 * Before the fix: the command line did not know the options. */
TEST_CASE("tadah data balance refuses the options of the retired rescaling",
          "[data_balance][retired]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  write_file("db4.dat", 2, -11.2);
  for (const std::string key : {"RESCALE", "THRESHOLD", "FORCE", "STRESS"}) {
    INFO(key);
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
}
