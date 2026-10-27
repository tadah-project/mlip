// Messages of MLIP's structures, cells, transformations, neighbour cache
// and datasets: each names what it is about (the structure, the cell, the
// supercell), never the class or the function of the code, and an internal
// error says so.
#include "catch2/catch.hpp"

#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/nn_cache.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_transformations.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace tadah::mlip;

namespace {

/* Whether @p s contains @p part. */
bool contains(const std::string &s, const std::string &part) {
  return s.find(part) != std::string::npos;
}

/* The message @p f throws, or "no exception". */
std::string thrown(const std::function<void()> &f) {
  try {
    f();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "no exception";
}

/* Two Ta atoms in the cell of rows @p a, @p b, @p c, labelled @p label. */
Structure two_ta(const std::string &label, double a[3], double b[3],
                 double c[3]) {
  tadah::core::PeriodicTable::initialize();
  Structure st;
  st.label = label;
  st.cell.load(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  const auto ta = tadah::core::PeriodicTable::find_by_symbol("Ta");
  st.add_atom(Atom(ta, 0.0, 0.0, 0.0));
  st.add_atom(Atom(ta, 1.5, 1.5, 1.5));
  return st;
}

}  // namespace

/* Pins: the refusals of a comparison of structures (Structure::is_the_same,
 * which tadah data dedup runs) name the structure, not the function: "structure
 * 'flat' (2 atoms) has a singular cell (zero, non-finite or coplanar lattice
 * vectors); its positions cannot be compared through the cell", "the
 * threshold 2.9 A is too large for structure 'cube' ..." and "the threshold
 * must be a finite number of at least 0".
 * Before the fix: each began "Structure::is_the_same: ". */
TEST_CASE("A comparison of structures names the structure, not the function",
          "[messages][is_the_same]") {
  double x[3] = {10, 0, 0}, y[3] = {0, 10, 0}, z[3] = {0, 0, 10},
         flat[3] = {10, 10, 0};
  const Structure cube = two_ta("cube", x, y, z);
  const Structure planar = two_ta("flat", x, y, flat);
  CHECK(thrown([&] { (void)planar.is_the_same(cube, 1e-6); }) ==
        "structure 'flat' (2 atoms) has a singular cell (zero, non-finite or "
        "coplanar lattice vectors); its positions cannot be compared through "
        "the cell");
  CHECK(thrown([&] { (void)cube.is_the_same(cube, 2.9); })
            .rfind("the threshold 2.9 A is too large for structure 'cube': ",
                   0) == 0);
  CHECK(thrown([&] { (void)cube.is_the_same(cube, -1.0); }) ==
        "the threshold must be a finite number of at least 0");
}

/* Pins: the cells of the PC_OBSERVABLE building blocks and the supercell
 * refuse their arguments naming what they build, with the value as given:
 * "a bcc cell needs a0 > 0, got -1.5", "an isolated-atom cell needs a box
 * size > 0, got 0", "a supercell needs every factor >= 1, got 0 1 1"; the
 * cells this version does not build are internal errors ("internal error:
 * an fcc cell is not built by this version", "... an hcp cell ...").
 * Before the fix: "cells::bcc: a0 must be > 0", "cells::isolated_atom:
 * box_size must be > 0", "StructureTransformation::replicate: every factor
 * must be >= 1, got 0 1 1." and "cells::fcc: not implemented in MVP. First
 * observable that needs FCC (e.g. phase_transition_energy) must land its
 * consumer before this stub may be filled in.". */
TEST_CASE("The cells and the supercell name what they build",
          "[messages][cell_builder]") {
  tadah::core::PeriodicTable::initialize();
  CHECK(thrown([] { (void)cells::bcc("Ta", -1.5); }) ==
        "a bcc cell needs a0 > 0, got -1.5");
  CHECK(thrown([] { (void)cells::isolated_atom("Ta", 0.0); }) ==
        "an isolated-atom cell needs a box size > 0, got 0");
  CHECK(thrown([] { (void)cells::fcc("Ta", 4.0); }) ==
        "internal error: an fcc cell is not built by this version");
  CHECK(thrown([] { (void)cells::hcp("Ta", 3.0, 1.6); }) ==
        "internal error: an hcp cell is not built by this version");
  Structure st = cells::bcc("Ta", 3.3);
  CHECK(thrown([&] { StructureTransformation::replicate(st, 0, 1, 1); }) ==
        "a supercell needs every factor >= 1, got 0 1 1");
}

/* Pins: a mirror plane (h k l) whose normal is too short to normalise (a
 * cell of 1e15 A, plane (1 0 0)) is refused naming the plane: "the plane (1
 * 0 0) of this cell has a normal of length near 0".
 * Before the fix: "Plane normal is near zero. Check cell or (h,k,l).", an
 * instruction. */
TEST_CASE("A mirror plane without a normal is named",
          "[messages][transform]") {
  double x[3] = {1e15, 0, 0}, y[3] = {0, 1e15, 0}, z[3] = {0, 0, 1e15};
  Structure huge = two_ta("huge", x, y, z);
  CHECK(thrown([&] { StructureTransformation::mirrorHkl(huge, 1, 0, 0); }) ==
        "the plane (1 0 0) of this cell has a normal of length near 0");
}

/* Pins: a cached neighbour list whose cutoff does not exceed its skin is an
 * internal error naming both as given: "internal error: the cached
 * neighbour list was built with cutoff 1.5 A, which does not exceed the
 * skin 2 A; a cached list is built at the cutoff plus the skin".
 * Before the fix: "nn_cache::is_fresh: the cached neighbour list was built
 * with cutoff 1.500000 A, ...; build it with NNFinder at cutoff = rcut +
 * skin". */
TEST_CASE("A cached neighbour list within its skin is an internal error",
          "[messages][nn_cache]") {
  double x[3] = {3.3, 0, 0}, y[3] = {0, 3.3, 0}, z[3] = {0, 0, 3.3};
  Structure st = two_ta("bcc", x, y, z);
  nn_cache::State cache;
  cache.skin = 2.0;
  NNFinder nnf;
  nnf.calc(st, 1.5);
  nn_cache::capture(st, cache);
  CHECK(thrown([&] { (void)nn_cache::is_fresh(st, cache); }) ==
        "internal error: the cached neighbour list was built with cutoff 1.5 "
        "A, which does not exceed the skin 2 A; a cached list is built at "
        "the cutoff plus the skin");
}

/* Pins: a dataset file that does not exist is refused with "DBFILE does not
 * exist: nosuch.tadah", and no log line names the function that found it.
 * Before the fix: an INFO line "StructureDB::find_unique_elements: DBFILE
 * does not exist: nosuch.tadah" came before the refusal. */
TEST_CASE("A dataset file that does not exist is named once",
          "[messages][structure_db]") {
  auto &logger = tadah::core::Logger::instance();
  const auto level = logger.getLogLevel();
  logger.setLogLevel(tadah::core::LogLevel::INFO);
  std::ostringstream out;
  std::streambuf *old = std::cout.rdbuf(out.rdbuf());
  const std::string what =
      thrown([] { (void)StructureDB::find_unique_elements("nosuch.tadah"); });
  std::cout.rdbuf(old);
  logger.setLogLevel(level);
  INFO("log:\n" << out.str());
  CHECK(what == "DBFILE does not exist: nosuch.tadah");
  CHECK(out.str().find("StructureDB::") == std::string::npos);
}

#include <tadah/core/context.h>
#include <tadah/mlip/intercepts.h>

/* Pins: the intercept constraints of a configuration (check_bias_constraints)
 * name the keys as a configuration writes them: NORM true with a linear
 * model needs BIAS true ("NORM true with MODEL Kern_Linear requires BIAS
 * true: ...; BIAS true or NORM false is accepted."), and BIAS true needs an
 * ATOM line ("BIAS true requires an ATOM line (BIAS adds one intercept
 * column per species)"); FIXINDEX under NORM true ends "; NORM false is
 * accepted.".
 * Before the fix: "Configuration: NORM=true with MODEL=Kern_Linear requires
 * BIAS=true. ... Set BIAS=true or NORM=false.", "Configuration: BIAS=true
 * requires a non-empty ATOM list (...)" and "... Set NORM false.". */
TEST_CASE("The intercept constraints name the keys as a configuration "
          "writes them",
          "[messages][bias]") {
  tadah::core::Context norm;
  norm.add("NORM", "true");
  norm.add("BIAS", "false");
  norm.add("MODEL", "M_KRR");
  norm.add("MODEL", "Kern_Linear");
  const std::string n = thrown([&] { check_bias_constraints(norm); });
  CHECK(n.rfind("NORM true with MODEL Kern_Linear requires BIAS true: ", 0) ==
        0);
  const std::string n_tail = "; BIAS true or NORM false is accepted.";
  CHECK(n.size() > n_tail.size());
  CHECK(n.compare(n.size() - n_tail.size(), n_tail.size(), n_tail) == 0);
  tadah::core::Context bias;
  bias.add("BIAS", "true");
  CHECK(thrown([&] { check_bias_constraints(bias); }) ==
        "BIAS true requires an ATOM line (BIAS adds one intercept column per "
        "species)");
  tadah::core::Context fixed;
  fixed.add("NORM", "true");
  fixed.add("BIAS", "true");
  fixed.add("ATOM", "Ta");
  fixed.add("MODEL", "M_BLR");
  fixed.add("MODEL", "BF_Linear");
  fixed.add("FIXINDEX", "1");
  fixed.add("FIXWEIGHT", "0.5");
  const std::string f = thrown([&] { check_bias_constraints(fixed); });
  CHECK(f.rfind("FIXINDEX/FIXWEIGHT is not supported with NORM true: ", 0) ==
        0);
  const std::string f_tail = "; NORM false is accepted.";
  CHECK(f.size() > f_tail.size());
  CHECK(f.compare(f.size() - f_tail.size(), f_tail.size(), f_tail) == 0);
}

#include <tadah/core/element.h>
#include <set>

/* Pins: an ATOM or WATOM line that does not fit the elements of the
 * datasets is refused naming both and what is accepted: "the ATOM line of
 * the training configuration names Ta Nb, and the datasets have Nb Ta; an
 * ATOM line that lists these elements in this order, or none, is
 * accepted"; "WATOM of the training configuration has 3 values for the 2
 * elements of the datasets (Nb Ta); one WATOM value for each element of
 * the datasets, or no WATOM line, is accepted"; "WATOM in the pair form
 * takes '<symbol> <symbol> <value>' triples, and the line has 4 entries".
 * Before the fix: "\nMismatch between elements in datasets and ATOM in the
 * context file.\nPlease either update the ATOM in the context file or
 * remove ATOM\nkey completely. ...", the WATOM form likewise, and "\nWATOM
 * pair format expects ... triples.\nGot a number of entries that is not a
 * multiple of three.\n". */
TEST_CASE("An ATOM or WATOM line that does not fit the datasets is named",
          "[messages][structure_db]") {
  tadah::core::PeriodicTable::initialize();
  std::set<tadah::core::Element> nbta = {tadah::core::Element("Nb"),
                                         tadah::core::Element("Ta")};
  tadah::core::Context atom;
  atom.add("ATOM", "Ta");
  atom.add("ATOM", "Nb");
  CHECK(thrown([&] { StructureDB::check_atom_key(atom, nbta); }) ==
        "the ATOM line of the training configuration names Ta Nb, and the "
        "datasets have Nb Ta; an ATOM line that lists these elements in "
        "this order, or none, is accepted");
  tadah::core::Context watom;
  for (const char *v : {"1", "2", "3"}) watom.add("WATOM", v);
  CHECK(thrown([&] { StructureDB::check_watom_key(watom, nbta); }) ==
        "WATOM of the training configuration has 3 values for the 2 "
        "elements of the datasets (Nb Ta); one WATOM value for each element "
        "of the datasets, or no WATOM line, is accepted");
  tadah::core::Context pairs;
  for (const char *v : {"Nb", "Ta", "1", "2"}) pairs.add("WATOM", v);
  CHECK(thrown([&] { StructureDB::check_watom_key(pairs, nbta); }) ==
        "WATOM in the pair form takes '<symbol> <symbol> <value>' triples, "
        "and the line has 4 entries");
}

#include <tadah/core/maths.h>
#include <tadah/mlip/design_matrix/column_block.h>
#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_rbf.h>

/* Pins: the layout refusals of a non-linear kernel name the kernel and the
 * keys, not the class of the code: "kernel Kern_RBF has no basis vectors
 * (SBASIS, BASIS): the design-matrix columns of a non-linear kernel are its
 * basis vectors" and "the basis vectors of kernel Kern_RBF have 3
 * components, and the descriptors have 5 (DSIZE): the basis (BASIS) does
 * not belong to these descriptors".
 * Before the fix: both began "DesignMatrix: " ("DesignMatrix: kernel ...
 * has no basis vectors (SBASIS/BASIS): ...", "... have 3 components but
 * the descriptor has 5 (DSIZE): ..."). */
TEST_CASE("The layout refusals of a kernel name the kernel and the keys",
          "[messages][kernel]") {
  ColumnLayout layout;
  ColumnBlock d2;
  d2.descriptor_label = "D2_Blip";
  d2.family = "D2";
  d2.col_offset = 0;
  d2.col_count = 5;
  layout.blocks.push_back(d2);
  DM_Kern_RBF kernel;
  const tadah::core::Context none;
  CHECK(thrown([&] { (void)kernel.phi_layout(layout, none); }) ==
        "kernel Kern_RBF has no basis vectors (SBASIS, BASIS): the "
        "design-matrix columns of a non-linear kernel are its basis vectors");
  kernel.set_basis(tadah::core::Matrix(size_t(3), size_t(2)));
  CHECK(thrown([&] { (void)kernel.phi_layout(layout, none); }) ==
        "the basis vectors of kernel Kern_RBF have 3 components, and the "
        "descriptors have 5 (DSIZE): the basis (BASIS) does not belong to "
        "these descriptors");
}

#include <tadah/mlip/key_storage/key_manager.h>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <unistd.h>

namespace {
/* The value of the environment variable @p name, if set. */
std::optional<std::string> env_of(const char *name) {
  const char *v = std::getenv(name);
  return v ? std::optional<std::string>(v) : std::nullopt;
}
void restore_env(const char *name, const std::optional<std::string> &v) {
  if (v)
    setenv(name, v->c_str(), 1);
  else
    unsetenv(name);
}
/* What getServiceKey(MP) prints, and returns, given @p typed on standard
 * input, with HOME a fresh directory (with .config) and no
 * MP_REST_API_KEY. */
struct Prompted {
  std::string out, err, value, home;
};
Prompted prompt_with(const std::string &typed) {
  static int counter = 0;
  namespace fs = std::filesystem;
  const fs::path home = fs::temp_directory_path() /
                        ("tadah_keys_" + std::to_string(::getpid()) + "_" +
                         std::to_string(counter++));
  fs::create_directories(home / ".config");
  const auto old_home = env_of("HOME");
  const auto old_key = env_of("MP_REST_API_KEY");
  setenv("HOME", home.c_str(), 1);
  unsetenv("MP_REST_API_KEY");
  std::istringstream in(typed);
  std::ostringstream out, err;
  std::streambuf *old_in = std::cin.rdbuf(in.rdbuf());
  std::streambuf *old_out = std::cout.rdbuf(out.rdbuf());
  std::streambuf *old_err = std::cerr.rdbuf(err.rdbuf());
  Prompted p;
  p.home = home.string();
  try {
    p.value = KeyManager::getServiceKey(ServiceType::MP, true);
  } catch (...) {
    std::cin.rdbuf(old_in);
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    restore_env("HOME", old_home);
    restore_env("MP_REST_API_KEY", old_key);
    fs::remove_all(home);
    throw;
  }
  std::cin.rdbuf(old_in);
  std::cout.rdbuf(old_out);
  std::cerr.rdbuf(old_err);
  restore_env("HOME", old_home);
  restore_env("MP_REST_API_KEY", old_key);
  fs::remove_all(home);
  p.out = out.str();
  p.err = err.str();
  return p;
}
}  // namespace

/* Pins: the key prompt of an online structure source states its options
 * and what each does, impersonally, and names the keys file it writes,
 * $HOME/.config/tadah/keys:
 *  - every run prints "No key found for Materials Project
 *    (MP_REST_API_KEY).", "The website of the service says how a key is
 *    obtained." and the menu "Options:", "[1] store the key in the keys
 *    file (kept for later runs)", ..., "Choice [1..4]: ";
 *  - 1 then a key: "The key (Ctrl+C cancels):" and "The key is stored in
 *    <HOME>/.config/tadah/keys.";
 *  - 3 then a key: "The key is exported for this terminal session by:",
 *    "  export MP_REST_API_KEY=\"abc\"", "A line in ~/.bashrc or ~/.zshrc
 *    exports it in every shell, such as the one this command adds:";
 *  - 4: "Aborted; the key stays missing.", and 1 with no key "No key
 *    entered; the key stays missing.".
 * Before the fix: "Please select an option:", "Please enter the key now",
 * "[Info] Key stored in context file (~/.context/tadah/keys)." (a path the
 * storage does not write), "To export this key for your current TERMINAL
 * session:", "add the same line to your ~/.bashrc", "Aborting. Key remains
 * missing.". */
TEST_CASE("The key prompt states its options and names the keys file",
          "[messages][keys]") {
  const std::string menu =
      "Options:\n"
      "[1] store the key in the keys file (kept for later runs)\n"
      "[2] set the environment variable for this run only\n"
      "[3] show the commands that export it in a shell (for several runs)\n"
      "[4] abort (nothing is done)\n"
      "\nChoice [1..4]: ";
  const std::string head =
      "\nNo key found for Materials Project (MP_REST_API_KEY).\n"
      "The website of the service says how a key is obtained.\n\n";
  {
    const Prompted p = prompt_with("1\nabc\n");
    INFO(p.out << p.err);
    CHECK(p.value == "abc");
    CHECK(p.out == head + menu + "The key (Ctrl+C cancels):\n> " +
                       "The key is stored in " + p.home +
                       "/.config/tadah/keys.\n");
  }
  {
    const Prompted p = prompt_with("3\nabc\n");
    INFO(p.out << p.err);
    CHECK(p.value == "abc");
    CHECK(p.out.rfind(head + menu +
                          "This option neither stores nor sets the key: it "
                          "shows the commands that export it in a shell.\n\n"
                          "The key (Ctrl+C cancels):\n> ",
                      0) == 0);
    CHECK(contains(p.out, "The key is exported for this terminal session "
                          "by:\n\n  export MP_REST_API_KEY=\"abc\"\n"));
    CHECK(contains(p.out, "A line in ~/.bashrc or ~/.zshrc exports it in "
                          "every shell, such as the one this command "
                          "adds:\n\n  echo 'export MP_REST_API_KEY=\"abc\"' "
                          ">> ~/.bashrc\n"));
    CHECK_FALSE(contains(p.out, "your"));
    CHECK_FALSE(contains(p.out, "Please"));
  }
  {
    const Prompted p = prompt_with("4\n");
    CHECK(p.value.empty());
    CHECK(p.out == head + menu);
    CHECK(p.err == "Aborted; the key stays missing.\n");
  }
  {
    const Prompted p = prompt_with("1\n\n");
    CHECK(p.value.empty());
    CHECK(p.err == "No key entered; the key stays missing.\n");
  }
}
