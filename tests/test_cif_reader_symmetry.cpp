// test_cif_reader_symmetry.cpp
//
// How CifReader turns the symmetry record and the site occupancies of a CIF
// into a Structure. gemmi::make_small_structure_from_block reads the
// symmetry-operator loop, the Hall symbol, the Hermann-Mauguin (H-M) symbol
// and the space-group number; CifReader expands the listed sites to every
// atom of the unit cell (get_all_unit_cell_sites).
//
// Contract pinned here:
//   - a space group given by any one of the H-M symbol (old or new tag), the
//     Hall symbol, the space-group number or a symmetry-operator loop (old
//     or new tag) gives every atom of the unit cell;
//   - a CIF with no symmetry record is P 1: the sites as listed, and two
//     rows at one position modulo a lattice vector are both kept;
//   - when the symmetry record contradicts itself (gemmi's
//     SmallStructure::check_spacegroup() is not empty), the structure
//     follows the operator loop and a WARNING names the file and quotes the
//     diagnostic; a consistent or absent record prints nothing;
//   - a site with _atom_site_occupancy below 1 is refused, naming the site
//     and its occupancy: Tadah has no representation for a disordered or
//     partly vacant site, and two species sharing a site would become
//     coincident atoms.
//
// Fixtures: tests_data/cif/ (one file per row of the investigation table,
// tmp/known-issues/report_leftovers.md section 6). The expected atoms are
// hand-written from the Wyckoff positions (bcc Fe 2a, NaCl 4a + 4b, hcp Mg
// 2c); none is computed with the reader under test.

#include "catch2/catch.hpp"

#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_readers/cif_reader.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using tadah::core::LogLevel;
using tadah::core::Logger;
using tadah::core::PeriodicTable;
using namespace tadah::mlip;

namespace {

using V3 = std::array<double, 3>;

const std::string kCifDir = "./tests_data/cif/";

struct Site {
  std::string symbol;
  V3 frac;
};

// bcc Fe, I m -3 m, Fe on 2a: the corner and the body centre.
const std::vector<Site> kFeBcc = {{"Fe", {0.0, 0.0, 0.0}},
                                  {"Fe", {0.5, 0.5, 0.5}}};
// Rock salt, F m -3 m: Na on 4a, Cl on 4b.
const std::vector<Site> kNaCl = {
    {"Na", {0.0, 0.0, 0.0}}, {"Na", {0.0, 0.5, 0.5}},
    {"Na", {0.5, 0.0, 0.5}}, {"Na", {0.5, 0.5, 0.0}},
    {"Cl", {0.5, 0.5, 0.5}}, {"Cl", {0.5, 0.0, 0.0}},
    {"Cl", {0.0, 0.5, 0.0}}, {"Cl", {0.0, 0.0, 0.5}}};
// hcp Mg, P 6_3/m m c, Mg on 2c.
const std::vector<Site> kMgHcp = {{"Mg", {1.0 / 3.0, 2.0 / 3.0, 0.25}},
                                  {"Mg", {2.0 / 3.0, 1.0 / 3.0, 0.75}}};

double dot(const V3 &u, const V3 &v) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}
V3 cross(const V3 &u, const V3 &v) {
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
          u[0] * v[1] - u[1] * v[0]};
}
V3 cell_row(const Structure &st, int i) {
  return {st.cell(i, 0), st.cell(i, 1), st.cell(i, 2)};
}
double wrap_unit(double x) { return x - std::round(x); }

// Fractional coordinates of atom n, f = (cell^T)^{-1} r with row i of
// Structure::cell the lattice vector a_i, written with the reciprocal
// vectors: f_k = r . (a_{k+1} x a_{k+2}) / (a_0 . (a_1 x a_2)).
V3 frac_of(const Structure &st, size_t n) {
  const V3 r = {st.atoms[n].position[0], st.atoms[n].position[1],
                st.atoms[n].position[2]};
  const V3 a0 = cell_row(st, 0), a1 = cell_row(st, 1), a2 = cell_row(st, 2);
  const double V = dot(a0, cross(a1, a2));
  return {dot(r, cross(a1, a2)) / V, dot(r, cross(a2, a0)) / V,
          dot(r, cross(a0, a1)) / V};
}

// The structure has one atom per site, and every atom sits on one of
// `sites` (same element, same fractional position modulo 1 to 1e-6), each
// site used once.
void check_sites(const Structure &st, const std::vector<Site> &sites) {
  CHECK(st.natoms() == sites.size());
  if (st.natoms() != sites.size()) return;
  std::vector<bool> used(sites.size(), false);
  for (size_t n = 0; n < st.natoms(); ++n) {
    const V3 f = frac_of(st, n);
    INFO("atom " << n << " (" << st.atoms[n].symbol << ") frac = " << f[0]
                 << " " << f[1] << " " << f[2]);
    bool matched = false;
    for (size_t s = 0; s < sites.size() && !matched; ++s) {
      if (used[s] || sites[s].symbol != st.atoms[n].symbol) continue;
      bool same = true;
      for (int k = 0; k < 3; ++k)
        same = same && std::abs(wrap_unit(f[k] - sites[s].frac[k])) < 1e-6;
      if (same) used[s] = matched = true;
    }
    CHECK(matched);
  }
}

bool contains(const std::string &text, const std::string &what) {
  return text.find(what) != std::string::npos;
}
std::string lower(std::string s) {
  for (auto &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string slurp(const std::string &path) {
  std::ifstream in(path);
  REQUIRE(in.is_open());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Runs `f` with std::cout (the Logger's console) and std::cerr captured and
// the Logger at WARNING, so a WARNING is seen whichever of the two the
// reader prints it to; returns what was printed. The streams and the log
// level are restored before returning or rethrowing, so no Catch output is
// captured.
template <class F>
std::string capture_output(F &&f) {
  auto &log = Logger::instance();
  const LogLevel old_level = log.getLogLevel();
  log.setLogLevel(LogLevel::WARNING);
  std::ostringstream out, err;
  std::streambuf *old_out = std::cout.rdbuf(out.rdbuf());
  std::streambuf *old_err = std::cerr.rdbuf(err.rdbuf());
  const auto restore = [&] {
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    log.setLogLevel(old_level);
  };
  try {
    f();
  } catch (...) {
    restore();
    throw;
  }
  restore();
  return out.str() + err.str();
}

// The three ways a CIF reaches CifReader.
enum class Route { File, Contents, Selector };
const Route kRoutes[] = {Route::File, Route::Contents, Route::Selector};

const char *route_name(Route r) {
  switch (r) {
    case Route::File:
      return "CifReader::read";
    case Route::Contents:
      return "CifReader::parseCifContents (the CODReader path)";
    case Route::Selector:
      return "StructureReaderSelector (-s x.cif)";
  }
  return "?";
}

// What one read returned and printed.
struct Read {
  Structure st;
  std::string log;
};

Read read_cif(const std::string &path, Route route) {
  const std::string text = route == Route::Contents ? slurp(path) : "";
  Read r;
  r.log = capture_output([&] {
    switch (route) {
      case Route::File: {
        CifReader reader;
        reader.read(path);
        r.st = reader.getStructure();
        break;
      }
      case Route::Contents: {
        CifReader reader;
        reader.parseCifContents(text);
        r.st = reader.getStructure();
        break;
      }
      case Route::Selector: {
        auto reader = StructureReaderSelector::getReader(path);
        reader->read(path);
        r.st = reader->getStructure();
        break;
      }
    }
  });
  return r;
}

Read parse_cif_text(const std::string &text) {
  Read r;
  r.log = capture_output([&] {
    CifReader reader;
    reader.parseCifContents(text);
    r.st = reader.getStructure();
  });
  return r;
}

// A cubic cell (a = 2.8665 A, bcc Fe) or rock-salt cell (a = 5.64 A) with
// an occupancy column; `sites` holds the _atom_site rows.
std::string cif_with_occupancy(const std::string &block, double a,
                               const std::string &hm,
                               const std::string &sites) {
  std::ostringstream s;
  s << "data_" << block << "\n"
    << "_cell_length_a " << a << "\n_cell_length_b " << a
    << "\n_cell_length_c " << a << "\n"
    << "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
    << "_symmetry_space_group_name_H-M '" << hm << "'\n"
    << "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
    << "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
    << "_atom_site_occupancy\n"
    << sites;
  return s.str();
}

}  // namespace

// ========================================================== space group ====

/* Pins: bcc Fe (a = 2.8665 A, Fe on 2a) gives both atoms of the unit cell,
 * (0,0,0) and (1/2,1/2,1/2), whichever way the CIF gives the space group:
 * the old (_symmetry_space_group_name_H-M) or new
 * (_space_group_name_H-M_alt) H-M tag alone, the Hall symbol -I 4 2 3
 * alone, the H-M symbol with all 96 operators of I m -3 m, or the
 * I-centring operators alone under the old or new loop tag (gemmi then
 * follows the loop, space group "I 1", which places the same two atoms).
 * Each file is read through CifReader::read, CifReader::parseCifContents
 * (the CODReader path) and StructureReaderSelector (-s x.cif).
 * Before the fix: passes as well (a pin). The symbol-only files are right
 * today only because gemmi swallows the std::length_error that its own
 * "S." step throws on an empty operator list and then tries the Hall and
 * H-M lookups; the explicit lookup must keep these results. */
TEST_CASE("CIF reader: bcc Fe gives both atoms of the cell, whether the "
          "space group comes from an H-M symbol, a Hall symbol or an "
          "operator loop",
          "[structure_readers][cif][symmetry]") {
  PeriodicTable::initialize();
  for (const char *name :
       {"fe_bcc_hm_only.cif", "fe_bcc_newtag_hm_only.cif",
        "fe_bcc_hall_only.cif", "fe_bcc_full_loop.cif",
        "fe_bcc_with_loop.cif", "fe_bcc_newloop.cif"}) {
    for (const Route route : kRoutes) {
      INFO(name << " via " << route_name(route));
      check_sites(read_cif(kCifDir + name, route).st, kFeBcc);
    }
  }
}

/* Pins: an H-M symbol alone expands other lattices too: rock-salt NaCl
 * ('F m -3 m', Na1 at the origin, Cl1 at the body centre) gives the 8 atoms
 * of the conventional cell, 4 Na on 4a and 4 Cl on 4b; hcp Mg
 * ('P 63/m m c', a non-orthogonal cell) gives 2 atoms on 2c,
 * (1/3,2/3,1/4) and (2/3,1/3,3/4).
 * Before the fix: passes as well (a pin), by the same swallowed exception
 * as the bcc Fe symbol-only files. */
TEST_CASE("CIF reader: an H-M symbol alone gives the full cell of rock-salt "
          "NaCl and hcp Mg",
          "[structure_readers][cif][symmetry]") {
  PeriodicTable::initialize();
  for (const Route route : kRoutes) {
    INFO(route_name(route));
    {
      INFO("nacl_hm_only.cif");
      check_sites(read_cif(kCifDir + "nacl_hm_only.cif", route).st, kNaCl);
    }
    {
      INFO("mg_hcp_hm_only.cif");
      check_sites(read_cif(kCifDir + "mg_hcp_hm_only.cif", route).st,
                  kMgHcp);
    }
  }
}

/* Pins: a CIF whose only symmetry record is the space-group number
 * (_symmetry_Int_Tables_number 229) gives the full bcc cell: 2 atoms, at
 * (0,0,0) and (1/2,1/2,1/2), through every route. The number names the
 * space group in its standard setting, as the symbols do.
 * Before the fix: gemmi's lookup order "S.H2" has no number step, so the
 * structure stayed P 1 and the reader returned the asymmetric unit: the
 * count CHECK fails with "1 == 2" on each of the three routes. */
TEST_CASE("CIF reader: a space-group number alone gives the full cell",
          "[structure_readers][cif][symmetry]") {
  PeriodicTable::initialize();
  for (const Route route : kRoutes) {
    INFO("fe_bcc_number_only.cif via " << route_name(route));
    check_sites(read_cif(kCifDir + "fe_bcc_number_only.cif", route).st,
                kFeBcc);
  }
}

/* Pins: a CIF with no symmetry record at all is P 1 by convention: the
 * sites as listed, 1 atom for fe_bcc_none.cif. Two _atom_site rows that
 * name one position modulo a lattice vector (fe_frac1_dup.cif: Fe1 at
 * x = 0, Fe2 at x = 1, Fe3 at the body centre) are both kept, 3 atoms with
 * Fe1 and Fe2 on one site modulo 1: the reader does not merge distinct
 * rows (gemmi merges only the images of one site, within 0.4 A), and the
 * r = 0 pair is NNFinder's to refuse (the coincident-atoms item).
 * Before the fix: passes as well (a pin); it keeps the number lookup from
 * giving a space group to a CIF that states none. */
TEST_CASE("CIF reader: a CIF without a symmetry record is read as P 1, "
          "every row kept",
          "[structure_readers][cif][symmetry]") {
  PeriodicTable::initialize();
  for (const Route route : kRoutes) {
    INFO(route_name(route));
    {
      INFO("fe_bcc_none.cif");
      check_sites(read_cif(kCifDir + "fe_bcc_none.cif", route).st,
                  {{"Fe", {0.0, 0.0, 0.0}}});
    }
    {
      INFO("fe_frac1_dup.cif");
      check_sites(read_cif(kCifDir + "fe_frac1_dup.cif", route).st,
                  {{"Fe", {0.0, 0.0, 0.0}},
                   {"Fe", {1.0, 0.0, 0.0}},
                   {"Fe", {0.5, 0.5, 0.5}}});
    }
  }
}

// ==================================================== the gemmi diagnostic ====

/* Pins: when gemmi's SmallStructure::check_spacegroup() reports that the
 * symmetry record contradicts itself, the reader prints a WARNING that
 * quotes the diagnostic, and reads the structure from the operator loop:
 *   - fe_bcc_mismatch.cif: H-M 'P m -3 m' with the I-centring operators
 *     (space group I 1): 2 atoms, and the diagnostic "space group from H-M
 *     symbol (P m -3 m) differs";
 *   - fe_bcc_with_loop.cif and fe_bcc_newloop.cif: H-M 'I m -3 m' with only
 *     the two I-centring operators: 2 atoms, and the diagnostic names
 *     'I m -3 m' (the loop is I 1, so a general position would give 2 atoms
 *     instead of 96).
 * Through CifReader::read and StructureReaderSelector the WARNING names
 * the file; through parseCifContents (the CODReader path, no file) it is
 * still printed. Both std::cout and std::cerr are captured with the Logger
 * at WARNING, so either way of printing it is seen.
 * Before the fix: CifReader discarded the string check_spacegroup()
 * returned and printed nothing: every CHECK on the captured output fails
 * (it is empty), the atom counts pass. */
TEST_CASE("CIF reader: an inconsistent symmetry record gives a WARNING "
          "naming the file and quoting gemmi's diagnostic",
          "[structure_readers][cif][symmetry][warning]") {
  PeriodicTable::initialize();
  struct Case {
    const char *file;
    const char *symbol;  // the H-M symbol the diagnostic quotes
  };
  const Case cases[] = {{"fe_bcc_mismatch.cif", "P m -3 m"},
                        {"fe_bcc_with_loop.cif", "I m -3 m"},
                        {"fe_bcc_newloop.cif", "I m -3 m"}};
  for (const Case &c : cases) {
    const std::string path = kCifDir + c.file;
    for (const Route route : kRoutes) {
      INFO(c.file << " via " << route_name(route));
      const Read r = read_cif(path, route);
      INFO("printed:\n" << r.log);
      check_sites(r.st, kFeBcc);
      CHECK(contains(r.log, "WARNING"));
      CHECK(contains(r.log, "differs"));
      CHECK(contains(r.log, std::string("(") + c.symbol + ")"));
      if (route != Route::Contents) CHECK(contains(r.log, path));
    }
  }
}

/* Pins: a consistent or absent symmetry record prints no warning: the bcc
 * Fe files with the H-M symbol alone (old and new tag), the Hall symbol
 * alone, the H-M symbol with the full loop, or the number alone (once the
 * number sets the space group, the two agree), the file without any
 * symmetry record, the two-rows-on-one-site file, NaCl, hcp Mg, and the
 * cell-convention fixtures (H-M symbol, number and full operator loop, all
 * agreeing).
 * Before the fix: passes as well (nothing is printed); it keeps the WARNING
 * to records that contradict themselves, not records that are merely
 * terse. */
TEST_CASE("CIF reader: a consistent or absent symmetry record gives no "
          "warning",
          "[structure_readers][cif][symmetry][warning]") {
  PeriodicTable::initialize();
  const std::vector<std::string> paths = {
      kCifDir + "fe_bcc_hm_only.cif",
      kCifDir + "fe_bcc_newtag_hm_only.cif",
      kCifDir + "fe_bcc_hall_only.cif",
      kCifDir + "fe_bcc_full_loop.cif",
      kCifDir + "fe_bcc_number_only.cif",
      kCifDir + "fe_bcc_none.cif",
      kCifDir + "fe_frac1_dup.cif",
      kCifDir + "nacl_hm_only.cif",
      kCifDir + "mg_hcp_hm_only.cif",
      "./tests_data/cell_convention/cif_hcp_mg.cif",
      "./tests_data/cell_convention/cif_triclinic_sio.cif"};
  for (const auto &path : paths) {
    for (const Route route : kRoutes) {
      INFO(path << " via " << route_name(route));
      const Read r = read_cif(path, route);
      INFO("printed:\n" << r.log);
      CHECK_FALSE(contains(lower(r.log), "warning"));
    }
  }
}

// ============================================================ occupancy ====

/* Pins: a site with _atom_site_occupancy below 1 is refused with a
 * std::runtime_error naming the site and its occupancy: "site Fe1 has
 * occupancy 0.5" for disordered fcc FeNi (feni_disordered.cif, F m -3 m,
 * Fe1 and Ni1 sharing 4a at 0.5 each), through every route; through
 * CifReader::read and StructureReaderSelector the message also names the
 * file. A single partly vacant site (Fe1 at 0.75 in bcc Fe) is refused the
 * same way, "site Fe1 has occupancy 0.75": Tadah has no representation
 * for either.
 * Before the fix: the occupancy column was ignored and nothing was
 * thrown: FeNi was read as 8 atoms (Fe and Ni coincident in pairs on the 4
 * fcc sites) and the vacancy as 2 Fe atoms, so each case fails with
 * "accepted: 8 atoms" (or "accepted: 2 atoms"). */
TEST_CASE("CIF reader: a partly occupied site is refused, naming the site "
          "and its occupancy",
          "[structure_readers][cif][occupancy]") {
  PeriodicTable::initialize();
  const std::string path = kCifDir + "feni_disordered.cif";
  for (const Route route : kRoutes) {
    INFO("feni_disordered.cif via " << route_name(route));
    try {
      const Read r = read_cif(path, route);
      FAIL_CHECK("accepted: " << r.st.natoms() << " atoms");
    } catch (const std::runtime_error &e) {
      const std::string what = e.what();
      INFO("message: " << what);
      CHECK(contains(what, "site Fe1 has occupancy 0.5"));
      if (route != Route::Contents) CHECK(contains(what, path));
    }
  }

  const std::string vacancy = cif_with_occupancy(
      "Fe_bcc_vacancy", 2.8665, "I m -3 m", "Fe1 Fe 0 0 0 0.75\n");
  INFO("bcc Fe with Fe1 at occupancy 0.75 (parseCifContents)");
  try {
    const Read r = parse_cif_text(vacancy);
    FAIL_CHECK("accepted: " << r.st.natoms() << " atoms");
  } catch (const std::runtime_error &e) {
    const std::string what = e.what();
    INFO("message: " << what);
    CHECK(contains(what, "site Fe1 has occupancy 0.75"));
  }
}

/* Pins: full occupancy is accepted however the column writes it: "1.0",
 * "1", or "?" (unknown, which CIF and gemmi take as 1). Rock-salt NaCl
 * with an occupancy column gives its 8 atoms, and nothing is printed.
 * Before the fix: passes as well (the column was ignored); it keeps the
 * refusal to occupancies below 1. */
TEST_CASE("CIF reader: an occupancy of 1, written or unknown, is accepted",
          "[structure_readers][cif][occupancy]") {
  PeriodicTable::initialize();
  const std::vector<std::array<const char *, 2>> occupancies = {
      {"1.0", "1.0"}, {"1", "1"}, {"1.0", "?"}};
  for (const auto &occ : occupancies) {
    INFO("Na1 occupancy " << occ[0] << ", Cl1 occupancy " << occ[1]);
    const std::string text = cif_with_occupancy(
        "NaCl_occupancy", 5.64, "F m -3 m",
        std::string("Na1 Na 0 0 0 ") + occ[0] + "\n" +
            "Cl1 Cl 0.5 0.5 0.5 " + occ[1] + "\n");
    Read r;
    try {
      r = parse_cif_text(text);
    } catch (const std::exception &e) {
      FAIL_CHECK("refused: " << e.what());
      continue;
    }
    INFO("printed:\n" << r.log);
    check_sites(r.st, kNaCl);
    CHECK_FALSE(contains(lower(r.log), "warning"));
  }
}

/* Pins: an occupancy that is not a number, or above 1, is refused, naming
 * the site: 'abc' ("site Na1 has occupancy 'abc', which is not a number"),
 * 1.5 ("site Na1 has occupancy 1.5"), and a quoted '0.5' is read as 0.5 and
 * refused as a partly occupied site ("site Na1 has occupancy 0.5").
 * Before the fix: the occupancy column was ignored (and gemmi reads a value
 * that is not a number, or a quoted one, as 1), so all three were accepted
 * as rock-salt NaCl (8 atoms). */
TEST_CASE("CIF reader: an occupancy that is not a number, or above 1, is refused",
          "[structure_readers][cif][occupancy]") {
  PeriodicTable::initialize();
  const std::vector<std::array<std::string, 2>> cases = {
      {"abc", "site Na1 has occupancy 'abc'"},
      {"1.5", "site Na1 has occupancy 1.5"},
      {"'0.5'", "site Na1 has occupancy 0.5"}};
  for (const auto &c : cases) {
    INFO("Na1 occupancy " << c[0]);
    const std::string text = cif_with_occupancy(
        "NaCl_bad_occupancy", 5.64, "F m -3 m",
        "Na1 Na 0 0 0 " + c[0] + "\nCl1 Cl 0.5 0.5 0.5 1\n");
    try {
      const Read r = parse_cif_text(text);
      FAIL_CHECK("accepted: " << r.st.natoms() << " atoms");
    } catch (const std::runtime_error &e) {
      INFO("message: " << e.what());
      CHECK(contains(e.what(), c[1]));
    }
  }
}
