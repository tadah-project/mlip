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
//   - a space-group number alone (no operator loop, no symbol) names the
//     group, not the setting: the reader takes the number's H-M symbol in
//     the setting that fits the cell angles (hexagonal or rhombohedral
//     axes) and origin choice 2, as gemmi does for an H-M symbol, and, when
//     the number has more than one tabulated setting, a WARNING names the
//     setting used;
//   - an operator loop that is incomplete is not completed or replaced by
//     the number: gemmi falls back to the H-M symbol, or to P 1 when there
//     is none, and the WARNING says which; a complete loop that is no
//     tabulated setting expands the sites with its own operators;
//   - a site with _atom_site_occupancy below 1 is refused, naming the site
//     and its occupancy: Tadah has no representation for a disordered or
//     partly vacant site, and two species sharing a site would become
//     coincident atoms. Occupancies within 1e-6 of 1 are taken as 1.
//
// Fixtures: tests_data/cif/ (one file per row of the investigation table,
// tmp/known-issues/report_leftovers.md section 6). The expected atoms are
// hand-written from the Wyckoff positions (bcc Fe 2a, NaCl 4a + 4b, hcp Mg
// 2c, Bi 2c of R -3 m:R and 6c of R -3 m:H, diamond Si 8a of F d -3 m
// origin choice 2) or from the definition of the group (the 96 images of a
// general position of I m -3 m); none is computed with the reader under
// test.

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
// Bi, R -3 m on rhombohedral axes (R -3 m:R), Bi on 2c (x,x,x) and
// (-x,-x,-x), x = 0.2339.
const double kBiX = 0.2339;
const std::vector<Site> kBiRh = {{"Bi", {kBiX, kBiX, kBiX}},
                                 {"Bi", {-kBiX, -kBiX, -kBiX}}};
// The same crystal on hexagonal axes (R -3 m:H), Bi on 6c (0,0,z) with
// z = x: (0,0,+-z) and the same with the R-centring translations
// (2/3,1/3,1/3) and (1/3,2/3,2/3).
const std::vector<Site> kBiHex = {
    {"Bi", {0.0, 0.0, kBiX}},
    {"Bi", {0.0, 0.0, -kBiX}},
    {"Bi", {2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 + kBiX}},
    {"Bi", {2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 - kBiX}},
    {"Bi", {1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0 + kBiX}},
    {"Bi", {1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0 - kBiX}}};
// Diamond Si, F d -3 m origin choice 2, Si on 8a: (1/8,1/8,1/8) and
// (7/8,7/8,7/8), each with the F-centring translations.
const std::vector<Site> kSiDiamond = {
    {"Si", {0.125, 0.125, 0.125}}, {"Si", {0.125, 0.625, 0.625}},
    {"Si", {0.625, 0.125, 0.625}}, {"Si", {0.625, 0.625, 0.125}},
    {"Si", {0.875, 0.875, 0.875}}, {"Si", {0.875, 0.375, 0.375}},
    {"Si", {0.375, 0.875, 0.375}}, {"Si", {0.375, 0.375, 0.875}}};

// The 96 images of the general position p in I m -3 m: the point group
// m -3 m is the 6 permutations of (x,y,z), each with the 8 choices of
// signs (48 operations), and the I centring adds (1/2,1/2,1/2) to each.
std::vector<Site> im3m_general_position(const std::string &symbol,
                                        const V3 &p) {
  const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                           {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  std::vector<Site> out;
  for (const auto &perm : perms)
    for (int signs = 0; signs < 8; ++signs)
      for (const double t : {0.0, 0.5}) {
        V3 f;
        for (int k = 0; k < 3; ++k)
          f[k] = (((signs >> k) & 1) ? -1.0 : 1.0) * p[perm[k]] + t;
        out.push_back({symbol, f});
      }
  return out;
}

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
size_t count_of(const std::string &text, const std::string &what) {
  size_t n = 0;
  for (size_t at = text.find(what); at != std::string::npos;
       at = text.find(what, at + what.size()))
    ++n;
  return n;
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

// How the reader names the source in a WARNING or an error: the file, or,
// through parseCifContents (no file), the data block.
std::string source_of(const std::string &path, Route route,
                      const std::string &block) {
  return route == Route::Contents ? "CIF text (data_" + block + ")"
                                  : "CIF file " + path;
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

// The number-only fixtures with more than one tabulated setting, and the
// setting (gemmi's SpaceGroup::xhm()) the cell must be built with.
struct NumberOnlyCase {
  const char *file;
  const char *block;    // the data block, without "data_"
  int number;           // _symmetry_Int_Tables_number
  const char *setting;  // the setting that fits the cell
  const std::vector<Site> *sites;
};
const NumberOnlyCase kNumberOnlyCases[] = {
    {"bi_rh_number_only.cif", "Bi_rh_number_only", 166, "R -3 m:R", &kBiRh},
    {"bi_hex_number_only.cif", "Bi_hex_number_only", 166, "R -3 m:H",
     &kBiHex},
    {"si_diamond_number_only.cif", "Si_diamond_number_only", 227,
     "F d -3 m:2", &kSiDiamond}};

/* Pins: a CIF whose only symmetry record is the space-group number is
 * expanded in the setting of that number that fits the cell, through every
 * route:
 *   - bi_rh_number_only.cif: 166 on rhombohedral axes (a = 4.7236 A,
 *     alpha = beta = gamma = 57.35 deg), Bi1 at (x,x,x), x = 0.2339: the
 *     setting is R -3 m:R, and Bi on 2c gives 2 atoms, (x,x,x) and
 *     (-x,-x,-x);
 *   - bi_hex_number_only.cif: the same crystal on hexagonal axes
 *     (alpha = 90, gamma = 120 deg), Bi1 at (0,0,z): R -3 m:H, Bi on 6c,
 *     6 atoms;
 *   - si_diamond_number_only.cif: 227, a = 5.431 A, Si1 at (1/8,1/8,1/8):
 *     F d -3 m origin choice 2 (the reference setting, origin at the
 *     inversion centre), Si on 8a, 8 atoms.
 * The number names the space-group type, not the setting. The reader
 * resolves it as gemmi resolves an H-M symbol given alone: the number's
 * H-M symbol, with the cell angles choosing hexagonal or rhombohedral axes
 * and origin choice 2 preferred (find_spacegroup_by_name(hm, alpha, gamma,
 * "2")).
 * Before the fix: the number was looked up with find_spacegroup_by_number,
 * which returns gemmi's first table entry for it (the CCP4 setting,
 * R -3 m:H or F d -3 m:1) whatever the cell. Bi on rhombohedral axes was
 * expanded with the hexagonal-setting operators, 36 atoms (the count CHECK
 * fails with "36 == 2"), and Si with origin choice 1, in which
 * (1/8,1/8,1/8) is a 16-fold site, 16 atoms ("16 == 8"), on each of the
 * three routes. Bi on hexagonal axes passes (6 atoms, a pin). */
TEST_CASE("CIF reader: a space-group number alone is expanded in the "
          "setting that fits the cell, origin choice 2",
          "[structure_readers][cif][symmetry]") {
  PeriodicTable::initialize();
  for (const NumberOnlyCase &c : kNumberOnlyCases) {
    for (const Route route : kRoutes) {
      INFO(c.file << " (" << c.setting << ") via " << route_name(route));
      check_sites(read_cif(kCifDir + c.file, route).st, *c.sites);
    }
  }
}

/* Pins: when the space-group number is the only symmetry record and the
 * number has more than one tabulated setting, the reader prints one
 * WARNING naming the source, the number and the setting the cell was built
 * with. The text is
 *   WARNING: <source>: the space group is given only by its number (<n>),
 *   which has more than one setting; the unit cell is built with space
 *   group <setting>, the one that fits the cell angles (origin choice 2
 *   where the group has two).
 * with <source> "CIF file <path>" (CifReader::read and
 * StructureReaderSelector) or "CIF text (data_<block>)" (parseCifContents),
 * and <setting> gemmi's SpaceGroup::xhm(): R -3 m:R for
 * bi_rh_number_only.cif, R -3 m:H for bi_hex_number_only.cif, F d -3 m:2
 * for si_diamond_number_only.cif. The test checks "WARNING: <source>: ",
 * "only by its number (<n>)", "the unit cell is built with space group
 * <setting>", and that "WARNING" appears once (the record is otherwise
 * consistent). The number of a group with a single setting (229,
 * fe_bcc_number_only.cif) leaves no choice and prints nothing (the
 * no-warning test below).
 * Before the fix: nothing was printed (gemmi's check_spacegroup() finds the
 * number consistent with the setting it was looked up in), so every CHECK
 * on the captured output fails, on each route. */
TEST_CASE("CIF reader: a space-group number alone gives a WARNING naming "
          "the setting used",
          "[structure_readers][cif][symmetry][warning]") {
  PeriodicTable::initialize();
  for (const NumberOnlyCase &c : kNumberOnlyCases) {
    const std::string path = kCifDir + c.file;
    for (const Route route : kRoutes) {
      INFO(c.file << " via " << route_name(route));
      const Read r = read_cif(path, route);
      INFO("printed:\n" << r.log);
      CHECK(contains(r.log,
                     "WARNING: " + source_of(path, route, c.block) + ": "));
      CHECK(contains(r.log, "only by its number (" +
                                std::to_string(c.number) + ")"));
      CHECK(contains(r.log, std::string("the unit cell is built with space "
                                        "group ") +
                                c.setting));
      CHECK(count_of(r.log, "WARNING") == 1);
    }
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
 * number sets the space group, the two agree, and 229 has a single
 * tabulated setting, I m -3 m, so the number leaves no choice to report;
 * compare the number-only WARNING test below), the file without any
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

/* Pins: an incomplete operator loop is neither completed nor replaced by
 * the number. The loop {x,y,z; x+1/2,y+1/2,z+1/2; y,z,x} (the threefold
 * y,z,x without its square z,x,y) names no group, so
 *   - with the H-M symbol 'I m -3 m' (fe_bcc_partial_loop_hm.cif) gemmi
 *     falls back to the symbol, and Fe1 on the general position
 *     (0.08,0.20,0.32) gives all 96 images of I m -3 m (the 48 signed
 *     permutations of (x,y,z), and each plus (1/2,1/2,1/2)); the WARNING
 *     says "the unit cell is built with space group I m -3 m";
 *   - with only the number 229 (fe_bcc_partial_loop_number.cif) gemmi does
 *     not look at the number, and the cell is P 1: 1 atom, the site as
 *     listed; the WARNING says "the unit cell is built with no symmetry
 *     (P 1, the sites as listed)".
 * Both WARNINGs name the source and quote gemmi's "symops list is
 * incomplete or incorrect". Through every route.
 * Before the fix: passes as well (a pin). The number is resolved only when
 * there is no operator loop, so the fix must leave both as they are. */
TEST_CASE("CIF reader: an incomplete operator loop falls back to the H-M "
          "symbol, or to P 1 when there is only the number, with a WARNING",
          "[structure_readers][cif][symmetry][warning]") {
  PeriodicTable::initialize();
  const V3 general = {0.08, 0.20, 0.32};
  struct Case {
    const char *file;
    const char *block;
    const char *built_with;
    std::vector<Site> sites;
  };
  const std::vector<Case> cases = {
      {"fe_bcc_partial_loop_hm.cif", "Fe_bcc_partial_loop_hm",
       "the unit cell is built with space group I m -3 m.",
       im3m_general_position("Fe", general)},
      {"fe_bcc_partial_loop_number.cif", "Fe_bcc_partial_loop_number",
       "the unit cell is built with no symmetry (P 1, the sites as listed).",
       {{"Fe", general}}}};
  for (const Case &c : cases) {
    const std::string path = kCifDir + c.file;
    for (const Route route : kRoutes) {
      INFO(c.file << " via " << route_name(route));
      const Read r = read_cif(path, route);
      INFO("printed:\n" << r.log);
      check_sites(r.st, c.sites);
      CHECK(contains(r.log,
                     "WARNING: " + source_of(path, route, c.block) + ": "));
      CHECK(contains(r.log, "symops list is incomplete or incorrect"));
      CHECK(contains(r.log, c.built_with));
    }
  }
}

/* Pins: the WARNING on an inconsistent record says what the cell was built
 * with and names the source:
 *   - fe_bcc_mismatch.cif (H-M 'P m -3 m' with only the I-centring
 *     operators): "the unit cell is built with space group I 1." (the
 *     loop's group); through parseCifContents (no file) the source is the
 *     data block, "WARNING: CIF text (data_Fe_bcc_mismatch): ", through
 *     CifReader::read and the selector "WARNING: CIF file <path>: ";
 *   - p2_shifted_origin_loop.cif (a complete loop that is no tabulated
 *     setting: P 2 with the twofold axis through (1/4,0,z),
 *     {x,y,z; -x+1/2,-y,z}): gemmi's "space group from symops not found in
 *     the table" and "the unit cell is built with the 2 operators of the
 *     symmetry-operator loop (no tabulated space group)."; Fe1 at
 *     (0.1,0.2,0.3) gives 2 atoms, the second at (0.4,0.8,0.3).
 * Before the fix: passes as well (a pin of the WARNING text). */
TEST_CASE("CIF reader: the WARNING names the symmetry the cell was built "
          "with and the file or data block",
          "[structure_readers][cif][symmetry][warning]") {
  PeriodicTable::initialize();
  for (const Route route : kRoutes) {
    INFO(route_name(route));
    {
      const std::string path = kCifDir + "fe_bcc_mismatch.cif";
      INFO("fe_bcc_mismatch.cif");
      const Read r = read_cif(path, route);
      INFO("printed:\n" << r.log);
      check_sites(r.st, kFeBcc);
      CHECK(contains(r.log, "WARNING: " +
                                source_of(path, route, "Fe_bcc_mismatch") +
                                ": "));
      CHECK(contains(r.log,
                     "the unit cell is built with space group I 1."));
    }
    {
      const std::string path = kCifDir + "p2_shifted_origin_loop.cif";
      INFO("p2_shifted_origin_loop.cif");
      const Read r = read_cif(path, route);
      INFO("printed:\n" << r.log);
      check_sites(r.st, {{"Fe", {0.1, 0.2, 0.3}}, {"Fe", {0.4, 0.8, 0.3}}});
      CHECK(contains(r.log,
                     "WARNING: " +
                         source_of(path, route, "P2_shifted_origin_loop") +
                         ": "));
      CHECK(contains(r.log, "space group from symops not found in the table"));
      CHECK(contains(r.log,
                     "the unit cell is built with the 2 operators of the "
                     "symmetry-operator loop (no tabulated space group)."));
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

/* Pins: an occupancy within 1e-6 of 1, or unknown, is taken as 1: '.'
 * (CIF's "inapplicable", which gemmi, like '?', reads as no value),
 * 1.0000005 (1 + 5e-7) and 0.9999995 (1 - 5e-7) each give rock-salt NaCl's
 * 8 atoms and print nothing.
 * Before the fix: passes as well (a pin of the 1e-6 tolerance and of '.'). */
TEST_CASE("CIF reader: an occupancy within 1e-6 of 1, or '.', is accepted",
          "[structure_readers][cif][occupancy]") {
  PeriodicTable::initialize();
  for (const char *occ : {".", "1.0000005", "0.9999995"}) {
    INFO("Na1 occupancy " << occ);
    const std::string text = cif_with_occupancy(
        "NaCl_occupancy_near_1", 5.64, "F m -3 m",
        std::string("Na1 Na 0 0 0 ") + occ + "\nCl1 Cl 0.5 0.5 0.5 1\n");
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

/* Pins: an occupancy more than 1e-6 from 1 is refused, and the message
 * quotes the occupancy as the CIF writes it, so that it shows why:
 *   - 0.999998 (1 - 2e-6): "site Na1 has occupancy 0.999998; Tadah has no
 *     representation for a partly occupied site ...";
 *   - 1.000002 (1 + 2e-6): "site Na1 has occupancy 1.000002, above 1,
 *     which is not a valid occupancy."
 * Before the fix: both are refused, but the value is streamed with the
 * default precision (6 significant digits), so the second message reads
 * "site Na1 has occupancy 1, above 1, which is not a valid occupancy.",
 * which contradicts itself: the CHECK for "site Na1 has occupancy
 * 1.000002" fails; the rest passes. */
TEST_CASE("CIF reader: an occupancy 2e-6 from 1 is refused, quoting the "
          "value as written",
          "[structure_readers][cif][occupancy]") {
  PeriodicTable::initialize();
  const std::vector<std::array<std::string, 3>> cases = {
      {"0.999998", "site Na1 has occupancy 0.999998",
       "no representation for a partly occupied site"},
      {"1.000002", "site Na1 has occupancy 1.000002", "above 1"}};
  for (const auto &c : cases) {
    INFO("Na1 occupancy " << c[0]);
    const std::string text = cif_with_occupancy(
        "NaCl_occupancy_off_1", 5.64, "F m -3 m",
        "Na1 Na 0 0 0 " + c[0] + "\nCl1 Cl 0.5 0.5 0.5 1\n");
    try {
      const Read r = parse_cif_text(text);
      FAIL_CHECK("accepted: " << r.st.natoms() << " atoms");
    } catch (const std::runtime_error &e) {
      const std::string what = e.what();
      INFO("message: " << what);
      CHECK(contains(what, "CIF text (data_NaCl_occupancy_off_1): "));
      CHECK(contains(what, c[1]));
      CHECK(contains(what, c[2]));
    }
  }
}
