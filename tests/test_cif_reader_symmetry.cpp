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
//     or new tag) gives every atom of the unit cell; a tag whose value is
//     unknown ('?') does not hide the other tag of the pair;
//   - a CIF with no symmetry record is P 1: the sites as listed, and two
//     rows at one position modulo a lattice vector are both kept;
//   - when the symmetry record contradicts itself (gemmi's
//     SmallStructure::check_spacegroup() is not empty), the structure
//     follows the record read first and a WARNING names the file, quotes
//     the diagnostic and says what the cell was built with; a consistent
//     or absent record prints nothing;
//   - the H-M symbol is normalised before any lookup and before gemmi's
//     consistency check, with or without an operator loop: surrounding
//     blanks trimmed, ICSD's ' S' / ' Z' (either case) read as origin
//     choice 1 / 2, '_' removed (spglib's P2_1/c), subscripts in
//     parentheses closed up (P2(1)/c, P6(3)/mmc) and a trailing
//     ' (No. N)' dropped; a value of blanks is no symbol, one that this
//     otherwise leaves empty cannot be read;
//   - the order: the operator loop (a tabulated setting, or its own
//     operators when they form a group in no table); else (no loop, or a
//     loop that forms no group) the Hall symbol (one in no tabulated
//     setting expands the sites with its own operators), else the H-M
//     symbol, in the setting it names or else origin choice 2 (also for
//     the current symbols such as 'C c c e') and hexagonal or rhombohedral
//     axes from the cell angles; else, without a loop only, the number;
//     else P 1. A symbol that cannot be read (or an H-M symbol that is a
//     number) is refused, quoting it ("cannot read"); a refusal from a
//     record read first says so ("read before");
//   - a space-group number alone is accepted only in 1..230 and gives the
//     setting its H-M symbol would; a WARNING names the setting used when
//     the cell could be in another setting of the group: another origin
//     choice, or another choice of axes that fits the cell (not the R
//     groups' other axes, the larger centred cells of P 1 or I 4/m m m, or
//     gemmi's extra CCP4 origins of 22, 23, 94 and 197);
//   - an incomplete operator loop is not replaced by the number: with no
//     symbol the cell is P 1, with the WARNING;
//   - a monoclinic group, however it was given, must fit the cell: the
//     angles between its unique axis and the other two axes (b: alpha,
//     gamma; c: alpha, beta; a: beta, gamma) are 90 within 1e-3 degrees,
//     or the CIF is refused, naming the group and the angles as read; so
//     must an R group its axes (hexagonal: alpha = beta = 90, gamma = 120;
//     rhombohedral: alpha = beta = gamma), those it names or, without, the
//     cell's. Operators in no tabulated setting are checked by the
//     tabulated group with their rotations and centring, else (an R or H
//     centring gemmi does not tabulate, such as the reverse setting) on
//     hexagonal axes, else by the tabulated primitive group with their
//     rotations: a supercell's extra translations (x+1/3, ...) are no R
//     centring. A refusal says what can change the result;
//   - with a loop or a Hall symbol, the H-M symbol is compared with the
//     group used by its operations, so an equivalent name of the same
//     setting ('C c c b:1' for C c c a:1) is no inconsistency;
//   - a site with _atom_site_occupancy below 1 is refused, naming the site
//     and its occupancy: Tadah has no representation for a disordered or
//     partly vacant site, and two species sharing a site would become
//     coincident atoms. Occupancies within 1e-6 of 1 are taken as 1.
//
// Fixtures: tests_data/cif/ (one file per row of the investigation table,
// tmp/known-issues/report_leftovers.md section 6, and
// si_diamond_icsd_z_loop.cif, an ICSD-style record with the 192 operators
// of F d -3 m:2); the space-group policy tests write their CIF inline
// (cif_cell). The expected atoms are hand-written from the Wyckoff
// positions (bcc Fe 2a, NaCl 4a + 4b, hcp Mg 2c, Bi 2c of R -3 m:R and 6c
// of R -3 m:H, diamond Si 8a of F d -3 m origin choice 1 and 2) or from
// the definition of the group (the 96 images of a general position of
// I m -3 m, the 4 images of a general position under the operators of a
// monoclinic setting); none is computed with the reader under test.

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
#include <cstdio>
#include <cstdlib>
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
  bool warns;  // whether the cell could be in another setting of the group
};
const NumberOnlyCase kNumberOnlyCases[] = {
    {"bi_rh_number_only.cif", "Bi_rh_number_only", 166, "R -3 m:R", &kBiRh,
     false},
    {"bi_hex_number_only.cif", "Bi_hex_number_only", 166, "R -3 m:H",
     &kBiHex, false},
    {"si_diamond_number_only.cif", "Si_diamond_number_only", 227,
     "F d -3 m:2", &kSiDiamond, true}};

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
 * cell could be in another tabulated setting of the group (another origin
 * choice, another choice of axes), the reader prints one WARNING naming
 * the source, the number and the setting the cell was built with. The
 * text is
 *   WARNING: <source>: the space group is given only by its number (<n>),
 *   and the cell could be in more than one of its settings; the unit cell
 *   is built with space group <setting>, its standard setting (origin
 *   choice 2 where the group has two): give the H-M symbol if the cell is
 *   in another.
 * with <source> "CIF file <path>" (CifReader::read and
 * StructureReaderSelector) or "CIF text (data_<block>)" (parseCifContents),
 * and <setting> gemmi's SpaceGroup::xhm(): F d -3 m:2 for
 * si_diamond_number_only.cif (origin choice 1 is the same cell). The test
 * checks "WARNING: <source>: ", "only by its number (<n>)", "the unit cell
 * is built with space group <setting>", and that "WARNING" appears once
 * (the record is otherwise consistent). The hexagonal and rhombohedral
 * axes of R -3 m (166) are two cells, which the cell angles tell apart:
 * bi_rh_number_only.cif (R -3 m:R) and bi_hex_number_only.cif (R -3 m:H)
 * print nothing. The number of a group with a single setting (229,
 * fe_bcc_number_only.cif) leaves no choice and prints nothing (the
 * no-warning test below).
 * Before the fix: nothing was printed (gemmi's check_spacegroup() finds the
 * number consistent with the setting it was looked up in), so every CHECK
 * on the captured output fails, on each route.
 * Review round 5: the Bi files printed the WARNING (every setting of 166
 * counted, then every setting with the letter R), so count_of == 0 fails
 * for them. */
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
      if (!c.warns) {
        CHECK(count_of(r.log, "WARNING") == 0);
        continue;
      }
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

namespace {

// A CIF with cell a, b, c, alpha, beta, gamma, the symmetry record
// `symmetry` (tag lines, and an operator loop if any) and the _atom_site
// rows `sites`. The cell is written with 15 significant digits, so that a
// value such as 90.00101 reaches the CIF as given (the stream's default of
// 6 digits would write 90.001).
std::string cif_cell(const std::string &block, double a, double b, double c,
                     double alpha, double beta, double gamma,
                     const std::string &symmetry, const std::string &sites) {
  std::ostringstream s;
  s.precision(15);
  s << "data_" << block << "\n"
    << "_cell_length_a " << a << "\n_cell_length_b " << b
    << "\n_cell_length_c " << c << "\n_cell_angle_alpha " << alpha
    << "\n_cell_angle_beta " << beta << "\n_cell_angle_gamma " << gamma
    << "\n" << symmetry
    << "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
    << "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
    << sites;
  return s.str();
}

// Whether `text` writes `value` so that it reads back exactly: some number
// in it (a run of characters from a digit that does not continue another
// number) parses with std::strtod to `value`. A message that prints
// 90.00101 as 90.001 does not mention 90.00101; one that prints it with
// 17 significant digits (90.001009999999994) or as the shortest form that
// reads back (90.00101) does.
bool mentions_value(const std::string &text, double value) {
  for (size_t i = 0; i < text.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(text[i]))) continue;
    if (i > 0 && (std::isdigit(static_cast<unsigned char>(text[i - 1])) ||
                  text[i - 1] == '.'))
      continue;
    if (std::strtod(text.c_str() + i, nullptr) == value) return true;
  }
  return false;
}

// Reads `text` through parseCifContents expecting it to be accepted: true
// with `r` filled, or false after a failed check quoting the refusal.
bool accepted(const std::string &text, Read &r) {
  try {
    r = parse_cif_text(text);
  } catch (const std::exception &e) {
    FAIL_CHECK("refused: " << e.what());
    return false;
  }
  return true;
}

// Reads `text` through parseCifContents expecting it to be refused with a
// std::runtime_error: its message, or "" after a failed check saying how
// many atoms were read instead.
std::string refusal(const std::string &text) {
  try {
    const Read r = parse_cif_text(text);
    FAIL_CHECK("accepted: " << r.st.natoms() << " atoms");
  } catch (const std::runtime_error &e) {
    return e.what();
  }
  return "";
}

// Fe1 at the general position (0.1,0.2,0.3) and its images under the
// operators of a monoclinic setting:
//   P 1 21/c 1: x,y,z; -x,y+1/2,-z+1/2; -x,-y,-z; x,-y+1/2,z+1/2;
//   P 1 21/m 1: x,y,z; -x,y+1/2,-z;     -x,-y,-z; x,-y+1/2,z;
//   P 1 1 21/m: x,y,z; -x,-y,z+1/2;     -x,-y,-z; x,y,-z+1/2;
//   P 21/m 1 1: x,y,z; x+1/2,-y,-z;     -x,-y,-z; -x+1/2,y,z;
// and under -P 2ybc (x+1/4,y,z), P 1 21/c 1 with the origin moved by a/4
// (no tabulated setting): x,y,z; -x+1/2,y+1/2,-z+1/2; -x+1/2,-y,-z;
// x,-y+1/2,z+1/2.
const std::vector<Site> kFeP21c = {{"Fe", {0.1, 0.2, 0.3}},
                                   {"Fe", {-0.1, 0.7, 0.2}},
                                   {"Fe", {-0.1, -0.2, -0.3}},
                                   {"Fe", {0.1, 0.3, 0.8}}};
const std::vector<Site> kFeP21mB = {{"Fe", {0.1, 0.2, 0.3}},
                                    {"Fe", {-0.1, 0.7, -0.3}},
                                    {"Fe", {-0.1, -0.2, -0.3}},
                                    {"Fe", {0.1, 0.3, 0.3}}};
const std::vector<Site> kFeP21mC = {{"Fe", {0.1, 0.2, 0.3}},
                                    {"Fe", {-0.1, -0.2, 0.8}},
                                    {"Fe", {-0.1, -0.2, -0.3}},
                                    {"Fe", {0.1, 0.2, 0.2}}};
const std::vector<Site> kFeP21mA = {{"Fe", {0.1, 0.2, 0.3}},
                                    {"Fe", {0.6, -0.2, -0.3}},
                                    {"Fe", {-0.1, -0.2, -0.3}},
                                    {"Fe", {0.4, 0.2, 0.3}}};
const std::vector<Site> kFeP21cShifted = {{"Fe", {0.1, 0.2, 0.3}},
                                          {"Fe", {0.4, 0.7, 0.2}},
                                          {"Fe", {0.4, -0.2, -0.3}},
                                          {"Fe", {0.1, 0.3, 0.8}}};
// Diamond Si, F d -3 m origin choice 1, Si on 8a: (0,0,0) and
// (1/4,1/4,1/4), each with the F-centring translations.
const std::vector<Site> kSiDiamondOrigin1 = {
    {"Si", {0.0, 0.0, 0.0}},    {"Si", {0.0, 0.5, 0.5}},
    {"Si", {0.5, 0.0, 0.5}},    {"Si", {0.5, 0.5, 0.0}},
    {"Si", {0.25, 0.25, 0.25}}, {"Si", {0.25, 0.75, 0.75}},
    {"Si", {0.75, 0.25, 0.75}}, {"Si", {0.75, 0.75, 0.25}}};

const std::string kFeGeneral = "Fe1 Fe 0.1 0.2 0.3\n";

std::string hm_line(const std::string &hm) {
  return "_symmetry_space_group_name_H-M '" + hm + "'\n";
}
std::string hall_line(const std::string &hall) {
  return "_symmetry_space_group_name_Hall '" + hall + "'\n";
}
std::string number_line(const std::string &n) {
  return "_symmetry_Int_Tables_number " + n + "\n";
}

}  // namespace

/* Pins: an H-M symbol with the ICSD origin suffix (' S' origin choice 1,
 * ' Z' origin choice 2) is read as that setting, and the number next to it
 * adds nothing (no "only by its number" WARNING): diamond Si (a = 5.431 A)
 * as 'F d -3 m S' with Si at (0,0,0) and as 'F d -3 m Z' with Si at
 * (1/8,1/8,1/8), each with _symmetry_Int_Tables_number 227, gives the 8
 * atoms of the cell. An H-M symbol nobody knows ('F d -3 m Q'), with the
 * number, is refused, naming the symbol: the setting cannot be told.
 * Before the fix: gemmi does not read the suffixes, so the number was used
 * in origin choice 2 with a WARNING that the group was given only by its
 * number: 16 atoms 1.92 A apart for the 'S' file; the unknown symbol was
 * accepted the same way. */
TEST_CASE("CIF reader: an ICSD origin suffix is read, and an unknown H-M "
          "symbol is refused",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  const std::string n227 = "_symmetry_Int_Tables_number 227\n";
  struct Case { std::string hm, x; };
  for (const Case &c : {Case{"F d -3 m S", "0"}, Case{"F d -3 m Z", "0.125"}}) {
    INFO(c.hm);
    const std::string text = cif_cell(
        "Si_icsd", 5.431, 5.431, 5.431, 90, 90, 90,
        "_symmetry_space_group_name_H-M '" + c.hm + "'\n" + n227,
        "Si1 Si " + c.x + " " + c.x + " " + c.x + "\n");
    Read r;
    try {
      r = parse_cif_text(text);
    } catch (const std::exception &e) {
      FAIL_CHECK("refused: " << e.what());
      continue;
    }
    INFO("printed:\n" << r.log);
    CHECK(r.st.natoms() == 8);
    CHECK_FALSE(contains(r.log, "only by its number"));
  }
  const std::string unknown = cif_cell(
      "Si_unknown_symbol", 5.431, 5.431, 5.431, 90, 90, 90,
      "_symmetry_space_group_name_H-M 'F d -3 m Q'\n" + n227,
      "Si1 Si 0.125 0.125 0.125\n");
  try {
    const Read r = parse_cif_text(unknown);
    FAIL_CHECK("accepted: " << r.st.natoms() << " atoms");
  } catch (const std::runtime_error &e) {
    INFO("message: " << e.what());
    CHECK(contains(e.what(), "F d -3 m Q"));
  }
}

/* Pins: a monoclinic space-group number alone (14) is used in its standard
 * setting, unique axis b, only when the cell has that form (alpha = gamma =
 * 90): a cell with beta = 110 gives the 4 atoms of P 1 21/c 1 and the
 * WARNING naming it; a cell with gamma = 110 (unique axis c) is refused,
 * naming the setting (P 1 21/c 1), its unique axis b and the angle that
 * breaks it (gamma 110).
 * Before the fix: the number was expanded in P 1 21/c 1 whatever the cell,
 * so the gamma = 110 cell gave 4 atoms with two different environments (not
 * one crystal), with a WARNING saying the setting fitted the cell angles.
 * Updated for the space-group policy: the cell is now checked against the
 * monoclinic setting however the group was given (the monoclinic tests
 * below), so the refusal names the setting rather than saying the group
 * came "only by its number (14)". Before that fix: the refusal read "the
 * space group is given only by its number (14), a monoclinic group, and
 * the cell (alpha 90, beta 90, gamma 110) is not in its standard setting,
 * unique axis b ...", which does not name P 1 21/c 1: that CHECK fails. */
TEST_CASE("CIF reader: a monoclinic space-group number alone needs the "
          "unique-axis-b cell",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  const std::string n14 = "_symmetry_Int_Tables_number 14\n";
  {
    const std::string b_unique = cif_cell("P21c_b", 5, 6, 7, 90, 110, 90, n14,
                                          "Fe1 Fe 0.1 0.2 0.3\n");
    Read r;
    REQUIRE_NOTHROW(r = parse_cif_text(b_unique));
    INFO("printed:\n" << r.log);
    CHECK(r.st.natoms() == 4);
    CHECK(contains(r.log, "only by its number (14)"));
    CHECK(contains(r.log, "P 1 21/c 1"));
  }
  const std::string c_unique = cif_cell("P21c_c", 5, 6, 7, 90, 90, 110, n14,
                                        "Fe1 Fe 0.1 0.2 0.3\n");
  try {
    const Read r = parse_cif_text(c_unique);
    FAIL_CHECK("accepted: " << r.st.natoms() << " atoms");
  } catch (const std::runtime_error &e) {
    INFO("message: " << e.what());
    CHECK(contains(e.what(), "CIF text (data_P21c_c): "));
    CHECK(contains(e.what(), "P 1 21/c 1"));
    CHECK(contains(e.what(), "unique axis b"));
    CHECK(mentions_value(e.what(), 110.0));
  }
}

// ============================================== the space-group symbol ====

/* Pins: without an operator loop, an H-M symbol written as spglib, other
 * programs or people write it is normalised before it is looked up, and
 * then agrees with the number next to it, so nothing is printed:
 *   - 'P2_1/c' (spglib; '_' removed), 'P2(1)/c' (the subscript in
 *     parentheses closed up), 'P 21/c (No. 14)' (the trailing number
 *     dropped) and '  P 21/c  ' (surrounding blanks trimmed), each with
 *     _symmetry_Int_Tables_number 14 in a cell with beta = 110: the 4 atoms
 *     of P 1 21/c 1 from Fe1 at (0.1,0.2,0.3);
 *   - 'P6(3)/mmc' with the number 194, hcp Mg (a = 3.2094 A, c = 5.2108 A,
 *     gamma = 120, Mg1 at (1/3,2/3,1/4)): the 2 atoms on 2c.
 * Before the fix: gemmi's find_spacegroup_by_name reads none of these
 * forms, so 'P2_1/c', 'P2(1)/c', 'P 21/c (No. 14)' and 'P6(3)/mmc' were
 * refused ("the space-group symbol 'P2_1/c' names no space group gemmi
 * knows; ..."): four "refused: ..." failures. '  P 21/c  ' passes (a pin:
 * gemmi skips the blanks). */
TEST_CASE("CIF reader: an H-M symbol in spglib or annotated form is "
          "normalised before it is looked up",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  for (const char *hm :
       {"P2_1/c", "P2(1)/c", "P 21/c (No. 14)", "  P 21/c  "}) {
    INFO("H-M '" << hm << "' with the number 14");
    Read r;
    if (!accepted(cif_cell("P21c_hm", 5, 6, 7, 90, 110, 90,
                           hm_line(hm) + number_line("14"), kFeGeneral),
                  r))
      continue;
    INFO("printed:\n" << r.log);
    check_sites(r.st, kFeP21c);
    CHECK_FALSE(contains(lower(r.log), "warning"));
  }
  INFO("hcp Mg, H-M 'P6(3)/mmc' with the number 194");
  Read r;
  if (!accepted(cif_cell("Mg_hcp_subscript", 3.2094, 3.2094, 5.2108, 90, 90,
                         120, hm_line("P6(3)/mmc") + number_line("194"),
                         "Mg1 Mg 0.333333333 0.666666667 0.25\n"),
                r))
    return;
  INFO("printed:\n" << r.log);
  check_sites(r.st, kMgHcp);
  CHECK_FALSE(contains(lower(r.log), "warning"));
}

/* Pins: ICSD's origin suffix is read in either case, and with an operator
 * loop as well as without one:
 *   - without a loop, lower-case 'F d -3 m z' with Si1 at (1/8,1/8,1/8)
 *     and 'F d -3 m s' with Si1 at the origin, each with the number 227
 *     (a = 5.431 A), give the 8 atoms of diamond Si on 8a of origin
 *     choice 2 and origin choice 1;
 *   - si_diamond_icsd_z_loop.cif, 'F d -3 m Z' with the number 227 and all
 *     192 operators of F d -3 m:2, gives the 8 atoms and prints nothing:
 *     the symbol, read as F d -3 m:2, agrees with the loop. Through every
 *     route.
 * Before the fix: only an upper-case suffix was read, and only without a
 * loop: 'F d -3 m z' and 'F d -3 m s' were refused ("the space-group
 * symbol 'F d -3 m z' names no space group gemmi knows; ..."); with the
 * loop the 8 atoms were right (gemmi follows the loop) but gemmi's
 * consistency check, given the symbol as written, printed a false "the
 * symmetry record is inconsistent (gemmi: "H-M symbol (F d -3 m Z) not
 * found in the table")" WARNING, on each of the three routes. */
TEST_CASE("CIF reader: an ICSD origin suffix is read in either case, with "
          "or without an operator loop",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  struct Case {
    const char *hm, *x;
    const std::vector<Site> *sites;
  };
  for (const Case &c : {Case{"F d -3 m z", "0.125", &kSiDiamond},
                        Case{"F d -3 m s", "0", &kSiDiamondOrigin1}}) {
    INFO("H-M '" << c.hm << "' with the number 227, no operator loop");
    Read r;
    const std::string x = c.x;
    if (!accepted(cif_cell("Si_icsd_lower_case", 5.431, 5.431, 5.431, 90, 90,
                           90, hm_line(c.hm) + number_line("227"),
                           "Si1 Si " + x + " " + x + " " + x + "\n"),
                  r))
      continue;
    INFO("printed:\n" << r.log);
    check_sites(r.st, *c.sites);
    CHECK_FALSE(contains(lower(r.log), "warning"));
  }
  const std::string path = kCifDir + "si_diamond_icsd_z_loop.cif";
  for (const Route route : kRoutes) {
    INFO("si_diamond_icsd_z_loop.cif via " << route_name(route));
    const Read r = read_cif(path, route);
    INFO("printed:\n" << r.log);
    check_sites(r.st, kSiDiamond);
    CHECK_FALSE(contains(lower(r.log), "warning"));
  }
}

/* Pins: without an operator loop, a Hall symbol that gives a complete
 * group gemmi does not tabulate expands the sites with its own operators,
 * as such an operator loop does (p2_shifted_origin_loop.cif): Hall
 * '-P 2ybc (x+1/4,y,z)', P 1 21/c 1 with the origin moved by a/4, in a
 * cell with beta = 110, gives 4 atoms from Fe1 at (0.1,0.2,0.3): (0.1,0.2,
 * 0.3), (0.4,0.7,0.2), (0.4,0.8,0.7) and (0.1,0.3,0.8). Whether a WARNING
 * is printed is not pinned here.
 * Before the fix: the Hall symbol was looked up only among the tabulated
 * settings, and the CIF was refused ("the space-group symbol '-P 2ybc
 * (x+1/4,y,z)' names no space group gemmi knows; ..."). */
TEST_CASE("CIF reader: a Hall symbol of no tabulated setting expands the "
          "sites with its own operators",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  Read r;
  if (!accepted(cif_cell("P21c_hall_shifted", 5, 6, 7, 90, 110, 90,
                         hall_line("-P 2ybc (x+1/4,y,z)"), kFeGeneral),
                r))
    return;
  INFO("printed:\n" << r.log);
  check_sites(r.st, kFeP21cShifted);
}

/* Pins: a symbol that names nothing readable once normalised is refused
 * with a std::runtime_error that names the source, quotes the symbol as
 * written and says the reader cannot read it ("cannot read"):
 *   - the H-M symbol 'F d -3 m Q' with the number 227 (the number does not
 *     stand in for the symbol: the setting cannot be told);
 *   - the Hall symbol '-Q 2ybc' (Q is no lattice symbol);
 *   - the Hall symbol '-P 2ybc (x+1/4,y,z' (the change of basis is not
 *     closed).
 * Before the fix: all three were refused, naming the source and the
 * symbol, but saying "names no space group gemmi knows": the CHECK for
 * "cannot read" fails three times. */
TEST_CASE("CIF reader: a space-group symbol that cannot be read is refused, "
          "quoting it",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  struct Case {
    const char *block, *symbol;
    std::string text;
  };
  const std::vector<Case> cases = {
      {"Si_unknown_symbol", "F d -3 m Q",
       cif_cell("Si_unknown_symbol", 5.431, 5.431, 5.431, 90, 90, 90,
                hm_line("F d -3 m Q") + number_line("227"),
                "Si1 Si 0.125 0.125 0.125\n")},
      {"Hall_unknown_lattice", "-Q 2ybc",
       cif_cell("Hall_unknown_lattice", 5, 6, 7, 90, 110, 90,
                hall_line("-Q 2ybc"), kFeGeneral)},
      {"Hall_open_basis", "-P 2ybc (x+1/4,y,z",
       cif_cell("Hall_open_basis", 5, 6, 7, 90, 110, 90,
                hall_line("-P 2ybc (x+1/4,y,z"), kFeGeneral)}};
  for (const Case &c : cases) {
    INFO("symbol '" << c.symbol << "'");
    const std::string what = refusal(c.text);
    INFO("message: " << what);
    if (what.empty()) continue;
    CHECK(contains(what, std::string("CIF text (data_") + c.block + "): "));
    CHECK(contains(what, std::string("'") + c.symbol + "'"));
    CHECK(contains(what, "cannot read"));
  }
}

/* Pins: an H-M symbol that is a number is no H-M symbol and is refused,
 * quoting it and saying the reader cannot read it: '14' and '1003', each
 * without a loop, a Hall symbol or a number, in a cell with beta = 110.
 * gemmi's name lookup would take it as a CCP4 code, which is the IT number
 * only up to 230 (1003 is P 1 1 2).
 * Before the fix: gemmi's lookup was used as it is, so '14' gave the 4
 * atoms of P 1 21/c 1 and '1003' the 2 atoms of P 1 1 2, without a
 * message ("accepted: 4 atoms", "accepted: 2 atoms"). */
TEST_CASE("CIF reader: an H-M symbol that is a number is refused",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  for (const char *hm : {"14", "1003"}) {
    INFO("H-M '" << hm << "'");
    const std::string what = refusal(
        cif_cell("Numeric_hm", 5, 6, 7, 90, 110, 90, hm_line(hm), kFeGeneral));
    INFO("message: " << what);
    if (what.empty()) continue;
    CHECK(contains(what, "CIF text (data_Numeric_hm): "));
    CHECK(contains(what, std::string("'") + hm + "'"));
    CHECK(contains(what, "cannot read"));
  }
}

// ============================================ the space-group number ====

/* Pins: a space-group number alone is accepted only in 1..230; 0, 231,
 * 999, -5 and 1003 (a CCP4 code, P 1 1 2 in gemmi's table, not an IT
 * number) are refused with a std::runtime_error that names the source and
 * the number as given. (An unknown number, '?', is no number: the null
 * test below.)
 * Before the fix: a number that is no IT number was not refused: 231, 999
 * and -5 were ignored and the CIF read silently as P 1, the 1 atom as
 * listed ("accepted: 1 atoms"); 1003 was looked up as gemmi's CCP4 code
 * and expanded in P 1 1 2, 2 atoms ("accepted: 2 atoms"), with the number
 * WARNING and a WARNING that the number (1003) differs.
 * Review round 5: 0 was taken as no number (gemmi reads '0' and '?' alike)
 * and the CIF read as P 1, 1 atom ("accepted: 1 atoms"). */
TEST_CASE("CIF reader: a space-group number outside 1..230 is refused",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  for (const char *n : {"0", "231", "999", "-5", "1003"}) {
    INFO("number " << n << " alone");
    const std::string what =
        refusal(cif_cell("Fe_bad_number", 2.8665, 2.8665, 2.8665, 90, 90, 90,
                         number_line(n), kFeGeneral));
    INFO("message: " << what);
    if (what.empty()) continue;
    CHECK(contains(what, "CIF text (data_Fe_bad_number): "));
    CHECK(contains(what, n));
  }
}

/* Pins: the "more than one setting" WARNING of a number alone is printed
 * when the cell could be in another tabulated setting of the group: those
 * whose change of basis from the setting used has determinant +-1 (an
 * origin shift, a permutation of the axes, another monoclinic cell
 * choice). gemmi's table also lists groups in larger centred cells (A 1 to
 * I 1, A -1 to I -1, F 4/m m m: determinant 2 or 4) and R groups on both
 * hexagonal and rhombohedral axes (determinant 3), which are other cells,
 * told apart by the cell itself.
 *   - Nothing is printed for P 1 (1) and P -1 (2) in a triclinic cell and
 *     I 4/m m m (139) in a tetragonal one: Fe1 at (0.1,0.2,0.3) gives 1
 *     atom in P 1 and 2, (x,y,z) and (-x,-y,-z), in P -1; Fe1 at the
 *     origin gives 2 in I 4/m m m, (0,0,0) and (1/2,1/2,1/2).
 *   - The C-centred orthorhombic groups whose other axis settings are A-
 *     or B-centred warn, naming the C setting, in a 3 x 4 x 5 A cell: C m m
 *     2 (35) with Fe1 at (0,0,0.1): (0,0,0.1), (1/2,1/2,0.1); C c c 2 (37)
 *     with Fe1 at (0,0,0.1): those and (0,0,0.6), (1/2,1/2,0.6); C m m m
 *     (65) with Fe1 at the origin: (0,0,0), (1/2,1/2,0); C c c m (66) with
 *     Fe1 at the origin: (0,0,0), (0,0,1/2), (1/2,1/2,0), (1/2,1/2,1/2).
 *   - Only the settings that fit the cell count, and only ITA's: P 21/m
 *     (11) in a cell with beta = 110 prints nothing (its unique-axis-a and
 *     -c settings need beta = 90), in a cell with all angles 90 it warns,
 *     naming P 1 21/m 1 (Fe1 at (0.1,0.2,0.3), the 4 atoms of P 1 21/m 1
 *     either way); F 2 2 2 (22), I 2 2 2 (23), P 42 21 2 (94) and I 2 3
 *     (197), whose other table entries are CCP4's extra origins (no ITA
 *     origin choice), print nothing: Fe1 at the origin gives the 4 atoms
 *     of the F lattice for 22 and (0,0,0), (1/2,1/2,1/2) for the others.
 *   - I b c a (73), whose other ITA setting I c a b differs from it by
 *     what gemmi encodes as an origin shift (x+1/4,y-1/4,z+1/4), warns,
 *     naming I b c a: Fe1 at the origin gives 8 atoms, every point with
 *     coordinates 0 or 1/2 (the images of the origin under -x+1/2,-y,z+1/2;
 *     -x,y+1/2,-z+1/2 and the I centring).
 * The numbers 227 and 14 warn too (the number-only WARNING test and the
 * monoclinic number test); 166 does not (the number-only WARNING test).
 * Before the fix: the WARNING counted every table entry of the number, so
 * every case expecting none printed it: with the cases of this test the
 * CHECK_FALSE fails eight times (1, 2, 139, 22, 23, 94, 197 and 11 with
 * beta = 110; the test began with the first three).
 * Review round 5: the WARNING counted the settings with the letter of the
 * setting used, so 35, 37, 65 and 66 (one C setting each) printed nothing
 * and the CHECKs for their WARNING fail (with the cases of this test, 12
 * CHECKs for those four, and the CHECK_FALSE for 22, 23, 94, 197 and 11
 * with beta = 110, which round 6 added). Review round 6: every
 * setting with the same determinant counted, whatever the cell, so 11 with
 * beta = 110, 22, 23, 94 and 197 warned: the CHECK_FALSE fails five
 * times. Review round 7: an entry with the rotations of the setting used
 * counted only as an origin choice (1 or 2), so I c a b did not, and 73
 * printed nothing: the CHECKs for its WARNING fail. */
TEST_CASE("CIF reader: a number alone warns only when the cell could be in "
          "another setting of its group",
          "[structure_readers][cif][spacegroup][warning]") {
  PeriodicTable::initialize();
  struct Case {
    const char *number;
    double a, b, c, alpha, beta, gamma;
    const char *site;
    std::vector<Site> sites;
    const char *warns_with;  // the setting the WARNING names, or nullptr
  };
  const std::vector<Case> cases = {
      {"1", 4.2, 5.1, 6.3, 81.5, 96.2, 103.7, "Fe1 Fe 0.1 0.2 0.3\n",
       {{"Fe", {0.1, 0.2, 0.3}}}, nullptr},
      {"2", 4.2, 5.1, 6.3, 81.5, 96.2, 103.7, "Fe1 Fe 0.1 0.2 0.3\n",
       {{"Fe", {0.1, 0.2, 0.3}}, {"Fe", {-0.1, -0.2, -0.3}}}, nullptr},
      {"139", 2.9, 2.9, 4.1, 90, 90, 90, "Fe1 Fe 0 0 0\n", kFeBcc, nullptr},
      {"22", 3, 4, 5, 90, 90, 90, "Fe1 Fe 0 0 0\n",
       {{"Fe", {0.0, 0.0, 0.0}},
        {"Fe", {0.0, 0.5, 0.5}},
        {"Fe", {0.5, 0.0, 0.5}},
        {"Fe", {0.5, 0.5, 0.0}}},
       nullptr},
      {"23", 3, 4, 5, 90, 90, 90, "Fe1 Fe 0 0 0\n", kFeBcc, nullptr},
      {"94", 3, 3, 5, 90, 90, 90, "Fe1 Fe 0 0 0\n", kFeBcc, nullptr},
      {"197", 5, 5, 5, 90, 90, 90, "Fe1 Fe 0 0 0\n", kFeBcc, nullptr},
      {"11", 5, 6, 7, 90, 110, 90, "Fe1 Fe 0.1 0.2 0.3\n", kFeP21mB,
       nullptr},
      {"11", 5, 6, 7, 90, 90, 90, "Fe1 Fe 0.1 0.2 0.3\n", kFeP21mB,
       "P 1 21/m 1"},
      {"73", 3, 4, 5, 90, 90, 90, "Fe1 Fe 0 0 0\n",
       {{"Fe", {0.0, 0.0, 0.0}},
        {"Fe", {0.0, 0.0, 0.5}},
        {"Fe", {0.0, 0.5, 0.0}},
        {"Fe", {0.0, 0.5, 0.5}},
        {"Fe", {0.5, 0.0, 0.0}},
        {"Fe", {0.5, 0.0, 0.5}},
        {"Fe", {0.5, 0.5, 0.0}},
        {"Fe", {0.5, 0.5, 0.5}}},
       "I b c a"},
      {"35", 3, 4, 5, 90, 90, 90, "Fe1 Fe 0 0 0.1\n",
       {{"Fe", {0.0, 0.0, 0.1}}, {"Fe", {0.5, 0.5, 0.1}}}, "C m m 2"},
      {"37", 3, 4, 5, 90, 90, 90, "Fe1 Fe 0 0 0.1\n",
       {{"Fe", {0.0, 0.0, 0.1}},
        {"Fe", {0.0, 0.0, 0.6}},
        {"Fe", {0.5, 0.5, 0.1}},
        {"Fe", {0.5, 0.5, 0.6}}},
       "C c c 2"},
      {"65", 3, 4, 5, 90, 90, 90, "Fe1 Fe 0 0 0\n",
       {{"Fe", {0.0, 0.0, 0.0}}, {"Fe", {0.5, 0.5, 0.0}}}, "C m m m"},
      {"66", 3, 4, 5, 90, 90, 90, "Fe1 Fe 0 0 0\n",
       {{"Fe", {0.0, 0.0, 0.0}},
        {"Fe", {0.0, 0.0, 0.5}},
        {"Fe", {0.5, 0.5, 0.0}},
        {"Fe", {0.5, 0.5, 0.5}}},
       "C c c m"}};
  for (const Case &c : cases) {
    INFO("number " << c.number << " alone");
    Read r;
    if (!accepted(cif_cell("Number_settings", c.a, c.b, c.c, c.alpha,
                           c.beta, c.gamma, number_line(c.number), c.site),
                  r))
      continue;
    INFO("printed:\n" << r.log);
    check_sites(r.st, c.sites);
    if (!c.warns_with) {
      CHECK_FALSE(contains(lower(r.log), "warning"));
      continue;
    }
    CHECK(count_of(r.log, "WARNING") == 1);
    CHECK(contains(r.log, std::string("only by its number (") + c.number +
                              ")"));
    CHECK(contains(r.log, std::string("built with space group ") +
                              c.warns_with + ","));
  }
}

// ================================================= monoclinic settings ====

/* Pins: a monoclinic group must fit the cell however it is given: the two
 * angles its unique axis needs to be 90 (unique axis b: alpha and gamma;
 * c: alpha and beta; a: beta and gamma) are 90 within 1e-3 degrees, or the
 * CIF is refused with a std::runtime_error naming the source, the setting
 * (gemmi's xhm()), its unique axis and the cell angles. Fe1 at
 * (0.1,0.2,0.3), a = 5, b = 6, c = 7 A:
 *   - full H-M symbols: 'P 1 21/m 1' with beta = 110, 'P 1 1 21/m' with
 *     gamma = 110 and 'P 21/m 1 1' with alpha = 110 each give their 4
 *     atoms; 'P 1 21/m 1' with gamma = 110, 'P 1 1 21/m' with beta = 110
 *     and 'P 21/m 1 1' with beta = 110 are refused;
 *   - with gamma = 110, the short symbol 'P 21/m' with the number 11
 *     (gemmi reads it as P 1 21/m 1), the Hall symbol '-P 2yb' and the
 *     operator loop of P 1 21/m 1 (x,y,z; -x,y+1/2,-z; -x,-y,-z;
 *     x,-y+1/2,z) are each refused, naming P 1 21/m 1 and unique axis b.
 * The number alone is the monoclinic number test above.
 * Before the fix: only a number alone was checked against the cell, so
 * the six cells that do not fit were each accepted without a message, 4
 * atoms in a cell that is not the setting's, with two different
 * environments ("accepted: 4 atoms", six times). The three cells that fit
 * pass (pins). */
TEST_CASE("CIF reader: a monoclinic space group must fit the cell, "
          "however it is given",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  struct Case {
    const char *block;
    std::string symmetry;
    double alpha, beta, gamma;
    const char *setting;  // gemmi's xhm()
    char axis;
    double bad_angle;                // 0 when the cell fits
    const std::vector<Site> *sites;  // when the cell fits
  };
  const std::string p21m_loop =
      "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n'-x,y+1/2,-z'\n"
      "'-x,-y,-z'\n'x,-y+1/2,z'\n";
  const std::vector<Case> cases = {
      {"P21m_b_fits", hm_line("P 1 21/m 1"), 90, 110, 90, "P 1 21/m 1", 'b',
       0, &kFeP21mB},
      {"P21m_c_fits", hm_line("P 1 1 21/m"), 90, 90, 110, "P 1 1 21/m", 'c',
       0, &kFeP21mC},
      {"P21m_a_fits", hm_line("P 21/m 1 1"), 110, 90, 90, "P 21/m 1 1", 'a',
       0, &kFeP21mA},
      {"P21m_b_gamma", hm_line("P 1 21/m 1"), 90, 90, 110, "P 1 21/m 1", 'b',
       110, nullptr},
      {"P21m_c_beta", hm_line("P 1 1 21/m"), 90, 110, 90, "P 1 1 21/m", 'c',
       110, nullptr},
      {"P21m_a_beta", hm_line("P 21/m 1 1"), 90, 110, 90, "P 21/m 1 1", 'a',
       110, nullptr},
      {"P21m_short_gamma", hm_line("P 21/m") + number_line("11"), 90, 90, 110,
       "P 1 21/m 1", 'b', 110, nullptr},
      {"P21m_hall_gamma", hall_line("-P 2yb"), 90, 90, 110, "P 1 21/m 1", 'b',
       110, nullptr},
      {"P21m_loop_gamma", p21m_loop, 90, 90, 110, "P 1 21/m 1", 'b', 110,
       nullptr}};
  for (const Case &c : cases) {
    const std::string text = cif_cell(c.block, 5, 6, 7, c.alpha, c.beta,
                                      c.gamma, c.symmetry, kFeGeneral);
    INFO(c.block << ": alpha " << c.alpha << ", beta " << c.beta
                 << ", gamma " << c.gamma << ", symmetry\n"
                 << c.symmetry);
    if (c.sites) {
      Read r;
      if (!accepted(text, r)) continue;
      INFO("printed:\n" << r.log);
      check_sites(r.st, *c.sites);
      continue;
    }
    const std::string what = refusal(text);
    INFO("message: " << what);
    if (what.empty()) continue;
    CHECK(contains(what, std::string("CIF text (data_") + c.block + "): "));
    CHECK(contains(what, c.setting));
    CHECK(contains(what, std::string("unique axis ") + c.axis));
    CHECK(mentions_value(what, c.bad_angle));
  }
}

/* Pins: the monoclinic check allows 1e-3 degrees and its message prints
 * the angles as read, so that it shows why the cell does not fit. With the
 * number 14 alone (P 1 21/c 1, unique axis b) and beta = 110:
 *   - alpha = 90.0004 (within 1e-3) gives the 4 atoms of P 1 21/c 1 from
 *     Fe1 at (0.1,0.2,0.3), with the number WARNING naming P 1 21/c 1;
 *   - alpha = 90.00101 (1.01e-3 off) is refused, naming P 1 21/c 1, and
 *     the message writes alpha so that it reads back as 90.00101 (6
 *     significant digits would print 90.001, which looks within 1e-3).
 * Before the fix: a number alone was checked to 1e-6 and the angles
 * printed with 6 significant digits: alpha = 90.0004 was refused
 * ("refused: ... the cell (alpha 90.0004, beta 110, gamma 90) is not in
 * its standard setting, unique axis b ..."), and the refusal for 90.00101
 * printed "alpha 90.001" and did not name P 1 21/c 1: both CHECKs on it
 * fail. */
TEST_CASE("CIF reader: the monoclinic check allows 1e-3 degrees and prints "
          "the angles as read",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  const std::string n14 = number_line("14");
  {
    INFO("alpha 90.0004");
    Read r;
    if (accepted(cif_cell("P21c_alpha_within", 5, 6, 7, 90.0004, 110, 90, n14,
                          kFeGeneral),
                 r)) {
      INFO("printed:\n" << r.log);
      check_sites(r.st, kFeP21c);
      CHECK(contains(r.log, "only by its number (14)"));
      CHECK(contains(r.log, "P 1 21/c 1"));
    }
  }
  INFO("alpha 90.00101");
  const std::string what = refusal(
      cif_cell("P21c_alpha_off", 5, 6, 7, 90.00101, 110, 90, n14, kFeGeneral));
  INFO("message: " << what);
  if (what.empty()) return;
  CHECK(contains(what, "CIF text (data_P21c_alpha_off): "));
  CHECK(contains(what, "P 1 21/c 1"));
  CHECK(mentions_value(what, 90.00101));
}

// ================================================ review round 5 ====

namespace {

// Fe1 at (0.1,0.2,0.3) under the operators of B 1 1 2/m (x,y,z; -x,-y,z;
// -x,-y,-z; x,y,-z) and the B centring (1/2,0,1/2).
const std::vector<Site> kFeB112m = {
    {"Fe", {0.1, 0.2, 0.3}},   {"Fe", {-0.1, -0.2, 0.3}},
    {"Fe", {-0.1, -0.2, -0.3}}, {"Fe", {0.1, 0.2, -0.3}},
    {"Fe", {0.6, 0.2, 0.8}},   {"Fe", {0.4, -0.2, 0.8}},
    {"Fe", {0.4, -0.2, 0.2}},  {"Fe", {0.6, 0.2, 0.2}}};

// Three operators of P 1 21/c 1 (x,y,z; -x,y+1/2,-z+1/2; -x,-y,-z): no
// group (their products are missing), so gemmi falls back to the symbols.
const std::string kIncompleteP21cLoop =
    "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n'-x,y+1/2,-z+1/2'\n"
    "'-x,-y,-z'\n";

// The operators of -P 2ybc (x+1/4,y,z): P 1 21/c 1 with the origin moved
// by a/4, a complete group in no tabulated setting.
const std::string kShiftedP21cLoop =
    "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n'-x+1/2,y+1/2,-z+1/2'\n"
    "'-x+1/2,-y,-z'\n'x,-y+1/2,z+1/2'\n";

}  // namespace

/* Pins: a monoclinic group in no tabulated setting (an origin moved from
 * the tabulated one) must fit the cell as its tabulated setting does: the
 * cell constraints come from the rotations alone. With gamma = 110, the
 * Hall symbol '-P 2ybc (x+1/4,y,z)' and the operator loop of that group
 * (x,y,z; -x+1/2,y+1/2,-z+1/2; -x+1/2,-y,-z; x,-y+1/2,z+1/2), each with
 * Fe1 at (0.1,0.2,0.3), are refused with a std::runtime_error naming the
 * source, unique axis b and the angle (110). With beta = 110 they are read:
 * 4 atoms each, (0.1,0.2,0.3), (0.4,0.7,0.2), (0.4,0.8,0.7) and
 * (0.1,0.3,0.8).
 * Centred groups in no tabulated setting are checked the same way (the
 * check uses the primitive group with their rotations): with Fe1 at
 * (0.1,0.2,0.3), the Hall symbol '-B 2y' (B 1 2/m 1, unique axis b) and
 * the 8 operators of that group with gamma = 110, and the Hall symbol
 * '-C 2' (C 1 1 2/m, unique axis c) with beta = 110, are refused, naming
 * the unique axis and the angle.
 * Before the fix: the check looked only at a tabulated group, so the two
 * P cases were accepted, 4 atoms each, with only the WARNING that the
 * group is in no table ("accepted: 4 atoms", twice; with the cases of
 * this test the three centred ones too, five in all). Review round 6: the
 * centred ones were looked up with their centring, which gemmi tabulates
 * for none of these settings, and accepted, 8 atoms each ("accepted: 8
 * atoms", three times). */
TEST_CASE("CIF reader: a monoclinic group in no tabulated setting must fit "
          "the cell",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  struct Case {
    const char *block;
    std::string symmetry;
    double beta, gamma;
    char axis;
  };
  const std::string b12m1_loop =
      "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n'-x,y,-z'\n'-x,-y,-z'\n"
      "'x,-y,z'\n'x+1/2,y,z+1/2'\n'-x+1/2,y,-z+1/2'\n'-x+1/2,-y,-z+1/2'\n"
      "'x+1/2,-y,z+1/2'\n";
  for (const Case &c :
       {Case{"P21c_hall_shifted_gamma", hall_line("-P 2ybc (x+1/4,y,z)"), 90,
             110, 'b'},
        Case{"P21c_loop_shifted_gamma", kShiftedP21cLoop, 90, 110, 'b'},
        Case{"B12m1_hall_gamma", hall_line("-B 2y"), 90, 110, 'b'},
        Case{"B12m1_loop_gamma", b12m1_loop, 90, 110, 'b'},
        Case{"C112m_hall_beta", hall_line("-C 2"), 110, 90, 'c'}}) {
    INFO(c.block);
    const std::string what = refusal(cif_cell(c.block, 5, 6, 7, 90, c.beta,
                                              c.gamma, c.symmetry,
                                              kFeGeneral));
    INFO("message: " << what);
    if (what.empty()) continue;
    CHECK(contains(what, std::string("CIF text (data_") + c.block + "): "));
    CHECK(contains(what, std::string("unique axis ") + c.axis));
    CHECK(mentions_value(what, 110.0));
  }
  for (const Case &c :
       {Case{"P21c_hall_shifted_beta", hall_line("-P 2ybc (x+1/4,y,z)"), 110,
             90, 'b'},
        Case{"P21c_loop_shifted_beta", kShiftedP21cLoop, 110, 90, 'b'}}) {
    INFO(c.block);
    Read r;
    if (!accepted(cif_cell(c.block, 5, 6, 7, 90, c.beta, c.gamma, c.symmetry,
                           kFeGeneral),
                  r))
      continue;
    check_sites(r.st, kFeP21cShifted);
  }
}

/* Pins: when the operator loop forms no group, the symbols are read as
 * they are without a loop (normalised, Hall first, a symbol that cannot
 * be read refused), the loop given as kIncompleteP21cLoop in a cell with
 * beta = 110 and Fe1 at (0.1,0.2,0.3):
 *   - with the H-M symbol 'P2_1/c': the 4 atoms of P 1 21/c 1 (a pin:
 *     without the normalised symbol the cell is P 1, 1 atom);
 *   - with the H-M symbol '1003' (a number, which gemmi takes as the CCP4
 *     code of P 1 1 2), 'P 21/c Q', or the Hall symbol '-Q 2ybc': refused,
 *     quoting the symbol, "cannot read".
 * Before the fix: gemmi's own fallback was used, so '1003' was read as
 * the CCP4 code of P 1 1 2 (unique axis c), which this cell (beta = 110)
 * does not fit: refused by the monoclinic check, without "cannot read";
 * 'P 21/c Q' and '-Q 2ybc' gave the 1 atom of P 1, with WARNINGs quoting
 * gemmi's diagnostics ("accepted: 1 atoms" twice). */
TEST_CASE("CIF reader: when the operator loop forms no group, the symbols "
          "are read as without a loop",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  {
    INFO("H-M 'P2_1/c' with the incomplete loop");
    Read r;
    if (accepted(cif_cell("P21c_partial_loop_spglib", 5, 6, 7, 90, 110, 90,
                          hm_line("P2_1/c") + kIncompleteP21cLoop,
                          kFeGeneral),
                 r)) {
      INFO("printed:\n" << r.log);
      check_sites(r.st, kFeP21c);
    }
  }
  struct Case {
    const char *block, *symbol;
    std::string symmetry;
  };
  const std::vector<Case> cases = {
      {"Partial_loop_numeric_hm", "1003",
       hm_line("1003") + kIncompleteP21cLoop},
      {"Partial_loop_unknown_hm", "P 21/c Q",
       hm_line("P 21/c Q") + kIncompleteP21cLoop},
      {"Partial_loop_unknown_hall", "-Q 2ybc",
       hall_line("-Q 2ybc") + kIncompleteP21cLoop}};
  for (const Case &c : cases) {
    INFO("symbol '" << c.symbol << "' with the incomplete loop");
    const std::string what = refusal(
        cif_cell(c.block, 5, 6, 7, 90, 110, 90, c.symmetry, kFeGeneral));
    INFO("message: " << what);
    if (what.empty()) continue;
    CHECK(contains(what, std::string("CIF text (data_") + c.block + "): "));
    CHECK(contains(what, std::string("'") + c.symbol + "'"));
    CHECK(contains(what, "cannot read"));
  }
}

/* Pins: an H-M value that normalisation leaves empty is a symbol that
 * cannot be read, not a missing one: '(No. 229)' alone and '_' with the
 * number 229, in a bcc cell (a = 2.8665 A) with Fe1 at the origin, are
 * refused, quoting the value, "cannot read". A blank or unknown value
 * ('?') is no symbol: '?' with the number 229 gives the 2 atoms of bcc Fe
 * and prints nothing, and an unknown number ('?') alone is no number: P 1,
 * 1 atom, nothing printed (pins).
 * Before the fix: '(No. 229)' was read as no symbol and the CIF as P 1, 1
 * atom; '_' as no symbol, so the number gave 2 atoms ("accepted: 1
 * atoms", "accepted: 2 atoms"). */
TEST_CASE("CIF reader: an H-M value that normalises to nothing is refused",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  struct Case {
    const char *block, *symbol;
    std::string symmetry;
  };
  const std::vector<Case> cases = {
      {"Hm_number_note_only", "(No. 229)", hm_line("(No. 229)")},
      {"Hm_underscore", "_", hm_line("_") + number_line("229")}};
  for (const Case &c : cases) {
    INFO("H-M '" << c.symbol << "'");
    const std::string what =
        refusal(cif_cell(c.block, 2.8665, 2.8665, 2.8665, 90, 90, 90,
                         c.symmetry, "Fe1 Fe 0 0 0\n"));
    INFO("message: " << what);
    if (what.empty()) continue;
    CHECK(contains(what, std::string("CIF text (data_") + c.block + "): "));
    CHECK(contains(what, std::string("'") + c.symbol + "'"));
    CHECK(contains(what, "cannot read"));
  }
  {
    INFO("H-M ? with the number 229");
    Read r;
    if (accepted(cif_cell("Hm_null", 2.8665, 2.8665, 2.8665, 90, 90, 90,
                          "_symmetry_space_group_name_H-M ?\n" +
                              number_line("229"),
                          "Fe1 Fe 0 0 0\n"),
                 r)) {
      INFO("printed:\n" << r.log);
      check_sites(r.st, kFeBcc);
      CHECK_FALSE(contains(lower(r.log), "warning"));
    }
  }
  {
    INFO("number ? alone");
    Read r;
    if (accepted(cif_cell("Number_null", 2.8665, 2.8665, 2.8665, 90, 90, 90,
                          number_line("?"), "Fe1 Fe 0 0 0\n"),
                 r)) {
      INFO("printed:\n" << r.log);
      check_sites(r.st, {{"Fe", {0.0, 0.0, 0.0}}});
      CHECK_FALSE(contains(lower(r.log), "warning"));
    }
  }
}

/* Pins: a short monoclinic symbol is read in the setting gemmi gives it,
 * which is unique axis b except for the B-centred 'B 2', 'B m', 'B b',
 * 'B n', 'B 2/m', 'B 2/b' and 'B 2/n' (unique axis c, the PDB
 * convention; each but 'B 2/m' with beta = 110 is refused, naming
 * B 1 1 <the rest> and unique axis c): with gamma =
 * 110, 'B 2/m' gives the 8 atoms of B 1 1 2/m from Fe1 at (0.1,0.2,0.3);
 * with beta = 110 it is refused, naming B 1 1 2/m and unique axis c.
 * 'B 21' is read as B 1 21 1, unique axis b: refused with gamma = 110,
 * naming B 1 21 1 and unique axis b. (The P short symbols: the monoclinic
 * tests above.)
 * Before the fix: passes as well (a pin of what the docs now say). */
TEST_CASE("CIF reader: a B-centred short monoclinic symbol means unique "
          "axis c",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  {
    INFO("'B 2/m' with gamma = 110");
    Read r;
    if (accepted(cif_cell("B2m_gamma", 5, 6, 7, 90, 90, 110,
                          hm_line("B 2/m"), kFeGeneral),
                 r)) {
      INFO("printed:\n" << r.log);
      check_sites(r.st, kFeB112m);
    }
  }
  {
    INFO("'B 2/m' with beta = 110");
    const std::string what = refusal(
        cif_cell("B2m_beta", 5, 6, 7, 90, 110, 90, hm_line("B 2/m"),
                 kFeGeneral));
    INFO("message: " << what);
    if (!what.empty()) {
      CHECK(contains(what, "B 1 1 2/m"));
      CHECK(contains(what, "unique axis c"));
      CHECK(mentions_value(what, 110.0));
    }
  }
  for (const char *hm : {"B 2/n", "B n", "B 2", "B m", "B b", "B 2/b"}) {
    INFO("'" << hm << "' with beta = 110");
    const std::string what = refusal(cif_cell(
        "Bn_beta", 5, 6, 7, 90, 110, 90, hm_line(hm), kFeGeneral));
    INFO("message: " << what);
    if (!what.empty()) {
      CHECK(contains(what, std::string("B 1 1 ") + (hm + 2)));
      CHECK(contains(what, "unique axis c"));
    }
  }
  INFO("'B 21' with gamma = 110");
  const std::string what = refusal(
      cif_cell("B21_gamma", 5, 6, 7, 90, 90, 110, hm_line("B 21"),
               kFeGeneral));
  INFO("message: " << what);
  if (what.empty()) return;
  CHECK(contains(what, "B 1 21 1"));
  CHECK(contains(what, "unique axis b"));
  CHECK(mentions_value(what, 110.0));
}

/* Pins: a refusal advises only what can change the result. The operator
 * loop is read before the symbols, and the Hall symbol before the H-M
 * symbol, so a refusal that comes from the loop or the Hall symbol while
 * the CIF also has an H-M symbol says so ("read before"), instead of
 * advising to give the H-M symbol (which is there already):
 *   - the loop of P 1 21/m 1 (x,y,z; -x,y+1/2,-z; -x,-y,-z; x,-y+1/2,z)
 *     with the H-M symbol 'P 1 1 21/m', gamma = 110: refused (the loop's
 *     group needs gamma = 90);
 *   - the Hall symbol '-P 2yb' with 'P 1 1 21/m', gamma = 110: refused;
 *   - the Hall symbol '-Q 4 2 3' (unreadable) with 'I m -3 m': refused.
 * Each message names the record it comes from and contains "read before".
 * Before the fix: the first two advised "give the full H-M symbol of the
 * setting the cell is in (e.g. 'P 1 1 21/m' for unique axis c)" and the
 * third "give a Hall symbol in standard form ..., the H-M symbol, or the
 * symmetry-operator loop": the CHECK for "read before" fails three
 * times. */
TEST_CASE("CIF reader: a refusal from a record read first says so",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  const std::string p21m_loop =
      "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n'-x,y+1/2,-z'\n"
      "'-x,-y,-z'\n'x,-y+1/2,z'\n";
  struct Case {
    const char *block, *record;
    std::string symmetry;
    double gamma;
  };
  const std::vector<Case> cases = {
      {"Loop_before_hm", "symmetry-operator loop",
       hm_line("P 1 1 21/m") + p21m_loop, 110},
      {"Hall_before_hm", "Hall symbol '-P 2yb'",
       hall_line("-P 2yb") + hm_line("P 1 1 21/m"), 110},
      {"Bad_hall_before_hm", "Hall symbol '-Q 4 2 3'",
       hall_line("-Q 4 2 3") + hm_line("I m -3 m"), 90}};
  for (const Case &c : cases) {
    INFO(c.block);
    const std::string what = refusal(
        cif_cell(c.block, 5, 6, 7, 90, 90, c.gamma, c.symmetry, kFeGeneral));
    INFO("message: " << what);
    if (what.empty()) continue;
    CHECK(contains(what, std::string("CIF text (data_") + c.block + "): "));
    CHECK(contains(what, c.record));
    CHECK(contains(what, "read before"));
  }
}

// ================================================ review round 6 ====

/* Pins: a symbol that names no origin choice gets origin choice 2 also in
 * its current spelling: C c c a (68) is written 'C c c e' since ITA 2002
 * (spglib and pymatgen write 'Ccce'), and A b a a and B b c b likewise
 * 'A e a a' and 'B b e b'. In a 6 x 7 x 8 A cell with Fe1 at the origin,
 * 'Ccce', 'C c c e' and 'C c c a' (each with the number 68) and the number
 * 68 alone all give the 8 atoms of origin choice 2, the origin's images
 * under -x+1/2,-y,z; -x,y,-z+1/2 and the C centring: every point with
 * coordinates 0 or 1/2. 'C c c e:1' gives origin choice 1, 4 atoms:
 * (0,0,0), (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0). 'A e a a' and 'B b e b'
 * (A b a a and B b c b, the same group on permuted axes) give the same 8
 * atoms: the set is unchanged by permuting the axes.
 * Before the fix: gemmi's name lookup applies its origin preference only
 * to the main table, and finds 'C c c e' among the other names, origin
 * choice 1: 'Ccce' and 'C c c e' gave the 4 atoms of origin choice 1, a
 * different crystal, without a message (check_sites fails twice; 'A e a a'
 * and 'B b e b', added in review round 9, fail the same way: four times
 * in all). */
TEST_CASE("CIF reader: the current spelling of a symbol gets origin "
          "choice 2 as well",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  std::vector<Site> origin2;
  for (double x : {0.0, 0.5})
    for (double y : {0.0, 0.5})
      for (double z : {0.0, 0.5}) origin2.push_back({"Fe", {x, y, z}});
  const std::vector<Site> origin1 = {{"Fe", {0.0, 0.0, 0.0}},
                                     {"Fe", {0.0, 0.5, 0.5}},
                                     {"Fe", {0.5, 0.0, 0.5}},
                                     {"Fe", {0.5, 0.5, 0.0}}};
  struct Case {
    std::string symmetry;
    const std::vector<Site> *sites;
  };
  const std::vector<Case> cases = {
      {hm_line("Ccce") + number_line("68"), &origin2},
      {hm_line("C c c e") + number_line("68"), &origin2},
      {hm_line("C c c a") + number_line("68"), &origin2},
      {number_line("68"), &origin2},
      {hm_line("C c c e:1"), &origin1},
      {hm_line("A e a a"), &origin2},
      {hm_line("B b e b"), &origin2}};
  for (const Case &c : cases) {
    INFO(c.symmetry);
    Read r;
    if (!accepted(cif_cell("Ccce", 6, 7, 8, 90, 90, 90, c.symmetry,
                           "Fe1 Fe 0 0 0\n"),
                  r))
      continue;
    INFO("printed:\n" << r.log);
    check_sites(r.st, *c.sites);
  }
}

/* Pins: a tag whose value is unknown ('?') or blank does not hide the
 * other tag of its pair: in a bcc cell (a = 2.8665 A) with Fe1 at the
 * origin,
 *   - _space_group_name_H-M_alt ? with _symmetry_space_group_name_H-M
 *     'I m -3 m';
 *   - _space_group_IT_number ? with _symmetry_Int_Tables_number 229;
 *   - _space_group_name_Hall '-I 4 2 3' with
 *     _symmetry_space_group_name_Hall ?;
 *   - _space_group_name_Hall '' with _symmetry_space_group_name_Hall
 *     '-I 4 2 3', and _space_group_name_H-M_alt ' ' (or '.') with
 *     _symmetry_space_group_name_H-M 'I m -3 m'
 * each give the 2 atoms of bcc Fe and print nothing.
 * Before the fix: gemmi takes the first H-M and number tag present, null
 * or blank or not, and the last Hall tag present, so the first three and
 * the blank and '.' H-M cases were read as P 1, 1 atom, without a message
 * (check_sites fails five times); the blank new Hall tag passed (gemmi
 * took the old tag, the last). Review round 7: a blank value was taken as
 * known, so the two blank cases were read as P 1 (check_sites fails
 * twice). */
TEST_CASE("CIF reader: an unknown value does not hide the other tag of its "
          "pair",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  for (const std::string &symmetry :
       {std::string("_space_group_name_H-M_alt ?\n") +
            "_symmetry_space_group_name_H-M 'I m -3 m'\n",
        std::string("_space_group_IT_number ?\n") +
            "_symmetry_Int_Tables_number 229\n",
        std::string("_space_group_name_Hall '-I 4 2 3'\n") +
            "_symmetry_space_group_name_Hall ?\n",
        std::string("_space_group_name_Hall ''\n") +
            "_symmetry_space_group_name_Hall '-I 4 2 3'\n",
        std::string("_space_group_name_H-M_alt ' '\n") +
            "_symmetry_space_group_name_H-M 'I m -3 m'\n",
        std::string("_space_group_name_H-M_alt .\n") +
            "_symmetry_space_group_name_H-M 'I m -3 m'\n"}) {
    INFO(symmetry);
    Read r;
    if (!accepted(cif_cell("Null_tag", 2.8665, 2.8665, 2.8665, 90, 90, 90,
                           symmetry, "Fe1 Fe 0 0 0\n"),
                  r))
      continue;
    INFO("printed:\n" << r.log);
    check_sites(r.st, kFeBcc);
    CHECK_FALSE(contains(lower(r.log), "warning"));
  }
}

// ================================================ review round 7 ====

/* Pins: the space-group number is read only when it is used, and a value
 * that is not an integer is refused naming the source and quoting it; one
 * that is not used gives a WARNING; a blank one is no number. In a bcc
 * cell (a = 2.8665 A) with Fe1 at the origin:
 *   - _symmetry_Int_Tables_number 229.0 with the H-M symbol 'I m -3 m',
 *     with or without _space_group_IT_number ? before it: the symbol gives
 *     the group, 2 atoms of bcc Fe, and a WARNING quotes '229.0' ("is not
 *     an integer");
 *   - _space_group_IT_number '' with _symmetry_Int_Tables_number 229, and
 *     _symmetry_Int_Tables_number '' with 'I m -3 m': 2 atoms, nothing
 *     printed;
 *   - a quoted number, _symmetry_Int_Tables_number '229', alone or with
 *     'I m -3 m': 2 atoms, nothing printed (a quoted value is read as
 *     written, as every other symmetry value is);
 *   - _symmetry_Int_Tables_number 229.0 alone, with or without
 *     _space_group_IT_number ? before it: refused, naming the source and
 *     quoting '229.0'.
 * Before the fix: the reader parsed the number of the second tag whenever
 * the first was unknown, so '229.0' after '?' threw gemmi's
 * std::invalid_argument "not an integer: 229." naming neither the file nor
 * the tag, even with the symbol. Review round 8: gemmi parsed the first
 * number tag present before the reader ran, so '229.0' and '' as the only
 * or first number were refused ("gemmi cannot read the data block: not an
 * integer: 229." or ": '"), even when the symbol gives the group: three
 * "refused" failures and a missing quote ('229.0' after '?' was read
 * then, as the first tag gemmi parsed was '?'). Review round 9: a quoted
 * number ('229') was parsed with its quotes, "not an integer": refused
 * alone, and a WARNING with the symbol. */
TEST_CASE("CIF reader: a space-group number that is not an integer is "
          "refused only when it is used, naming the source",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  const auto bcc = [](const char *block, const std::string &symmetry) {
    return cif_cell(block, 2.8665, 2.8665, 2.8665, 90, 90, 90, symmetry,
                    "Fe1 Fe 0 0 0\n");
  };
  struct Accepted {
    const char *block;
    std::string symmetry;
    bool warns;
  };
  for (const Accepted &c :
       {Accepted{"Number_decimal_hm_new_null",
                 "_space_group_IT_number ?\n" + number_line("229.0") +
                     hm_line("I m -3 m"),
                 true},
        Accepted{"Number_decimal_hm",
                 number_line("229.0") + hm_line("I m -3 m"), true},
        Accepted{"Number_blank_new", "_space_group_IT_number ''\n" +
                                         number_line("229"),
                 false},
        Accepted{"Number_blank_hm",
                 "_symmetry_Int_Tables_number ''\n" + hm_line("I m -3 m"),
                 false},
        Accepted{"Number_quoted", "_symmetry_Int_Tables_number '229'\n",
                 false},
        Accepted{"Number_quoted_hm",
                 "_symmetry_Int_Tables_number '229'\n" + hm_line("I m -3 m"),
                 false}}) {
    INFO(c.block);
    Read r;
    if (!accepted(bcc(c.block, c.symmetry), r)) continue;
    INFO("printed:\n" << r.log);
    check_sites(r.st, kFeBcc);
    if (c.warns) {
      CHECK(count_of(r.log, "WARNING") == 1);
      CHECK(contains(r.log, "'229.0'"));
      CHECK(contains(r.log, "is not an integer"));
    } else {
      CHECK_FALSE(contains(lower(r.log), "warning"));
    }
  }
  for (const char *block : {"Number_decimal_new_null", "Number_decimal_old"}) {
    INFO(block);
    const std::string symmetry =
        (std::string(block) == "Number_decimal_new_null"
             ? std::string("_space_group_IT_number ?\n")
             : std::string()) +
        number_line("229.0");
    std::string what;
    try {
      const Read r = parse_cif_text(bcc(block, symmetry));
      FAIL_CHECK("accepted: " << r.st.natoms() << " atoms");
    } catch (const std::exception &e) {
      what = e.what();
    }
    INFO("message: " << what);
    CHECK(contains(what, std::string("CIF text (data_") + block + "): "));
    CHECK(contains(what, "'229.0'"));
  }
}

/* Pins: an H-M symbol that names a setting explicitly (origin choice,
 * axes) and differs from the operator loop's is reported:
 * si_diamond_icsd_s_loop.cif, the 192 operators of F d -3 m:2 with ICSD's
 * 'F d -3 m S' (origin choice 1), gives the loop's 8 atoms and one WARNING
 * naming F d -3 m:1 and F d -3 m:2, through every route; 'F d -3 m Z'
 * (si_diamond_icsd_z_loop.cif) prints nothing (the ICSD suffix test
 * above).
 * Before the fix: gemmi's check compares the H-M symbols without their
 * setting, so 'F d -3 m S' printed nothing (count_of == 1 fails). Review
 * round 8: the test read the CIF only through parseCifContents. */
TEST_CASE("CIF reader: an H-M symbol naming another setting than the loop's "
          "gives a WARNING",
          "[structure_readers][cif][spacegroup][warning]") {
  PeriodicTable::initialize();
  const std::string path = kCifDir + "si_diamond_icsd_s_loop.cif";
  for (const Route route : kRoutes) {
    INFO("si_diamond_icsd_s_loop.cif via " << route_name(route));
    const Read r = read_cif(path, route);
    INFO("printed:\n" << r.log);
    check_sites(r.st, kSiDiamond);
    CHECK(count_of(r.log, "WARNING") == 1);
    CHECK(contains(r.log, "WARNING: " +
                              source_of(path, route,
                                        "Si_diamond_icsd_s_loop") +
                              ": "));
    CHECK(contains(r.log, "F d -3 m:1"));
    CHECK(contains(r.log, "F d -3 m:2"));
  }
}

// ================================================ review round 8 ====

namespace {

// The 12 operators of R -3 m on rhombohedral axes (R -3 m:R).
const std::string kRm3mRLoop =
    "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n'z,x,y'\n'y,z,x'\n"
    "'-y,-x,-z'\n'-x,-z,-y'\n'-z,-y,-x'\n'-x,-y,-z'\n'-z,-x,-y'\n"
    "'-y,-z,-x'\n'y,x,z'\n'x,z,y'\n'z,y,x'\n";

// Bi in its rhombohedral and hexagonal cells (bi_rh_number_only.cif,
// bi_hex_number_only.cif), Bi1 at x = 0.2339 on 2c or 6c.
std::string bi_rh(const char *block, const std::string &symmetry) {
  return cif_cell(block, 4.7236, 4.7236, 4.7236, 57.35, 57.35, 57.35,
                  symmetry, "Bi1 Bi 0.2339 0.2339 0.2339\n");
}
std::string bi_hex(const char *block, const std::string &symmetry) {
  return cif_cell(block, 4.5332, 4.5332, 11.7967, 90, 90, 120, symmetry,
                  "Bi1 Bi 0 0 0.2339\n");
}

}  // namespace

/* Pins: an R group must fit the axes of the cell, as a monoclinic one
 * must: on hexagonal axes (R ... :H) the cell has alpha = beta = 90 and
 * gamma = 120 degrees, on rhombohedral axes (:R) alpha = beta = gamma,
 * each within 1e-3, or the CIF is refused with a std::runtime_error naming
 * the source, the setting and the angles:
 *   - the rhombohedral Bi cell (alpha = beta = gamma = 57.35) with the H-M
 *     symbol 'R -3 m:H', or ICSD's 'R -3 m H', is refused, naming
 *     R -3 m:H and 57.35;
 *   - the hexagonal Bi cell (gamma = 120) with 'R -3 m:R' is refused,
 *     naming R -3 m:R and 120;
 *   - 'R -3 m', which names no axes, gives the axes of the cell: the 2
 *     atoms of R -3 m:R in the rhombohedral cell and the 6 of R -3 m:H in
 *     the hexagonal one (pins).
 * Before the fix: the setting the symbol names was used whatever the cell:
 * 'R -3 m:H' and 'R -3 m H' gave 36 atoms 0.78 A apart in the
 * rhombohedral cell, and 'R -3 m:R' 6 atoms 1.06 A apart in the hexagonal
 * one, without a message ("accepted: 36 atoms" twice, "accepted: 6
 * atoms"). */
TEST_CASE("CIF reader: an R group must fit the axes of the cell",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  struct Case {
    std::string text;
    const char *block, *setting;
    double angle;
  };
  for (const Case &c :
       {Case{bi_rh("Bi_rh_colon_H", hm_line("R -3 m:H")), "Bi_rh_colon_H",
             "R -3 m:H", 57.35},
        Case{bi_rh("Bi_rh_icsd_H", hm_line("R -3 m H")), "Bi_rh_icsd_H",
             "R -3 m:H", 57.35},
        Case{bi_hex("Bi_hex_colon_R", hm_line("R -3 m:R")), "Bi_hex_colon_R",
             "R -3 m:R", 120}}) {
    INFO(c.block);
    const std::string what = refusal(c.text);
    INFO("message: " << what);
    if (what.empty()) continue;
    CHECK(contains(what, std::string("CIF text (data_") + c.block + "): "));
    CHECK(contains(what, c.setting));
    CHECK(mentions_value(what, c.angle));
  }
  {
    INFO("'R -3 m' in the rhombohedral cell");
    Read r;
    if (accepted(bi_rh("Bi_rh_plain", hm_line("R -3 m")), r))
      check_sites(r.st, kBiRh);
  }
  INFO("'R -3 m' in the hexagonal cell");
  Read r;
  if (accepted(bi_hex("Bi_hex_plain", hm_line("R -3 m")), r))
    check_sites(r.st, kBiHex);
}

/* Pins: ICSD's ' H' / ' R' suffix of an R group's H-M symbol names its
 * axes, like ':H' / ':R', also for the WARNING on a symbol that names
 * another setting than the loop's: the rhombohedral Bi cell with the 12
 * operators of R -3 m:R and the H-M symbol 'R -3 m R' prints nothing (2
 * atoms); with 'R -3 m H' it gives the 2 atoms of the loop and one WARNING
 * naming R -3 m:H and R -3 m:R.
 * Before the fix: only ':' marked an explicit setting, so 'R -3 m H'
 * printed nothing (count_of == 1 fails). */
TEST_CASE("CIF reader: ICSD's H and R suffixes name the axes",
          "[structure_readers][cif][spacegroup][warning]") {
  PeriodicTable::initialize();
  {
    INFO("'R -3 m R' with the R -3 m:R loop");
    Read r;
    if (accepted(bi_rh("Bi_rh_loop_R", hm_line("R -3 m R") + kRm3mRLoop), r)) {
      INFO("printed:\n" << r.log);
      check_sites(r.st, kBiRh);
      CHECK_FALSE(contains(lower(r.log), "warning"));
    }
  }
  INFO("'R -3 m H' with the R -3 m:R loop");
  Read r;
  if (!accepted(bi_rh("Bi_rh_loop_H", hm_line("R -3 m H") + kRm3mRLoop), r))
    return;
  INFO("printed:\n" << r.log);
  check_sites(r.st, kBiRh);
  CHECK(count_of(r.log, "WARNING") == 1);
  CHECK(contains(r.log, "R -3 m:H"));
  CHECK(contains(r.log, "R -3 m:R"));
}

/* Pins: an unknown value of the new operator-loop tag does not hide the
 * old one: _space_group_symop_operation_xyz ? with the loop
 * _symmetry_equiv_pos_as_xyz x,y,z; x+1/2,y+1/2,z+1/2 (the I centring),
 * in a bcc cell (a = 2.8665 A) with Fe1 at the origin, gives the 2 atoms
 * of bcc Fe and prints nothing.
 * Before the fix: gemmi takes the first loop tag present, so the loop was
 * the single operator '?', which forms no group: P 1, 1 atom, with a
 * WARNING (check_sites fails). */
TEST_CASE("CIF reader: an unknown value of the new loop tag does not hide "
          "the old loop",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  Read r;
  if (!accepted(cif_cell("Loop_null_new", 2.8665, 2.8665, 2.8665, 90, 90, 90,
                         "_space_group_symop_operation_xyz ?\n"
                         "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n"
                         "'x+1/2,y+1/2,z+1/2'\n",
                         "Fe1 Fe 0 0 0\n"),
                r))
    return;
  INFO("printed:\n" << r.log);
  check_sites(r.st, kFeBcc);
  CHECK_FALSE(contains(lower(r.log), "warning"));
}

/* Pins: an error gemmi raises while parsing the text names the source:
 *   - two_blocks.cif (data_a, then data_b) through CifReader::read and the
 *     selector: "CIF file <path>: ";
 *   - the same text, and a CIF with an unterminated quoted string, through
 *     parseCifContents: "CIF text: ".
 * Before the fix: gemmi's messages ("single data block expected, got 2",
 * "string:4:...: unterminated 'string'") named neither (four CHECKs
 * fail). */
TEST_CASE("CIF reader: a CIF gemmi cannot parse is refused naming the "
          "source",
          "[structure_readers][cif]") {
  PeriodicTable::initialize();
  const std::string path = kCifDir + "two_blocks.cif";
  for (const Route route : {Route::File, Route::Selector}) {
    INFO("two_blocks.cif via " << route_name(route));
    std::string what;
    try {
      read_cif(path, route);
      FAIL_CHECK("accepted");
    } catch (const std::exception &e) {
      what = e.what();
    }
    INFO("message: " << what);
    CHECK(contains(what, "CIF file " + path + ": "));
  }
  for (const std::string &text :
       {slurp(path),
        std::string("data_q\n_cell_length_a 'unterminated\n")}) {
    std::string what;
    try {
      parse_cif_text(text);
      FAIL_CHECK("accepted");
    } catch (const std::exception &e) {
      what = e.what();
    }
    INFO("message: " << what);
    CHECK(contains(what, "CIF text: "));
  }
}

// ================================================ review round 9 ====

namespace {

// R 3 on hexagonal axes in the reverse setting (centring (1/3,2/3,1/3),
// (2/3,1/3,2/3)), which gemmi does not tabulate.
const std::string kR3ReverseLoop =
    "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n'-y,x-y,z'\n'-x+y,-x,z'\n"
    "'x+1/3,y+2/3,z+1/3'\n'-y+1/3,x-y+2/3,z+1/3'\n'-x+y+1/3,-x+2/3,z+1/3'\n"
    "'x+2/3,y+1/3,z+2/3'\n'-y+2/3,x-y+1/3,z+2/3'\n'-x+y+2/3,-x+1/3,z+2/3'\n";

}  // namespace

/* Pins: an R group must fit the axes of the cell however it is given, in a
 * tabulated setting or not, and the refusal advises what can change it:
 *   - in the rhombohedral Bi cell: the Hall symbol '-R 3 2"' (R -3 m:H),
 *     the same with the origin moved, '-R 3 2" (x,y,z+1/4)' (in no table),
 *     and the 9 operators of R 3 on hexagonal axes in the reverse setting
 *     (in no table) are refused, each naming hexagonal axes and 57.35;
 *   - in the hexagonal Bi cell, the 12 operators of R -3 m:R are refused,
 *     naming rhombohedral axes and 120;
 *   - in a cell that fits neither axes (a = b = 4.5332, c = 11.7967 A,
 *     gamma = 119.5), the number 166 and the H-M symbol 'R -3 m' (axes
 *     from the cell) are refused, naming 119.5 and saying to correct the
 *     cell ("correct the cell"), not to give the symbol without its axes.
 * Before the fix: the origin-moved Hall symbol (18 atoms) and the reverse
 * loop (9 atoms) were accepted, with only the WARNING that the group is in
 * no table: their rotations were looked up without their centring, as
 * P -3 m 1 and P 3, which have no axes to check ("accepted", twice); the
 * last two advised "give the H-M symbol without its axes (e.g. 'R -3 m',
 * which takes them from the cell)" (the CHECK for "correct the cell"
 * fails twice). The others pass (pins of the Hall and loop routes). */
TEST_CASE("CIF reader: an R group must fit the axes of the cell however it "
          "is given",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  struct Case {
    const char *block;
    std::string text;
    const char *axes;
    double angle;
    bool correct_cell;
  };
  const std::string off_cell_site = "Bi1 Bi 0 0 0.2339\n";
  const std::vector<Case> cases = {
      {"Bi_rh_hall_H", bi_rh("Bi_rh_hall_H", hall_line("-R 3 2\"")),
       "hexagonal axes", 57.35, false},
      {"Bi_rh_hall_H_shift",
       bi_rh("Bi_rh_hall_H_shift", hall_line("-R 3 2\" (x,y,z+1/4)")),
       "hexagonal axes", 57.35, false},
      {"Bi_rh_reverse", bi_rh("Bi_rh_reverse", kR3ReverseLoop),
       "hexagonal axes", 57.35, false},
      {"Bi_hex_loop_R", bi_hex("Bi_hex_loop_R", kRm3mRLoop),
       "rhombohedral axes", 120, false},
      {"Bi_off_number",
       cif_cell("Bi_off_number", 4.5332, 4.5332, 11.7967, 90, 90, 119.5,
                number_line("166"), off_cell_site),
       "hexagonal axes", 119.5, true},
      {"Bi_off_hm",
       cif_cell("Bi_off_hm", 4.5332, 4.5332, 11.7967, 90, 90, 119.5,
                hm_line("R -3 m"), off_cell_site),
       "hexagonal axes", 119.5, true}};
  for (const Case &c : cases) {
    INFO(c.block);
    const std::string what = refusal(c.text);
    INFO("message: " << what);
    if (what.empty()) continue;
    CHECK(contains(what, std::string("CIF text (data_") + c.block + "): "));
    CHECK(contains(what, c.axes));
    CHECK(mentions_value(what, c.angle));
    if (c.correct_cell) {
      CHECK(contains(what, "correct the cell"));
      CHECK_FALSE(contains(what, "without its axes"));
    }
  }
}

/* Pins: with an operator loop that forms a group, the H-M symbol is only
 * compared with it, and a WARNING reports one that cannot be read or is a
 * number: fe_bcc_full_loop.cif (the 96 operators of I m -3 m) with the
 * H-M symbol replaced by 'F d -3 m Q', '(No. 229)', '1003' or '229' gives
 * the 2 atoms of bcc Fe and one WARNING, which for the two numbers says
 * "is a number".
 * Before the fix: '229' printed nothing (gemmi's check took it as the
 * CCP4 code of I m -3 m, the loop's group) and '1003' a WARNING that the
 * group differs, P 1 1 2: the CHECKs for "is a number" fail twice and
 * count_of == 1 once. */
TEST_CASE("CIF reader: with a loop, an H-M symbol that is a number or "
          "cannot be read gives a WARNING",
          "[structure_readers][cif][spacegroup][warning]") {
  PeriodicTable::initialize();
  const std::string text = slurp(kCifDir + "fe_bcc_full_loop.cif");
  const std::string line = "_symmetry_space_group_name_H-M 'I m -3 m'";
  REQUIRE(text.find(line) != std::string::npos);
  for (const char *hm : {"F d -3 m Q", "(No. 229)", "1003", "229"}) {
    INFO("H-M '" << hm << "' with the loop of I m -3 m");
    std::string t = text;
    t.replace(t.find(line), line.size(),
              std::string("_symmetry_space_group_name_H-M '") + hm + "'");
    Read r;
    if (!accepted(t, r)) continue;
    INFO("printed:\n" << r.log);
    check_sites(r.st, kFeBcc);
    CHECK(count_of(r.log, "WARNING") == 1);
    if (std::isdigit(static_cast<unsigned char>(hm[0])))
      CHECK(contains(r.log, "is a number"));
  }
}

/* Pins: an H-M symbol that names another setting than the Hall symbol's
 * is reported as one that differs from the loop's is: diamond Si
 * (a = 5.431 A, Si1 at (1/8,1/8,1/8)) with the Hall symbol '-F 4vw 2vw 3'
 * (F d -3 m:2) and the H-M symbol 'F d -3 m S' (origin choice 1) gives the
 * 8 atoms and one WARNING naming F d -3 m:1 and F d -3 m:2.
 * Before the fix: passes as well (a pin of the Hall route). */
TEST_CASE("CIF reader: an H-M symbol naming another setting than the Hall "
          "symbol's gives a WARNING",
          "[structure_readers][cif][spacegroup][warning]") {
  PeriodicTable::initialize();
  Read r;
  if (!accepted(cif_cell("Si_hall_S", 5.431, 5.431, 5.431, 90, 90, 90,
                         hall_line("-F 4vw 2vw 3") + hm_line("F d -3 m S"),
                         "Si1 Si 0.125 0.125 0.125\n"),
                r))
    return;
  INFO("printed:\n" << r.log);
  check_sites(r.st, kSiDiamond);
  CHECK(count_of(r.log, "WARNING") == 1);
  CHECK(contains(r.log, "F d -3 m:1"));
  CHECK(contains(r.log, "F d -3 m:2"));
}

// =============================================== review round 10 ====

/* Pins: the extra translations of a supercell are no R centring: in a
 * loop, x+1/3 and x+2/3 make the cell 3 times the primitive one, and the
 * cell is checked by the axes of the rotations (none for P 1):
 *   - the 3 operators x,y,z; x+1/3,y,z; x+2/3,y,z (P 1 in a 3 x 1 x 1
 *     supercell, a = 8.6, b = c = 2.8665 A, all angles 90) with Fe1 at the
 *     origin: 3 atoms, (0,0,0), (1/3,0,0), (2/3,0,0);
 *   - the 6 operators VESTA writes for that supercell of bcc Fe (those 3
 *     and x+1/6,y+1/2,z+1/2; x+1/2,y+1/2,z+1/2; x+5/6,y+1/2,z+1/2): 6
 *     atoms;
 *   - the 4 operators of P 1 2/m 1 (x,y,z; -x,y,-z; -x,-y,-z; x,-y,z),
 *     each also with x+1/3 and x+2/3 (a 3 x 1 x 1 supercell, a = 15, b = 6,
 *     c = 7 A): with beta = 100, the 12 atoms from Fe1 at (0.1,0.2,0.3);
 *     with gamma = 110 refused, naming unique axis b and 110.
 * Before the fix: a centring vector with a third was taken as an R
 * centring on hexagonal axes, so each was refused ("... R-centred on
 * hexagonal axes) is on hexagonal axes ... gamma 90"): three "refused"
 * failures, and the last refusal names no unique axis. */
TEST_CASE("CIF reader: a supercell's translations by thirds are no R "
          "centring",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  const std::string loop_head = "loop_\n_symmetry_equiv_pos_as_xyz\n";
  {
    INFO("P 1 in a 3 x 1 x 1 supercell");
    Read r;
    if (accepted(cif_cell("Sc311_p1", 8.6, 2.8665, 2.8665, 90, 90, 90,
                          loop_head + "'x,y,z'\n'x+1/3,y,z'\n'x+2/3,y,z'\n",
                          "Fe1 Fe 0 0 0\n"),
                 r))
      check_sites(r.st, {{"Fe", {0.0, 0.0, 0.0}},
                         {"Fe", {1.0 / 3.0, 0.0, 0.0}},
                         {"Fe", {2.0 / 3.0, 0.0, 0.0}}});
  }
  {
    INFO("VESTA's 6 operators for the 3 x 1 x 1 supercell of bcc Fe");
    Read r;
    if (accepted(cif_cell("Sc311_vesta", 8.6, 2.8665, 2.8665, 90, 90, 90,
                          loop_head +
                              "'x,y,z'\n'x+1/3,y,z'\n'x+2/3,y,z'\n"
                              "'x+1/6,y+1/2,z+1/2'\n'x+1/2,y+1/2,z+1/2'\n"
                              "'x+5/6,y+1/2,z+1/2'\n",
                          "Fe1 Fe 0 0 0\n"),
                 r))
      check_sites(r.st, {{"Fe", {0.0, 0.0, 0.0}},
                         {"Fe", {1.0 / 3.0, 0.0, 0.0}},
                         {"Fe", {2.0 / 3.0, 0.0, 0.0}},
                         {"Fe", {1.0 / 6.0, 0.5, 0.5}},
                         {"Fe", {0.5, 0.5, 0.5}},
                         {"Fe", {5.0 / 6.0, 0.5, 0.5}}});
  }
  std::string mono = loop_head;
  std::vector<Site> mono_sites;
  for (const char *t : {"", "+1/3", "+2/3"}) {
    for (const char *op : {"x%s,y,z", "-x%s,y,-z", "-x%s,-y,-z", "x%s,-y,z"}) {
      char buf[32];
      std::snprintf(buf, sizeof buf, op, t);
      mono += std::string("'") + buf + "'\n";
    }
  }
  for (double t : {0.0, 1.0 / 3.0, 2.0 / 3.0}) {
    mono_sites.push_back({"Fe", {0.1 + t, 0.2, 0.3}});
    mono_sites.push_back({"Fe", {-0.1 + t, 0.2, -0.3}});
    mono_sites.push_back({"Fe", {-0.1 + t, -0.2, -0.3}});
    mono_sites.push_back({"Fe", {0.1 + t, -0.2, 0.3}});
  }
  {
    INFO("P 1 2/m 1 in a 3 x 1 x 1 supercell, beta = 100");
    Read r;
    if (accepted(cif_cell("Sc311_mono", 15, 6, 7, 90, 100, 90, mono,
                          kFeGeneral),
                 r))
      check_sites(r.st, mono_sites);
  }
  INFO("P 1 2/m 1 in a 3 x 1 x 1 supercell, gamma = 110");
  const std::string what = refusal(
      cif_cell("Sc311_mono_gamma", 15, 6, 7, 90, 90, 110, mono, kFeGeneral));
  INFO("message: " << what);
  if (what.empty()) return;
  CHECK(contains(what, "unique axis b"));
  CHECK(mentions_value(what, 110.0));
}

/* Pins: a refusal advises correcting the cell when no setting of the group
 * fits it, and another setting when one does:
 *   - 'R -3 m H' and 'R -3 m:H' in a cell with gamma = 119.99 (a = b =
 *     4.5332, c = 11.7967 A), which fits neither hexagonal nor
 *     rhombohedral axes: "correct the cell", not "without its axes";
 *   - 'R -3 m:R' in the hexagonal Bi cell (which fits hexagonal axes):
 *     "without its axes";
 *   - 'P 1 21/m 1' with alpha = 100 and gamma = 110 (a = 5, b = 6, c = 7
 *     A), which fits no unique axis: "correct the cell", not "full H-M
 *     symbol"; with only gamma = 110 (unique axis c fits): "full H-M
 *     symbol".
 * Before the fix: a symbol naming its axes was always told to drop them,
 * and a monoclinic symbol to give the setting the cell is in, which a
 * second read refuses as well (the CHECKs for "correct the cell" fail
 * three times). */
TEST_CASE("CIF reader: a refusal advises correcting the cell when no "
          "setting fits it",
          "[structure_readers][cif][spacegroup]") {
  PeriodicTable::initialize();
  struct Case {
    std::string text;
    const char *advice, *not_advice;
  };
  const std::string bi_site = "Bi1 Bi 0 0 0.2339\n";
  const std::vector<Case> cases = {
      {cif_cell("R_named_H_off", 4.5332, 4.5332, 11.7967, 90, 90, 119.99,
                hm_line("R -3 m H"), bi_site),
       "correct the cell", "without its axes"},
      {cif_cell("R_colon_H_off", 4.5332, 4.5332, 11.7967, 90, 90, 119.99,
                hm_line("R -3 m:H"), bi_site),
       "correct the cell", "without its axes"},
      {bi_hex("R_colon_R_hex", hm_line("R -3 m:R")), "without its axes",
       "correct the cell"},
      {cif_cell("P21m_no_axis", 5, 6, 7, 100, 90, 110, hm_line("P 1 21/m 1"),
                kFeGeneral),
       "correct the cell", "full H-M symbol"},
      {cif_cell("P21m_axis_c", 5, 6, 7, 90, 90, 110, hm_line("P 1 21/m 1"),
                kFeGeneral),
       "full H-M symbol", "correct the cell"}};
  for (const Case &c : cases) {
    const std::string what = refusal(c.text);
    INFO("message: " << what);
    if (what.empty()) continue;
    CHECK(contains(what, c.advice));
    CHECK_FALSE(contains(what, c.not_advice));
  }
}

/* Pins: an H-M symbol is compared with the group used by its operations,
 * so a name gemmi tabulates for the same operations is no inconsistency:
 * the Hall symbol 'C 2 2 -1ac' (C c c a:1) with the H-M symbol
 * 'C c c b:1', or ICSD's 'C c c b S', in a 5 x 6 x 7 A cell with Fe1 at
 * the origin, gives the 4 atoms of origin choice 1, (0,0,0), (0,1/2,1/2),
 * (1/2,0,1/2), (1/2,1/2,0), and prints nothing.
 * Before the fix: gemmi's check compares the H-M strings, C c c b and
 * C c c a, and printed "space group from H-M symbol (C c c b:1) differs"
 * (the CHECK_FALSE fails twice). */
TEST_CASE("CIF reader: an equivalent name of the same setting is no "
          "inconsistency",
          "[structure_readers][cif][spacegroup][warning]") {
  PeriodicTable::initialize();
  for (const char *hm : {"C c c b:1", "C c c b S"}) {
    INFO("Hall 'C 2 2 -1ac' with H-M '" << hm << "'");
    Read r;
    if (!accepted(cif_cell("Cccb1", 5, 6, 7, 90, 90, 90,
                           hall_line("C 2 2 -1ac") + hm_line(hm),
                           "Fe1 Fe 0 0 0\n"),
                  r))
      continue;
    INFO("printed:\n" << r.log);
    check_sites(r.st, {{"Fe", {0.0, 0.0, 0.0}},
                       {"Fe", {0.0, 0.5, 0.5}},
                       {"Fe", {0.5, 0.0, 0.5}},
                       {"Fe", {0.5, 0.5, 0.0}}});
    CHECK_FALSE(contains(lower(r.log), "warning"));
  }
}
