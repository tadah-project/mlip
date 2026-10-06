// Catch2 v2 unit tests for compute_data_stats (Phase 3).
//
// Builds tiny synthetic StructureDBs in C++ — no external DBFILE
// dependency — and verifies:
//   * single-element bcc-like 2-atom cell -> 1 per_pair, peaks at NN
//   * binary rocksalt-like 8-atom cell    -> 3 per_pair (Na-Na, Na-Cl, Cl-Cl)
//   * material-class detector flags "metal_like" vs "covalent_or_ionic"

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/core/stats/data_stats_io.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/data_stats_compute.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure_db.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace tadah::mlip;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::core::stats::DataStats;
using tadah::core::stats::IDX_P50;

static Context make_ctx_with_rcut(double rcut) {
  // RCUTMAX is an internal key — addInternal must be used.
  // RCUT2B is allowed via add (postprocess derives RCUTMAX from it).
  Context c;
  c.add<double>("RCUT2B", rcut);
  c.add<bool>("FORCE", false);
  c.add<bool>("STRESS", false);
  c.add<bool>("BIAS", false);
  return c;
}

static Structure make_bcc_ta_cell(double a) {
  // bcc primitive: 2 atoms at (0,0,0) and (a/2, a/2, a/2) in a cubic cell.
  PeriodicTable::initialize();
  auto ta = PeriodicTable::find_by_symbol("Ta");
  Structure st;
  st.label = "synthetic_bcc_Ta";
  st.energy = -16.0;
  st.cell(0,0) = a;  st.cell(0,1) = 0;  st.cell(0,2) = 0;
  st.cell(1,0) = 0;  st.cell(1,1) = a;  st.cell(1,2) = 0;
  st.cell(2,0) = 0;  st.cell(2,1) = 0;  st.cell(2,2) = a;
  Atom a0(ta, 0.0, 0.0, 0.0);
  Atom a1(ta, 0.5*a, 0.5*a, 0.5*a);
  st.add_atom(a0);
  st.add_atom(a1);
  return st;
}

static Structure make_rocksalt_cell(double a) {
  // 8-atom rocksalt: Na at FCC positions, Cl at FCC + (a/2, 0, 0).
  PeriodicTable::initialize();
  auto na = PeriodicTable::find_by_symbol("Na");
  auto cl = PeriodicTable::find_by_symbol("Cl");
  Structure st;
  st.label = "synthetic_rocksalt";
  st.energy = -32.0;
  st.cell(0,0) = a;  st.cell(0,1) = 0;  st.cell(0,2) = 0;
  st.cell(1,0) = 0;  st.cell(1,1) = a;  st.cell(1,2) = 0;
  st.cell(2,0) = 0;  st.cell(2,1) = 0;  st.cell(2,2) = a;
  // Na FCC sublattice
  st.add_atom(Atom(na, 0,    0,    0));
  st.add_atom(Atom(na, 0.5*a,0.5*a,0));
  st.add_atom(Atom(na, 0.5*a,0,    0.5*a));
  st.add_atom(Atom(na, 0,    0.5*a,0.5*a));
  // Cl FCC sublattice (shifted by (a/2, 0, 0))
  st.add_atom(Atom(cl, 0.5*a,0,    0));
  st.add_atom(Atom(cl, 0,    0.5*a,0));
  st.add_atom(Atom(cl, 0,    0,    0.5*a));
  st.add_atom(Atom(cl, 0.5*a,0.5*a,0.5*a));
  return st;
}

TEST_CASE("compute_data_stats: single bcc Ta cell produces one per_pair with NN peak") {
  Context c = make_ctx_with_rcut(6.0);
  StructureDB stdb;
  stdb.add(make_bcc_ta_cell(/*a=*/3.30));
  // (NN_bcc = a*sqrt(3)/2 ≈ 2.86 Å)

  auto d = compute_data_stats(stdb, c, false);
  REQUIRE(d.n_atoms == 2);
  REQUIRE(d.n_structures == 1);
  REQUIRE(d.elements.size() == 1);
  REQUIRE(d.elements[0] == "Ta");

  REQUIRE(d.per_pair.size() == 1);
  const auto& pp = d.per_pair[0];
  REQUIRE(pp.elements.first  == "Ta");
  REQUIRE(pp.elements.second == "Ta");
  REQUIRE(pp.r_min > 2.5);
  REQUIRE(pp.r_min < 3.0);

  // At least one detected KDE peak should bracket the NN+2NN shells
  // (analytical NN = 2.858 Å, 2NN = a = 3.30 Å). With 116 pairs in the
  // 2-atom cubic cell (58 neighbours per atom within 6 Å: 8 + 6 + 12 +
  // 24 + 8), Silverman's bandwidth merges the two shells into a single
  // peak around their mean. We assert this leftmost
  // peak lies in the [2.7, 3.5] window — anywhere else would mean
  // the radial GTO would miss the dominant 1NN/2NN region.
  REQUIRE_FALSE(pp.r_peaks.empty());
  bool nn_region_peak_found = false;
  for (const auto& pk : pp.r_peaks) {
    if (pk.position >= 2.7 && pk.position <= 3.5) {
      nn_region_peak_found = true;
      break;
    }
  }
  REQUIRE(nn_region_peak_found);
}

TEST_CASE("compute_data_stats: rocksalt cell produces three per_pair entries") {
  Context c = make_ctx_with_rcut(6.0);
  StructureDB stdb;
  stdb.add(make_rocksalt_cell(/*a=*/5.64));        // typical NaCl lattice constant

  auto d = compute_data_stats(stdb, c, false);
  REQUIRE(d.n_atoms == 8);
  REQUIRE(d.elements.size() == 2);
  // sorted lex: Cl, Na
  REQUIRE(d.elements[0] == "Cl");
  REQUIRE(d.elements[1] == "Na");

  // Three element-pair buckets: Cl-Cl, Cl-Na, Na-Na (canonical pair
  // ordering is lex-ascending, so first <= second).
  REQUIRE(d.per_pair.size() == 3);
  bool found_clcl = false, found_clna = false, found_nana = false;
  for (const auto& pp : d.per_pair) {
    if (pp.elements.first == "Cl" && pp.elements.second == "Cl") found_clcl = true;
    if (pp.elements.first == "Cl" && pp.elements.second == "Na") found_clna = true;
    if (pp.elements.first == "Na" && pp.elements.second == "Na") found_nana = true;
  }
  REQUIRE(found_clcl);
  REQUIRE(found_clna);
  REQUIRE(found_nana);

  // Cl-Na nearest-neighbour distance is a/2 ≈ 2.82 Å.
  for (const auto& pp : d.per_pair) {
    if (pp.elements.first == "Cl" && pp.elements.second == "Na") {
      REQUIRE(pp.r_min == Approx(2.82).margin(0.05));
    }
  }
}

TEST_CASE("compute_data_stats: classifier flags pure metal vs binary as different classes") {
  Context cTa = make_ctx_with_rcut(6.0);
  StructureDB stdbTa;
  stdbTa.add(make_bcc_ta_cell(3.30));
  auto dTa = compute_data_stats(stdbTa, cTa, false);
  // The 2-atom bcc cell has only ~14 first/second neighbours per atom
  // under PBC at a=3.30 with cutoff 6.0 — well above the dimer
  // threshold of 10. With multiple resolved peaks, classifier picks
  // metal_like.
  REQUIRE(dTa.class_label == "metal_like");

  Context cNa = make_ctx_with_rcut(6.0);
  StructureDB stdbNa;
  stdbNa.add(make_rocksalt_cell(5.64));
  auto dNa = compute_data_stats(stdbNa, cNa, false);
  REQUIRE(dNa.class_label == "covalent_or_ionic");
}

TEST_CASE("DataStats save / load round-trips through the text format") {
  Context c = make_ctx_with_rcut(6.0);
  StructureDB stdb;
  stdb.add(make_bcc_ta_cell(3.30));
  auto d = compute_data_stats(stdb, c, false);

  std::stringstream ss;
  tadah::core::stats::save(d, ss);
  auto d2 = tadah::core::stats::load(ss);

  REQUIRE(d2.n_atoms == d.n_atoms);
  REQUIRE(d2.n_structures == d.n_structures);
  REQUIRE(d2.elements == d.elements);
  REQUIRE(d2.class_label == d.class_label);
  REQUIRE(d2.per_pair.size() == d.per_pair.size());
  REQUIRE(d2.per_pair[0].elements == d.per_pair[0].elements);
  REQUIRE(d2.per_pair[0].n_pairs == d.per_pair[0].n_pairs);
  for (std::size_t i = 0; i < 11; ++i) {
    REQUIRE(d2.per_pair[0].r_percentiles[i] == Approx(d.per_pair[0].r_percentiles[i]));
  }
  REQUIRE(d2.per_pair[0].r_peaks.size() == d.per_pair[0].r_peaks.size());
  for (std::size_t i = 0; i < d.per_pair[0].r_peaks.size(); ++i) {
    REQUIRE(d2.per_pair[0].r_peaks[i].position == d.per_pair[0].r_peaks[i].position);
    REQUIRE(d2.per_pair[0].r_peaks[i].fwhm     == d.per_pair[0].r_peaks[i].fwhm);
  }
}

// The neighbour counts and pair distances describe what the model sees: a
// pair enters a descriptor only when its distance is strictly below the
// cutoff (DescriptorsCalc skips rij_sq >= rc^2). A simple cubic lattice with
// a = 2.5 at the cutoff 5 = 2a has 26 neighbours strictly inside (6 at a,
// 12 at a sqrt 2, 8 at a sqrt 3) and 6 more exactly at 2a. A neighbour list
// may hold pairs a rounding margin beyond its cutoff, so the audit must
// apply the strict test itself: 26 per atom, not 32.
TEST_CASE("compute_data_stats counts neighbours strictly inside the cutoff",
          "[data_stats][nn]") {
  PeriodicTable::initialize();
  auto ta = PeriodicTable::find_by_symbol("Ta");
  const double a = 2.5;
  Structure st;
  st.label = "simple_cubic_Ta";
  st.energy = -8.0;
  st.cell.load(a, 0.0, 0.0,
               0.0, a, 0.0,
               0.0, 0.0, a);
  st.add_atom(Atom(ta, 0.0, 0.0, 0.0));
  StructureDB stdb;
  stdb.add(st);

  Context c = make_ctx_with_rcut(2.0 * a);
  auto d = compute_data_stats(stdb, c, false);
  REQUIRE(d.n_atoms == 1);

  REQUIRE(d.nn_count_percentiles_per_element.count("Ta") == 1);
  // One atom: every percentile and the mean are its count, the spread is 0.
  const auto &counts = d.nn_count_percentiles_per_element.at("Ta");
  for (std::size_t k = tadah::core::stats::IDX_P1;
       k <= tadah::core::stats::IDX_MEAN; ++k) {
    INFO("summary slot " << k);
    CHECK(counts[k] == 26.0);
  }
  CHECK(counts[tadah::core::stats::IDX_STD] == 0.0);

  REQUIRE(d.per_pair.size() == 1);
  CHECK(d.per_pair[0].n_pairs == 26);
  CHECK(d.per_pair[0].r_min == Approx(a).epsilon(1e-12));
  CHECK(d.per_pair[0].r_max < 2.0 * a);
  CHECK(d.per_pair[0].r_max == Approx(a * std::sqrt(3.0)).epsilon(1e-12));
}

// compute_data_stats builds the missing neighbour lists in an OpenMP loop.
// An exception thrown inside an OpenMP region cannot leave it (the runtime
// calls std::terminate), so a structure the neighbour search refuses (here
// a non-finite position) must reach the caller as that exception, not end
// the process or yield statistics built from a broken list.
TEST_CASE("compute_data_stats carries a neighbour-list error out of its "
          "parallel loop",
          "[data_stats][nn][invalid]") {
  PeriodicTable::initialize();
  StructureDB stdb;
  stdb.add(make_bcc_ta_cell(3.30));
  Structure bad = make_bcc_ta_cell(3.30);
  bad.label = "nan_position";
  bad.atoms[1].position[1] = std::numeric_limits<double>::quiet_NaN();
  stdb.add(bad);
  stdb.add(make_bcc_ta_cell(3.40));
  REQUIRE(stdb.size() == 3);

  Context c = make_ctx_with_rcut(6.0);
  CHECK_THROWS_AS(compute_data_stats(stdb, c, false), std::invalid_argument);
  // The error names the structure (1-based, as dataset_stats counts).
  CHECK_THROWS_WITH(compute_data_stats(stdb, c, false),
                    Catch::Matchers::Contains("structure 2 of 3 ('nan_position')"));
}

// A structure may arrive with a neighbour list built for a smaller cutoff
// (e.g. by an earlier task). The audit counts neighbours out to RCUTMAX, so
// it rebuilds such a list instead of counting from it: bcc a = 3.3 has 58
// neighbours per atom within 6 A (8 + 6 + 12 + 24 + 8), 116 pairs in the
// 2-atom cell, while a list at 3.0 A holds only the 8 first neighbours.
TEST_CASE("compute_data_stats rebuilds a neighbour list built with a smaller "
          "cutoff",
          "[data_stats][nn]") {
  PeriodicTable::initialize();
  Structure st = make_bcc_ta_cell(3.30);
  NNFinder().calc(st, 3.0);
  REQUIRE(st.nn_size(0) == 8);
  StructureDB stdb;
  stdb.add(st);

  Context c = make_ctx_with_rcut(6.0);
  auto d = compute_data_stats(stdb, c, false);
  REQUIRE(d.per_pair.size() == 1);
  CHECK(d.per_pair[0].n_pairs == 116);
  const auto &counts = d.nn_count_percentiles_per_element.at("Ta");
  CHECK(counts[tadah::core::stats::IDX_MEAN] == 58.0);
}

/* Pins: compute_data_stats counts a pair of atoms at distance 0 like any
 * other pair strictly inside the cutoff, so two atoms at one position give
 * per_pair r_min 0. It reports such a structure rather than refusing it:
 * dataset_stats is the tool that audits a dataset, so it must show what
 * training, prediction and HPO refuse.
 *
 * bcc Ta (a = 3.3) with a third Ta on top of atom 1, cutoff 6 A. In the
 * 2-atom cell each atom has 58 neighbours within 6 A (8 + 6 + 12 + 24 + 8),
 * 26 on its own simple-cubic sublattice and 32 on the other. Atoms 1 and 3
 * each see 58 + 26 + 1 (the other's images plus the pair at distance 0),
 * atom 2 sees 58 + 32, so the cell holds 2 x 85 + 90 = 260 directed pairs,
 * 2 of them (1 -> 3 and 3 -> 1) at distance 0 (counted by brute force as
 * well).
 *
 * Before the fix: the pair loop skipped rij <= 0
 * (data_stats_compute.cpp, `if (rij <= 0.0 || rij >= rcut_audit)
 * continue;`), so r_min was the first-neighbour distance a sqrt(3)/2 =
 * 2.858 A and n_pairs 258. */
TEST_CASE("compute_data_stats counts a pair of coincident atoms, so r_min "
          "is 0",
          "[data_stats][nn][coincident]") {
  PeriodicTable::initialize();
  Structure st = make_bcc_ta_cell(3.30);
  st.label = "coincident_Ta";
  st.add_atom(Atom(PeriodicTable::find_by_symbol("Ta"), 0.0, 0.0, 0.0));
  REQUIRE(st.natoms() == 3);
  StructureDB stdb;
  stdb.add(st);

  Context c = make_ctx_with_rcut(6.0);
  DataStats d;
  // The audit's opt-in (dataset_stats); without it the pair is refused, as
  // by suggest_bounds and check_bounds (their tests).
  REQUIRE_NOTHROW(d = compute_data_stats(stdb, c, false, 0, /*list_coincident=*/true));
  REQUIRE(d.n_atoms == 3);
  REQUIRE(d.per_pair.size() == 1);
  INFO("r_min " << d.per_pair[0].r_min << ", n_pairs "
                << d.per_pair[0].n_pairs);
  CHECK(d.per_pair[0].r_min == 0.0);
  CHECK(d.per_pair[0].n_pairs == 260);
  CHECK(d.per_pair[0].r_max < 6.0);
}
