// test_nn_list_guard.cpp
//
// A neighbour list serves every cutoff up to the one it was built at, and no
// larger one: a pair between the list cutoff and the model cutoff is not in
// the list, so a descriptor computed from it silently loses those pairs.
// DescriptorsCalc and the dPhi/dtheta walkers (HPO's analytical gradients)
// read Structure::nn and filter it by their own cutoff; they must refuse a
// list built at a smaller cutoff than they need (std::logic_error naming the
// neighbour list) and accept one built at that cutoff or above.
//
// Every cell here is thick (all heights >= the largest list cutoff used,
// atoms inside the cell), so the lists themselves are complete; only the
// cutoff comparison is under test.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/analytics/dphi_dtheta_walk.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/cutoffs.h>
#include <tadah/models/descriptors/d2/d2_bp.h>
#include <tadah/models/descriptors/dm/dm_ead.h>

#include "nn_oracle.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::StDescriptors;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;
using tadah::models::Cut_Cos;
using tadah::models::D2_BP;
using tadah::models::DM_EAD;

// Outcome of a call: did it throw std::logic_error (or a type derived from
// it), something else, or nothing; and the message.
struct Outcome {
  bool logic_error = false;
  bool other = false;
  std::string what;
};

template <class F>
Outcome run(F &&f) {
  Outcome o;
  try {
    f();
  } catch (const std::logic_error &e) {
    o.logic_error = true;
    o.what = e.what();
  } catch (const std::exception &e) {
    o.other = true;
    o.what = e.what();
  }
  return o;
}

// The refusal: std::logic_error whose message names the neighbour list.
#define CHECK_REFUSES_SHORT_LIST(expr)                                  \
  do {                                                                  \
    const Outcome o_ = run([&]() { (void)(expr); });                    \
    INFO(#expr << " -> logic_error=" << o_.logic_error                  \
               << " other=" << o_.other << " what=\"" << o_.what << "\"");\
    CHECK(o_.logic_error);                                              \
    CHECK(o_.what.find("neighbour list") != std::string::npos);         \
  } while (0)

// 2x2x2 bcc Ta supercell (a = 3.3, 16 atoms): every height 6.6 A is above
// every list cutoff used below, so each list holds all its pairs.
Structure bcc_2x2x2() {
  PeriodicTable::initialize();
  const double a = 3.3;
  Structure prim;
  prim.label = "bcc Ta 2x2x2";
  prim.cell.load(a, 0.0, 0.0,
                 0.0, a, 0.0,
                 0.0, 0.0, a);
  const auto ta = PeriodicTable::find_by_symbol("Ta");
  prim.add_atom(Atom(ta, 0.0, 0.0, 0.0));
  prim.add_atom(Atom(ta, 0.5 * a, 0.5 * a, 0.5 * a));
  Structure st = nn_oracle::supercell(prim, 2, 2, 2);
  // Small fixed displacements so no two neighbours are equivalent.
  for (std::size_t i = 0; i < st.natoms(); ++i) {
    const double t = static_cast<double>(i);
    st.atoms[i].position[0] += 0.03 * std::sin(1.3 * t + 0.2);
    st.atoms[i].position[1] += 0.03 * std::sin(2.1 * t + 0.7);
    st.atoms[i].position[2] += 0.03 * std::sin(0.7 * t + 1.1);
  }
  return st;
}

// Trainer context: D2_BP (RCUT2B = rc2b) and, when rcmb > 0, DM_EAD with
// Lmax 1 (RCUTMB = rcmb); FORCE and STRESS on so every branch of
// DescriptorsCalc::calc runs.
Context model_context(double rc2b, double rcmb) {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 73);
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_BP");
  c.add("TYPE2B", 2);
  c.add("TYPE2B", 2);
  c.add("TYPE2B", "Ta");
  c.add("TYPE2B", "Ta");
  c.add("CGRID2B", "2.8 3.6");
  c.add("SGRID2B", "1.0 1.0");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCUT2B", rc2b);
  if (rcmb > 0.0) {
    c.add("INITMB", "true");
    c.add("TYPEMB", "DM_EAD");
    c.add("TYPEMB", 1);
    c.add("TYPEMB", 2);
    c.add("TYPEMB", 2);
    c.add("TYPEMB", "Ta");
    c.add("TYPEMB", "Ta");
    c.add("CGRIDMB", "2.6 3.4");
    c.add("SGRIDMB", "1.0 1.0");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB", rcmb);
  } else {
    c.add("INITMB", "false");
    c.add("TYPEMB", "DM_Dummy");
    c.add("TYPEMB", "Ta");
    c.add("TYPEMB", "Ta");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB", rc2b);
  }
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", "true");
  c.add("STRESS", "true");
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  return c;
}

// Descriptor-only contexts for the walkers (the walker tests' pattern).
Context d2_context(double rcut) {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 73);
  c.add("INIT2B", "true");
  c.add("VERBOSE", 0);
  c.add("SGRID2B", "1.0 1.0");
  c.add("CGRID2B", "2.8 3.6");
  c.add("RCUT2B", rcut);
  c.add("TYPE2B", "D2_BP");
  c.add("TYPE2B", 2);
  c.add("TYPE2B", 2);
  c.add("TYPE2B", "Ta");
  c.add("TYPE2B", "Ta");
  return c;
}

Context dm_context(double rcut) {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 73);
  c.add("INITMB", "true");
  c.add("VERBOSE", 0);
  c.add("SGRIDMB", "1.0 1.0");
  c.add("CGRIDMB", "2.6 3.4");
  c.add("RCUTMB", rcut);
  c.add("TYPEMB", "DM_EAD");
  c.add("TYPEMB", 1);
  c.add("TYPEMB", 2);
  c.add("TYPEMB", 2);
  c.add("TYPEMB", "Ta");
  c.add("TYPEMB", "Ta");
  return c;
}

StructureDB one_structure_db(const Structure &st) {
  StructureDB db;
  db.dbidx = {0, 1};
  db.structures.push_back(st);
  return db;
}

double max_abs_aed_diff(const StDescriptors &a, const StDescriptors &b,
                        std::size_t natoms, double &scale) {
  double diff = 0.0;
  scale = 0.0;
  for (std::size_t i = 0; i < natoms; ++i) {
    const auto &x = a.get_aed(i);
    const auto &y = b.get_aed(i);
    REQUIRE(x.size() == y.size());
    for (std::size_t k = 0; k < static_cast<std::size_t>(x.size()); ++k) {
      diff = std::max(diff, std::fabs(x[k] - y[k]));
      scale = std::max(scale, std::fabs(y[k]));
    }
  }
  return diff;
}

}  // namespace

TEST_CASE("DescriptorsCalc refuses a neighbour list shorter than the model "
          "cutoff",
          "[nn][nn_guard][descriptors]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;

  SECTION("two-body cutoff 5 (RCUTMAX 5)") {
    Context ctx = model_context(5.0, 0.0);
    Trainer tr(ctx, ws);
    REQUIRE(tr.context.get<double>("RCUTMAX") == 5.0);

    Structure st = bcc_2x2x2();
    NNFinder nnf;
    nnf.calc(st, 4.0);
    // The short list really lacks pairs the model uses.
    const auto ref = nn_oracle::reference(st, 5.0);
    REQUIRE(nn_oracle::table_set(st).size() < ref.inside.size());
    CHECK_REFUSES_SHORT_LIST(tr.dc.calc(st));

    // A list at the model cutoff, or above it, is accepted and gives the
    // same descriptors.
    Structure st5 = st, st65 = st;
    nnf.calc(st5, 5.0);
    nnf.calc(st65, 6.5);
    StDescriptors d5, d65;
    REQUIRE_NOTHROW(d5 = tr.dc.calc(st5));
    REQUIRE_NOTHROW(d65 = tr.dc.calc(st65));
    double scale = 0.0;
    const double diff = max_abs_aed_diff(d5, d65, st.natoms(), scale);
    INFO("max |aed(5.0 list) - aed(6.5 list)| = " << diff << ", scale "
                                                 << scale);
    REQUIRE(scale > 0.0);
    CHECK(diff <= 1e-12 * scale);
  }

  SECTION("many-body cutoff 5 above the two-body cutoff 3.5") {
    Context ctx = model_context(3.5, 5.0);
    Trainer tr(ctx, ws);
    REQUIRE(tr.context.get<double>("RCUTMAX") == 5.0);

    Structure st = bcc_2x2x2();
    NNFinder nnf;
    nnf.calc(st, 4.0);  // covers the two-body cutoff, not the many-body one
    CHECK_REFUSES_SHORT_LIST(tr.dc.calc(st));

    Structure st5 = st, st65 = st;
    nnf.calc(st5, 5.0);
    nnf.calc(st65, 6.5);
    StDescriptors d5, d65;
    REQUIRE_NOTHROW(d5 = tr.dc.calc(st5));
    REQUIRE_NOTHROW(d65 = tr.dc.calc(st65));
    double scale = 0.0;
    const double diff = max_abs_aed_diff(d5, d65, st.natoms(), scale);
    INFO("max |aed(5.0 list) - aed(6.5 list)| = " << diff << ", scale "
                                                 << scale);
    REQUIRE(scale > 0.0);
    CHECK(diff <= 1e-12 * scale);
  }
}

TEST_CASE("dPhi/dtheta walkers refuse a neighbour list shorter than the "
          "descriptor cutoff",
          "[nn][nn_guard][dphi_dtheta_walk]") {
  PeriodicTable::initialize();
  const double rc = 5.0;

  Structure base = bcc_2x2x2();
  StructureDB short_db = one_structure_db(base);
  StructureDB full_db = one_structure_db(base);
  NNFinder nnf;
  nnf.calc(short_db.structures[0], 4.0);
  nnf.calc(full_db.structures[0], rc);
  const std::size_t natoms = base.natoms();
  const std::vector<std::size_t> force_row_start = {0};

  SECTION("two-body walkers") {
    Context c = d2_context(rc);
    D2_BP d2(c);
    Cut_Cos fc(rc);
    d2.set_fcut(&fc, false);
    REQUIRE(d2.get_rcut() == rc);
    const std::size_t n_cols = d2.size();
    std::vector<double> e(n_cols, 0.0), f(3 * natoms * n_cols, 0.0);

    CHECK_REFUSES_SHORT_LIST(tadah::mlip::assemble_dphi_train_energy_rows_d2(
        short_db, d2, "CGRID2B", 1, e.data(), n_cols));
    CHECK_REFUSES_SHORT_LIST(tadah::mlip::assemble_dphi_train_force_rows_d2(
        short_db, d2, "CGRID2B", 1, f.data(), n_cols, force_row_start));
    CHECK_REFUSES_SHORT_LIST(tadah::mlip::assemble_dphi_train_both_rows_d2(
        short_db, d2, "CGRID2B", 1, e.data(), f.data(), n_cols,
        force_row_start));

    CHECK_NOTHROW(tadah::mlip::assemble_dphi_train_energy_rows_d2(
        full_db, d2, "CGRID2B", 1, e.data(), n_cols));
    CHECK_NOTHROW(tadah::mlip::assemble_dphi_train_force_rows_d2(
        full_db, d2, "CGRID2B", 1, f.data(), n_cols, force_row_start));
    CHECK_NOTHROW(tadah::mlip::assemble_dphi_train_both_rows_d2(
        full_db, d2, "CGRID2B", 1, e.data(), f.data(), n_cols,
        force_row_start));
  }

  SECTION("many-body walkers and the rho cache") {
    Context c = dm_context(rc);
    DM_EAD dm(c);
    Cut_Cos fc(rc);
    dm.set_fcut(&fc, false);
    REQUIRE(dm.get_rcut() == rc);
    const std::size_t n_cols = dm.size();
    REQUIRE(n_cols > 0);
    std::vector<double> e(n_cols, 0.0), f(3 * natoms * n_cols, 0.0);

    CHECK_REFUSES_SHORT_LIST(tadah::mlip::assemble_dphi_train_energy_rows_dm(
        short_db, dm, "CGRIDMB", 1, e.data(), n_cols));
    CHECK_REFUSES_SHORT_LIST(tadah::mlip::assemble_dphi_train_force_rows_dm(
        short_db, dm, "CGRIDMB", 1, f.data(), n_cols, force_row_start));
    CHECK_REFUSES_SHORT_LIST(tadah::mlip::assemble_dphi_train_both_rows_dm(
        short_db, dm, "CGRIDMB", 1, e.data(), f.data(), n_cols,
        force_row_start));
    {
      tadah::mlip::DMRhoCache cache;
      CHECK_REFUSES_SHORT_LIST(
          tadah::mlip::build_dm_rho_cache(short_db, dm, cache, n_cols));
    }
    {
      // A cache built from the full list, walked with the short one.
      tadah::mlip::DMRhoCache cache;
      REQUIRE_NOTHROW(
          tadah::mlip::build_dm_rho_cache(full_db, dm, cache, n_cols));
      CHECK_REFUSES_SHORT_LIST(
          tadah::mlip::assemble_dphi_train_both_rows_dm_cached(
              short_db, dm, cache, "CGRIDMB", 1, e.data(), f.data(), n_cols,
              force_row_start));
      CHECK_NOTHROW(tadah::mlip::assemble_dphi_train_both_rows_dm_cached(
          full_db, dm, cache, "CGRIDMB", 1, e.data(), f.data(), n_cols,
          force_row_start));
    }

    CHECK_NOTHROW(tadah::mlip::assemble_dphi_train_energy_rows_dm(
        full_db, dm, "CGRIDMB", 1, e.data(), n_cols));
    CHECK_NOTHROW(tadah::mlip::assemble_dphi_train_force_rows_dm(
        full_db, dm, "CGRIDMB", 1, f.data(), n_cols, force_row_start));
    CHECK_NOTHROW(tadah::mlip::assemble_dphi_train_both_rows_dm(
        full_db, dm, "CGRIDMB", 1, e.data(), f.data(), n_cols,
        force_row_start));
  }
}

// DescriptorsCalc::calc(StructureDB) runs its structures in a parallel loop
// (parallel_for), so it checks every list before the loop: the refusal
// names the first structure whose list is short, whatever the thread order.
TEST_CASE("DescriptorsCalc on a StructureDB refuses a short list before its "
          "loop and names the structure",
          "[nn][nn_guard][descriptors][stdb]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  Context ctx = model_context(5.0, 0.0);
  Trainer tr(ctx, ws);

  NNFinder nnf;
  StructureDB db;
  Structure s1 = bcc_2x2x2();
  s1.label = "full list";
  nnf.calc(s1, 5.0);
  Structure s2 = bcc_2x2x2();
  s2.label = "short list at 4";
  nnf.calc(s2, 4.0);
  Structure s3 = bcc_2x2x2();
  s3.label = "another short list";
  nnf.calc(s3, 3.0);
  db.add(s1);
  db.add(s2);
  db.add(s3);
  REQUIRE(db.size() == 3);

  CHECK_REFUSES_SHORT_LIST(tr.dc.calc(db));
  const Outcome o = run([&]() { (void)tr.dc.calc(db); });
  INFO("what: " << o.what);
  CHECK(o.what.find("'short list at 4'") != std::string::npos);

  // With every list at the model cutoff the database is computed.
  StructureDB full;
  for (std::size_t i = 0; i < db.size(); ++i) {
    Structure s = db(i);
    nnf.calc(s, 5.0);
    full.add(s);
  }
  CHECK_NOTHROW(tr.dc.calc(full));
}
