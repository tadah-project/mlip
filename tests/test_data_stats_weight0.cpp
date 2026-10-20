// test_data_stats_weight0.cpp
//
// The statistics of the data leave out the energies and forces a structure
// does not have: a structure stored with eweight 0 has no energy of its own,
// one with fweight 0 no forces (the values stored for them are
// placeholders; tadah data convert stores zero forces with fweight 0 for a
// file that gives none). compute_data_stats (tadah analysis dataset_stats, and the
// statistics suggest_bounds and check_bounds read) takes the energies per
// atom of the structures with an energy and the force magnitudes of those
// with forces; the design matrix's standard deviations of the reference
// energies, forces and stresses (logged by a fit) those with that weight.
//
// Data: three bcc Ta cells of 2 atoms, energies per atom -8, -7 and a
// placeholder 100 (eweight 0); force magnitudes 0.3 and 0.4 on the first
// two, a placeholder 90 on the third (fweight 0).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/data_stats_compute.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/design_matrix.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>

#include <cmath>

using namespace tadah::mlip;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::core::stats::IDX_MEAN;

namespace {

Structure cell(double a, double e_per_atom, double f) {
  PeriodicTable::initialize();
  const auto ta = PeriodicTable::find_by_symbol("Ta");
  Structure st;
  st.label = "bcc";
  st.energy = 2 * e_per_atom;
  st.cell.set_zero();
  st.cell(0, 0) = st.cell(1, 1) = st.cell(2, 2) = a;
  st.stress.set_zero();
  st.add_atom(Atom(ta, 0.0, 0.0, 0.0, f, 0.0, 0.0));
  st.add_atom(Atom(ta, 0.5 * a, 0.5 * a, 0.5 * a, -f, 0.0, 0.0));
  return st;
}

StructureDB data() {
  StructureDB db;
  db.add(cell(3.30, -8.0, 0.3));
  db.add(cell(3.32, -7.0, 0.4));
  Structure p = cell(3.31, 100.0, 90.0);
  p.eweight = 0.0;
  p.fweight = 0.0;
  db.add(p);
  db.dbidx = {0, 3};
  return db;
}

}  // namespace

/* Pins: the energy per atom statistics of compute_data_stats are those of
 * -8 and -7 (mean -7.5), the force magnitude statistics those of 0.3 and
 * 0.4 (mean 0.35).
 * Before the fix: the placeholders were included (energy mean 28.33,
 * force mean 30.23). */
TEST_CASE("compute_data_stats leaves out the energies and forces a structure "
          "does not have",
          "[data_stats][weight0]") {
  Context c;
  c.add<double>("RCUT2B", 5.0);
  c.add<bool>("FORCE", false);
  c.add<bool>("STRESS", false);
  c.add<bool>("BIAS", false);
  StructureDB db = data();
  const auto d = compute_data_stats(db, c, false);
  CHECK(d.e_per_atom_percentiles[IDX_MEAN] == Approx(-7.5).epsilon(1e-12));
  CHECK(d.f_norm_percentiles[IDX_MEAN] == Approx(0.35).epsilon(1e-12));
}

/* Pins: the design matrix's standard deviation of the reference energies
 * per atom is that of -8 and -7 (sqrt(0.5), n - 1 in the denominator), and
 * of the forces that of the 12 force components of the first two cells
 * (0.3, 0, 0, -0.3, 0, 0, 0.4, 0, 0, -0.4, 0, 0).
 * Before the fix: the placeholder cell was included. */
TEST_CASE("The design matrix's standard deviations leave out the energies "
          "and forces a structure does not have",
          "[data_stats][weight0]") {
  PeriodicTable::initialize();
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1);
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_Blip 3 3 Ta Ta");
  c.add("CGRID2B", "2.8 3.5 4.2");
  c.add("SGRID2B", "2.0 2.0 2.0");
  c.add("TYPEMB", "DM_Dummy Ta Ta");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 5.0);
  c.add("RCUTMB", 5.0);
  c.add("DIMER", "false 0 false");
  c.add("FORCE", "true");
  c.add("STRESS", "false");
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  tadah::mlip::memory::MLIPWorkspaceManager ws;
  Trainer tr(c, ws);
  StructureDB db = data();
  tr.nnf.calc(db, tr.context);
  DesignMatrix<DM_Function_Base &> dm(*tr.fb, tr.context);
  Normaliser norm(tr.context);
  dm.build(db, norm, tr.dc, false);
  CHECK(dm.e_std_dev == Approx(std::sqrt(0.5)).epsilon(1e-12));
  // 12 components, mean 0, sum of squares 2 (0.09 + 0.16) = 0.5
  CHECK(dm.f_std_dev == Approx(std::sqrt(0.5 / 11)).epsilon(1e-12));
}
