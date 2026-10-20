// test_nn_replication_invariance.cpp
//
// A periodic structure and any supercell of it describe the same infinite
// crystal, so a model must predict the same energy per atom, the same force
// on every copy of an atom, and the same (intensive) stress for both. The
// supercells here are built with a test-local loop along the cell rows
// (nn_oracle::supercell) and are thick (every height above the cutoff);
// the original cells are thin (a height below the cutoff, so the neighbour
// search needs several images of each atom), triclinic, or hold atoms
// outside the cell. Any periodic image the neighbour search drops or adds
// in the small cell shows up as a difference from its supercell.
//
// The model is linear (BF_Linear) in a two-body (D2_BP) and a many-body
// (DM_EAD, Lmax 1) descriptor with different cutoffs, FORCE and STRESS on,
// and fixed weights (no training, so nothing depends on a fit).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/trainer.h>

#include "nn_oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::force_type;
using tadah::core::Matrix3d;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::StDescriptors;
using tadah::mlip::Structure;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

constexpr double kRcut2b = 5.0;
constexpr double kRcutMb = 4.5;
constexpr double kRcutMax = 5.0;

Context replication_context() {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 73);
  c.add("INIT2B", "true");
  c.add("INITMB", "true");
  c.add("TYPE2B", "D2_BP");
  c.add("TYPE2B", 3);
  c.add("TYPE2B", 3);
  c.add("TYPE2B", "Ta");
  c.add("TYPE2B", "Ta");
  c.add("CGRID2B", "2.6 3.4 4.4");
  c.add("SGRID2B", "1.5 1.0 0.8");
  c.add("TYPEMB", "DM_EAD");
  c.add("TYPEMB", 1);
  c.add("TYPEMB", 2);
  c.add("TYPEMB", 2);
  c.add("TYPEMB", "Ta");
  c.add("TYPEMB", "Ta");
  c.add("CGRIDMB", "2.5 3.5");
  c.add("SGRIDMB", "1.2 0.9");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", kRcut2b);
  c.add("RCUTMB", kRcutMb);
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

// The model with every weight fixed (FIXINDEX/FIXWEIGHT).
struct ReplicationModel {
  MLIPWorkspaceManager ws;
  Context ctx;
  std::unique_ptr<Trainer> tr;

  ReplicationModel() : ctx(replication_context()) {
    std::size_t n_w = 0;
    {
      Context probe_ctx = replication_context();
      Trainer probe(probe_ctx, ws);
      n_w = probe.fb->get_phi_cols(probe.context);
    }
    REQUIRE(n_w > 0);
    ctx.add("FIXINDEX", "1-" + std::to_string(n_w));
    for (std::size_t k = 0; k < n_w; ++k)
      ctx.add("FIXWEIGHT", 2e-5 * std::cos(1.7 * static_cast<double>(k) + 0.3));
    tr = std::make_unique<Trainer>(ctx, ws);
    tr->provision_fixed_weights();
    REQUIRE(tr->context.get<double>("RCUTMAX") == kRcutMax);
  }
};

struct Prediction {
  double energy_per_atom = 0.0;
  std::vector<force_type> forces;
  Matrix3d stress;
};

// Neighbour list at the model's largest cutoff, then energy, forces and
// stress through DescriptorsCalc and the model.
Prediction predict(Trainer &tr, Structure st) {
  NNFinder().calc(st, kRcutMax);
  const StDescriptors d = tr.dc.calc(st);
  Prediction p;
  p.energy_per_atom = tr.model->epredict(d) / static_cast<double>(st.natoms());
  for (std::size_t a = 0; a < st.natoms(); ++a)
    p.forces.push_back(tr.model->fpredict(a, d, st));
  p.stress = tr.model->spredict(d, st);
  return p;
}

// Structure with the given rows and fractional positions (r = f . A).
Structure make_cell(const std::array<std::array<double, 3>, 3> &rows,
                    const std::vector<std::array<double, 3>> &frac,
                    const std::string &label) {
  PeriodicTable::initialize();
  Structure st;
  st.label = label;
  st.cell.load(rows[0][0], rows[0][1], rows[0][2],
               rows[1][0], rows[1][1], rows[1][2],
               rows[2][0], rows[2][1], rows[2][2]);
  const auto ta = PeriodicTable::find_by_symbol("Ta");
  for (const auto &f : frac) {
    double r[3];
    for (int c = 0; c < 3; ++c)
      r[c] = f[0] * rows[0][c] + f[1] * rows[1][c] + f[2] * rows[2][c];
    st.add_atom(Atom(ta, r[0], r[1], r[2]));
  }
  return st;
}

double min_height(const Structure &st) {
  const auto L = nn_oracle::lattice(st);
  return static_cast<double>(std::min({L.h[0], L.h[1], L.h[2]}));
}

double max_abs(const Matrix3d &m) {
  double v = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) v = std::max(v, std::fabs(m(i, j)));
  return v;
}

// Compare the prediction for st with the one for its n0 x n1 x n2 supercell:
// the energy per atom (1e-12 relative), the force on every copy of every
// atom (1e-10 eV/A) and the stress (1e-10, relative to its size when it
// exceeds 1).
void check_against_supercell(Trainer &tr, const Structure &st, int n0, int n1,
                             int n2) {
  const Structure sc = nn_oracle::supercell(st, n0, n1, n2);
  REQUIRE(sc.natoms() == st.natoms() * n0 * n1 * n2);
  INFO(st.label << ": min height " << min_height(st) << " A, supercell "
                << n0 << "x" << n1 << "x" << n2 << " min height "
                << min_height(sc) << " A, cutoff " << kRcutMax);
  // The small cell needs several images; the supercell few.
  REQUIRE(min_height(st) < kRcutMax);
  REQUIRE(min_height(sc) >= kRcutMax);

  const Prediction p = predict(tr, st);
  const Prediction q = predict(tr, sc);

  INFO("E/atom: cell " << p.energy_per_atom << ", supercell "
                       << q.energy_per_atom);
  CHECK(p.energy_per_atom == Approx(q.energy_per_atom).epsilon(1e-12));

  double f_diff = 0.0, f_scale = 0.0;
  std::size_t idx = 0;
  for (int x = 0; x < n0; ++x)
    for (int y = 0; y < n1; ++y)
      for (int z = 0; z < n2; ++z)
        for (std::size_t i = 0; i < st.natoms(); ++i, ++idx)
          for (int k = 0; k < 3; ++k) {
            f_diff = std::max(f_diff,
                              std::fabs(q.forces[idx][k] - p.forces[i][k]));
            f_scale = std::max(f_scale, std::fabs(p.forces[i][k]));
          }
  INFO("forces: max |F(copy) - F(atom)| = " << f_diff << " eV/A, max |F| = "
                                            << f_scale);
  CHECK(f_diff <= 1e-10);

  double s_diff = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      s_diff = std::max(s_diff, std::fabs(p.stress(a, b) - q.stress(a, b)));
  const double s_scale = std::max(max_abs(p.stress), max_abs(q.stress));
  INFO("stress: max diff " << s_diff << ", max |stress| " << s_scale);
  CHECK(s_diff <= 1e-10 * std::max(1.0, s_scale));
}

// The triclinic cell of the second and third sections.
const std::array<std::array<double, 3>, 3> kTriclinicRows = {
    {{3.2, 0.0, 0.0}, {1.3, 3.0, 0.0}, {0.6, 0.9, 3.1}}};
const std::vector<std::array<double, 3>> kTriclinicFrac = {
    {0.10, 0.15, 0.20}, {0.60, 0.55, 0.70}};

}  // namespace

TEST_CASE("Replicated thin and triclinic cells predict the same per-atom "
          "energy forces and stress",
          "[nn][replication][model]") {
  PeriodicTable::initialize();
  ReplicationModel model;
  Trainer &tr = *model.tr;

  SECTION("thin orthogonal cell") {
    // Edges 3.1, 3.4, 2.9 A: every height below the cutoff 5, and atoms
    // placed so that some neighbours (inside both cutoffs) are two images
    // away along a2, where no lattice vector shorter than the cutoff
    // reaches.
    const Structure st = make_cell(
        {{{3.1, 0.0, 0.0}, {0.0, 3.4, 0.0}, {0.0, 0.0, 2.9}}},
        {{0.12, 0.07, 0.21}, {0.33, 0.41, 0.74}}, "thin orthogonal");
    check_against_supercell(tr, st, 2, 2, 2);
  }

  SECTION("thin triclinic cell") {
    const Structure st =
        make_cell(kTriclinicRows, kTriclinicFrac, "thin triclinic");
    check_against_supercell(tr, st, 3, 3, 3);
  }

  SECTION("unwrapped copy of the triclinic cell") {
    // The same crystal with the atoms moved by lattice vectors out of the
    // cell: atom 0 by 2 a0 - a2, atom 1 by -a1 + 3 a2.
    const Structure wrapped =
        make_cell(kTriclinicRows, kTriclinicFrac, "thin triclinic");
    std::vector<std::array<double, 3>> frac = kTriclinicFrac;
    frac[0][0] += 2.0;
    frac[0][2] -= 1.0;
    frac[1][1] -= 1.0;
    frac[1][2] += 3.0;
    const Structure st = make_cell(kTriclinicRows, frac, "unwrapped triclinic");
    check_against_supercell(tr, st, 3, 3, 3);

    // And the same per-atom prediction as the wrapped cell.
    const Prediction p = predict(tr, st);
    const Prediction w = predict(tr, wrapped);
    INFO("E/atom: unwrapped " << p.energy_per_atom << ", wrapped "
                              << w.energy_per_atom);
    CHECK(p.energy_per_atom == Approx(w.energy_per_atom).epsilon(1e-12));
    double f_diff = 0.0;
    for (std::size_t i = 0; i < st.natoms(); ++i)
      for (int k = 0; k < 3; ++k)
        f_diff = std::max(f_diff, std::fabs(p.forces[i][k] - w.forces[i][k]));
    INFO("forces: max |F(unwrapped) - F(wrapped)| = " << f_diff);
    CHECK(f_diff <= 1e-10);
    double s_diff = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        s_diff = std::max(s_diff, std::fabs(p.stress(a, b) - w.stress(a, b)));
    CHECK(s_diff <= 1e-10 * std::max(1.0, max_abs(w.stress)));
  }
}
