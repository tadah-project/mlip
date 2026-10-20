// test_error_scales.cpp
//
// What a fit minimises: the sum over the observations of the training data
// of priority x (residual / error scale)^2, plus the LAMBDA penalty.
//
//  - the energy of a structure of N atoms is one observation, its residual
//    divided by N (ENORM ATOM) or sqrt(N) (ENORM SQRT), error scale ESIGMA
//    (meV/atom); each force component one, error scale sqrt(a^2 + (b M)^2)
//    in eV/A (FSIGMA a b, M the RMS of the structure's own reference force
//    components); the stress tensor nine entries (each off-diagonal entry
//    twice), error scale SSIGMA a b in GPa;
//  - the priority of an observation is the structure's eweight, fweight or
//    sweight times the WDBFILE value of its dataset;
//  - a row that is zero in every column is not an observation.
//
// The oracles do not compute the rule: they compare fits that the rule
// makes equal (an exact periodic copy of a cell and the cell with a
// priority; every error scale times c and LAMBDA times c^2; labels and
// absolute error scales times c and coefficients times c; a cell and its
// rotated copy), and fits that a setting must leave alone.
//
// Data: Ta cells of 2 to 4 atoms (and periodic copies of up to 54), D2_Blip
// with 3 functions, Cut_Cos at 5 A, INITMB false (DSIZE 3), BIAS false,
// OALGO 1. The labels are arbitrary smooth numbers, so every fit has
// residuals.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/design_matrix.h>
#include <tadah/mlip/design_matrix/functions/basis_functions/dm_bf_linear.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/row_weights.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/dc_selector.h>

#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Catch::Matchers::Contains;
using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

void set_key(Context &c, const std::string &key, const std::string &value) {
  if (c.has_stored(key)) c.remove(key);
  c.add(key, value);
}

Context make_context(bool force, bool stress) {
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
  c.add("FORCE", force ? "true" : "false");
  c.add("STRESS", stress ? "true" : "false");
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  return c;
}

/* A cell with lattice rows a (a 3x3 row-major list), atoms at fractional
 * coordinates frac, and smooth labels set by the seed s. */
Structure make_cell(const std::array<double, 9> &a,
                    const std::vector<std::array<double, 3>> &frac,
                    std::size_t s) {
  Structure st;
  st.label = "cell " + std::to_string(s);
  st.cell.load(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
  for (const auto &f : frac) {
    double r[3];
    for (int k = 0; k < 3; ++k)
      r[k] = f[0] * st.cell(0, k) + f[1] * st.cell(1, k) + f[2] * st.cell(2, k);
    st.add_atom(Atom(Element("Ta"), r[0], r[1], r[2], 0, 0, 0));
  }
  const double ds = static_cast<double>(s);
  st.energy = -8.1 * static_cast<double>(st.natoms()) +
              0.37 * std::sin(1.3 * ds + 0.2);
  for (std::size_t i = 0; i < st.natoms(); ++i)
    for (int k = 0; k < 3; ++k)
      st.atoms[i].force[k] =
          0.25 * std::sin(0.7 * static_cast<double>(i) + 1.1 * k + 0.3 * ds);
  for (int x = 0; x < 3; ++x)
    for (int y = x; y < 3; ++y) {
      st.stress(x, y) = 0.01 * std::cos(0.9 * x + 1.7 * y + 0.4 * ds);
      st.stress(y, x) = st.stress(x, y);
    }
  return st;
}

/* A 2-atom bcc cell, rattled (its forces and its off-diagonal stress are
 * not zero). */
Structure cell_2() {
  return make_cell({3.30, 0, 0, 0.04, 3.31, 0, -0.03, 0.05, 3.29},
                   {{0.01, 0.02, -0.01}, {0.52, 0.47, 0.53}}, 1);
}

/* The other cells of the training data. */
Structure cell_3() {
  return make_cell({3.45, 0, 0, 0.12, 3.40, 0, -0.08, 0.10, 3.47},
                   {{0.00, 0.01, 0.02}, {0.48, 0.52, 0.05}, {0.27, 0.55, 0.51}},
                   2);
}
Structure cell_4() {
  return make_cell({4.10, 0, 0, 0.0, 4.05, 0, 0.15, 0.0, 4.20},
                   {{0.0, 0.0, 0.0},
                    {0.51, 0.49, 0.02},
                    {0.48, 0.03, 0.50},
                    {0.02, 0.52, 0.47}},
                   3);
}

/* The exact periodic copy of @p st, n0 x n1 x n2 cells: the same atoms in
 * each, so the same per-atom descriptors; the energy times the number of
 * copies, the same forces on each copy of an atom and the same stress. */
Structure periodic_copy(const Structure &st, int n0, int n1, int n2) {
  Structure out;
  out.label = st.label + " copy";
  out.cell.load(n0 * st.cell(0, 0), n0 * st.cell(0, 1), n0 * st.cell(0, 2),
                n1 * st.cell(1, 0), n1 * st.cell(1, 1), n1 * st.cell(1, 2),
                n2 * st.cell(2, 0), n2 * st.cell(2, 1), n2 * st.cell(2, 2));
  for (int i = 0; i < n0; ++i)
    for (int j = 0; j < n1; ++j)
      for (int l = 0; l < n2; ++l)
        for (const Atom &a : st.atoms) {
          double r[3];
          for (int k = 0; k < 3; ++k)
            r[k] = a.position[k] + i * st.cell(0, k) + j * st.cell(1, k) +
                   l * st.cell(2, k);
          out.add_atom(Atom(Element("Ta"), r[0], r[1], r[2], a.force[0],
                            a.force[1], a.force[2]));
        }
  out.energy = st.energy * n0 * n1 * n2;
  out.stress = st.stress;
  out.eweight = st.eweight;
  out.fweight = st.fweight;
  out.sweight = st.sweight;
  return out;
}

/* @p st rotated by the rotation matrix R: its lattice rows, positions and
 * forces R v, its stress R S R^T. */
Structure rotated(const Structure &st, const std::array<double, 9> &R) {
  auto rot = [&](const double v[3], double out[3]) {
    for (int i = 0; i < 3; ++i)
      out[i] = R[3 * i] * v[0] + R[3 * i + 1] * v[1] + R[3 * i + 2] * v[2];
  };
  Structure out;
  out.label = st.label + " rotated";
  double rows[3][3];
  for (int i = 0; i < 3; ++i) {
    const double v[3] = {st.cell(i, 0), st.cell(i, 1), st.cell(i, 2)};
    rot(v, rows[i]);
  }
  out.cell.load(rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1],
                rows[1][2], rows[2][0], rows[2][1], rows[2][2]);
  for (const Atom &a : st.atoms) {
    const double p[3] = {a.position[0], a.position[1], a.position[2]};
    const double f[3] = {a.force[0], a.force[1], a.force[2]};
    double pr[3], fr[3];
    rot(p, pr);
    rot(f, fr);
    out.add_atom(Atom(Element("Ta"), pr[0], pr[1], pr[2], fr[0], fr[1], fr[2]));
  }
  for (int x = 0; x < 3; ++x)
    for (int y = 0; y < 3; ++y) {
      double v = 0;
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          v += R[3 * x + k] * st.stress(k, l) * R[3 * y + l];
      out.stress(x, y) = v;
    }
  out.energy = st.energy;
  return out;
}

/* The rotation by @p angle about the unit axis (1, 2, 3)/sqrt(14). */
std::array<double, 9> rotation(double angle) {
  const double n = std::sqrt(14.0);
  const double u[3] = {1 / n, 2 / n, 3 / n};
  const double c = std::cos(angle), s = std::sin(angle), t = 1 - c;
  return {t * u[0] * u[0] + c,        t * u[0] * u[1] - s * u[2],
          t * u[0] * u[2] + s * u[1], t * u[0] * u[1] + s * u[2],
          t * u[1] * u[1] + c,        t * u[1] * u[2] - s * u[0],
          t * u[0] * u[2] - s * u[1], t * u[1] * u[2] + s * u[0],
          t * u[2] * u[2] + c};
}

StructureDB db_of(const std::vector<Structure> &cells) {
  StructureDB db;
  for (const Structure &st : cells) db.add(st);
  db.dbidx = {0, db.size()};
  return db;
}

/* The WEIGHTS a fit of @p db with the configuration @p c writes. */
std::vector<double> fitted(const Context &c, const StructureDB &db) {
  Context cc = c;
  MLIPWorkspaceManager ws;
  Trainer tr(cc, ws);
  StructureDB d = db;
  tr.train(d);
  const Context pot = tr.get_param_file();
  std::vector<double> w;
  for (std::size_t i = 0; i < pot.size("WEIGHTS"); ++i)
    w.push_back(pot.get<double>("WEIGHTS", i));
  return w;
}

/* What a fit of @p db with @p c throws, "" when it does not throw. */
std::string fit_refusal(const Context &c, const StructureDB &db) {
  try {
    fitted(c, db);
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

void check_same(const std::vector<double> &a, const std::vector<double> &b,
                double rel) {
  REQUIRE(a.size() == b.size());
  REQUIRE_FALSE(a.empty());
  double scale = 0;
  for (double v : b) scale = std::max(scale, std::fabs(v));
  for (std::size_t i = 0; i < a.size(); ++i) {
    INFO("coefficient " << i << ": " << a[i] << " against " << b[i]);
    CHECK(std::fabs(a[i] - b[i]) <= rel * scale);
  }
}

}  // namespace

/* Pins (ENORM ATOM, the default): replacing a 2-atom cell by its exact
 * periodic copy of k cells (2 x 1 x 1, k = 2, and 3 x 3 x 3, k = 27: 54
 * atoms) leaves the cost of its energy and of its stress as they are and
 * multiplies that of its forces by k (k copies of each component): the fit
 * equals the fit of the cell with fweight k. With energies only (FORCE and
 * STRESS false) the copy fits as the cell itself. 1e-9 relative.
 * Before the fix: the energy residual was the total energy's, so the
 * copy's energy cost k^2 times the cell's and the fits differed. */
TEST_CASE("ENORM ATOM: an exact periodic copy of a cell costs what the cell "
          "costs, its forces k times",
          "[error_scales][copy]") {
  PeriodicTable::initialize();
  for (const std::array<int, 3> n : {std::array<int, 3>{2, 1, 1},
                                     std::array<int, 3>{3, 3, 3}}) {
    const int k = n[0] * n[1] * n[2];
    DYNAMIC_SECTION("k = " << k) {
      const Structure a = cell_2();
      const Structure copy = periodic_copy(a, n[0], n[1], n[2]);
      REQUIRE(copy.natoms() == 2u * static_cast<unsigned>(k));
      {
        const Context c = make_context(false, false);
        check_same(fitted(c, db_of({copy, cell_3(), cell_4()})),
                   fitted(c, db_of({a, cell_3(), cell_4()})), 1e-9);
      }
      Context c = make_context(true, true);
      Structure prioritised = a;
      prioritised.fweight = k;
      check_same(fitted(c, db_of({copy, cell_3(), cell_4()})),
                 fitted(c, db_of({prioritised, cell_3(), cell_4()})), 1e-9);
    }
  }
}

/* Pins (ENORM SQRT): the copy of k cells costs k times the cell's energy
 * and forces and the cell's stress: the fit equals the fit of the cell with
 * eweight k and fweight k. 1e-9 relative.
 * Before the fix: ENORM was not a key (Context::add threw "Key not
 * allowed: ENORM"). */
TEST_CASE("ENORM SQRT: an exact periodic copy costs k times the cell's "
          "energy and forces",
          "[error_scales][copy]") {
  PeriodicTable::initialize();
  for (const std::array<int, 3> n : {std::array<int, 3>{2, 1, 1},
                                     std::array<int, 3>{3, 3, 3}}) {
    const int k = n[0] * n[1] * n[2];
    DYNAMIC_SECTION("k = " << k) {
      Context c = make_context(true, true);
      set_key(c, "ENORM", "SQRT");
      const Structure a = cell_2();
      Structure prioritised = a;
      prioritised.eweight = k;
      prioritised.fweight = k;
      check_same(
          fitted(c, db_of({periodic_copy(a, n[0], n[1], n[2]), cell_3(),
                           cell_4()})),
          fitted(c, db_of({prioritised, cell_3(), cell_4()})), 1e-9);
    }
  }
}

/* Pins: only the ratios of the error scales and LAMBDA shape the fit:
 * every error scale times c fits as LAMBDA times c^2 (the data term of the
 * sum is divided by c^2). ESIGMA 600, FSIGMA 6 3, SSIGMA 210 15 with LAMBDA
 * 0.1 fit as ESIGMA 200, FSIGMA 2 1, SSIGMA 70 5 with LAMBDA 0.9 (c = 3: a
 * and b both times c, so each error scale sqrt(a^2 + (b M)^2) is times c).
 * 1e-9 relative. The scales are large enough for LAMBDA to tell: LAMBDA
 * 0.1 and 0.9 at the same scales give other coefficients (by more than
 * 1e-6 relative).
 * Before the fix: ESIGMA, FSIGMA and SSIGMA were not keys. */
TEST_CASE("Every error scale times c fits as LAMBDA times c^2",
          "[error_scales]") {
  PeriodicTable::initialize();
  const StructureDB db = db_of({cell_2(), cell_3(), cell_4()});
  Context c1 = make_context(true, true);
  set_key(c1, "ESIGMA", "200");
  set_key(c1, "FSIGMA", "2 1");
  set_key(c1, "SSIGMA", "70 5");
  set_key(c1, "LAMBDA", "0.9");
  Context c3 = make_context(true, true);
  set_key(c3, "ESIGMA", "600");
  set_key(c3, "FSIGMA", "6 3");
  set_key(c3, "SSIGMA", "210 15");
  set_key(c3, "LAMBDA", "0.1");
  check_same(fitted(c3, db), fitted(c1, db), 1e-9);
  /* LAMBDA alone changes the fit (the comparison is not empty). */
  Context other = c1;
  set_key(other, "LAMBDA", "0.1");
  const std::vector<double> w1 = fitted(c1, db), wo = fitted(other, db);
  REQUIRE(w1.size() == wo.size());
  double diff = 0, scale = 0;
  for (std::size_t i = 0; i < w1.size(); ++i) {
    diff = std::max(diff, std::fabs(w1[i] - wo[i]));
    scale = std::max(scale, std::fabs(w1[i]));
  }
  CHECK(diff > 1e-6 * scale);
}

/* Pins: the labels (energies, forces, stresses) times c = 2.5 with the
 * absolute parts a of ESIGMA, FSIGMA and SSIGMA times c (the relative parts
 * b unchanged: M is times c with the labels) give coefficients times c, at
 * LAMBDA 0 (a penalty on the coefficients, which are times c, would need
 * LAMBDA / c^2). 1e-9 relative.
 * Before the fix: ESIGMA, FSIGMA and SSIGMA were not keys. */
TEST_CASE("Labels and absolute error scales times c give coefficients times c",
          "[error_scales]") {
  PeriodicTable::initialize();
  const double k = 2.5;
  std::vector<Structure> cells = {cell_2(), cell_3(), cell_4()};
  Context c = make_context(true, true);
  set_key(c, "ESIGMA", "2");
  set_key(c, "FSIGMA", "0.2 0.1");
  set_key(c, "SSIGMA", "0.7 0.05");
  const std::vector<double> w = fitted(c, db_of(cells));
  for (Structure &st : cells) {
    st.energy *= k;
    for (Atom &a : st.atoms)
      for (int i = 0; i < 3; ++i) a.force[i] *= k;
    for (int x = 0; x < 3; ++x)
      for (int y = 0; y < 3; ++y) st.stress(x, y) *= k;
  }
  set_key(c, "ESIGMA", "5");
  set_key(c, "FSIGMA", "0.5 0.1");
  set_key(c, "SSIGMA", "1.75 0.05");
  std::vector<double> wk = fitted(c, db_of(cells));
  for (double &v : wk) v /= k;
  check_same(wk, w, 1e-9);
}

/* Pins: the stress of a structure is observed by the nine entries of its
 * tensor (each off-diagonal entry twice), whose squared residual, the
 * Frobenius norm, does not depend on the orientation of the cell: the
 * fit of a sheared 3-atom cell rotated by 0.7 rad about (1, 2, 3) equals
 * the fit of the cell (STRESS and FORCE true; descriptors and energies are
 * invariant, forces rotate). 1e-9 relative.
 * Before the fix: the six entries xx, xy, xz, yy, yz and zz were each
 * observed once, which a rotation changes (the fits differed). */
TEST_CASE("The stress observations are those of the tensor: a rotated cell "
          "fits as the cell",
          "[error_scales][stress]") {
  PeriodicTable::initialize();
  Context c = make_context(true, true);
  set_key(c, "SSIGMA", "0.05");  // stresses matter in the fit
  const Structure a = cell_3();
  const Structure r = rotated(a, rotation(0.7));
  check_same(fitted(c, db_of({r, cell_2(), cell_4()})),
             fitted(c, db_of({a, cell_2(), cell_4()})), 1e-9);
}

/* Pins: the relative part of FSIGMA uses the structure's own reference
 * forces only: with FSIGMA 0.1 0.5 the weighted design-matrix rows of a
 * cell are the same, bit for bit, whatever the forces of another cell
 * (times 3 here), and those of the other cell's force rows change. An
 * absolute part a = 0 is refused (a cell with zero forces would have the
 * error scale 0), and so is a negative relative part, each naming the key.
 * Before the fix: FSIGMA was not a key. */
TEST_CASE("FSIGMA a b: the relative part reads the structure's own forces; "
          "a > 0 and b >= 0",
          "[error_scales][relative]") {
  PeriodicTable::initialize();
  Context c = make_context(true, false);
  set_key(c, "FSIGMA", "0.1 0.5");
  auto weighted_rows = [&](StructureDB db) {
    tadah::models::DC_Selector dcs(c);
    tadah::mlip::DescriptorsCalc<> dc(c, *dcs.d2b, *dcs.dmb, *dcs.c2b,
                                      *dcs.cmb);
    tadah::mlip::NNFinder nnf;
    nnf.calc(db, c);
    tadah::mlip::DM_BF_Linear bf(c);
    tadah::mlip::DesignMatrix<tadah::mlip::DM_BF_Linear> dm(bf, c);
    tadah::mlip::Normaliser norm(c);
    dm.build(db, norm, dc, false);
    std::vector<std::vector<double>> rows;
    const auto &P = dm.getPhi();
    for (std::size_t i = 0; i < P.rows(); ++i) {
      std::vector<double> r;
      for (std::size_t j = 0; j < P.cols(); ++j) r.push_back(P(i, j));
      rows.push_back(r);
    }
    return rows;
  };
  const Structure a = cell_3(), b = cell_4();
  Structure b3 = b;
  for (Atom &at : b3.atoms)
    for (int k = 0; k < 3; ++k) at.force[k] *= 3.0;
  const auto r1 = weighted_rows(db_of({a, b}));
  const auto r2 = weighted_rows(db_of({a, b3}));
  REQUIRE(r1.size() == r2.size());
  const std::size_t rows_a = 1 + 3 * a.natoms();
  for (std::size_t i = 0; i < rows_a; ++i) {
    INFO("row " << i << " of the first cell");
    CHECK(r1[i] == r2[i]);
  }
  bool changed = false;
  for (std::size_t i = rows_a + 1; i < rows_a + 1 + 3 * b.natoms(); ++i)
    changed = changed || r1[i] != r2[i];
  CHECK(changed);

  const StructureDB db = db_of({a, b});
  set_key(c, "FSIGMA", "0 0.5");
  CHECK_THAT(fit_refusal(c, db), Contains("FSIGMA 0 0.5") && Contains("> 0"));
  set_key(c, "FSIGMA", "0.1 -0.5");
  CHECK_THAT(fit_refusal(c, db),
             Contains("FSIGMA 0.1 -0.5") && Contains(">= 0"));
  set_key(c, "SSIGMA", "-1");
  set_key(c, "FSIGMA", "0.1");
  set_key(c, "STRESS", "true");
  CHECK_THAT(fit_refusal(c, db), Contains("SSIGMA -1") && Contains("> 0"));
  set_key(c, "SSIGMA", "0.5");
  set_key(c, "ESIGMA", "0");
  CHECK_THAT(fit_refusal(c, db), Contains("ESIGMA 0") && Contains("> 0"));
}

/* Pins: a fit refuses an error scale other than its default, compared by
 * value, for a quantity that is off: FSIGMA 0.2 with FORCE false, SSIGMA 1
 * with STRESS false, ESIGMA 2 and ENORM SQRT with ENERGY false, each naming
 * the key and the switch; it accepts the defaults in both forms (FSIGMA
 * 0.1 and FSIGMA 0.1 0, as every potential has them), and FSIGMA 0.2 with
 * FORCE true; a fit with ENERGY, FORCE and STRESS all false, which observes
 * nothing, is refused.
 * Before the fix: none of these keys existed. */
TEST_CASE("A fit refuses an error scale for a quantity that is off",
          "[error_scales][switches]") {
  PeriodicTable::initialize();
  const StructureDB db = db_of({cell_2(), cell_3(), cell_4()});
  Context c = make_context(false, false);
  set_key(c, "FSIGMA", "0.2");
  CHECK_THAT(fit_refusal(c, db),
             Contains("FSIGMA 0.2") && Contains("FORCE false"));
  set_key(c, "FSIGMA", "0.1");
  CHECK(fit_refusal(c, db).empty());
  set_key(c, "FSIGMA", "0.1 0");
  CHECK(fit_refusal(c, db).empty());
  set_key(c, "SSIGMA", "1");
  CHECK_THAT(fit_refusal(c, db),
             Contains("SSIGMA 1") && Contains("STRESS false"));
  set_key(c, "SSIGMA", "0.5 0");
  CHECK(fit_refusal(c, db).empty());

  Context f = make_context(true, false);
  set_key(f, "FSIGMA", "0.2");
  CHECK(fit_refusal(f, db).empty());
  set_key(f, "ENERGY", "false");
  set_key(f, "ESIGMA", "2");
  CHECK_THAT(fit_refusal(f, db),
             Contains("ESIGMA 2") && Contains("ENERGY false"));
  set_key(f, "ESIGMA", "1");
  set_key(f, "ENORM", "SQRT");
  CHECK_THAT(fit_refusal(f, db),
             Contains("ENORM SQRT") && Contains("ENERGY false"));
  set_key(f, "ENORM", "ATOM");
  CHECK(fit_refusal(f, db).empty());

  Context none = make_context(false, false);
  set_key(none, "ENERGY", "false");
  CHECK_THAT(fit_refusal(none, db),
             Contains("ENERGY") && Contains("FORCE") && Contains("STRESS"));
}

/* Pins: ENERGY false fits as every structure's eweight 0 (its energies are
 * no observations: the fits agree to 1e-12 relative), the design matrix
 * counts no energy observation (DesignMatrix::n_observations(): the force
 * rows only; count_observations of its rows without the factors: no
 * energy, every force component), and the potential records ENERGY false.
 * Before the fix: ENERGY was not a key. */
TEST_CASE("ENERGY false: no energy is an observation", "[error_scales]") {
  PeriodicTable::initialize();
  std::vector<Structure> cells = {cell_2(), cell_3(), cell_4()};
  Context c = make_context(true, false);
  set_key(c, "ENERGY", "false");
  const std::vector<double> w = fitted(c, db_of(cells));
  Context with_e = make_context(true, false);
  for (Structure &st : cells) st.eweight = 0.0;
  check_same(w, fitted(with_e, db_of(cells)), 1e-12);

  StructureDB db = db_of({cell_2(), cell_3(), cell_4()});
  tadah::models::DC_Selector dcs(c);
  tadah::mlip::DescriptorsCalc<> dc(c, *dcs.d2b, *dcs.dmb, *dcs.c2b, *dcs.cmb);
  tadah::mlip::NNFinder nnf;
  nnf.calc(db, c);
  tadah::mlip::DM_BF_Linear bf(c);
  tadah::mlip::DesignMatrix<tadah::mlip::DM_BF_Linear> dm(bf, c);
  tadah::mlip::Normaliser norm(c);
  dm.build(db, norm, dc, true);
  CHECK(dm.n_observations() == 3 * (2 + 3 + 4));
  const tadah::mlip::ObservationCount n = tadah::mlip::count_observations(
      dm.ws->Phi_cpy, 3 + 3 * (2 + 3 + 4), dm.ws->wfactors, dm.getTlabels());
  CHECK(n.total == dm.n_observations());
  CHECK(n.by_type[0] == 0);
  CHECK(n.by_type[1] == 3 * (2 + 3 + 4));
  CHECK(n.by_type[2] == 0);

  Context cc = c;
  MLIPWorkspaceManager ws;
  Trainer tr(cc, ws);
  StructureDB d = db_of({cell_2(), cell_3(), cell_4()});
  tr.train(d);
  const Context pot = tr.get_param_file();
  CHECK(pot.get<std::string>("ENERGY") == "false");
}

/* Pins: a row of the design matrix that is zero in every column carries no
 * information and is not an observation: a Ta dimer along x in a 20 A box
 * has force rows along y and z that the descriptors make zero, so with
 * FORCE true its observations are its energy and the two x components (3,
 * not 7).
 * Before the fix: every row of a non-zero weight was counted (7). */
TEST_CASE("A design-matrix row that is zero in every column is not an "
          "observation",
          "[error_scales][observations]") {
  PeriodicTable::initialize();
  Structure dimer;
  dimer.cell.load(20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0);
  dimer.add_atom(Atom(Element("Ta"), 5, 5, 5, 0.3, 0, 0));
  dimer.add_atom(Atom(Element("Ta"), 7.9, 5, 5, -0.3, 0, 0));
  dimer.energy = -16.5;
  StructureDB db = db_of({dimer});
  Context c = make_context(true, false);
  tadah::models::DC_Selector dcs(c);
  tadah::mlip::DescriptorsCalc<> dc(c, *dcs.d2b, *dcs.dmb, *dcs.c2b, *dcs.cmb);
  tadah::mlip::NNFinder nnf;
  nnf.calc(db, c);
  tadah::mlip::DM_BF_Linear bf(c);
  tadah::mlip::DesignMatrix<tadah::mlip::DM_BF_Linear> dm(bf, c);
  tadah::mlip::Normaliser norm(c);
  dm.build(db, norm, dc, false);
  CHECK(dm.n_observations() == 3);
}

namespace {

/* A perfect 2-atom bcc cell of lattice constant @p a, both atoms moved by
 * @p t (a rigid translation, A): its forces are zero by symmetry (and in
 * its labels), its energy a smooth function of a. */
Structure bcc_cell(double a, const std::array<double, 3> &t) {
  Structure st;
  st.label = "bcc " + std::to_string(a);
  st.cell.load(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
  st.add_atom(Atom(Element("Ta"), t[0], t[1], t[2], 0, 0, 0));
  st.add_atom(Atom(Element("Ta"), 0.5 * a + t[0], 0.5 * a + t[1],
                   0.5 * a + t[2], 0, 0, 0));
  st.energy = 2.0 * (-11.8 + 4.0 * (a - 3.3) * (a - 3.3)) +
              0.02 * std::sin(7.0 * a);
  return st;
}

/* 46 bcc cells, a = 2.60 to 3.95 A, moved by @p t. */
StructureDB bcc_set(const std::array<double, 3> &t) {
  std::vector<Structure> cells;
  for (int k = 0; k < 46; ++k) cells.push_back(bcc_cell(2.60 + 0.03 * k, t));
  return db_of(cells);
}

const std::array<double, 3> kAtOrigin{{0.0, 0.0, 0.0}};
const std::array<double, 3> kMoved{{0.1234, 0.5678, 0.9012}};

/* The observations of a DesignMatrix build of @p db in @p c, from the
 * descriptors (precomputed false) or from precomputed descriptors
 * (precomputed true), and the data rows of its design matrix with a value
 * other than exactly 0. */
std::pair<std::size_t, std::size_t> observations_of(const Context &c,
                                                    const StructureDB &db,
                                                    bool precomputed) {
  Context cc = c;
  StructureDB d = db;
  tadah::models::DC_Selector dcs(cc);
  tadah::mlip::DescriptorsCalc<> dc(cc, *dcs.d2b, *dcs.dmb, *dcs.c2b, *dcs.cmb);
  tadah::mlip::NNFinder nnf;
  nnf.calc(d, cc);
  tadah::mlip::DM_BF_Linear bf(cc);
  tadah::mlip::DesignMatrix<tadah::mlip::DM_BF_Linear> dm(bf, cc);
  tadah::mlip::Normaliser norm(cc);
  if (precomputed) {
    tadah::mlip::StDescriptorsDB st = dc.calc(d);
    dm.build(st, d, false);
  } else {
    dm.build(d, norm, dc, false);
  }
  const tadah::core::phi_type &Phi = dm.getPhi();
  const std::size_t data_rows = Phi.rows() - Phi.cols();
  std::size_t exact = 0;
  for (std::size_t i = 0; i < data_rows; ++i) {
    bool nonzero = false;
    for (std::size_t j = 0; j < Phi.cols() && !nonzero; ++j)
      nonzero = Phi(i, j) != 0.0;
    exact += nonzero ? 1 : 0;
  }
  return {dm.n_observations(), exact};
}

/* The potential of a fit of @p db in @p c: of tadah train (Trainer::train)
 * or of the C++ API's train(StDescriptorsDB, StructureDB). */
Context potential_of(Context c, StructureDB db, bool api) {
  MLIPWorkspaceManager ws;
  Trainer tr(c, ws);
  if (api) {
    tr.nnf.calc(db, tr.context);
    tadah::mlip::StDescriptorsDB st = tr.dc.calc(db);
    tr.model->train(st, db);
  } else {
    tr.train(db);
  }
  return tr.get_param_file();
}

double value_of(const Context &c, const std::string &key, std::size_t i) {
  return c.get<double>(key, i);
}

}  // namespace

/* Pins: a force that is zero by symmetry is not an observation, however
 * its design row rounds. The force rows of a perfect 2-atom bcc cell are
 * sums over neighbours that cancel; their rounding leaves values of about
 * 1e-16 against energy rows of order 1 (the data rows other than exactly
 * 0 outnumber the energies, which the CHECK on `exact` shows: a test of
 * exact zeros counts these rows, more of them after a rigid translation).
 * 46 such cells with FORCE true have 46 observations, the energies, in
 * both builds of the design matrix, at the origin and moved by (0.1234,
 * 0.5678, 0.9012) A.
 * Before the fix: a row counted unless every value was exactly 0, so the
 * count was above 46 and changed under the translation. */
TEST_CASE("A force row that is zero by symmetry is not an observation, "
          "however its sums round",
          "[error_scales][observations]") {
  PeriodicTable::initialize();
  const Context c = make_context(true, false);
  for (const std::array<double, 3> &t : {kAtOrigin, kMoved}) {
    for (bool precomputed : {false, true}) {
      DYNAMIC_SECTION("translation " << t[0] << " " << t[1] << " " << t[2]
                                     << (precomputed ? ", precomputed"
                                                     : "")) {
        const auto [n_obs, exact] = observations_of(c, bcc_set(t), precomputed);
        CHECK(exact > 46);
        CHECK(n_obs == 46);
      }
    }
  }
}

/* Pins: the forces of the bcc cells above, zero by symmetry, carry no
 * information, so a fit with FORCE true is the fit with FORCE false, and a
 * rigid translation of the atoms changes nothing (1e-9 relative):
 *  - OALGO 3, LAMBDA 0: WEIGHTS;
 *  - OALGO 3, LAMBDA -1 (the evidence approximation, whose noise precision
 *    counts the observations, the 46 energies): ALPHA, BETA and WEIGHTS;
 * through tadah train (Trainer::train) and the C++ API's train of
 * precomputed descriptors.
 * Before the fix: the count of observations took most of the force rows,
 * by their rounding, and the translation changed BETA and the WEIGHTS of
 * LAMBDA -1 (and the noise variance of the fit's energies, which the
 * potential held then). */
TEST_CASE("Forces zero by symmetry change neither the fit nor the "
          "evidence, nor does a rigid translation",
          "[error_scales][observations]") {
  PeriodicTable::initialize();
  for (const char *lambda : {"0", "-1"}) {
    for (bool api : {false, true}) {
      DYNAMIC_SECTION("LAMBDA " << lambda << (api ? ", C++ API" : "")) {
        Context energies = make_context(false, false);
        set_key(energies, "OALGO", "3");
        set_key(energies, "LAMBDA", lambda);
        Context forces = energies;
        set_key(forces, "FORCE", "true");
        const Context ref = potential_of(energies, bcc_set(kAtOrigin), api);
        std::vector<double> w_ref;
        for (std::size_t i = 0; i < ref.size("WEIGHTS"); ++i)
          w_ref.push_back(value_of(ref, "WEIGHTS", i));
        for (const std::array<double, 3> &t : {kAtOrigin, kMoved}) {
          INFO("translation " << t[0] << " " << t[1] << " " << t[2]);
          const Context pot = potential_of(forces, bcc_set(t), api);
          std::vector<double> w;
          for (std::size_t i = 0; i < pot.size("WEIGHTS"); ++i)
            w.push_back(value_of(pot, "WEIGHTS", i));
          check_same(w, w_ref, 1e-9);
          if (std::string(lambda) == "-1") {
            for (const char *key : {"ALPHA", "BETA"}) {
              INFO(key);
              const double r = value_of(ref, key, 0);
              CHECK(std::fabs(value_of(pot, key, 0) - r) <=
                    1e-9 * std::fabs(r));
            }
          }
        }
      }
    }
  }
}

/* Pins: the priority of a structure's observations is its weight in the
 * data file times the WDBFILE value of its dataset, applied in the fit (the
 * C++ API too, which applies no load transforms): two datasets with WDBFILE
 * 1 0.5 fit as the second dataset's eweight, fweight and sweight halved
 * without WDBFILE (1e-12 relative); WDBFILE 1 0 fits as the first dataset
 * alone. WDBFILE leaves the weights of the data as they are
 * (apply_load_transforms does not change them: WDBFILE 0 takes a dataset
 * out of the fit only, not out of the metrics, the filters or ESHIFT); a
 * WDBFILE whose count is not the number of datasets, or that has a
 * negative value, is refused by the fit.
 * Before the fix: the fit of the API ignored WDBFILE (the coefficients of
 * WDBFILE 1 0.5 were those without it), and apply_load_transforms
 * multiplied the weights of the data by it (WDBFILE 1 0 left eweight 0). */
TEST_CASE("WDBFILE: a priority per dataset, applied in the fit",
          "[error_scales][priorities]") {
  PeriodicTable::initialize();
  auto two_sets = [](double scale2) {
    StructureDB db;
    db.add(cell_2());
    db.add(cell_3());
    Structure b = cell_4(), d = cell_3();
    d.label = "cell 5";
    d.energy += 0.11;
    for (Structure *st : {&b, &d}) {
      st->eweight *= scale2;
      st->fweight *= scale2;
      st->sweight *= scale2;
      db.add(*st);
    }
    db.dbidx = {0, 2, 4};
    return db;
  };
  Context c = make_context(true, true);
  set_key(c, "LAMBDA", "1e-4");
  Context half = c;
  set_key(half, "WDBFILE", "1 0.5");
  check_same(fitted(half, two_sets(1.0)), fitted(c, two_sets(0.5)), 1e-12);

  Context out = c;
  set_key(out, "WDBFILE", "1 0");
  StructureDB first;
  first.add(cell_2());
  first.add(cell_3());
  first.dbidx = {0, 2};
  check_same(fitted(out, two_sets(1.0)), fitted(c, first), 1e-12);

  StructureDB loaded = two_sets(1.0);
  tadah::mlip::apply_load_transforms(loaded, out);
  for (std::size_t s = 0; s < loaded.size(); ++s) {
    INFO("structure " << s);
    CHECK(loaded(s).eweight == 1.0);
    CHECK(loaded(s).fweight == 1.0);
    CHECK(loaded(s).sweight == 1.0);
  }

  Context three = c;
  set_key(three, "WDBFILE", "1 1 1");
  CHECK_THAT(fit_refusal(three, two_sets(1.0)),
             Contains("WDBFILE") && Contains("3") && Contains("2 datasets"));
  Context negative = c;
  set_key(negative, "WDBFILE", "1 -0.5");
  CHECK_THAT(fit_refusal(negative, two_sets(1.0)),
             Contains("WDBFILE") && Contains("-0.5") && Contains("dataset 2"));
}

/* Pins: a trained potential has a key line of each error scale, of ENORM
 * and of ENERGY, the defaults included (ESIGMA 1, FSIGMA 0.1 0, SSIGMA 0.5
 * 0, ENORM ATOM, ENERGY true), a given value as given, and no line of a
 * retired key.
 * Before the fix: the potential had EWEIGHT 1.0, FWEIGHT 1.0 and SWEIGHT
 * 1.0 lines and none of the error scales. */
TEST_CASE("A trained potential records its error scales, ENORM and ENERGY",
          "[error_scales][potential]") {
  PeriodicTable::initialize();
  auto pot_of = [](Context c) {
    MLIPWorkspaceManager ws;
    Trainer tr(c, ws);
    StructureDB d = db_of({cell_2(), cell_3(), cell_4()});
    tr.train(d);
    return tr.get_param_file();
  };
  const Context pot = pot_of(make_context(false, false));
  CHECK(pot("ESIGMA") == std::vector<std::string>{"1"});
  CHECK(pot("FSIGMA") == (std::vector<std::string>{"0.1", "0"}));
  CHECK(pot("SSIGMA") == (std::vector<std::string>{"0.5", "0"}));
  CHECK(pot("ENORM") == std::vector<std::string>{"ATOM"});
  CHECK(pot("ENERGY") == std::vector<std::string>{"true"});
  for (const char *key : {"EWEIGHT", "FWEIGHT", "SWEIGHT", "WDBFILE_AUTO",
                          "EWEIGHT_TEMP"}) {
    INFO(key);
    CHECK_FALSE(pot.has_stored(key));
  }
  std::ostringstream text;
  pot.write_potential(text, 17);
  CHECK(text.str().find("EWEIGHT") == std::string::npos);

  Context given = make_context(true, false);
  set_key(given, "FSIGMA", "0.2 0.05");
  set_key(given, "ESIGMA", "0.5");
  set_key(given, "ENORM", "SQRT");
  const Context p2 = pot_of(given);
  CHECK(p2("FSIGMA") == (std::vector<std::string>{"0.2", "0.05"}));
  CHECK(p2("ESIGMA") == std::vector<std::string>{"0.5"});
  CHECK(p2("ENORM") == std::vector<std::string>{"SQRT"});
}

/* Pins: every fit writes one INFO line naming each error scale, ENORM and
 * ENERGY with its value and whether it is the default; a scale of a
 * quantity that is off is said to be inert.
 * Before the fix: no such line (and no such keys). */
TEST_CASE("A fit names its error scales in an INFO line",
          "[error_scales][info]") {
  PeriodicTable::initialize();
  Context c = make_context(true, false);
  set_key(c, "FSIGMA", "0.2 0.05");
  set_key(c, "VERBOSE", "2");
  auto &log = tadah::core::Logger::instance();
  const auto old = log.getLogLevel();
  std::ostringstream captured;
  std::streambuf *buf = std::cout.rdbuf(captured.rdbuf());
  try {
    fitted(c, db_of({cell_2(), cell_3(), cell_4()}));
  } catch (...) {
    std::cout.rdbuf(buf);
    log.setLogLevel(old);
    throw;
  }
  std::cout.rdbuf(buf);
  log.setLogLevel(old);
  std::string line;
  {
    std::istringstream in(captured.str());
    for (std::string l; std::getline(in, l);)
      if (l.find("error scales") != std::string::npos) line = l;
  }
  INFO(captured.str());
  CHECK_THAT(line, Contains("ESIGMA 1 meV/atom (default)") &&
                       Contains("ENORM ATOM (default)") &&
                       Contains("ENERGY true (default)") &&
                       Contains("FSIGMA 0.2 0.05 eV/") &&
                       Contains("SSIGMA 0.5 0 GPa (default; STRESS false)"));
  CHECK(line.find("FSIGMA 0.2 0.05 eV/Å (default") == std::string::npos);
}
