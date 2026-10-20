// test_row_weights.cpp
//
// The one unit that computes the weight of every row of a fit, its
// observations and its effective parameters (row_weights.h).
//
//  - a row's weight is W = p m / sigma^2 and its factor sqrt(W) =
//    sqrt(p m) / sigma: p the priority (the structure's eweight, fweight or
//    sweight times the WDBFILE value of its dataset), m 2 for the
//    off-diagonal stress rows (xy, xz, yz: the nine entries of the tensor)
//    and 1 for the others; sigma N^q ESIGMA/1000 eV for the energy of a
//    structure of N atoms (q 1 under ENORM ATOM, 1/2 under SQRT),
//    sqrt(a^2 + (b M)^2) for a force component (FSIGMA a b, M the RMS of the
//    structure's reference force components) and for a stress entry
//    (SSIGMA in GPa, 1 GPa = 6.241509074e-3 eV/A^3, M the RMS of the nine
//    entries of its reference stress);
//  - the effective parameters of each quantity, gamma_q = tr(H^-1 H_q) with
//    H = sum_i W_i phi_i phi_i^T + lambda I_pen over the directions the
//    solve keeps, per quantity and per dataset; they sum to sum_kept
//    s_i^2/(s_i^2 + lambda), the rank at lambda 0, and the prior's share is
//    the rest of the rank.
//
// The references are the formulas above evaluated here, in the order the
// header documents (bitwise), and an independent SVD of the weighted
// design matrix (LAPACK dgesvd) for the effective parameters.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/lapack.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/design_matrix.h>
#include <tadah/mlip/design_matrix/functions/basis_functions/dm_bf_linear.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/row_weights.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/dc_selector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Catch::Matchers::Contains;
using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::RowWeights;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;

std::uint64_t bits(double v) {
  std::uint64_t u;
  std::memcpy(&u, &v, sizeof u);
  return u;
}

void set_key(Context &c, const std::string &key, const std::string &value) {
  if (c.has_stored(key)) c.remove(key);
  c.add(key, value);
}

/* A cubic cell of @p n atoms on a simple-cubic grid with edge @p a, with
 * forces and a stress that are not zero, and the given priorities. */
Structure cell_of(std::size_t n, double a, double ew, double fw, double sw) {
  Structure st;
  st.cell.load(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
  for (std::size_t i = 0; i < n; ++i) {
    const double x = 0.37 * i, y = 0.11 * i, z = 0.53 * i;
    st.add_atom(Atom(Element("Ta"), std::fmod(x, a), std::fmod(y, a),
                     std::fmod(z, a), 0.3 * std::sin(1.0 + i),
                     -0.2 * std::cos(0.5 * i), 0.1 * std::sin(2.0 * i)));
  }
  st.energy = -8.0 * static_cast<double>(n);
  for (int x = 0; x < 3; ++x)
    for (int y = x; y < 3; ++y) {
      st.stress(x, y) = 0.01 * (1 + x) - 0.004 * y;
      st.stress(y, x) = st.stress(x, y);
    }
  st.eweight = ew;
  st.fweight = fw;
  st.sweight = sw;
  return st;
}

double force_rms(const Structure &st) {
  double s = 0;
  for (const Atom &a : st.atoms)
    for (int k = 0; k < 3; ++k) s += a.force[k] * a.force[k];
  return std::sqrt(s / (3.0 * static_cast<double>(st.natoms())));
}

double stress_rms(const Structure &st) {
  double s = 0;
  for (int x = 0; x < 3; ++x)
    for (int y = 0; y < 3; ++y) s += st.stress(x, y) * st.stress(x, y);
  return std::sqrt(s / 9.0);
}

/* The singular values of the m x n column-major matrix @p a. */
std::vector<double> singular_values(std::vector<double> a, int m, int n) {
  char jobu = 'N', jobvt = 'N';
  int lda = m, ldu = 1, ldvt = 1, info = 0, lwork = -1;
  std::vector<double> s(static_cast<std::size_t>(std::min(m, n)));
  double u = 0, vt = 0, wq = 0;
  tadah::core::dgesvd_(&jobu, &jobvt, &m, &n, a.data(), &lda, s.data(), &u,
                       &ldu, &vt, &ldvt, &wq, &lwork, &info);
  lwork = static_cast<int>(wq);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  tadah::core::dgesvd_(&jobu, &jobvt, &m, &n, a.data(), &lda, s.data(), &u,
                       &ldu, &vt, &ldvt, work.data(), &lwork, &info);
  REQUIRE(info == 0);
  return s;
}

}  // namespace

/* Pins: the row factors, bit for bit, of a 2-, a 4- and a 54-atom cell in
 * two datasets (WDBFILE 1 0.25) with priorities of their own, under ENORM
 * ATOM and SQRT, ESIGMA 2, FSIGMA 0.2 0.3 and SSIGMA 0.7 0.05: energy
 * sqrt(p_E)/(N^q ESIGMA/1000), force sqrt(p_F)/sqrt(a^2 + (b M_F)^2),
 * stress sqrt(p_S m)/sqrt(a'^2 + (b M_S)^2), a' = 0.7 x 6.241509074e-3,
 * m = 2 off the diagonal; and their sigmas and priorities. ENERGY false
 * makes every energy factor 0; WDBFILE 0 every factor of its dataset. */
TEST_CASE("RowWeights: the factor of each row, bit for bit",
          "[row_weights]") {
  PeriodicTable::initialize();
  StructureDB db;
  db.add(cell_of(2, 3.3, 0.5, 2.0, 3.0));
  db.add(cell_of(4, 4.1, 4.0, 0.25, 1.0));
  db.add(cell_of(54, 9.9, 1.0, 1.0, 1.0));
  db.dbidx = {0, 2, 3};
  const double wdb[2] = {1.0, 0.25};
  const std::size_t dataset[3] = {0, 0, 1};
  for (const std::string norm : {"ATOM", "SQRT"}) {
    DYNAMIC_SECTION("ENORM " << norm) {
      Context c;
      c.add("FORCE", "true");
      c.add("STRESS", "true");
      c.add("ESIGMA", "2");
      c.add("FSIGMA", "0.2 0.3");
      c.add("SSIGMA", "0.7 0.05");
      c.add("ENORM", norm);
      c.add("WDBFILE", "1 0.25");
      const RowWeights rw(c, db);
      for (std::size_t s = 0; s < db.size(); ++s) {
        INFO("structure " << s);
        const Structure &st = db(s);
        const double N = static_cast<double>(st.natoms());
        const double d = wdb[dataset[s]];
        CHECK(rw.dataset_priority(s) == d);
        const double pE = st.eweight * d, pF = st.fweight * d,
                     pS = st.sweight * d;
        CHECK(bits(rw.energy_priority(s)) == bits(pE));
        const double Nq = norm == "ATOM" ? N : std::sqrt(N);
        const double sE = Nq * (2.0 / 1000.0);
        CHECK(bits(rw.energy_sigma(s)) == bits(sE));
        CHECK(bits(rw.energy_factor(s)) == bits(std::sqrt(pE) / sE));
        const double bF = 0.3 * force_rms(st);
        const double sF = std::sqrt(0.2 * 0.2 + bF * bF);
        CHECK(bits(rw.force_sigma(s)) == bits(sF));
        CHECK(bits(rw.force_factor(s)) == bits(std::sqrt(pF) / sF));
        const double aS = 0.7 * 6.241509074e-3;
        const double bS = 0.05 * stress_rms(st);
        const double sS = std::sqrt(aS * aS + bS * bS);
        CHECK(bits(rw.stress_sigma(s)) == bits(sS));
        for (int x = 0; x < 3; ++x)
          for (int y = x; y < 3; ++y) {
            INFO("stress " << x << y);
            const double m = x == y ? 1.0 : 2.0;
            CHECK(bits(rw.stress_factor(s, x, y)) ==
                  bits(std::sqrt(pS * m) / sS));
          }
      }
    }
  }
  SECTION("ENERGY false, WDBFILE 0") {
    Context c;
    c.add("FORCE", "true");
    c.add("ENERGY", "false");
    c.add("WDBFILE", "1 0");
    const RowWeights rw(c, db);
    for (std::size_t s = 0; s < 3; ++s) {
      CHECK(rw.energy_factor(s) == 0.0);
      CHECK(rw.energy_priority(s) == 0.0);
    }
    CHECK(rw.force_factor(0) > 0.0);
    CHECK(rw.force_factor(2) == 0.0);
    CHECK(rw.stress_factor(2, 0, 1) == 0.0);
  }
  SECTION("a structure whose forces are zero has the absolute scale") {
    StructureDB z;
    Structure st = cell_of(2, 3.3, 1, 1, 1);
    for (Atom &a : st.atoms) a.force = tadah::core::Vec3d(0, 0, 0);
    z.add(st);
    Context c;
    c.add("FSIGMA", "0.2 5");
    const RowWeights rw(c, z);
    CHECK(rw.force_sigma(0) == 0.2);
  }
}

/* Pins: the effective parameters of a design matrix whose energy rows
 * (type 0) span columns 0 and 1 and whose force rows (type 1) span
 * columns 2 and 3, in two datasets: at LAMBDA 0 gamma_E = 2, gamma_F = 2,
 * no stress, the rank 4, no prior share; at LAMBDA 0.3 (OALGO 3, every
 * column penalised) each block's sum of s^2/(s^2 + 0.3) over its own
 * singular values (an independent SVD of the block), the prior's share 4
 * minus their sum; the per-dataset split sums to the per-type values; with
 * the first column penalised and the others not (OALGO 1, LAMBDA 0.3) the
 * energy block's share is that of its 2 x 2 normal matrix. 1e-12. */
TEST_CASE("Effective parameters per quantity and dataset", "[row_weights]") {
  // rows: 3 energies (datasets 0, 0, 1), 4 forces (datasets 0, 1, 1, 1)
  const double E[3][2] = {{1.0, 0.5}, {0.2, 2.0}, {1.5, -0.7}};
  const double F[4][2] = {{0.3, 1.1}, {-0.4, 0.9}, {2.0, 0.1}, {0.6, -1.3}};
  tadah::core::phi_type Phi(7, 4);
  Phi.set_zero();
  tadah::core::t_type labels(7);
  std::vector<std::size_t> ds = {0, 0, 1, 0, 1, 1, 1};
  for (int i = 0; i < 3; ++i) {
    Phi(i, 0) = E[i][0];
    Phi(i, 1) = E[i][1];
    labels(i) = 0;
  }
  for (int i = 0; i < 4; ++i) {
    Phi(3 + i, 2) = F[i][0];
    Phi(3 + i, 3) = F[i][1];
    labels(3 + i) = 1;
  }
  auto block_sv = [](const double (*b)[2], int rows) {
    std::vector<double> a;
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < rows; ++i) a.push_back(b[i][j]);
    return singular_values(a, rows, 2);
  };
  const auto sE = block_sv(E, 3), sF = block_sv(F, 4);
  {
    const auto g = tadah::mlip::effective_parameters(Phi, labels, ds, 2, 0.0,
                                                     {}, 3, 1e-8);
    CHECK(g.by_type[0] == Approx(2.0).epsilon(1e-12));
    CHECK(g.by_type[1] == Approx(2.0).epsilon(1e-12));
    CHECK(g.by_type[2] == 0.0);
    CHECK(g.retained == 4);
    CHECK(g.total == Approx(4.0).epsilon(1e-12));
    CHECK(std::fabs(g.prior) < 1e-12);
  }
  {
    const double lam = 0.3;
    const auto g = tadah::mlip::effective_parameters(Phi, labels, ds, 2, lam,
                                                     {}, 3, 1e-8);
    double gE = 0, gF = 0;
    for (double s : sE) gE += s * s / (s * s + lam);
    for (double s : sF) gF += s * s / (s * s + lam);
    CHECK(g.by_type[0] == Approx(gE).epsilon(1e-12));
    CHECK(g.by_type[1] == Approx(gF).epsilon(1e-12));
    CHECK(g.total == Approx(gE + gF).epsilon(1e-12));
    CHECK(g.prior == Approx(4.0 - gE - gF).epsilon(1e-12));
    REQUIRE(g.by_dataset.size() == 2);
    for (int q = 0; q < 3; ++q)
      CHECK(g.by_dataset[0][q] + g.by_dataset[1][q] ==
            Approx(g.by_type[q]).margin(1e-12));
    CHECK(g.by_dataset[0][0] > 0.0);
    CHECK(g.by_dataset[1][1] > g.by_dataset[0][1]);
  }
  {
    // OALGO 1, LAMBDA 0.3 on column 0 only (columns 1-3 unpenalised): the
    // energy block's H_E + lambda e0 e0^T, its share tr((H_E + L)^-1 H_E).
    double h[2][2] = {{0, 0}, {0, 0}};
    for (int i = 0; i < 3; ++i)
      for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b) h[a][b] += E[i][a] * E[i][b];
    const double m00 = h[0][0] + 0.3, m01 = h[0][1], m11 = h[1][1];
    const double det = m00 * m11 - m01 * m01;
    // tr(M^-1 H) with M^-1 = [[m11, -m01], [-m01, m00]] / det
    const double gE =
        (m11 * h[0][0] - m01 * h[1][0] - m01 * h[0][1] + m00 * h[1][1]) / det;
    const auto g = tadah::mlip::effective_parameters(Phi, labels, ds, 2, 0.3,
                                                     {1, 2, 3}, 1, 1e-8);
    CHECK(g.by_type[0] == Approx(gE).epsilon(1e-12));
    CHECK(g.by_type[1] == Approx(2.0).epsilon(1e-12));
    CHECK(g.prior == Approx(2.0 - gE).epsilon(1e-12));
  }
}

/* Pins: the effective parameters of a fit's weighted design matrix (two
 * datasets of Ta cells, FORCE and STRESS true) sum to sum_kept
 * s_i^2/(s_i^2 + lambda), s_i the singular values of the weighted matrix
 * (an independent SVD), at LAMBDA 0 (= the rank, 3), at LAMBDA 1e-2 and at
 * the lambda the evidence approximation found (LAMBDA -1: ALPHA/BETA of the
 * fit), OALGO 3; and a column repeated (rank 3 of 4 columns) gives 3
 * retained directions. 1e-10 relative. */
TEST_CASE("Effective parameters of a fit sum to sum s^2/(s^2 + lambda)",
          "[row_weights]") {
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
  c.add("STRESS", "true");
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("OALGO", 3);
  c.add("ESIGMA", "5");
  c.add("FSIGMA", "0.2 0.1");
  c.add("SSIGMA", "0.3");
  StructureDB db;
  db.add(cell_of(2, 3.3, 1, 1, 1));
  db.add(cell_of(3, 3.6, 1, 0.5, 1));
  db.add(cell_of(4, 4.1, 2, 1, 1));
  db.dbidx = {0, 2, 3};
  tadah::mlip::NNFinder nnf;
  nnf.calc(db, c);
  tadah::models::DC_Selector dcs(c);
  tadah::mlip::DescriptorsCalc<> dc(c, *dcs.d2b, *dcs.dmb, *dcs.c2b, *dcs.cmb);
  tadah::mlip::DM_BF_Linear bf(c);
  tadah::mlip::DesignMatrix<tadah::mlip::DM_BF_Linear> dm(bf, c);
  dm.augment = false;
  tadah::mlip::Normaliser norm(c);
  dm.build(db, norm, dc, false);
  const tadah::core::phi_type Phi = dm.getPhi();
  const tadah::core::t_type labels = dm.getTlabels();
  const std::vector<std::size_t> ds = dm.row_datasets();
  REQUIRE(ds.size() == Phi.rows());
  std::vector<double> a(Phi.rows() * Phi.cols());
  for (std::size_t j = 0; j < Phi.cols(); ++j)
    for (std::size_t i = 0; i < Phi.rows(); ++i) a[i + Phi.rows() * j] = Phi(i, j);
  const auto s = singular_values(a, static_cast<int>(Phi.rows()),
                                 static_cast<int>(Phi.cols()));
  auto invariant = [&](double lam) {
    double sum = 0;
    for (double v : s)
      if (v > 1e-8 * s[0]) sum += v * v / (v * v + lam);
    return sum;
  };

  // The lambda of the evidence approximation: a fit with LAMBDA -1.
  double ea_lambda = 0;
  {
    Context f = c;
    f.add("LAMBDA", -1);
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    tadah::mlip::Trainer tr(f, ws);
    StructureDB d = db;
    tr.train(d);
    const Context pot = tr.get_param_file();
    ea_lambda = pot.get<double>("ALPHA") / pot.get<double>("BETA");
  }
  for (const double lam : {0.0, 1e-2, ea_lambda}) {
    INFO("lambda " << lam);
    const auto g = tadah::mlip::effective_parameters(Phi, labels, ds, 2, lam,
                                                     {}, 3, 1e-8);
    CHECK(g.total == Approx(invariant(lam)).epsilon(1e-10));
    CHECK(g.by_type[0] + g.by_type[1] + g.by_type[2] ==
          Approx(g.total).epsilon(1e-12));
    CHECK(g.retained == 3);
    CHECK(g.total + g.prior == Approx(3.0).epsilon(1e-12));
    CHECK(g.by_type[0] > 0.0);
    CHECK(g.by_type[1] > 0.0);
    CHECK(g.by_type[2] > 0.0);
    if (lam == 0.0) CHECK(g.total == Approx(3.0).epsilon(1e-10));
  }
  // A repeated column: 4 columns of rank 3.
  tadah::core::phi_type P4(Phi.rows(), 4);
  for (std::size_t i = 0; i < Phi.rows(); ++i) {
    for (std::size_t j = 0; j < 3; ++j) P4(i, j) = Phi(i, j);
    P4(i, 3) = Phi(i, 0);
  }
  const auto g4 = tadah::mlip::effective_parameters(P4, labels, ds, 2, 0.0, {},
                                                    3, 1e-8);
  CHECK(g4.retained == 3);
  CHECK(g4.total == Approx(3.0).epsilon(1e-10));
}

/* Pins: the INFO line of a fit names each error scale, ENORM and ENERGY
 * with its value and whether it is the default, and the switch that makes
 * a scale inert. */
TEST_CASE("fit_scales_info names every error scale", "[row_weights]") {
  Context c;
  c.add("FORCE", "true");
  c.add("FSIGMA", "0.2");
  c.add("ENORM", "sqrt");
  const std::string line = tadah::mlip::fit_scales_info(c);
  CHECK_THAT(line, Contains("ESIGMA 1 meV per sqrt(atom) (default)") &&
                       Contains("ENORM SQRT") &&
                       Contains("ENERGY true (default)") &&
                       Contains("FSIGMA 0.2 0 eV/Å,") &&
                       Contains("SSIGMA 0.5 0 GPa (default; STRESS false)"));
  CHECK(line.find("ENORM SQRT (default)") == std::string::npos);
}
