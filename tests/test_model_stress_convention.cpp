// test_model_stress_convention.cpp
//
// The stored and predicted stress is the Cauchy stress, positive under
// tension, in eV/A^3:
//
//     sigma_ij = (1/V) dE/d(eps_ij),
//
// the derivative of the energy with respect to a homogeneous (symmetric)
// strain of the cell and the atoms, divided by the volume. That is what the
// models predict (M_Tadah_Base::spredict and stress_force_predict compute
// -(1/V) sum_pairs r_ij (x) F_ij), what the design matrix fits and what a
// Tadah dataset stores; LAMMPS reports its negative as `press`.
//
// The models are linear (M_BLR BF_Linear and M_KRR Kern_Linear) in a
// two-body and a many-body descriptor (the many-body term makes
// fd_ji != -fd_ij, so both halves of every pair force matter), with fixed
// weights: nothing depends on a fit. The cell is sheared and the atoms are
// displaced off their lattice sites, so all six stress components are
// non-zero and distinct.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/descriptors/dm/embed_spec.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::Matrix3d;
using tadah::core::PeriodicTable;
using tadah::core::t_type;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

constexpr double kRcut = 5.0;  // RCUT2B = RCUTMB = RCUTMAX

// Keys shared by both models: Ta, Cut_Cos at kRcut, FORCE and STRESS on (so
// DescriptorsCalc computes the descriptor derivatives), no bias, no
// normalisation.
void add_common_keys(Context &c) {
  c.add("ATOM", "Ta");
  c.add("WATOM", 73);
  c.add("INIT2B", "true");
  c.add("INITMB", "true");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", kRcut);
  c.add("RCUTMB", kRcut);
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", "true");
  c.add("STRESS", "true");
  c.add("EWEIGHT", 1.0);
  c.add("FWEIGHT", 1.0);
  c.add("SWEIGHT", 1.0);
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("LAMBDA", 0);
}

// M_BLR BF_Linear: D2_Blip (4 functions) + DM_EAD (Lmax 1, 2 functions).
Context blr_context() {
  Context c;
  add_common_keys(c);
  c.add("TYPE2B", "D2_Blip");
  c.add("TYPE2B", 4);
  c.add("TYPE2B", 4);
  c.add("TYPE2B", "Ta");
  c.add("TYPE2B", "Ta");
  c.add("CGRID2B", "2.4 3.0 3.6 4.2");
  c.add("SGRID2B", "1.0 1.0 1.0 1.0");
  c.add("TYPEMB", "DM_EAD");
  c.add("TYPEMB", 1);
  c.add("TYPEMB", 2);
  c.add("TYPEMB", 2);
  c.add("TYPEMB", "Ta");
  c.add("TYPEMB", "Ta");
  c.add("CGRIDMB", "2.5 3.5");
  c.add("SGRIDMB", "1.2 0.9");
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("OALGO", 1);
  return c;
}

// M_KRR Kern_Linear: D2_LJ (the pair term of the Tadah! 1.6.0 docs
// quickstart potential) + DM_Blip (4 functions).
Context krr_context() {
  Context c;
  add_common_keys(c);
  c.add("TYPE2B", "D2_LJ");
  c.add("TYPE2B", "Ta");
  c.add("TYPE2B", "Ta");
  c.add("TYPEMB", "DM_Blip");
  c.add("TYPEMB", 0);
  c.add("TYPEMB", 4);
  c.add("TYPEMB", 4);
  c.add("TYPEMB", "Ta");
  c.add("TYPEMB", "Ta");
  c.add("CGRIDMB", "2.5 3.0 3.5 4.0");
  c.add("SGRIDMB", "1.0 1.0 1.0 1.0");
  c.add("MODEL", "M_KRR");
  c.add("MODEL", "Kern_Linear");
  c.add("MODEL", 1);
  c.add("OALGO", 1);
  return c;
}

// bcc Ta (a = 3.1 A), the cell sheared to the rows
// 3.1 * [(1, 0, 0), (0.07, 1, 0), (0.03, -0.05, 1)] (V = 3.1^3, the shear is
// unimodular) and both atoms displaced off their sites by fixed vectors.
Structure sheared_cell() {
  Structure st;
  const double a = 3.1;
  st.cell.load(a, 0.0, 0.0,
               0.07 * a, a, 0.0,
               0.03 * a, -0.05 * a, a);
  // r = f0 a0 + f1 a1 + f2 a2 for f = (1/2, 1/2, 1/2), then displaced.
  const double hx = 0.5 * (1.0 + 0.07 + 0.03) * a;
  const double hy = 0.5 * (1.0 - 0.05) * a;
  const double hz = 0.5 * a;
  st.add_atom(Atom(Element("Ta"), 0.031, -0.042, 0.017, 0, 0, 0));
  st.add_atom(Atom(Element("Ta"), hx - 0.025, hy + 0.038, hz + 0.046,
                   0, 0, 0));
  return st;
}

// st deformed by F = I + e with the symmetric strain e_ij = e_ji = d/2
// (e_ii = d): every lattice vector (row of cell) and every position goes to
// F v.
Structure strained(const Structure &st, int i, int j, double d) {
  double F[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  F[i][j] += 0.5 * d;
  F[j][i] += 0.5 * d;
  auto map = [&F](double x, double y, double z, int c) {
    return F[c][0] * x + F[c][1] * y + F[c][2] * z;
  };
  Structure out = st;
  for (int r = 0; r < 3; ++r) {
    const double x = st.cell(r, 0), y = st.cell(r, 1), z = st.cell(r, 2);
    for (int c = 0; c < 3; ++c) out.cell(r, c) = map(x, y, z, c);
  }
  for (std::size_t a = 0; a < st.natoms(); ++a) {
    const auto &p = st.atoms[a].position;
    for (int c = 0; c < 3; ++c)
      out.atoms[a].position[c] = map(p[0], p[1], p[2], c);
  }
  return out;
}

// A model with the fixed weights w_k = scale_k * cos(1.7 k + 0.3). The
// scales bring every column's contribution to the stress of sheared_cell()
// to 1e-3 .. 0.3 eV/A^3 (the many-body columns are much larger per unit
// weight than the pair columns), so that each part of the force and stress
// sums matters and every component of the total is 0.06 .. 0.44 eV/A^3.
struct FixedModel {
  FixedModel(const FixedModel &) = delete;
  FixedModel &operator=(const FixedModel &) = delete;

  MLIPWorkspaceManager ws;
  Context ctx;
  std::unique_ptr<Trainer> tr;

  FixedModel(Context c, const std::vector<double> &scale)
      : ctx(std::move(c)) {
    tr = std::make_unique<Trainer>(ctx, ws);
    Structure st = sheared_cell();
    NNFinder().calc(st, kRcut);
    const std::size_t n = tr->dc.calc(st).dim();
    REQUIRE(n == scale.size());
    t_type w(n);
    for (std::size_t k = 0; k < n; ++k)
      w(k) = scale[k] * std::cos(1.7 * static_cast<double>(k) + 0.3);
    tr->model->set_weights(w);
  }

  double energy(Structure st) const {
    NNFinder().calc(st, kRcut);
    return tr->model->epredict(tr->dc.calc(st));
  }

  // M_Tadah_Base::predict with STRESS true; FORCE selects the path
  // (false: spredict, true: stress_force_predict).
  Structure predict(Structure st, bool force) const {
    NNFinder().calc(st, kRcut);
    Context pc = tr->context;
    pc.remove("FORCE");
    pc.add("FORCE", force ? "true" : "false");
    pc.remove("STRESS");
    pc.add("STRESS", "true");
    return tr->model->predict(pc, st, tr->dc);
  }
};

// FixedModel is neither copied nor moved: the Trainer keeps a pointer to
// the workspace manager next to it.
// Columns: D2_Blip (4), then DM_EAD (4).
std::unique_ptr<FixedModel> blr_model() {
  return std::make_unique<FixedModel>(
      blr_context(),
      std::vector<double>{0.02, 0.02, 0.02, 0.02, 1e-5, 1e-4, 0.05, 0.05});
}

// Columns: D2_LJ (2), then DM_Blip (4).
std::unique_ptr<FixedModel> krr_model() {
  return std::make_unique<FixedModel>(
      krr_context(), std::vector<double>{1.0, 100.0, 1e-5, 2e-5, 1e-4, 1e-3});
}

using ModelMaker = std::unique_ptr<FixedModel> (*)();

}  // namespace

// Pins the stress convention of the predicted stress: for both linear model
// families and both predict paths (STRESS true with FORCE false goes through
// spredict, with FORCE true through stress_force_predict) each of the six
// components sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz of
// the upper triangle equals (1/V) dE/d(eps) from central differences of the
// model's own energy (h = 1e-5 on a symmetric strain, the cell and the atoms
// moved together, the neighbour list rebuilt each time).
// Tolerance: the central difference is accurate to O(h^2) E'''/V plus the
// energy's rounding over 2hV. Measured here: at most 6.8e-10 (M_BLR) and
// 1.6e-9 eV/A^3 (M_KRR) at h = 1e-5, scaling as h^2 (1.6e-7 at h = 1e-4);
// two shipped potentials gave ~1e-9. 1e-7 leaves 60x
// headroom and is far below the components themselves (0.06 .. 0.44,
// |sigma| > 1e-2 checked), so a sign or factor error cannot pass.
// This passes before and after the fix. It fails if the sign of the
// predicted stress is flipped, e.g. `s(x,y) -= V_inv*0.5*...` changed to
// `+=` in M_Tadah_Base::spredict and stress_force_predict
// (MLIP/src/m_tadah_base.cpp), or if a factor (0.5, 1/V) is lost.
TEST_CASE("Predicted stress equals (1/V) dE/d(strain) for M_BLR and M_KRR",
          "[model][stress][stress_convention]") {
  PeriodicTable::initialize();
  const double h = 1e-5;
  const Structure st0 = sheared_cell();
  const double V = st0.get_volume();
  REQUIRE(V == Approx(3.1 * 3.1 * 3.1).epsilon(1e-13));

  const std::vector<std::pair<std::string, ModelMaker>> models = {
      {"M_BLR BF_Linear (D2_Blip + DM_EAD)", &blr_model},
      {"M_KRR Kern_Linear (D2_LJ + DM_Blip)", &krr_model}};
  const std::array<std::pair<int, int>, 6> comps = {
      {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {0, 2}}};

  for (const auto &m : models) {
    INFO(m.first);
    const std::unique_ptr<FixedModel> model = m.second();

    Matrix3d fd;
    for (const auto &ij : comps) {
      const double ep = model->energy(strained(st0, ij.first, ij.second, h));
      const double em = model->energy(strained(st0, ij.first, ij.second, -h));
      fd(ij.first, ij.second) = (ep - em) / (2.0 * h * V);
    }

    for (const bool force : {false, true}) {
      INFO("FORCE " << (force ? "true (stress_force_predict)"
                              : "false (spredict)"));
      const Structure p = model->predict(st0, force);
      for (const auto &ij : comps) {
        const int x = ij.first, y = ij.second;
        INFO("sigma(" << x << "," << y << "): predicted " << p.stress(x, y)
                      << ", (1/V) dE/d(eps) " << fd(x, y));
        CHECK(std::fabs(fd(x, y)) > 1e-2);
        CHECK(p.stress(x, y) == Approx(fd(x, y)).margin(1e-7));
      }
    }
  }
}

// Pins that a prediction's stress is a symmetric tensor: M_Tadah_Base::
// predict with STRESS true fills the lower triangle with the upper one,
// stress(y,x) = stress(x,y), for M_BLR BF_Linear and M_KRR Kern_Linear, on
// both predict paths (FORCE false: spredict; FORCE true:
// stress_force_predict) and through the StructureDB overload that `tadah
// predict` and HPO validation call. The off-diagonals of this sheared cell
// are well away from zero (|sigma_xy|, |sigma_yz|, |sigma_xz| > 1e-2,
// checked), so the comparison is not 0 == 0. The diagonal is the predicted
// one (it equals (1/V) dE/d(eps), pinned above).
// Before the fix: the predicted lower triangle stayed 0 (spredict and
// stress_force_predict only fill y >= x), so stress(1,0), stress(2,1) and
// stress(2,0) were 0 while stress(0,1), stress(1,2) and stress(0,2) were
// not; Structure::operator==, norm() and transformStress saw a non-symmetric
// tensor.
TEST_CASE("Predicted stress is symmetric for M_BLR and M_KRR",
          "[model][stress][stress_convention]") {
  PeriodicTable::initialize();
  const Structure st0 = sheared_cell();

  const std::vector<std::pair<std::string, ModelMaker>> models = {
      {"M_BLR BF_Linear (D2_Blip + DM_EAD)", &blr_model},
      {"M_KRR Kern_Linear (D2_LJ + DM_Blip)", &krr_model}};
  const std::array<std::pair<int, int>, 3> offdiag = {
      {{0, 1}, {1, 2}, {0, 2}}};

  auto check_symmetric = [&offdiag](const Matrix3d &s) {
    for (const auto &xy : offdiag) {
      const int x = xy.first, y = xy.second;
      INFO("stress(" << x << "," << y << ") = " << s(x, y) << ", stress("
                     << y << "," << x << ") = " << s(y, x));
      CHECK(std::fabs(s(x, y)) > 1e-2);
      CHECK(s(y, x) == s(x, y));
    }
  };

  for (const auto &m : models) {
    INFO(m.first);
    const std::unique_ptr<FixedModel> model = m.second();
    for (const bool force : {false, true}) {
      INFO("FORCE " << (force ? "true (stress_force_predict)"
                              : "false (spredict)"));
      check_symmetric(model->predict(st0, force).stress);
    }

    // The StructureDB overload (parallel over structures).
    StructureDB db;
    db.add(st0);
    db.dbidx = {0, 1};
    NNFinder().calc(db, kRcut);
    Context pc = model->tr->context;
    INFO("StructureDB overload, FORCE true");
    const StructureDB pred = model->tr->model->predict(pc, db, model->tr->dc);
    REQUIRE(pred.size() == 1);
    check_symmetric(pred(0).stress);
  }
}

TEST_CASE("DM_Embed cluster forces and all stress components equal energy differences", "[unified-dm][stress][force]") {
  PeriodicTable::initialize();auto c=blr_context();
  for(const char *key:{"TYPEMB","CGRIDMB","SGRIDMB","RCUTMB","RCTYPEMB"})c.remove(key);
  c.remove("WATOM");c.add("WATOM",1);
  tadah::models::EmbedSpec spec;spec.id="ta";spec.pairs={"Ta","Ta"};
  spec.form=tadah::models::RhoForm::NonNeg;spec.radius=kRcut;spec.cutoff="Cut_Poly";
  spec.density.basis=tadah::models::EmbedBasis::WendlandC4;
  spec.density.centres={2.2,3.2};spec.density.scales={0.6,0.7};spec.density.amplitudes={0.4,0.2};
  spec.embedding.basis=tadah::models::EmbedBasis::Gaussian;
  spec.embedding.centres={0,0.5,1};spec.embedding.scales={2,2,2};
  spec.envelope=tadah::models::EmbedEnvelope::Smoothstep2;spec.envelope_cutoff=1;
  c.append_block(tadah::models::embed_block(spec));
  FixedModel model(std::move(c),{.02,.02,.02,.02,1,2,3});
  const auto st=sheared_cell();const auto predicted=model.predict(st,true);const double h=1e-5;
  for(int x=0;x<3;++x)for(int y=x;y<3;++y) {
    double fd=(model.energy(strained(st,x,y,h))-model.energy(strained(st,x,y,-h)))/(2*h*st.get_volume());
    REQUIRE(predicted.stress(x,y)==Approx(fd).margin(1e-7));
  }
  for(std::size_t a=0;a<st.natoms();++a)for(int k=0;k<3;++k) {
    auto plus=st,minus=st;plus.atoms[a].position[k]+=h;minus.atoms[a].position[k]-=h;
    double fd=-(model.energy(plus)-model.energy(minus))/(2*h);
    REQUIRE(predicted.atoms[a].force[k]==Approx(fd).margin(1e-7));
  }
  for(int k=0;k<3;++k) {double total=0;for(std::size_t a=0;a<st.natoms();++a)total+=predicted.atoms[a].force[k];REQUIRE(std::abs(total)<1e-12);}
  auto translated=st;for(std::size_t a=0;a<st.natoms();++a)for(int k=0;k<3;++k)translated.atoms[a].position[k]+=0.37*(k+1);
  REQUIRE(model.energy(translated)==Approx(model.energy(st)).margin(1e-12));
}
