// test_trainer_sigma.cpp
//
// A potential used as the training configuration (`tadah train -c
// pot.tadah`, the -c potential of `tadah hpo`): the Trainer built from it
// fits a model of its own, whose potential carries the weight covariance
// of that fit only (OALGO 3: WCOV and WNOISE) or none (any other fit), and
// an M_KRR potential trains as an M_BLR one does; a Trainer whose weights
// are then set from FIXWEIGHT (provision_fixed_weights) holds no
// covariance of its earlier fit. (Until WCOV and WNOISE replaced it, the
// covariance was SIGMA, written into the model's Context by the fit.)
//
// Data: three sheared and rattled Ta cells of 3 and 4 atoms, D2_Blip with
// 3 functions, Cut_Cos at 5 A, INITMB false (DSIZE 3), FORCE true, LAMBDA
// 0. The targets are arbitrary smooth numbers.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/models/descriptors/dm/embed_spec.h>
#include <sstream>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

std::uint64_t sbits(double v) {
  std::uint64_t u;
  std::memcpy(&u, &v, sizeof u);
  return u;
}

Context make_context(const std::string &model, const std::string &fn,
                     const std::string &oalgo) {
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
  c.add("EWEIGHT", 1.0);
  c.add("FWEIGHT", 1.0);
  c.add("SWEIGHT", 1.0);
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", model);
  c.add("MODEL", fn);
  c.add("LAMBDA", 0);
  c.add("OALGO", oalgo);
  return c;
}

Structure make_cell(double a, std::array<double, 3> shear,
                    const std::vector<std::array<double, 3>> &frac,
                    std::size_t s) {
  Structure st;
  st.cell.load(a, 0.0, 0.0, shear[0] * a, a, 0.0, shear[1] * a, shear[2] * a,
               a);
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
  return st;
}

StructureDB training_db() {
  StructureDB db;
  db.add(make_cell(3.30, {0.05, 0.02, -0.03},
                   {{0.00, 0.01, 0.02}, {0.48, 0.52, 0.05}, {0.27, 0.55, 0.51}},
                   0));
  db.add(make_cell(3.45, {-0.04, 0.03, 0.06},
                   {{0.01, -0.02, 0.00}, {0.52, 0.47, 0.03},
                    {0.49, 0.02, 0.53}, {-0.03, 0.51, 0.48}},
                   1));
  db.add(make_cell(3.15, {0.07, -0.05, 0.02},
                   {{0.02, -0.01, 0.00}, {0.33, 0.31, 0.36}, {0.68, 0.64, 0.70}},
                   2));
  return db;
}

/* The potential a Trainer of @p c writes after training on the data. */
Context trained_potential(Context c) {
  MLIPWorkspaceManager ws;
  Trainer tr(c, ws);
  StructureDB db = training_db();
  tr.train(db);
  return tr.get_param_file();
}

void set_key(Context &c, const std::string &key, const std::string &value) {
  c.remove(key);
  c.add(key, value);
}

}  // namespace

TEST_CASE("DM_Embed train write read predict retains energies forces and stresses", "[unified-dm][train]") {
  PeriodicTable::initialize();auto c=make_context("M_BLR","BF_Linear","1");
  for(const char *key:{"TYPEMB","RCTYPEMB","RCUTMB"})c.remove(key);
  c.add("INITMB",true);c.add("STRESS",true);
  tadah::models::EmbedSpec spec;spec.id="ta";spec.pairs={"Ta","Ta"};
  spec.form=tadah::models::RhoForm::NonNeg;spec.cutoff="Cut_Poly";spec.radius=6;
  spec.density.basis=tadah::models::EmbedBasis::WendlandC4;
  spec.density.centres={2.2,3.2};spec.density.scales={0.6,0.7};spec.density.amplitudes={0.4,0.2};
  spec.embedding.basis=tadah::models::EmbedBasis::Gaussian;
  spec.embedding.centres={0,0.5,1};spec.embedding.scales={2,2,2};
  c.append_block(tadah::models::embed_block(spec));
  MLIPWorkspaceManager workspace;Trainer trainer(c,workspace);auto db=training_db();trainer.train(db);
  auto pot=trainer.get_param_file();
  auto reference=trainer.model->predict(pot,db,trainer.dc);
  std::ostringstream text;pot.write_potential(text,18);
  std::istringstream input(text.str());auto loaded=tadah::core::CoreUserConfigParser().parse(input);
  tadah::models::DC_Selector selector(loaded);
  tadah::mlip::DescriptorsCalc<> dc(loaded,*selector.d2b,*selector.dmb,*selector.c2b,*selector.cmb);
  std::unique_ptr<tadah::mlip::DM_Function_Base> fb(tadah::core::factory<tadah::mlip::DM_Function_Base,const Context&>(loaded.get<std::string>("MODEL",1),loaded));
  std::unique_ptr<tadah::mlip::M_Tadah_Base> model(tadah::core::factory<tadah::mlip::M_Tadah_Base,tadah::mlip::DM_Function_Base&,Context&>(loaded.get<std::string>("MODEL"),*fb,loaded));
  auto predicted=model->predict(loaded,db,dc);
  REQUIRE(predicted.size()==reference.size());
  for(std::size_t i=0;i<reference.size();++i) {
    REQUIRE(predicted(i).energy==Approx(reference(i).energy).margin(1e-11));
    for(std::size_t a=0;a<reference(i).natoms();++a)for(std::size_t k=0;k<3;++k)
      REQUIRE(predicted(i).atoms[a].force[k]==Approx(reference(i).atoms[a].force[k]).margin(1e-11));
    for(std::size_t a=0;a<3;++a)for(std::size_t b=0;b<3;++b)
      REQUIRE(predicted(i).stress(a,b)==Approx(reference(i).stress(a,b)).margin(1e-11));
  }
}

/* Pins: retraining from a potential (the potential of an OALGO 3 fit used
 * as the configuration, three times in a row) writes one WCOV of N*N+1 =
 * 10 values (DSIZE 3) and one WNOISE (E and F, 4 tokens), equal bit for bit
 * to the first fit's, and the same weights, for M_BLR BF_Linear and M_KRR
 * Kern_Linear.
 * Before the fix (WCOV and WNOISE): the potentials had no WCOV and no
 * WNOISE (this case, then named "... writes the SIGMA of its own fit",
 * checked SIGMA). Before fix/hpo-trainer-reuse (SIGMA, 10 values, was then
 * the covariance): M_BLR's retrains wrote 20, then 30 values (each fit
 * appended its SIGMA to the one in the configuration; predict --error then
 * read past its covariance matrix); M_KRR's retrain threw "This object is
 * already trained!" (the WEIGHTS of the potential marked it trained). */
TEST_CASE("Retraining from an OALGO 3 potential writes the covariance of "
          "its own fit",
          "[trainer][sigma]") {
  PeriodicTable::initialize();
  for (const auto &mf : {std::array<const char *, 2>{"M_BLR", "BF_Linear"},
                         std::array<const char *, 2>{"M_KRR", "Kern_Linear"}}) {
    DYNAMIC_SECTION(mf[0] << " " << mf[1]) {
      const Context first = trained_potential(make_context(mf[0], mf[1], "3"));
      REQUIRE(first.size("WCOV") == 10);
      REQUIRE(first.size("WNOISE") == 4);
      CHECK_FALSE(first.has_stored("SIGMA"));
      Context pot = first;
      for (int retrain = 1; retrain <= 2; ++retrain) {
        INFO("retrain " << retrain);
        std::string msg;
        try {
          pot = trained_potential(pot);
        } catch (const std::exception &e) {
          msg = e.what();
        }
        INFO("train: " << msg);
        REQUIRE(msg.empty());
        REQUIRE(pot.size("WCOV") == 10);
        for (std::size_t k = 0; k < 10; ++k)
          CHECK(sbits(pot.get<double>("WCOV", k)) ==
                sbits(first.get<double>("WCOV", k)));
        REQUIRE(pot.size("WNOISE") == 4);
        for (std::size_t k : {1, 3})
          CHECK(sbits(pot.get<double>("WNOISE", k)) ==
                sbits(first.get<double>("WNOISE", k)));
        REQUIRE(pot.size("WEIGHTS") == first.size("WEIGHTS"));
        for (std::size_t k = 0; k < first.size("WEIGHTS"); ++k)
          CHECK(sbits(pot.get<double>("WEIGHTS", k)) ==
                sbits(first.get<double>("WEIGHTS", k)));
      }
    }
  }
}

/* Pins: retraining an OALGO 3 potential with OALGO 1 (the potential edited
 * to OALGO 1 and used as the configuration) writes a potential without
 * WCOV, WNOISE or SIGMA, for M_BLR BF_Linear and M_KRR Kern_Linear; a
 * retrain with OALGO 3 and FIXINDEX pinning every weight (M_BLR, the fit
 * that solves nothing) writes the covariance of pinned weights, WCOV 3
 * and nine zeros, with a WNOISE (the noise of the pinned weights'
 * residuals), and its weights are the FIXWEIGHT values.
 * Before the fix (WCOV and WNOISE): the potentials had no WCOV, and the
 * pinned retrain no WNOISE (it wrote no covariance: the expectation of this
 * case, then named "... writes no SIGMA", for the pinned retrain before
 * WCOV). Before fix/hpo-trainer-reuse: the potentials kept the
 * configuration's 10 SIGMA values beside weights of another fit (and
 * predict --error used them); M_KRR's retrain threw "This object is
 * already trained!". */
TEST_CASE("Retraining an OALGO 3 potential without OALGO 3 writes no "
          "covariance",
          "[trainer][sigma]") {
  PeriodicTable::initialize();
  for (const auto &mf : {std::array<const char *, 2>{"M_BLR", "BF_Linear"},
                         std::array<const char *, 2>{"M_KRR", "Kern_Linear"}}) {
    DYNAMIC_SECTION(mf[0] << " " << mf[1]) {
      Context pot = trained_potential(make_context(mf[0], mf[1], "3"));
      REQUIRE(pot.size("WCOV") == 10);
      set_key(pot, "OALGO", "1");
      std::string msg;
      Context out;
      try {
        out = trained_potential(pot);
      } catch (const std::exception &e) {
        msg = e.what();
      }
      INFO("train: " << msg);
      REQUIRE(msg.empty());
      CHECK_FALSE(out.has_stored("SIGMA"));
      CHECK_FALSE(out.has_stored("WCOV"));
      CHECK_FALSE(out.has_stored("WNOISE"));
      CHECK(out.size("WEIGHTS") == 3);
    }
  }
  SECTION("M_BLR BF_Linear, FIXINDEX pinning every weight") {
    Context pot = trained_potential(make_context("M_BLR", "BF_Linear", "3"));
    REQUIRE(pot.size("WCOV") == 10);
    pot.add("FIXINDEX", "1-3");
    pot.add("FIXWEIGHT", "0.25 -0.5 0.75");
    const Context out = trained_potential(pot);
    CHECK_FALSE(out.has_stored("SIGMA"));
    REQUIRE(out.size("WCOV") == 10);
    CHECK(out.get<double>("WCOV", 0) == 3.0);
    for (std::size_t k = 1; k < 10; ++k) CHECK(out.get<double>("WCOV", k) == 0.0);
    CHECK(out.size("WNOISE") == 4);
    REQUIRE(out.size("WEIGHTS") == 3);
    CHECK(out.get<double>("WEIGHTS", 0) == 0.25);
    CHECK(out.get<double>("WEIGHTS", 1) == -0.5);
    CHECK(out.get<double>("WEIGHTS", 2) == 0.75);
  }
}

/* Pins: Trainer::provision_fixed_weights() leaves no covariance of an
 * earlier fit of its Trainer: an OALGO 3 fit of the Trainer (M_BLR
 * BF_Linear, DSIZE 3) gives the potential (get_param_file) WCOV (10
 * values) and WNOISE; FIXINDEX 1-3 and FIXWEIGHT 0.25 -0.5 0.75 are then
 * added to tr.context, the model is reset to be fitted again in place
 * (model->reset_trained()) and provision_fixed_weights() sets every
 * weight, with no fit: the potential has no WCOV, WNOISE or SIGMA, the
 * model is trained, and the weights are the FIXWEIGHT values; tr.context
 * has none of the three (a fit writes nothing into it).
 * Before the fix (WCOV and WNOISE): the first potential had no WCOV (the
 * fit wrote SIGMA into tr.context, which provision_fixed_weights removed:
 * the expectation of this case, then named "provision_fixed_weights
 * removes the SIGMA of an earlier fit of its Trainer", before WCOV). */
TEST_CASE("provision_fixed_weights leaves no covariance of an earlier fit "
          "of its Trainer",
          "[trainer][sigma]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  Context c = make_context("M_BLR", "BF_Linear", "3");
  Trainer tr(c, ws);
  StructureDB db = training_db();
  tr.train(db);
  CHECK_FALSE(tr.context.has_stored("SIGMA"));
  REQUIRE(tr.get_param_file().size("WCOV") == 10);
  REQUIRE(tr.get_param_file().has_stored("WNOISE"));

  tr.context.add("FIXINDEX", "1-3");
  tr.context.add("FIXWEIGHT", "0.25 -0.5 0.75");
  tr.model->reset_trained();
  REQUIRE_NOTHROW(tr.provision_fixed_weights());
  for (const char *key : {"SIGMA", "WCOV", "WNOISE"}) {
    INFO(key);
    CHECK_FALSE(tr.context.has_stored(key));
    CHECK_FALSE(tr.get_param_file().has_stored(key));
  }
  CHECK(tr.model->is_trained());
  const auto &w = tr.model->get_weights();
  REQUIRE(w.size() == 3);
  CHECK(w[0] == 0.25);
  CHECK(w[1] == -0.5);
  CHECK(w[2] == 0.75);
}
