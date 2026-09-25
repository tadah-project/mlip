// DescriptorsCalc: the DIMER path (DIMER true <bond length> <bond flag>).
//
// A DIMER model describes a pair of diatomic molecules, stored as four
// atoms: molecule I is atoms 0 and 1, molecule J is atoms 2 and 3. Each
// atom's descriptors are built from its distances to the atoms of the other
// molecule and, when the bond flag is true, to its bonded partner as well.
// The published Tadah!N2 potential is such a model.
//
// The expected descriptors are assembled here from the descriptor objects
// themselves, pair by pair, with the displacement vectors computed from the
// atom positions: the test checks that the DIMER path feeds the real
// geometry to the descriptors, not how the descriptors are defined.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure.h>
#include <tadah/models/dc_selector.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

using namespace tadah::core;
using namespace tadah::mlip;
using tadah::models::DC_Selector;

namespace {

constexpr double kBond = 1.1;
constexpr double kRcut = 6.0;

/* A two-body (D2_Blip) and a many-body (DM_Blip, Lmax 1) descriptor, for
 * the element pairs `pairs2b` and `pairsmb`; DIMER on or off. */
Context descriptor_context(bool dimer, bool bond, const std::string &pairs2b,
                           const std::string &pairsmb) {
  return CoreUserConfigParser().parse(std::vector<std::string>{
      "ATOM N", "BIAS false", "INIT2B true", "INITMB true",
      "TYPE2B D2_Blip 2 2 " + pairs2b, "CGRID2B 2.0 3.5", "SGRID2B 0.8 0.5",
      "TYPEMB DM_Blip 1 2 2 " + pairsmb, "CGRIDMB 1.5 3.0", "SGRIDMB 0.6 0.4",
      "RCTYPE2B Cut_Cos", "RCTYPEMB Cut_Cos",
      "RCUT2B " + std::to_string(kRcut), "RCUTMB " + std::to_string(kRcut),
      std::string("DIMER ") + (dimer ? "true " : "false ") +
          std::to_string(kBond) + (bond ? " true" : " false")});
}

/* Both descriptors for every element pair, as in the published Tadah!N2
 * potential. */
Context dimer_context(bool bond) {
  return descriptor_context(true, bond, "* *", "* *");
}

/* Two N2 molecules whose centres of mass are `d` apart along x. Molecule J
 * is tilted, so that no two displacement vectors are parallel. */
Structure n2_pair(double d) {
  PeriodicTable::initialize();
  Structure st;
  st.cell.load(40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0);
  const double c = 20.0;
  const double h = 0.5 * kBond;
  const Vec3d u(0.3, 0.1, std::sqrt(1.0 - 0.3 * 0.3 - 0.1 * 0.1));
  const Element N = PeriodicTable::find_by_symbol("N");
  st.add_atom(Atom(N, c, c, c - h, 0, 0, 0));
  st.add_atom(Atom(N, c, c, c + h, 0, 0, 0));
  st.add_atom(Atom(N, c + d - h * u[0], c - h * u[1], c - h * u[2], 0, 0, 0));
  st.add_atom(Atom(N, c + d + h * u[0], c + h * u[1], c + h * u[2], 0, 0, 0));
  return st;
}

/* Atoms whose distances enter the descriptors of atom a. */
std::vector<size_t> partners(size_t a, bool bond) {
  std::vector<size_t> p;
  for (size_t b = 0; b < 4; ++b) {
    if (b == a) continue;
    const bool bonded = (a / 2) == (b / 2);
    if (bonded && !bond) continue;
    p.push_back(b);
  }
  return p;
}

struct DimerDescriptors {
  Context ctx;
  DC_Selector dcs;
  DescriptorsCalc<> dc;

  explicit DimerDescriptors(bool bond)
      : ctx(dimer_context(bond)), dcs(ctx),
        dc(ctx, *dcs.d2b, *dcs.dmb, *dcs.c2b, *dcs.cmb) {}

  StDescriptors calc(Structure st) {
    NNFinder().calc(st, kRcut);
    return dc.calc(st);
  }

  /* Descriptors of atom a, summed pair by pair over its partners. */
  aed_type expected(const Structure &st, size_t a, bool bond, size_t dim) {
    aed_type aed(dim);
    aed.set_zero();
    rho_type rho(dcs.dmb->rhoi_size() + dcs.dmb->rhoip_size());
    rho.set_zero();
    const Vec3d &pa = st.atoms[a].position;
    const int Za = st.atoms[a].Z;
    for (size_t b : partners(a, bond)) {
      const Vec3d &pb = st.atoms[b].position;
      const int Zb = st.atoms[b].Z;
      Vec3d del;
      for (size_t k = 0; k < 3; ++k) del[k] = pa[k] - pb[k];
      const double rsq = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
      const double r = std::sqrt(rsq);
      dcs.d2b->calc_aed(Za, Zb, r, rsq, aed);
      dcs.dmb->calc_rho(Za, Zb, r, rsq, del, rho);
    }
    dcs.dmb->calc_aed(rho, aed);
    return aed;
  }
};

} // namespace

TEST_CASE("DescriptorsCalc DIMER: each atom's descriptors come from its "
          "distances to its partner atoms",
          "[descriptors_calc][dimer]") {
  for (const bool bond : {true, false}) {
    for (const double d : {3.2, 4.5}) {
      INFO("bond included: " << bond << ", separation: " << d);
      DimerDescriptors dd(bond);
      const Structure st = n2_pair(d);
      const StDescriptors st_d = dd.calc(st);
      REQUIRE(st_d.naed() == 4);
      for (size_t a = 0; a < 4; ++a) {
        INFO("atom " << a);
        const aed_type &got = st_d.get_aed(a);
        const aed_type want = dd.expected(st, a, bond, got.size());
        bool nonzero = false;
        for (size_t k = 0; k < got.size(); ++k) {
          INFO("descriptor " << k);
          CHECK(got(k) == Approx(want(k)).epsilon(1e-12).margin(1e-14));
          nonzero = nonzero || want(k) != 0.0;
        }
        CHECK(nonzero);
      }
    }
  }
}

TEST_CASE("DescriptorsCalc DIMER: the descriptors follow the separation of "
          "the molecules",
          "[descriptors_calc][dimer]") {
  DimerDescriptors dd(true);
  const StDescriptors near = dd.calc(n2_pair(3.2));
  const StDescriptors far = dd.calc(n2_pair(4.5));
  for (size_t a = 0; a < 4; ++a) {
    INFO("atom " << a);
    double diff = 0.0;
    for (size_t k = 0; k < near.dim(); ++k)
      diff += std::fabs(near.get_aed(a)(k) - far.get_aed(a)(k));
    CHECK(diff > 1e-3);
  }
}

TEST_CASE("DescriptorsCalc DIMER: descriptors that name specific element "
          "pairs are refused", "[descriptors_calc][dimer]") {
  /* calc_dimer evaluates every atom as Z = 1, whatever its element. A
   * descriptor declared for specific element pairs (`N N`) applies to no
   * pair of Z = 1 atoms, so every descriptor of every atom was zero and
   * the model predicted from zero descriptors without a word. Every
   * DescriptorsCalc constructor -- the one `tadah train` (Trainer) and
   * `tadah predict` (PredictEngine) use, with the descriptors of a
   * DC_Selector, and the one that builds its own -- refuses it. */
  const std::string msg =
      "DIMER evaluates every atom as Z=1; declare TYPE2B/TYPEMB with * * "
      "element pairs";
  PeriodicTable::initialize();
  for (const auto &pairs : {std::make_pair("N N", "N N"),
                            std::make_pair("N N", "* *"),
                            std::make_pair("* *", "N N"),
                            std::make_pair("* N", "* *"),
                            std::make_pair("* *", "N *")}) {
    INFO("TYPE2B ... " << pairs.first << ", TYPEMB ... " << pairs.second);
    Context ctx = descriptor_context(true, true, pairs.first, pairs.second);
    DC_Selector dcs(ctx);
    CHECK_THROWS_WITH(
        (DescriptorsCalc<>(ctx, *dcs.d2b, *dcs.dmb, *dcs.c2b, *dcs.cmb)),
        Catch::Contains(msg));
    Context own = descriptor_context(true, true, pairs.first, pairs.second);
    CHECK_THROWS_WITH((DescriptorsCalc<tadah::models::D2_Blip,
                                       tadah::models::DM_Blip,
                                       tadah::models::Cut_Cos,
                                       tadah::models::Cut_Cos>(own)),
                      Catch::Contains(msg));
  }

  /* `* *` is accepted, and so are specific pairs without DIMER. */
  for (const bool dimer : {true, false}) {
    INFO("DIMER " << dimer);
    const std::string pairs = dimer ? "* *" : "N N";
    Context ctx = descriptor_context(dimer, true, pairs, pairs);
    DC_Selector dcs(ctx);
    CHECK_NOTHROW(
        DescriptorsCalc<>(ctx, *dcs.d2b, *dcs.dmb, *dcs.c2b, *dcs.cmb));
  }
}
