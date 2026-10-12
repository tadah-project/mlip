// BIAS true: one intercept column per ATOM species.
//
// Every atom's descriptor starts with N = size(ATOM) one-hot intercept
// columns, in ATOM order (1.0 in the column of the atom's species, 0 in the
// others), followed by the two-body and the many-body descriptors. This is
// the layout pair_style tadah (MODELS DC_Selector), the ridge (intercepts
// are not penalised), the normaliser (intercepts are not normalised) and
// the NORM weight conversion use; MLIP's DescriptorsCalc gave every atom one
// shared intercept column instead (DSIZE = 1 + descriptors), so with two or
// more species the trained weights did not fit the LAMMPS descriptors, NORM
// true and LAMBDA > 0 treated the first descriptor column as an intercept,
// and HPO's analytical gradient read past each atom's descriptor.
//
// The reference of the descriptor checks is the same configuration with
// BIAS false: with BIAS true the descriptor of an atom must be its one-hot
// intercept block followed by the BIAS false descriptor, bit for bit (aed
// and force descriptors).
//
// The fits use synthetic Nb-Ta data: 2x2x2 bcc supercells (16 atoms, a =
// 3.3 A, rattled) with varied numbers of Nb atoms, and energies from a
// per-species constant plus a fixed linear model of the BIAS false
// descriptors plus a small deterministic "noise".

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/registry.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/intercepts.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/models/basis.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/st_descriptors_db.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/functions/kernels/kern_quadratic.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::DescriptorsCalc;
using tadah::mlip::DM_Function_Base;
using tadah::mlip::M_Tadah_Base;
using tadah::mlip::NNFinder;
using tadah::mlip::StDescriptors;
using tadah::mlip::StDescriptorsDB;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;
using tadah::models::DC_Selector;

constexpr double kRcut = 5.0;
constexpr int kNb = 41;
constexpr int kTa = 73;

void set(Context &c, const std::string &key, const std::string &value) {
  c.remove(key);
  c.add(key, value);
}

/* D2_BP (3 Gaussians) for every element pair; with `mb` also DM_EAD
 * (Lmax 1, 2 Gaussians: 4 descriptors; without it no TYPEMB line, as in a
 * written two-body potential); Cut_Cos at 5 A; M_BLR BF_Linear, LAMBDA 0,
 * OALGO 1, energies only. */
Context make_context(const std::vector<std::string> &atoms, bool bias,
                     bool mb = false, double rcut = kRcut) {
  Context c;
  for (const auto &a : atoms) c.add("ATOM", a);
  c.add("INIT2B", "true");
  c.add("INITMB", mb ? "true" : "false");
  c.add("TYPE2B", "D2_BP");
  c.add("TYPE2B", 3);
  c.add("TYPE2B", 3);
  c.add("TYPE2B", "*");
  c.add("TYPE2B", "*");
  c.add("CGRID2B", "2.4 3.1 3.9");
  c.add("SGRID2B", "1.0 1.2 0.9");
  if (mb) {
    c.add("TYPEMB", "DM_EAD");
    c.add("TYPEMB", 1);
    c.add("TYPEMB", 2);
    c.add("TYPEMB", 2);
    c.add("TYPEMB", "*");
    c.add("TYPEMB", "*");
    c.add("CGRIDMB", "2.6 3.6");
    c.add("SGRIDMB", "1.0 1.1");
  }
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", rcut);
  c.add("RCUTMB", rcut);
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("EWEIGHT", 1.0);
  c.add("FWEIGHT", 1.0);
  c.add("SWEIGHT", 1.0);
  c.add("BIAS", bias ? "true" : "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  return c;
}

void set_model(Context &c, const std::string &m0, const std::string &m1) {
  c.remove("MODEL");
  c.add("MODEL", m0);
  c.add("MODEL", m1);
}

/* A 2x2x2 bcc supercell (a = 3.3 A) whose 16 sites hold the given
 * elements, every atom moved deterministically by up to 0.1 A. */
Structure bcc_cell(const std::vector<std::string> &sites, unsigned seed) {
  Structure st;
  const double L = 6.6;
  st.cell.load(L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L);
  std::size_t i = 0;
  for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z)
        for (double h : {0.0, 0.5}) {
          const double n[3] = {double(x), double(y), double(z)};
          double r[3];
          for (int k = 0; k < 3; ++k)
            r[k] = 3.3 * (n[k] + h) +
                   0.1 * std::sin(1.3 * double(i) + 0.7 * k + 0.37 * seed);
          st.add_atom(Atom(PeriodicTable::find_by_symbol(sites[i]), r[0],
                           r[1], r[2], 0.0, 0.0, 0.0));
          ++i;
        }
  return st;
}

/* The 16 sites with n_nb Nb atoms and Ta on the others, the Nb sites drawn
 * by a Fisher-Yates shuffle of std::mt19937(seed) (fully specified). */
Structure tanb_cell(int n_nb, unsigned seed) {
  std::vector<std::string> sites(16, "Ta");
  std::vector<int> order(16);
  for (int k = 0; k < 16; ++k) order[k] = k;
  std::mt19937 rng(seed);
  for (int k = 15; k > 0; --k) {
    const int j = static_cast<int>(rng() % static_cast<std::uint32_t>(k + 1));
    std::swap(order[k], order[j]);
  }
  for (int k = 0; k < n_nb; ++k) sites[order[k]] = "Nb";
  return bcc_cell(sites, seed);
}

int count_of(const Structure &st, int Z) {
  int n = 0;
  for (const Atom &a : st.atoms) n += a.Z == Z;
  return n;
}

/* Training data: energy = -7.6 n_Nb - 8.9 n_Ta + w . (sum of the BIAS
 * false descriptors) + 0.01 sin(1.7 s + 0.3). */
StructureDB make_db(const std::vector<int> &n_nb, bool mb, unsigned seed0) {
  MLIPWorkspaceManager ws;
  Context c0 = make_context({"Nb", "Ta"}, false, mb);
  Trainer truth(c0, ws);
  const std::vector<double> w = {0.11, -0.07, 0.035, 0.02, -0.013, 0.008,
                                 -0.004};
  StructureDB db;
  for (std::size_t s = 0; s < n_nb.size(); ++s) {
    Structure st = tanb_cell(n_nb[s], seed0 + static_cast<unsigned>(s));
    NNFinder().calc(st, kRcut);
    const StDescriptors d = truth.dc.calc(st);
    double e = -7.6 * count_of(st, kNb) - 8.9 * count_of(st, kTa) +
               0.01 * std::sin(1.7 * double(s) + 0.3);
    for (std::size_t a = 0; a < d.naed(); ++a)
      for (std::size_t k = 0; k < d.dim(); ++k) e += w[k] * d.get_aed(a)(k);
    st.energy = e;
    db.add(st);
  }
  db.dbidx = {0, db.size()};
  return db;
}

const std::vector<int> kVaried = {0, 1, 3, 5, 8, 10, 12, 16, 2, 7, 9, 4,
                                  14, 6};

/* A potential as `tadah predict` loads it: written as a file and read
 * back, then DC_Selector, DescriptorsCalc, the function and the model. */
struct Loaded {
  Context ctx;
  DC_Selector dcs;
  DescriptorsCalc<> dc;
  std::unique_ptr<DM_Function_Base> fb;
  std::unique_ptr<M_Tadah_Base> model;
  explicit Loaded(const Context &pot)
      : ctx(reload(pot)), dcs(ctx),
        dc(ctx, *dcs.d2b, *dcs.dmb, *dcs.c2b, *dcs.cmb),
        fb(tadah::core::factory<DM_Function_Base, const Context &>(
            ctx.get<std::string>("MODEL", 1), ctx)),
        model(tadah::core::factory<M_Tadah_Base, DM_Function_Base &,
                                   Context &>(
            ctx.get<std::string>("MODEL", 0), *fb, ctx)) {}
  static Context reload(const Context &pot) {
    std::ostringstream os;
    pot.write_potential(os, 18);
    std::istringstream in(os.str());
    return tadah::core::CoreUserConfigParser().parse(in);
  }
  StructureDB predict(StructureDB db) {
    NNFinder().calc(db, kRcut);
    return model->predict(ctx, db, dc);
  }
};

std::vector<double> stored_doubles(const Context &c, const std::string &key) {
  std::vector<double> v;
  if (!c.has_stored(key)) return v;
  for (std::size_t i = 0; i < c.size(key); ++i)
    v.push_back(c.get<double>(key, i));
  return v;
}

/* Compares the BIAS true descriptors `d1` (with the intercept column of
 * each atom in `col`) with the BIAS false descriptors `d0`: the intercept
 * block one-hot, the rest equal bit for bit; force descriptors too when
 * present. Returns the number of entries that differ (first one in msg). */
std::size_t layout_mismatches(const StDescriptors &d1, const StDescriptors &d0,
                              std::size_t n_int, const std::vector<int> &col,
                              std::string &msg) {
  std::size_t bad = 0;
  auto note = [&](const std::string &m) {
    if (!bad++) msg = m;
  };
  if (d1.dim() != d0.dim() + n_int) {
    note("dim " + std::to_string(d1.dim()) + " != " +
         std::to_string(d0.dim()) + " + " + std::to_string(n_int));
    return bad;
  }
  for (std::size_t a = 0; a < d1.naed(); ++a) {
    const auto &e1 = d1.get_aed(a);
    const auto &e0 = d0.get_aed(a);
    for (std::size_t k = 0; k < n_int; ++k)
      if (e1(k) != (static_cast<int>(k) == col[a] ? 1.0 : 0.0))
        note("atom " + std::to_string(a) + " intercept column " +
             std::to_string(k) + " = " + std::to_string(e1(k)));
    for (std::size_t k = 0; k < d0.dim(); ++k)
      if (e1(n_int + k) != e0(k))
        note("atom " + std::to_string(a) + " aed(" +
             std::to_string(n_int + k) + ")");
  }
  if (d1.fd.size() != d0.fd.size()) {
    note("force descriptors of different shape");
    return bad;
  }
  for (std::size_t a = 0; a < d1.fd.size(); ++a)
    for (std::size_t jj = 0; jj < d1.fd[a].size(); ++jj)
      for (std::size_t x = 0; x < 3; ++x) {
        for (std::size_t k = 0; k < n_int; ++k)
          if (d1.fd[a][jj](k, x) != 0.0)
            note("fd of atom " + std::to_string(a) + " intercept row " +
                 std::to_string(k));
        for (std::size_t k = 0; k < d0.dim(); ++k)
          if (d1.fd[a][jj](n_int + k, x) != d0.fd[a][jj](k, x))
            note("fd of atom " + std::to_string(a) + ", neighbour " +
                 std::to_string(jj) + ", row " + std::to_string(n_int + k) +
                 ", direction " + std::to_string(x));
      }
  return bad;
}

std::vector<int> intercept_columns(const Structure &st,
                                   const std::vector<int> &Z_order) {
  std::vector<int> col;
  for (const Atom &a : st.atoms) {
    int c = -1;
    for (std::size_t k = 0; k < Z_order.size(); ++k)
      if (Z_order[k] == a.Z) c = static_cast<int>(k);
    col.push_back(c);
  }
  return col;
}

std::string refusal_of(const std::function<void()> &f) {
  try {
    f();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "<no exception>";
}

}  // namespace

// Pins: a Trainer with ATOM Nb Ta and BIAS true (D2_BP, 3 functions, and
// DM_EAD, 4) has DSIZE 9 = 2 + 3 + 4, the DC_Selector size (what
// pair_style tadah builds); the two-body block starts at column 2 and the
// many-body block at 5; DescriptorsCalc::layout() has an intercept block of
// 2 columns at offset 0, then D2_BP at 2 and DM_EAD at 5. With ATOM Ta
// (one species) DSIZE is 8 and the blocks start at 1 and 4, as before.
// Before the fix: DSIZE 8 against the DC_Selector's 9, the two-body block
// at column 1 and the many-body block at 4 (common_constructor overwrote
// the DC_Selector offsets with one intercept column), layout's intercept
// block 1 column; the one-species section passes.
TEST_CASE("DescriptorsCalc BIAS true: one intercept column per ATOM species "
          "in DSIZE and the column layout",
          "[descriptors_calc][bias]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  SECTION("ATOM Nb Ta") {
    Context c = make_context({"Nb", "Ta"}, true, true);
    Trainer tr(c, ws);
    CHECK(tr.DCS.bias == 2);
    CHECK(tr.DCS.descriptor_size() == 9);
    CHECK(tr.context.get<std::size_t>("DSIZE") == 9);
    CHECK(tr.dc.d2.get_fidx() == 2);
    CHECK(tr.dc.dm.get_fidx() == 5);
    const auto L = tr.dc.layout(tr.context);
    REQUIRE(L.blocks.size() == 3);
    CHECK(L.blocks[0].family == "bias");
    CHECK(L.blocks[0].col_offset == 0);
    CHECK(L.blocks[0].col_count == 2);
    CHECK(L.blocks[1].col_offset == 2);
    CHECK(L.blocks[1].col_count == 3);
    CHECK(L.blocks[2].col_offset == 5);
    CHECK(L.blocks[2].col_count == 4);
  }
  SECTION("ATOM Ta") {
    Context c = make_context({"Ta"}, true, true);
    Trainer tr(c, ws);
    CHECK(tr.DCS.descriptor_size() == 8);
    CHECK(tr.context.get<std::size_t>("DSIZE") == 8);
    CHECK(tr.dc.d2.get_fidx() == 1);
    CHECK(tr.dc.dm.get_fidx() == 4);
    const auto L = tr.dc.layout(tr.context);
    REQUIRE(L.blocks.size() == 3);
    CHECK(L.blocks[0].col_count == 1);
  }
}

// Pins: on a periodic Nb-Ta cell (6 Nb, 10 Ta) with FORCE true, the BIAS
// true descriptor of every atom is a one-hot intercept block (Nb: column 0,
// Ta: column 1, in ATOM order) followed by the BIAS false descriptor, bit
// for bit, for ATOM Nb Ta and for ATOM Ta Nb; its force descriptors have
// zero intercept rows and the BIAS false rows after them (the two-body rows
// expanded to x, y and z), and so do the stress descriptors (STRESS true).
// Before the fix: the descriptor had 1 + 7 entries (dim 8 != 7 + 2): one
// intercept column, 1.0 for Nb and Ta atoms alike.
TEST_CASE("DescriptorsCalc BIAS true: each atom's intercept is the column of "
          "its species and the BIAS false descriptor follows",
          "[descriptors_calc][bias]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  Structure st = tanb_cell(6, 11);
  NNFinder().calc(st, kRcut);
  REQUIRE(count_of(st, kNb) == 6);
  for (const char *fs : {"FORCE", "STRESS"}) {
    for (const auto &atoms : std::vector<std::vector<std::string>>{
             {"Nb", "Ta"}, {"Ta", "Nb"}}) {
      DYNAMIC_SECTION(fs << " true, ATOM " << atoms[0] << " " << atoms[1]) {
        Context c1 = make_context(atoms, true, true);
        Context c0 = make_context(atoms, false, true);
        set(c1, fs, "true");
        set(c0, fs, "true");
        Trainer t1(c1, ws), t0(c0, ws);
        const StDescriptors d1 = t1.dc.calc(st);
        const StDescriptors d0 = t0.dc.calc(st);
        REQUIRE(d0.dim() == 7);
        REQUIRE_FALSE(d1.fd.empty());
        const std::vector<int> order = {
            PeriodicTable::find_by_symbol(atoms[0]).Z,
            PeriodicTable::find_by_symbol(atoms[1]).Z};
        std::string msg;
        const std::size_t bad =
            layout_mismatches(d1, d0, 2, intercept_columns(st, order), msg);
        INFO("first difference: " << msg);
        CHECK(bad == 0);
      }
    }
  }
}

// Pins: with DIMER true (two molecules, Nb-Ta and Ta-Nb, four atoms) the
// intercept of each atom is the column of its own species (Nb 0, Ta 1)
// and the BIAS false DIMER descriptor follows bit for bit; with ATOM Ta
// (Ta2 molecules) one intercept column of 1.0, as before.
// Before the fix: one intercept column (dim 8 != 7 + 2), 1.0 for every
// atom; the one-species section passes.
TEST_CASE("DescriptorsCalc BIAS true with DIMER: each atom's intercept is "
          "the column of its own species",
          "[descriptors_calc][bias][dimer]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  auto dimer_context = [](const std::vector<std::string> &atoms, bool bias) {
    Context c = make_context(atoms, bias, true, 6.0);
    c.remove("DIMER");
    c.add("DIMER", "true");
    c.add("DIMER", 1.1);
    c.add("DIMER", "true");
    return c;
  };
  auto pair = [](const std::vector<std::string> &el) {
    Structure st;
    st.cell.load(40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0);
    const double c = 20.0, h = 0.55, d = 3.4;
    const double u[3] = {0.3, 0.1, std::sqrt(1.0 - 0.09 - 0.01)};
    const double p[4][3] = {{c, c, c - h},
                            {c, c, c + h},
                            {c + d - h * u[0], c - h * u[1], c - h * u[2]},
                            {c + d + h * u[0], c + h * u[1], c + h * u[2]}};
    for (int a = 0; a < 4; ++a)
      st.add_atom(Atom(PeriodicTable::find_by_symbol(el[a]), p[a][0], p[a][1],
                       p[a][2], 0.0, 0.0, 0.0));
    return st;
  };
  SECTION("ATOM Nb Ta, molecules Nb-Ta and Ta-Nb") {
    Context c1 = dimer_context({"Nb", "Ta"}, true);
    Context c0 = dimer_context({"Nb", "Ta"}, false);
    Trainer t1(c1, ws), t0(c0, ws);
    const Structure st = pair({"Nb", "Ta", "Ta", "Nb"});
    const StDescriptors d1 = t1.dc.calc(st), d0 = t0.dc.calc(st);
    std::string msg;
    const std::size_t bad = layout_mismatches(
        d1, d0, 2, intercept_columns(st, {kNb, kTa}), msg);
    INFO("first difference: " << msg);
    CHECK(bad == 0);
  }
  SECTION("ATOM Ta") {
    Context c1 = dimer_context({"Ta"}, true);
    Context c0 = dimer_context({"Ta"}, false);
    Trainer t1(c1, ws), t0(c0, ws);
    const Structure st = pair({"Ta", "Ta", "Ta", "Ta"});
    const StDescriptors d1 = t1.dc.calc(st), d0 = t0.dc.calc(st);
    std::string msg;
    const std::size_t bad =
        layout_mismatches(d1, d0, 1, intercept_columns(st, {kTa}), msg);
    INFO("first difference: " << msg);
    CHECK(bad == 0);
  }
}

// Pins: with a D2_mJoin two-body descriptor (D2_BP of 3 and of 2
// functions) and DM_EAD, on the periodic Nb-Ta cell with FORCE true and
// with STRESS true, ATOM Nb Ta, the BIAS true descriptor of every atom is
// its one-hot intercept block followed by the BIAS false descriptor, bit
// for bit, and so are its force (or stress) descriptors: zero intercept
// rows, then the BIAS false rows, the two-body rows expanded to x, y and z.
// Before the fix: D2_mJoin kept its own first index 0 (its set_fidx placed
// only its children), and DescriptorsCalc expands the two-body rows from
// d2.get_fidx(): rows 0 to 4 were expanded instead of 2 to 6, so the last
// two two-body rows (5 and 6) kept the radial derivative, unscaled, in x
// and 0 in y and z (2496 entries differed, the first fd row 5, x).
TEST_CASE("DescriptorsCalc BIAS true with a D2_mJoin: the two-body force "
          "rows follow the intercept columns",
          "[descriptors_calc][bias]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  Structure st = tanb_cell(6, 11);
  NNFinder().calc(st, kRcut);
  auto mjoin_context = [](bool bias) {
    Context c = make_context({"Nb", "Ta"}, bias, true);
    for (const char *k : {"TYPE2B", "CGRID2B", "SGRID2B", "RCTYPE2B",
                          "RCUT2B"})
      c.remove(k);
    c.add("TYPE2B", "D2_mJoin");
    for (const std::string &n : {"3", "2"}) {
      c.add("TYPE2B", "D2_BP");
      c.add("TYPE2B", n);
      c.add("TYPE2B", n);
      c.add("TYPE2B", "*");
      c.add("TYPE2B", "*");
      c.add("RCTYPE2B", "Cut_Cos");
      c.add("RCUT2B", kRcut);
    }
    c.add("CGRID2B", "2.4 3.1 3.9 2.8 3.5");
    c.add("SGRID2B", "1.0 1.2 0.9 1.1 0.8");
    return c;
  };
  for (const char *fs : {"FORCE", "STRESS"}) {
    DYNAMIC_SECTION(fs << " true") {
      Context c1 = mjoin_context(true);
      Context c0 = mjoin_context(false);
      set(c1, fs, "true");
      set(c0, fs, "true");
      Trainer t1(c1, ws), t0(c0, ws);
      const StDescriptors d1 = t1.dc.calc(st);
      const StDescriptors d0 = t0.dc.calc(st);
      REQUIRE(d0.dim() == 5 + 4);
      REQUIRE_FALSE(d1.fd.empty());
      std::string msg;
      const std::size_t bad = layout_mismatches(
          d1, d0, 2, intercept_columns(st, {kNb, kTa}), msg);
      INFO("first difference: " << msg);
      CHECK(bad == 0);
    }
  }
}

// Pins: with ATOM Ta and BIAS true (one species) the descriptor is a 1.0
// followed by the BIAS false descriptor, bit for bit, with zero intercept
// rows in the force descriptors, and DSIZE is 1 + the descriptor size:
// the single-species layout is unchanged.
// Before the fix: passes as well (a pin of the single-species layout,
// which the per-species layout must reproduce exactly).
TEST_CASE("DescriptorsCalc BIAS true with one species: the layout is "
          "unchanged",
          "[descriptors_calc][bias]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  Structure st = bcc_cell(std::vector<std::string>(16, "Ta"), 5);
  NNFinder().calc(st, kRcut);
  Context c1 = make_context({"Ta"}, true, true);
  Context c0 = make_context({"Ta"}, false, true);
  set(c1, "FORCE", "true");
  set(c0, "FORCE", "true");
  Trainer t1(c1, ws), t0(c0, ws);
  CHECK(t1.context.get<std::size_t>("DSIZE") ==
        t0.context.get<std::size_t>("DSIZE") + 1);
  const StDescriptors d1 = t1.dc.calc(st), d0 = t0.dc.calc(st);
  std::string msg;
  const std::size_t bad =
      layout_mismatches(d1, d0, 1, intercept_columns(st, {kTa}), msg);
  INFO("first difference: " << msg);
  CHECK(bad == 0);
}

// Pins: with BIAS true an atom of an element that ATOM does not list (W in
// an ATOM Nb Ta potential) has no intercept column: DescriptorsCalc::calc
// of the structure and of a StructureDB holding it throw
// std::runtime_error naming W and ATOM Nb Ta (DIMER too). With BIAS false
// the structure is computed (no intercept needed).
// Before the fix: every call computed descriptors (one shared intercept,
// 1.0 for the W atom too): no exception.
TEST_CASE("DescriptorsCalc BIAS true refuses an atom of an element not in "
          "ATOM, naming it",
          "[descriptors_calc][bias]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  std::vector<std::string> sites(16, "Ta");
  sites[3] = "Nb";
  sites[7] = "W";
  Structure st = bcc_cell(sites, 3);
  NNFinder().calc(st, kRcut);
  Context c1 = make_context({"Nb", "Ta"}, true);
  Trainer t1(c1, ws);
  const std::string m = refusal_of([&] { (void)t1.dc.calc(st); });
  INFO(m);
  CHECK(m.find("W") != std::string::npos);
  CHECK(m.find("ATOM Nb Ta") != std::string::npos);
  StructureDB db;
  db.add(bcc_cell(std::vector<std::string>(16, "Ta"), 4));
  db.add(st);
  NNFinder().calc(db, kRcut);
  const std::string mdb = refusal_of([&] { (void)t1.dc.calc(db); });
  INFO(mdb);
  CHECK(mdb.find("W") != std::string::npos);
  CHECK(mdb.find("ATOM Nb Ta") != std::string::npos);
  Context c0 = make_context({"Nb", "Ta"}, false);
  Trainer t0(c0, ws);
  CHECK_NOTHROW(t0.dc.calc(st));
  // DIMER: four atoms, one of them W.
  Context cd = make_context({"Nb", "Ta"}, true);
  cd.remove("DIMER");
  cd.add("DIMER", "true");
  cd.add("DIMER", 1.1);
  cd.add("DIMER", "true");
  Trainer td(cd, ws);
  Structure dim;
  dim.cell.load(40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0);
  const char *el[4] = {"Nb", "Ta", "W", "Nb"};
  const double x[4] = {20.0, 21.1, 23.4, 24.5};
  for (int a = 0; a < 4; ++a)
    dim.add_atom(Atom(PeriodicTable::find_by_symbol(el[a]), x[a], 20.0, 20.0,
                      0.0, 0.0, 0.0));
  const std::string md = refusal_of([&] { (void)td.dc.calc(dim); });
  INFO(md);
  CHECK(md.find("W") != std::string::npos);
  CHECK(md.find("ATOM Nb Ta") != std::string::npos);
}

// Pins: trained on varied Nb-Ta compositions at LAMBDA 0, M_BLR BF_Linear
// and M_KRR Kern_Linear with BIAS true predict the same energies on other
// cells with NORM true as with NORM false (1e-9 relative): the NORM weight
// conversion moves the mean correction onto every species' intercept, and
// the potential written and read back predicts the same.
// Before the fix: NORM true predicted +76810 eV against -126 eV with
// M_BLR (the normaliser and the conversion took the first descriptor
// column for the second species' intercept) and -8.1e5 eV with M_KRR (the
// normaliser left that column unnormalised, the M_KRR conversion divided
// its weight by its standard deviation).
TEST_CASE("BIAS true with two species: NORM true predicts what NORM false "
          "does",
          "[descriptors_calc][bias][norm]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB db = make_db(kVaried, true, 100);
  StructureDB test;
  for (int n : {4, 11, 15}) test.add(tanb_cell(n, 500 + n));
  for (const auto &model : std::vector<std::vector<std::string>>{
           {"M_BLR", "BF_Linear"}, {"M_KRR", "Kern_Linear"}}) {
    DYNAMIC_SECTION(model[0] << " " << model[1]) {
      std::vector<std::vector<double>> e(2);
      for (int norm = 0; norm < 2; ++norm) {
        Context c = make_context({"Nb", "Ta"}, true, true);
        set_model(c, model[0], model[1]);
        set(c, "NORM", norm ? "true" : "false");
        Trainer tr(c, ws);
        tr.train(db);
        Loaded pot(tr.get_param_file());
        const StructureDB pred = pot.predict(test);
        for (std::size_t s = 0; s < test.size(); ++s)
          e[norm].push_back(pred(s).energy);
      }
      for (std::size_t s = 0; s < test.size(); ++s) {
        CAPTURE(s, e[0][s], e[1][s]);
        CHECK(e[1][s] == Approx(e[0][s]).epsilon(1e-9));
      }
    }
  }
}

namespace {

/* Solves A x = b (A symmetric positive definite, small) by Gaussian
 * elimination with partial pivoting. */
std::vector<double> solve(std::vector<std::vector<double>> A,
                          std::vector<double> b) {
  const std::size_t n = b.size();
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < n; ++i)
      if (std::fabs(A[i][k]) > std::fabs(A[p][k])) p = i;
    std::swap(A[k], A[p]);
    std::swap(b[k], b[p]);
    for (std::size_t i = k + 1; i < n; ++i) {
      const double f = A[i][k] / A[k][k];
      for (std::size_t j = k; j < n; ++j) A[i][j] -= f * A[k][j];
      b[i] -= f * b[k];
    }
  }
  std::vector<double> x(n);
  for (std::size_t i = n; i-- > 0;) {
    double s = b[i];
    for (std::size_t j = i + 1; j < n; ++j) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  return x;
}

}  // namespace

// Pins: with ATOM Nb Ta, BIAS true and LAMBDA 0.5, the trained weights are
// the ridge solution that leaves exactly the two intercepts unpenalised:
// (Phi^T Phi + 0.5 diag(0, 0, 1, ..., 1)) w = Phi^T T, with Phi built here
// from the species counts and the summed BIAS false descriptors (1e-8
// relative).
// Before the fix: 1 + 3 weights instead of 2 + 3 (one shared intercept;
// the ridge left the first descriptor weight unpenalised as the second
// species' intercept).
TEST_CASE("BIAS true with two species: LAMBDA penalises every descriptor "
          "weight and no intercept",
          "[descriptors_calc][bias][lambda]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB db = make_db(kVaried, false, 200);
  const double lambda = 0.5;
  Context c = make_context({"Nb", "Ta"}, true);
  set(c, "LAMBDA", "0.5");
  Trainer tr(c, ws);
  tr.train(db);
  const auto &w = tr.model->get_weights();

  Context c0 = make_context({"Nb", "Ta"}, false);
  Trainer t0(c0, ws);
  const std::size_t D = 3, n = 2 + D;
  std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
  std::vector<double> b(n, 0.0);
  for (std::size_t s = 0; s < db.size(); ++s) {
    const StDescriptors d = t0.dc.calc(db(s));
    std::vector<double> row(n, 0.0);
    row[0] = count_of(db(s), kNb);
    row[1] = count_of(db(s), kTa);
    for (std::size_t a = 0; a < d.naed(); ++a)
      for (std::size_t k = 0; k < D; ++k) row[2 + k] += d.get_aed(a)(k);
    for (std::size_t i = 0; i < n; ++i) {
      b[i] += row[i] * db(s).energy;
      for (std::size_t j = 0; j < n; ++j) A[i][j] += row[i] * row[j];
    }
  }
  for (std::size_t k = 2; k < n; ++k) A[k][k] += lambda;
  const std::vector<double> ref = solve(A, b);
  REQUIRE(w.size() == n);
  for (std::size_t i = 0; i < n; ++i) {
    CAPTURE(i, w(i), ref[i]);
    CHECK(w(i) == Approx(ref[i]).epsilon(1e-8).margin(1e-10));
  }
}

// Pins: NORM true with BIAS false is refused when a linear model is
// trained (M_KRR Kern_Linear and M_BLR BF_Linear): the conversion of the
// normalised weights needs an intercept column to absorb the mean
// correction (std::runtime_error naming NORM and BIAS).
// Before the fix: both trained; M_KRR folded the correction into the first
// descriptor weight (column 0) and M_BLR dropped it: silently wrong
// potentials (M_KRR NORM true: +0.97 against -8.13 eV/atom on Ta data).
TEST_CASE("NORM true with BIAS false is refused for linear models",
          "[descriptors_calc][bias][norm]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB db = make_db(kVaried, false, 300);
  for (const auto &model : std::vector<std::vector<std::string>>{
           {"M_KRR", "Kern_Linear"}, {"M_BLR", "BF_Linear"}}) {
    DYNAMIC_SECTION(model[0] << " " << model[1]) {
      Context c = make_context({"Nb", "Ta"}, false);
      set_model(c, model[0], model[1]);
      set(c, "NORM", "true");
      Trainer tr(c, ws);
      const std::string m = refusal_of([&] { tr.train(db); });
      INFO(m);
      CHECK(m.find("NORM") != std::string::npos);
      CHECK(m.find("BIAS") != std::string::npos);
    }
  }
}

// Pins: a potential with ATOM Nb Ta and BIAS true whose WEIGHTS fit one
// shared intercept column (1 + 3 for D2_BP 3, DSIZE 2 + 3) builds, and is
// refused when it predicts (tadah predict's path), with a message naming
// WEIGHTS, DSIZE, BIAS and ATOM ("has 4 WEIGHTS", "5 components (DSIZE)")
// and saying to retrain ("since 1.3.0-beta.1" pair_style tadah used one
// intercept column per species with it); likewise an M_KRR Kern_RBF
// potential whose basis vectors have 1 + 3 entries ("of 4 values each").
// Before the fix: DSIZE was 4, so both predicted with the shared
// intercept (no exception; DSIZE 4 != 5). Before the one-guard rewording
// (review): the messages said "WEIGHTS has 4 values", "DSIZE 5" and "4
// entries", and the hint named no version.
TEST_CASE("A multi-species BIAS potential with one shared intercept column "
          "is refused when it predicts",
          "[descriptors_calc][bias][legacy]") {
  PeriodicTable::initialize();
  StructureDB test;
  test.add(tanb_cell(5, 77));
  SECTION("M_BLR BF_Linear") {
    Context pot = make_context({"Nb", "Ta"}, true);
    for (double w : {-8.2, 0.1, -0.05, 0.02}) pot.add("WEIGHTS", w);
    std::unique_ptr<Loaded> l;
    REQUIRE_NOTHROW(l.reset(new Loaded(pot)));
    const std::string m = refusal_of([&] { (void)l->predict(test); });
    INFO(m);
    CHECK(l->ctx.get<std::size_t>("DSIZE") == 5);
    CHECK(m.find("has 4 WEIGHTS") != std::string::npos);
    CHECK(m.find("5 components (DSIZE)") != std::string::npos);
    CHECK(m.find("since 1.3.0-beta.1") != std::string::npos);
    CHECK(m.find("BIAS true") != std::string::npos);
    CHECK(m.find("ATOM Nb Ta") != std::string::npos);
    CHECK(m.find("one intercept column shared by all species") !=
          std::string::npos);
    CHECK(m.find("retrain") != std::string::npos);
  }
  SECTION("M_KRR Kern_RBF") {
    Context pot = make_context({"Nb", "Ta"}, true);
    set_model(pot, "M_KRR", "Kern_RBF");
    pot.add("MODEL", 1);
    pot.add("MPARAMS", 1.5);
    pot.add("SBASIS", 3);
    for (int v = 0; v < 3; ++v)
      for (int k = 0; k < 4; ++k) pot.add("BASIS", k == 0 ? 1.0 : 0.3 * v + k);
    for (double w : {0.5, -0.2, 0.1}) pot.add("WEIGHTS", w);
    std::unique_ptr<Loaded> l;
    REQUIRE_NOTHROW(l.reset(new Loaded(pot)));
    const std::string m = refusal_of([&] { (void)l->predict(test); });
    INFO(m);
    CHECK(l->ctx.get<std::size_t>("DSIZE") == 5);
    CHECK(m.find("BASIS") != std::string::npos);
    CHECK(m.find("of 4 values each") != std::string::npos);
    CHECK(m.find("one intercept column shared by all species") !=
          std::string::npos);
  }
}

// Pins: a training configuration that is such a potential (ATOM Nb Ta,
// BIAS true, NORM true, 1 + 3 stale WEIGHTS, NMEAN and NSTDEV) trains, for
// M_BLR BF_Linear and M_KRR Kern_Linear: the potential it gives has 2 + 3
// WEIGHTS, NMEAN and NSTDEV (no stale value kept) and predicts without a
// refusal.
// Before the fix: M_BLR wrote 4 WEIGHTS and appended the 4 new NMEAN and
// NSTDEV values to the 4 stale ones (8 each); M_KRR threw "This object is
// already trained!" (MODELS, fixed in 88ec1bb) and, with that fixed, wrote
// 8 NMEAN and NSTDEV values.
TEST_CASE("A stale multi-species BIAS potential used as a training "
          "configuration trains into the per-species layout",
          "[descriptors_calc][bias][legacy]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB db = make_db(kVaried, false, 400);
  StructureDB test;
  test.add(tanb_cell(9, 99));
  for (const auto &model : std::vector<std::vector<std::string>>{
           {"M_BLR", "BF_Linear"}, {"M_KRR", "Kern_Linear"}}) {
    DYNAMIC_SECTION(model[0] << " " << model[1]) {
      Context c = make_context({"Nb", "Ta"}, true);
      set_model(c, model[0], model[1]);
      set(c, "NORM", "true");
      for (double w : {-8.2, 0.1, -0.05, 0.02}) c.add("WEIGHTS", w);
      for (double v : {1.0, 2.0, 3.0, 4.0}) c.add("NMEAN", v);
      for (double v : {0.0, 1.0, 1.5, 2.0}) c.add("NSTDEV", v);
      Trainer tr(c, ws);
      REQUIRE_NOTHROW(tr.train(db));
      const Context pot = tr.get_param_file();
      CHECK(stored_doubles(pot, "WEIGHTS").size() == 5);
      CHECK(stored_doubles(pot, "NMEAN").size() == 5);
      CHECK(stored_doubles(pot, "NSTDEV").size() == 5);
      Loaded l(pot);
      CHECK_NOTHROW(l.predict(test));
    }
  }
}

// Pins: with BIAS true the energy-weighted training compositions must
// determine the per-species intercepts. Every structure 8 Nb + 8 Ta (the
// species counts have rank 1 for 2 intercepts) is refused before the fit,
// for OALGO 1, 2, 3 and 4, with std::runtime_error naming the species, the
// rank and the remedies (BIAS false, varied compositions, FIXINDEX);
// pinning the Nb intercept with FIXINDEX 1 / FIXWEIGHT -7.6 makes the same
// data train; varied compositions train; with ATOM Ta (one intercept) data
// without any energy weight (EWEIGHT 0, forces fitted) are refused too.
// Before the fix: the fixed-stoichiometry data trained with an arbitrary
// split of the energy between the species (one shared intercept: no
// refusal); the EWEIGHT 0 fit trained with an undetermined intercept.
TEST_CASE("BIAS true refuses training data whose compositions cannot "
          "determine the per-species intercepts",
          "[descriptors_calc][bias][composition]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB fixed = make_db(std::vector<int>(10, 8), false, 600);
  for (int oalgo : {1, 2, 3, 4}) {
    DYNAMIC_SECTION("fixed 8 Nb + 8 Ta, OALGO " << oalgo) {
      Context c = make_context({"Nb", "Ta"}, true);
      set(c, "OALGO", std::to_string(oalgo));
      Trainer tr(c, ws);
      const std::string m = refusal_of([&] { tr.train(fixed); });
      INFO(m);
      CHECK(m.find("cannot determine") != std::string::npos);
      CHECK(m.find("Nb Ta") != std::string::npos);
      CHECK(m.find("rank 1") != std::string::npos);
      CHECK(m.find("BIAS false") != std::string::npos);
      CHECK(m.find("FIXINDEX") != std::string::npos);
    }
  }
  SECTION("fixed 8 Nb + 8 Ta, the Nb intercept pinned") {
    Context c = make_context({"Nb", "Ta"}, true);
    c.add("FIXINDEX", "1");
    c.add("FIXWEIGHT", -7.6);
    Trainer tr(c, ws);
    CHECK_NOTHROW(tr.train(fixed));
  }
  SECTION("varied compositions") {
    StructureDB varied = make_db(kVaried, false, 700);
    Context c = make_context({"Nb", "Ta"}, true);
    Trainer tr(c, ws);
    CHECK_NOTHROW(tr.train(varied));
  }
  SECTION("ATOM Ta, no energy weight") {
    StructureDB ta;
    for (unsigned s = 0; s < 4; ++s)
      ta.add(bcc_cell(std::vector<std::string>(16, "Ta"), 800 + s));
    ta.dbidx = {0, ta.size()};
    Context c = make_context({"Ta"}, true);
    set(c, "FORCE", "true");
    set(c, "EWEIGHT", "0");
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(ta); });
    INFO(m);
    CHECK(m.find("cannot determine") != std::string::npos);
    CHECK(m.find("energy weight") != std::string::npos);
  }
}

// Pins: FIXINDEX/FIXWEIGHT with NORM true is refused when a model is
// trained (M_BLR BF_Linear and M_KRR Kern_Linear, BIAS true, varied
// compositions, FIXINDEX 1 / FIXWEIGHT -7.6): std::runtime_error naming
// FIXINDEX/FIXWEIGHT and NORM true, the message tadah hpo refuses the
// configuration with at startup. The fit pins the weights of the
// normalised descriptors, and the conversion to raw-descriptor weights
// moves a pinned intercept by the descriptor means times the weights.
// Before the fix: both trained, and the written WEIGHTS[1] was not the
// FIXWEIGHT value (-7.649 against -7.6 on Nb2Ta2 cells), nor its WCOV row
// zero; tadah hpo refused the same file.
TEST_CASE("FIXINDEX/FIXWEIGHT with NORM true is refused when a model is "
          "trained", "[descriptors_calc][bias][norm][fixindex]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB db = make_db(kVaried, false, 350);
  for (const auto &model : std::vector<std::vector<std::string>>{
           {"M_KRR", "Kern_Linear"}, {"M_BLR", "BF_Linear"}}) {
    DYNAMIC_SECTION(model[0] << " " << model[1]) {
      Context c = make_context({"Nb", "Ta"}, true);
      set_model(c, model[0], model[1]);
      set(c, "NORM", "true");
      c.add("FIXINDEX", "1");
      c.add("FIXWEIGHT", -7.6);
      Trainer tr(c, ws);
      const std::string m = refusal_of([&] { tr.train(db); });
      INFO(m);
      CHECK(m.find("FIXINDEX/FIXWEIGHT is not supported with NORM true") !=
            std::string::npos);
      CHECK(m.find("Set NORM false") != std::string::npos);
    }
  }
}

// Pins: the remedies of the composition refusal (every structure 8 Nb +
// 8 Ta, BIAS true) are those the configuration can take: under NORM false
// BIAS false with explicit ESHIFT values or ESHIFT_ATOM, other
// compositions, or FIXINDEX/FIXWEIGHT on the intercepts, and never
// ESHIFT_DBATOM, whose least-squares fit of the species energies is
// singular on the compositions the check refuses (ESHIFT 0 0, with a
// WARNING); under NORM true (BF_Linear, which needs BIAS true there, and
// FIXINDEX refused) other compositions or NORM false, and neither BIAS
// false nor FIXINDEX.
// Before the fix: every refusal said "Use BIAS false (ESHIFT_ATOM,
// ESHIFT_DBATOM or ESHIFT remove a per-species reference energy), ... or
// pin intercepts with FIXINDEX/FIXWEIGHT", under NORM true too.
TEST_CASE("The composition refusal names only the remedies the "
          "configuration can take",
          "[descriptors_calc][bias][composition]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB fixed = make_db(std::vector<int>(10, 8), false, 650);
  SECTION("NORM false") {
    Context c = make_context({"Nb", "Ta"}, true);
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    CHECK(m.find("cannot determine") != std::string::npos);
    CHECK(m.find("ESHIFT_DBATOM") == std::string::npos);
    CHECK(m.find("Use BIAS false with explicit ESHIFT values") !=
          std::string::npos);
    CHECK(m.find("ESHIFT_ATOM") != std::string::npos);
    CHECK(m.find("pin intercepts with FIXINDEX/FIXWEIGHT") !=
          std::string::npos);
  }
  SECTION("NORM true") {
    Context c = make_context({"Nb", "Ta"}, true);
    set(c, "NORM", "true");
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    CHECK(m.find("cannot determine") != std::string::npos);
    CHECK(m.find("FIXINDEX") == std::string::npos);
    CHECK(m.find("BIAS false") == std::string::npos);
    CHECK(m.find("ESHIFT_DBATOM") == std::string::npos);
    CHECK(m.find("add energy-weighted structures of other compositions") !=
          std::string::npos);
    CHECK(m.find("NORM false") != std::string::npos);
  }
}

// Pins: the composition check reads FIXINDEX as the trainer does
// (core::parse_fixindex): on data whose every structure is 8 Nb + 8 Ta
// (ATOM Nb Ta, BIAS true), FIXINDEX given as separate values, '1 3' with
// FIXWEIGHT -7.6 0.0, pins the Nb intercept and the model trains (weight 1
// -7.6, weight 3 0.0); FIXINDEX '2-1' (with two FIXWEIGHT values) is refused
// as a descending range, not as data that cannot determine the intercepts.
// Before the fix: the check read FIXINDEX with the lenient
// core::parse_indices, which dropped '1 3' (the values joined into one
// malformed segment) and expanded '2-1' to nothing, so both were refused
// with "BIAS true: the training data cannot determine the per-species
// energy constants (the intercepts) of ATOM Nb Ta" (rank 1).
TEST_CASE("BIAS true: the composition check reads FIXINDEX as the trainer "
          "does",
          "[descriptors_calc][bias][composition][fixindex]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB fixed = make_db(std::vector<int>(10, 8), false, 600);
  SECTION("separate values pin the Nb intercept") {
    Context c = make_context({"Nb", "Ta"}, true);
    c.add("FIXINDEX", "1 3");
    c.add("FIXWEIGHT", -7.6);
    c.add("FIXWEIGHT", 0.0);
    REQUIRE(c.size("FIXINDEX") == 2);
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    REQUIRE(m == "<no exception>");
    CHECK(tr.model->get_weights()(0) == -7.6);
    CHECK(tr.model->get_weights()(2) == 0.0);
  }
  SECTION("a descending range is refused as such") {
    Context c = make_context({"Nb", "Ta"}, true);
    c.add("FIXINDEX", "2-1");
    c.add("FIXWEIGHT", -7.6);
    c.add("FIXWEIGHT", -8.9);
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    CHECK(m.find("FIXINDEX: index '2-1' is a descending range") !=
          std::string::npos);
    CHECK(m.find("cannot determine") == std::string::npos);
  }
}

// Pins: the composition check bounds FIXINDEX by the weight count of the
// fit (model_weight_count) and refuses an index past it with the fit's own
// message, before the fit, on data whose every structure is 8 Nb + 8 Ta
// (ATOM Nb Ta, BIAS true, D2_BP 3):
//   - M_BLR BF_Linear (5 weights): FIXINDEX 18446744073709551615, 6 and 0
//     (FIXWEIGHT 0.05) are refused as "FIXINDEX: index <i> is outside 1-5
//     (the model has 5 weights)", by Trainer::train (M_BLR::train) and by
//     require_intercepts_determined called as tadah train calls it (the
//     Trainer's context and function, before the fit);
//   - M_KRR Kern_Quadratic (SBASIS 6, its weights those of the basis the
//     fit draws): FIXINDEX 7 as "outside 1-6 (the model has 6 weights)".
// Before the fix: the check read FIXINDEX without the weight count and
// sized a vector<bool> by the largest index (pinned_weight.resize(i)):
// 18446744073709551615 threw std::length_error "vector<bool>::_M_fill_insert"
// (100000000000 asked for 12.5 GB), 6 and 7 passed it and the data were
// refused as "cannot determine the per-species energy constants (the
// intercepts)" (rank 1), so the fit's bound was never reached, and 0 was
// "FIXINDEX: index 0 is not a model weight (the weights are numbered from
// 1)", where the trainer and BIAS false say "outside 1-5".
TEST_CASE("BIAS true: the composition check bounds FIXINDEX by the fit's "
          "weight count",
          "[descriptors_calc][bias][composition][fixindex]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB fixed = make_db(std::vector<int>(10, 8), false, 600);
  for (const char *fi : {"18446744073709551615", "6", "0"}) DYNAMIC_SECTION(
      "M_BLR BF_Linear, FIXINDEX " << fi) {
    const std::string expected = std::string("FIXINDEX: index ") + fi +
                                 " is outside 1-5 (the model has 5 weights)";
    Context c = make_context({"Nb", "Ta"}, true);
    c.add("FIXINDEX", fi);
    c.add("FIXWEIGHT", 0.05);
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    CHECK(m == expected);
    Trainer te(c, ws);
    const std::string me = refusal_of([&] {
      tadah::mlip::require_intercepts_determined(fixed, te.context, *te.fb);
    });
    INFO(me);
    CHECK(me == expected);
  }
  SECTION("M_KRR Kern_Quadratic, FIXINDEX 7") {
    Context c = make_context({"Nb", "Ta"}, true);
    set_model(c, "M_KRR", "Kern_Quadratic");
    c.add("SBASIS", 6);
    c.add("FIXINDEX", "7");
    c.add("FIXWEIGHT", 0.05);
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    CHECK(m == "FIXINDEX: index 7 is outside 1-6 (the model has 6 weights)");
  }
}

// Pins: M_KRR implementation 2, whose solve (K + LAMBDA I) w = t fits the
// targets of its basis vectors (drawn for every kernel, Kern_Linear too),
// is not checked by the composition check, with Kern_Linear as with a
// non-linear kernel, on data whose every structure is 8 Nb + 8 Ta (ATOM Nb
// Ta, BIAS true, D2_BP 3, MODEL M_KRR Kern_Linear 2, SBASIS 5, LAMBDA
// 1e-3): require_intercepts_determined, which tadah train calls before the
// fit, passes without FIXINDEX and with FIXINDEX 99 / FIXWEIGHT 14 (it
// does not read FIXINDEX for implementation 2); Trainer::train refuses
// FIXINDEX 1 / FIXWEIGHT -7.6 and FIXINDEX 99 / FIXWEIGHT 14 with the
// solve's "MODEL M_KRR Kern_Linear 2 cannot pin weights".
// Before the fix: the check skipped implementation 2 for a non-linear
// kernel only: Kern_Linear 2 was refused as data that "cannot determine
// the per-species energy constants" (rank 1), and with FIXINDEX 99 as
// "FIXINDEX: index 99 is outside 1-5 (the model has 5 weights)" (FIXINDEX
// read, against DSIZE, for a solve that cannot pin weights: it fits the 5
// basis vectors' coefficients at once); Trainer::train (M_KRR::train2,
// which calls no check) trained with FIXINDEX 1 and 99 alike, the pins
// ignored.
TEST_CASE("BIAS true: M_KRR implementation 2 is not checked, with "
          "Kern_Linear too",
          "[descriptors_calc][bias][composition][fixindex][krr]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB fixed = make_db(std::vector<int>(10, 8), false, 600);
  auto krr2 = [] {
    Context c = make_context({"Nb", "Ta"}, true);
    c.remove("MODEL");
    c.add("MODEL", "M_KRR");
    c.add("MODEL", "Kern_Linear");
    c.add("MODEL", 2);
    c.add("SBASIS", 5);
    set(c, "LAMBDA", "1e-3");
    return c;
  };
  for (const char *fi : {"", "99"}) DYNAMIC_SECTION(
      "the composition check passes, FIXINDEX '" << fi << "'") {
    Context c = krr2();
    if (*fi) {
      c.add("FIXINDEX", fi);
      c.add("FIXWEIGHT", 14.0);
    }
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] {
      tadah::mlip::require_intercepts_determined(fixed, tr.context, *tr.fb);
    });
    INFO(m);
    CHECK(m == "<no exception>");
  }
  for (const char *fi : {"1", "99"}) DYNAMIC_SECTION(
      "Trainer::train refuses FIXINDEX " << fi) {
    Context c = krr2();
    c.add("FIXINDEX", fi);
    c.add("FIXWEIGHT", std::string(fi) == "1" ? -7.6 : 14.0);
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    CHECK(m.rfind("MODEL M_KRR Kern_Linear 2 cannot pin weights", 0) == 0);
  }
}

// Pins: the composition check takes the intercepts from the model's
// function (Function_Base::intercept_columns) on data whose every structure
// is 8 Nb + 8 Ta (ATOM Nb Ta, BIAS true, D2_BP 3: DSIZE 5):
//   - M_BLR BF_Polynomial2 (15 weights, the products (i,ii), i <= ii): its
//     intercepts are weights 1, (0,0), and 6, (1,1); unpinned the data are
//     refused, the message naming them ("weights 1 (Nb) and 6 (Ta)"); with
//     FIXINDEX 2 (the Nb x Ta product, which is 0 for every atom) they are
//     refused too, weight 2 being no intercept; with FIXINDEX 6 / FIXWEIGHT
//     -8.9 (the Ta constant pinned) the model trains, weight 6 -8.9;
//   - M_KRR Kern_Quadratic (SBASIS 6): refused, its intercepts being the
//     columns of the intercept basis vectors e0 and e1 (a pin: refused
//     before the fix as well);
//   - M_KRR Kern_RBF (SBASIS 6): trains, K(b, x) depending on every
//     component of x (no intercept column, nothing unpenalised to
//     determine).
// Before the fix: the check counted ATOM species whatever the function and
// took FIXINDEX 1 to N for the intercepts: BF_Polynomial2 unpinned was
// refused naming "weights 1 to 2", with FIXINDEX 2 it passed the check and
// trained an arbitrary split of the energy between the species, with
// FIXINDEX 6 it was refused ("rank 1, while 2 intercepts are fitted");
// Kern_RBF was refused.
TEST_CASE("BIAS true: the composition check takes the intercepts from the "
          "function",
          "[descriptors_calc][bias][composition]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB fixed = make_db(std::vector<int>(10, 8), false, 600);
  auto poly2 = [] {
    Context c = make_context({"Nb", "Ta"}, true);
    set_model(c, "M_BLR", "BF_Polynomial2");
    return c;
  };
  SECTION("BF_Polynomial2, no pin") {
    Context c = poly2();
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    CHECK(m.find("cannot determine") != std::string::npos);
    CHECK(m.find("rank 1") != std::string::npos);
    CHECK(m.find("weights 1 (Nb) and 6 (Ta)") != std::string::npos);
  }
  SECTION("BF_Polynomial2, FIXINDEX 2: no intercept") {
    Context c = poly2();
    c.add("FIXINDEX", "2");
    c.add("FIXWEIGHT", 0.0);
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    CHECK(m.find("cannot determine") != std::string::npos);
    CHECK(m.find("rank 1") != std::string::npos);
  }
  SECTION("BF_Polynomial2, FIXINDEX 6: the Ta constant") {
    Context c = poly2();
    c.add("FIXINDEX", "6");
    c.add("FIXWEIGHT", -8.9);
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    REQUIRE(m == "<no exception>");
    REQUIRE(tr.model->get_weights().size() == 15);
    CHECK(tr.model->get_weights()(5) == -8.9);
  }
  SECTION("M_KRR Kern_Quadratic (a pin)") {
    Context c = make_context({"Nb", "Ta"}, true);
    set_model(c, "M_KRR", "Kern_Quadratic");
    c.add("SBASIS", 6);
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    CHECK(m.find("cannot determine") != std::string::npos);
  }
  SECTION("M_KRR Kern_RBF") {
    Context c = make_context({"Nb", "Ta"}, true);
    set_model(c, "M_KRR", "Kern_RBF");
    c.add("SBASIS", 6);
    c.add("MPARAMS", 1.0);
    Trainer tr(c, ws);
    const std::string m = refusal_of([&] { tr.train(fixed); });
    INFO(m);
    CHECK(m == "<no exception>");
  }
}

// Pins: the KRR basis of a model with BIAS true starts with one intercept
// vector per species (ATOM Nb Ta: e0 and e1, then AEDs, each with exactly
// one intercept entry 1.0), for build_random_basis and prep_basis_for_krr
// (whose targets are 0 for the intercept vectors); with one species e0, as
// before.
// Before the fix: only e0 was seeded, the second basis vector was an AED
// (with the one shared intercept, b(0,1) = 1).
TEST_CASE("The KRR basis seeds one intercept vector per species",
          "[descriptors_calc][bias][basis]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  for (const auto &atoms : std::vector<std::vector<std::string>>{
           {"Nb", "Ta"}, {"Ta"}}) {
    DYNAMIC_SECTION("ATOM with " << atoms.size() << " species") {
      Context c = make_context(atoms, true);
      c.add("SBASIS", 6);
      Trainer tr(c, ws);
      StructureDB db;
      for (unsigned s = 0; s < 8; ++s)
        db.add(atoms.size() == 2
                   ? tanb_cell(static_cast<int>(2 * s), 900 + s)
                   : bcc_cell(std::vector<std::string>(16, "Ta"), 900 + s));
      NNFinder().calc(db, kRcut);
      StDescriptorsDB sd = tr.dc.calc(db);
      const std::size_t N = atoms.size();
      const std::size_t dim = sd(0).dim();
      tadah::mlip::Basis<DM_Function_Base &> basis(tr.context);
      for (int prep = 0; prep < 2; ++prep) {
        CAPTURE(prep);
        if (prep)
          basis.prep_basis_for_krr(sd, db);
        else
          basis.build_random_basis(6, sd);
        REQUIRE(basis.b.cols() == 6);
        REQUIRE(basis.b.rows() == dim);
        for (std::size_t v = 0; v < 6; ++v) {
          CAPTURE(v);
          if (v < N) {
            for (std::size_t k = 0; k < dim; ++k)
              CHECK(basis.b(k, v) == (k == v ? 1.0 : 0.0));
            if (prep) CHECK(basis.T(v) == 0.0);
          } else {
            double s = 0.0;
            for (std::size_t k = 0; k < N; ++k) s += basis.b(k, v);
            // an AED (build_random_basis) or a per-atom mean of AEDs of a
            // structure (prep_basis_for_krr): intercepts sum to 1
            CHECK(s == Approx(1.0).epsilon(1e-12));
          }
        }
      }
    }
  }
}

// Pins: Normaliser::learn excludes every intercept column from the
// zero-variance rule (ATOM Nb Ta: a Ta column that is 0 for every atom
// keeps standard deviation 0, as the always-1 Nb column does), and a
// constant descriptor column still gets the largest double.
// Before the fix: `if (bias) b++` excluded only column 0, so the constant
// Ta intercept column got the largest double.
TEST_CASE("Normaliser::learn leaves every intercept column out of the "
          "zero-variance rule",
          "[descriptors_calc][bias][norm]") {
  PeriodicTable::initialize();
  Context c;
  c.add("BIAS", "true");
  c.add("ATOM", "Nb");
  c.add("ATOM", "Ta");
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.addInternal("DSIZE", 4);
  Structure st;
  st.cell.load(20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0);
  for (int a = 0; a < 3; ++a)
    st.add_atom(Atom(tadah::core::Element("H"), 2.0 * a, 0.0, 0.0, 0.0, 0.0,
                     0.0));
  StDescriptors d(st, c);
  for (int a = 0; a < 3; ++a) {
    d.get_aed(a)(0) = 1.0;            // every atom Nb
    d.get_aed(a)(1) = 0.0;            // no Ta
    d.get_aed(a)(2) = 0.5 + a;        // a descriptor
    d.get_aed(a)(3) = 7.0;            // a constant descriptor
  }
  StDescriptorsDB db;
  db.add(d);
  tadah::mlip::Normaliser norm(c);
  norm.learn(db);
  CHECK(norm.std_dev[0] == 0.0);
  CHECK(norm.std_dev[1] == 0.0);
  CHECK(norm.std_dev[2] > 0.0);
  CHECK(norm.std_dev[3] == std::numeric_limits<double>::max());
}

namespace {

/* A potential trained by a Trainer from @p c on @p db, as its file holds
 * it (written and read back). */
Context trained_potential(Context c, StructureDB db) {
  MLIPWorkspaceManager ws;
  Trainer tr(c, ws);
  tr.train(db);
  return Loaded::reload(tr.get_param_file());
}

/* The basis vectors @p vectors (each of the same length) as BASIS values
 * of @p c, with SBASIS their number. */
void set_basis(Context &c, const std::vector<std::vector<double>> &vectors) {
  if (c.has_stored("BASIS")) c.remove("BASIS");
  set(c, "SBASIS", std::to_string(vectors.size()));
  for (const auto &v : vectors)
    for (double x : v) c.add("BASIS", x);
}

}  // namespace

// Pins: a Trainer built from a potential, as `tadah train -c pot.tadah`
// and a `tadah hpo -c` seed build one, reads none of the potential's fit:
// its context has no WEIGHTS, BASIS, NMEAN, NSTDEV or SIGMA. An M_KRR
// Kern_Quadratic potential (ATOM Nb Ta, BIAS true, NORM true, SBASIS 6,
// DSIZE 5) retrained with SBASIS 10 and with SBASIS 7 gives 10 and 7
// WEIGHTS, 10 x 5 and 7 x 5 BASIS values and 5 NMEAN; an M_BLR BF_Linear
// potential (5 WEIGHTS) with MODEL M_KRR Kern_RBF, MPARAMS 1.0 and SBASIS 6
// trains into 6 WEIGHTS; an M_BLR potential fitted with OALGO 3 (WCOV of
// 1 + 5 x 5 values, and WNOISE) retrained with OALGO 1 has no WCOV or
// WNOISE, and retrained with OALGO 3 the WCOV and WNOISE of its own fit
// (26 values, and E with its value: FORCE false; equal to the potential's:
// the same fit of the same data); the Trainer's context has no WCOV or
// WNOISE either.
// Before the fix: the Trainer built the model from the potential's fit:
// for the kernels it threw, "the potential has 6 WEIGHTS but SBASIS 10
// basis vectors (BASIS)", "BASIS has 30 values, which is not a multiple of
// SBASIS 7" and "the potential has WEIGHTS but no basis vectors (SBASIS and
// BASIS)"; the OALGO 1 fit kept the potential's SIGMA (26 values), and the
// OALGO 3 fit appended its own to it (52 values, the potential's first).
// Before WCOV and WNOISE (the covariance was SIGMA, which this case
// checked the same way): no potential had WCOV or WNOISE.
TEST_CASE("A Trainer built from a potential reads none of the potential's "
          "fit",
          "[descriptors_calc][bias][retrain]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  StructureDB db = make_db(kVaried, false, 400);
  const std::vector<std::string> fit_keys = {"WEIGHTS", "BASIS", "NMEAN",
                                             "NSTDEV", "SIGMA", "WCOV",
                                             "WNOISE"};
  auto build = [&](Context cfg, std::unique_ptr<Trainer> &tr) {
    return refusal_of([&] { tr.reset(new Trainer(cfg, ws)); });
  };
  SECTION("M_KRR Kern_Quadratic retrained with another SBASIS") {
    Context c = make_context({"Nb", "Ta"}, true);
    set_model(c, "M_KRR", "Kern_Quadratic");
    set(c, "NORM", "true");
    c.add("SBASIS", 6);
    const Context pot = trained_potential(c, db);
    REQUIRE(stored_doubles(pot, "WEIGHTS").size() == 6);
    REQUIRE(stored_doubles(pot, "BASIS").size() == 30);
    REQUIRE(stored_doubles(pot, "NMEAN").size() == 5);
    for (std::size_t s : std::vector<std::size_t>{10, 7}) {
      CAPTURE(s);
      Context cfg = pot;
      set(cfg, "SBASIS", std::to_string(s));
      std::unique_ptr<Trainer> tr;
      const std::string m = build(cfg, tr);
      INFO(m);
      REQUIRE(m == "<no exception>");
      for (const auto &k : fit_keys) {
        CAPTURE(k);
        CHECK_FALSE(tr->context.has_stored(k));
      }
      REQUIRE_NOTHROW(tr->train(db));
      const Context out = tr->get_param_file();
      CHECK(out.get<std::size_t>("SBASIS") == s);
      CHECK(stored_doubles(out, "WEIGHTS").size() == s);
      CHECK(stored_doubles(out, "BASIS").size() == 5 * s);
      CHECK(stored_doubles(out, "NMEAN").size() == 5);
    }
  }
  SECTION("an M_BLR BF_Linear potential trained as M_KRR Kern_RBF") {
    const Context lin = trained_potential(make_context({"Nb", "Ta"}, true), db);
    REQUIRE(stored_doubles(lin, "WEIGHTS").size() == 5);
    Context cfg = lin;
    set_model(cfg, "M_KRR", "Kern_RBF");
    set(cfg, "MPARAMS", "1.0");
    set(cfg, "SBASIS", "6");
    std::unique_ptr<Trainer> tr;
    const std::string m = build(cfg, tr);
    INFO(m);
    REQUIRE(m == "<no exception>");
    REQUIRE_NOTHROW(tr->train(db));
    CHECK(stored_doubles(tr->get_param_file(), "WEIGHTS").size() == 6);
  }
  SECTION("the covariance of an OALGO 3 potential") {
    Context c = make_context({"Nb", "Ta"}, true);
    set(c, "OALGO", "3");
    const Context pot = trained_potential(c, db);
    const std::vector<double> wcov = stored_doubles(pot, "WCOV");
    REQUIRE(wcov.size() == 26);
    REQUIRE(wcov[0] == 5.0);
    REQUIRE(pot.size("WNOISE") == 2);
    REQUIRE(pot.get<std::string>("WNOISE", 0) == "E");
    for (int oalgo : {1, 3}) {
      CAPTURE(oalgo);
      Context cfg = pot;
      set(cfg, "OALGO", std::to_string(oalgo));
      Trainer tr(cfg, ws);
      for (const auto &k : fit_keys) {
        CAPTURE(k);
        CHECK_FALSE(tr.context.has_stored(k));
      }
      tr.train(db);
      const Context out_pot = tr.get_param_file();
      const std::vector<double> out = stored_doubles(out_pot, "WCOV");
      if (oalgo == 1) {
        CHECK(out.empty());
        CHECK_FALSE(out_pot.has_stored("WNOISE"));
      } else {
        REQUIRE(out.size() == wcov.size());
        for (std::size_t i = 0; i < out.size(); ++i)
          CHECK(out[i] == Approx(wcov[i]).epsilon(1e-12).margin(1e-300));
        REQUIRE(out_pot.size("WNOISE") == 2);
        CHECK(out_pot.get<double>("WNOISE", 1) ==
              Approx(pot.get<double>("WNOISE", 1)).epsilon(1e-12));
      }
      CHECK_FALSE(out_pot.has_stored("SIGMA"));
    }
  }
}

// Pins: the FIXINDEX WARNING of a non-linear kernel (ATOM Nb Ta, BIAS
// true) names the intercepts of the basis its fit draws, whose first
// vectors are e_0 and e_1: for M_KRR Kern_Quadratic "its intercepts are
// weights 1 (Nb) and 2 (Ta)", for a training configuration and for a
// potential with one intercept component shared by the species (as tadah
// trained BIAS true before 1.6.0: BASIS vectors of 1 + 3 values, the first
// e_0); for Kern_RBF, which has no intercept column, "weights 1 to 2
// belong to its intercept basis vectors" (a pin).
// Before the fix: the WARNING described the kernel's own basis: the
// training configuration, whose kernel has none before its fit, got
// "weights 1 to 2 belong to its intercept basis vectors" without the
// intercepts, the potential "its intercepts are weights 1 (Nb)" (its stale
// e_0 alone).
TEST_CASE("The FIXINDEX WARNING of a non-linear kernel names the intercepts "
          "of the basis its fit draws",
          "[descriptors_calc][bias][fixindex]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  auto krr = [](const std::string &kernel) {
    Context c = make_context({"Nb", "Ta"}, true);
    set_model(c, "M_KRR", kernel);
    c.add("SBASIS", 6);
    c.add("FIXINDEX", "3");
    c.add("FIXWEIGHT", 0.1);
    return c;
  };
  auto warning = [&](Context c) {
    Trainer tr(c, ws);
    return tadah::mlip::fixindex_intercept_warning(tr.context, *tr.fb);
  };
  SECTION("Kern_Quadratic, a training configuration") {
    const std::string w = warning(krr("Kern_Quadratic"));
    INFO(w);
    CHECK(w.find("FIXINDEX with BIAS true") != std::string::npos);
    CHECK(w.find("its intercepts are weights 1 (Nb) and 2 (Ta)") !=
          std::string::npos);
  }
  SECTION("Kern_Quadratic, a potential with one shared intercept component") {
    Context c = krr("Kern_Quadratic");
    set_basis(c, {{1.0, 0.0, 0.0, 0.0},
                  {1.0, 0.2, 0.3, 0.1},
                  {1.0, 0.5, 0.1, 0.4}});
    for (double v : {-8.0, 0.3, -0.2}) c.add("WEIGHTS", v);
    const std::string w = warning(c);
    INFO(w);
    CHECK(w.find("its intercepts are weights 1 (Nb) and 2 (Ta)") !=
          std::string::npos);
  }
  SECTION("Kern_RBF (a pin)") {
    Context c = krr("Kern_RBF");
    c.add("MPARAMS", 1.0);
    const std::string w = warning(c);
    INFO(w);
    CHECK(w.find("weights 1 to 2 belong to its intercept basis vectors") !=
          std::string::npos);
  }
}

// Pins: the composition check of a non-linear kernel (ATOM Nb Ta, BIAS
// true) takes its intercepts from the basis its fit draws, whatever basis
// the kernel holds. MODEL M_KRR Kern_Quadratic 1: on data of one
// composition (every structure Nb8 Ta8) refused, naming "rank 1" and
// "weights 1 (Nb) and 2 (Ta)" (the intercept vectors e_0 and e_1 that
// start the draw), for the kernel of a Trainer before its fit (no basis)
// and for a kernel with a potential's basis vectors of another descriptor
// (6 values each, DSIZE 5); varied compositions pass. MODEL M_KRR
// Kern_Quadratic 2 is not checked, also with a potential's basis e_0, e_1.
// Kern_RBF is not refused (a pin). Without a MODEL that draws the basis, a
// kernel whose basis vectors do not have the descriptor's length is refused
// with std::logic_error, naming both lengths.
// Before the fix: the check took the kernel's own basis: none before the
// fit (no refusal), the potential's e_0, e_1 for implementation 2 (refused,
// "rank 1"); with the 6-value vectors it evaluated the kernel at the
// 5-entry descriptor, reading past it (refused as now with MODEL
// Kern_Quadratic 1, the values read past the descriptor multiplied by 0;
// no std::logic_error without MODEL).
TEST_CASE("The composition check of a non-linear kernel takes its "
          "intercepts from the basis its fit draws",
          "[descriptors_calc][bias][composition]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  const StructureDB fixed = make_db(std::vector<int>(10, 8), false, 600);
  const StructureDB varied = make_db(kVaried, false, 400);
  auto check = [](const StructureDB &db, const Context &c,
                  const tadah::models::Function_Base &f) {
    return refusal_of(
        [&] { tadah::mlip::require_intercepts_determined(db, c, f); });
  };
  auto krr = [](const std::string &kernel, int impl) {
    Context c = make_context({"Nb", "Ta"}, true);
    set_model(c, "M_KRR", kernel);
    c.add("MODEL", impl);
    c.add("SBASIS", 6);
    return c;
  };
  /* The keys the check reads, DSIZE 5 (2 intercepts + D2_BP 3), and a
   * kernel basis of e_0 and e_1 of `length` values each. */
  auto kernel_keys = [](std::size_t length, int impl) {
    Context k;
    k.add("ATOM", "Nb");
    k.add("ATOM", "Ta");
    k.add("BIAS", "true");
    k.add("EWEIGHT", 1.0);
    k.add("VERBOSE", 0);
    k.addInternal("DSIZE", 5);
    if (impl > 0) {
      k.add("MODEL", "M_KRR");
      k.add("MODEL", "Kern_Quadratic");
      k.add("MODEL", impl);
    }
    std::vector<double> e0(length, 0.0), e1(length, 0.0);
    e0[0] = 1.0;
    e1[1] = 1.0;
    set_basis(k, {e0, e1});
    return k;
  };
  SECTION("Kern_Quadratic 1, the kernel of a Trainer before its fit") {
    Context c = krr("Kern_Quadratic", 1);
    Trainer tr(c, ws);
    const std::string m = check(fixed, tr.context, *tr.fb);
    INFO(m);
    CHECK(m.find("cannot determine") != std::string::npos);
    CHECK(m.find("rank 1") != std::string::npos);
    CHECK(m.find("weights 1 (Nb) and 2 (Ta)") != std::string::npos);
    CHECK(check(varied, tr.context, *tr.fb) == "<no exception>");
  }
  SECTION("Kern_Quadratic 1, basis vectors of another descriptor") {
    const Context k = kernel_keys(6, 1);
    const tadah::models::Kern_Quadratic kq(k);
    const std::string m = check(fixed, k, kq);
    INFO(m);
    CHECK(m.find("rank 1") != std::string::npos);
    CHECK(m.find("weights 1 (Nb) and 2 (Ta)") != std::string::npos);
    CHECK(check(varied, k, kq) == "<no exception>");
  }
  SECTION("Kern_Quadratic 2 is not checked") {
    Context c = krr("Kern_Quadratic", 2);
    Trainer tr(c, ws);
    CHECK(check(fixed, tr.context, *tr.fb) == "<no exception>");
    const Context k = kernel_keys(5, 2);
    const tadah::models::Kern_Quadratic kq(k);
    const std::string m = check(fixed, k, kq);
    INFO(m);
    CHECK(m == "<no exception>");
  }
  SECTION("Kern_RBF 1 (a pin)") {
    Context c = krr("Kern_RBF", 1);
    c.add("MPARAMS", 1.0);
    Trainer tr(c, ws);
    CHECK(check(fixed, tr.context, *tr.fb) == "<no exception>");
  }
  SECTION("a kernel's basis of another length, no MODEL") {
    const Context k = kernel_keys(6, 0);
    const tadah::models::Kern_Quadratic kq(k);
    CHECK_THROWS_AS(tadah::mlip::require_intercepts_determined(varied, k, kq),
                    std::logic_error);
    const std::string m = check(varied, k, kq);
    INFO(m);
    CHECK(m.find("6 values") != std::string::npos);
    CHECK(m.find("DSIZE 5") != std::string::npos);
  }
}

// Pins: the weight count of a non-linear kernel's configuration
// (model_weight_count, which tadah hpo checks at startup) refuses SBASIS
// below the intercept vectors plus one drawn vector, the bound of the
// basis draw (Basis::check_sbasis): with ATOM Nb Ta and BIAS true, SBASIS 2
// of M_KRR Kern_RBF is refused naming "SBASIS 2", "at least 3 basis
// vectors", "the 2 intercept vectors" and the "training AEDs" (for
// implementation 2 the "training structures"), and SBASIS 3 gives 3; with
// ATOM Ta and BIAS true, or with BIAS false, SBASIS 1 is refused naming "at
// least 2 basis vectors" and "column 0 is the bias vector e0", and SBASIS 2
// gives 2 (a pin).
// Before the fix: SBASIS 2 on two species with BIAS true gave 2 (the bound
// was 2 whatever the species), a count every fit of the configuration
// refuses ("needs at least 3 basis vectors").
TEST_CASE("model_weight_count: SBASIS of a non-linear kernel needs one "
          "drawn vector after the intercept vectors",
          "[descriptors_calc][bias][basis]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  auto count = [&](const std::vector<std::string> &atoms, bool bias, int impl,
                   int sbasis) {
    Context c = make_context(atoms, bias);
    set_model(c, "M_KRR", "Kern_RBF");
    c.add("MODEL", impl);
    c.add("MPARAMS", 1.0);
    c.add("SBASIS", sbasis);
    Trainer tr(c, ws);
    std::size_t k = 0;
    const std::string m = refusal_of(
        [&] { k = tadah::mlip::model_weight_count(*tr.fb, tr.context); });
    return m == "<no exception>" ? std::to_string(k) : m;
  };
  for (int impl : {1, 2}) {
    CAPTURE(impl);
    const std::string m = count({"Nb", "Ta"}, true, impl, 2);
    INFO(m);
    CHECK(m.find("M_KRR Kern_RBF") != std::string::npos);
    CHECK(m.find("SBASIS 2") != std::string::npos);
    CHECK(m.find("at least 3 basis vectors") != std::string::npos);
    CHECK(m.find("the 2 intercept vectors") != std::string::npos);
    CHECK(m.find(impl == 1 ? "training AEDs" : "training structures") !=
          std::string::npos);
    CHECK(count({"Nb", "Ta"}, true, impl, 3) == "3");
  }
  for (const bool two : {false, true}) {
    CAPTURE(two);
    const std::vector<std::string> atoms =
        two ? std::vector<std::string>{"Nb", "Ta"}
            : std::vector<std::string>{"Ta"};
    const bool bias = !two;   // ATOM Ta with BIAS true, Nb Ta with BIAS false
    const std::string m = count(atoms, bias, 1, 1);
    INFO(m);
    CHECK(m.find("at least 2 basis vectors") != std::string::npos);
    CHECK(m.find("column 0 is the bias vector e0") != std::string::npos);
    CHECK(count(atoms, bias, 1, 2) == "2");
  }
}

// Pins: M_KRR Kern_Linear implementation 2 draws SBASIS basis vectors for
// its solve (the intercept vectors, then mean AEDs of training structures),
// so its weight count (model_weight_count, which tadah hpo's startup calls)
// checks SBASIS as a non-linear kernel's does, and is DSIZE, the descriptor
// weights the solve's coefficients make: with ATOM Ta and BIAS false (DSIZE
// 3) no SBASIS is refused naming "M_KRR Kern_Linear 2 needs SBASIS" and the
// "training structures"; SBASIS 1 is refused naming "M_KRR Kern_Linear 2",
// "SBASIS 1", "at least 2 basis vectors" and "column 0 is the bias vector
// e0", without the word "non-linear"; SBASIS 2 and 7 give 3. With ATOM Nb Ta
// and BIAS true (DSIZE 5) SBASIS 2 is refused naming "at least 3 basis
// vectors" and "the 2 intercept vectors", and SBASIS 3 gives 5. Kern_Linear
// implementation 1, which draws no basis, gives 3 without SBASIS (a pin).
// Before the fix: model_weight_count returned DSIZE for Kern_Linear 2
// before any SBASIS check: no SBASIS and SBASIS 1 gave 3, SBASIS 2 on Nb Ta
// gave 5, so tadah hpo's startup passed them and every evaluation threw
// ("Key not found in the Context object: SBASIS"; "SBASIS 1: a non-linear
// M_KRR model needs at least 2 basis vectors").
TEST_CASE("model_weight_count: M_KRR Kern_Linear 2 checks SBASIS and has "
          "DSIZE weights",
          "[descriptors_calc][bias][basis]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  // The count, or the refusal; sbasis < 0: no SBASIS.
  auto count = [&](const std::vector<std::string> &atoms, bool bias, int impl,
                   int sbasis) {
    Context c = make_context(atoms, bias);
    set_model(c, "M_KRR", "Kern_Linear");
    c.add("MODEL", impl);
    if (sbasis >= 0) c.add("SBASIS", sbasis);
    Trainer tr(c, ws);
    std::size_t k = 0;
    const std::string m = refusal_of(
        [&] { k = tadah::mlip::model_weight_count(*tr.fb, tr.context); });
    return m == "<no exception>" ? std::to_string(k) : m;
  };
  const std::vector<std::string> ta = {"Ta"};
  {
    const std::string m = count(ta, false, 2, -1);
    INFO(m);
    CHECK(m.find("M_KRR Kern_Linear 2 needs SBASIS") != std::string::npos);
    CHECK(m.find("training structures") != std::string::npos);
  }
  {
    const std::string m = count(ta, false, 2, 1);
    INFO(m);
    CHECK(m.find("M_KRR Kern_Linear 2") != std::string::npos);
    CHECK(m.find("SBASIS 1") != std::string::npos);
    CHECK(m.find("at least 2 basis vectors") != std::string::npos);
    CHECK(m.find("column 0 is the bias vector e0") != std::string::npos);
    CHECK(m.find("non-linear") == std::string::npos);
  }
  CHECK(count(ta, false, 2, 2) == "3");
  CHECK(count(ta, false, 2, 7) == "3");
  {
    const std::string m = count({"Nb", "Ta"}, true, 2, 2);
    INFO(m);
    CHECK(m.find("at least 3 basis vectors") != std::string::npos);
    CHECK(m.find("the 2 intercept vectors") != std::string::npos);
  }
  CHECK(count({"Nb", "Ta"}, true, 2, 3) == "5");
  CHECK(count(ta, false, 1, -1) == "3");
}

// Pins: with ATOM Nb Ta and BIAS true the basis draw takes SBASIS from 3
// (the intercept vectors e_0, e_1 and one drawn vector) to the candidates
// plus 2: build_random_basis (implementation 1; 8 structures of 16 atoms,
// 128 AEDs) refuses SBASIS 2 ("needs at least 3 basis vectors", "columns 0
// to 1 are the 2 intercept vectors") and 131 ("set SBASIS to at most
// 130"), and draws 130; prep_basis_for_krr (implementation 2, 8
// structures) refuses SBASIS 11 ("set SBASIS to at most 10") and draws 10.
// Before the fix: passes as well (a pin: Basis::check_sbasis has counted
// one intercept vector per species since the KRR basis seeded one per
// species; its bounds with two species had no test).
TEST_CASE("The KRR basis draw takes SBASIS from N + 1 to the candidates "
          "plus N with BIAS true on N species",
          "[descriptors_calc][bias][basis]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  Context c = make_context({"Nb", "Ta"}, true);
  c.add("SBASIS", 6);
  Trainer tr(c, ws);
  StructureDB db;
  for (unsigned s = 0; s < 8; ++s)
    db.add(tanb_cell(static_cast<int>(2 * s), 900 + s));
  NNFinder().calc(db, kRcut);
  StDescriptorsDB sd = tr.dc.calc(db);
  tadah::mlip::Basis<DM_Function_Base &> basis(tr.context);
  auto draw = [&](std::size_t s) {
    return refusal_of([&] { basis.build_random_basis(s, sd); });
  };
  {
    const std::string m = draw(2);
    INFO(m);
    CHECK(m.find("SBASIS 2") != std::string::npos);
    CHECK(m.find("needs at least 3 basis vectors") != std::string::npos);
    CHECK(m.find("columns 0 to 1 are the 2 intercept vectors") !=
          std::string::npos);
  }
  {
    const std::string m = draw(131);
    INFO(m);
    CHECK(m.find("set SBASIS to at most 130") != std::string::npos);
  }
  CHECK(draw(130) == "<no exception>");
  CHECK(basis.b.cols() == 130);
  auto prep = [&](int s) {
    set(tr.context, "SBASIS", std::to_string(s));
    return refusal_of([&] { basis.prep_basis_for_krr(sd, db); });
  };
  {
    const std::string m = prep(11);
    INFO(m);
    CHECK(m.find("set SBASIS to at most 10") != std::string::npos);
  }
  CHECK(prep(10) == "<no exception>");
  CHECK(basis.b.cols() == 10);
}
