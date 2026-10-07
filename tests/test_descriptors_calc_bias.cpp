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
#include <tadah/models/dc_selector.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
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
// Before the fix: one intercept column (dim 9 != 8 + 2), 1.0 for every
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
