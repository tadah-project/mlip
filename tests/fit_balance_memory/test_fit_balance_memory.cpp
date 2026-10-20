// test_fit_balance_memory.cpp
//
// The memory of the balance table of a fit (fit_balance.h). The table is
// computed after the fit, while the fit still holds its design matrix, and
// tadah train writes the potential after it: so the table reads the rows
// of the fit in blocks, without a copy of its n x p rows, and a table that
// cannot have its memory is none, saying why, rather than an exception
// that loses the fit.
//
// The executable replaces the global operator new, to count the bytes a
// table allocates and to refuse an allocation above a size: it is an
// executable of its own, so that no other test runs with it.

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/fit_balance.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <new>
#include <string>

namespace {

std::atomic<bool> g_counting{false};
std::atomic<std::size_t> g_bytes{0};
std::atomic<std::size_t> g_refuse_above{0};  // 0: no allocation refused

}  // namespace

void *operator new(std::size_t n) {
  if (g_counting.load(std::memory_order_relaxed)) {
    const std::size_t cap = g_refuse_above.load(std::memory_order_relaxed);
    if (cap != 0 && n > cap) throw std::bad_alloc();
    g_bytes.fetch_add(n, std::memory_order_relaxed);
  }
  if (void *p = std::malloc(n == 0 ? 1 : n)) return p;
  throw std::bad_alloc();
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::FitBalance;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

/* A bcc Ta supercell of 4 x 4 x 4 cubic cells (128 atoms, edge 13.2 Å)
 * with its atoms displaced by up to 0.1 Å, each structure @p k its own
 * way, with forces and an energy that are not zero. */
Structure displaced_bcc(std::size_t k) {
  Structure st;
  const double a = 3.3, L = 4 * a;
  st.cell.load(L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L);
  std::size_t i = 0;
  for (int x = 0; x < 4; ++x)
    for (int y = 0; y < 4; ++y)
      for (int z = 0; z < 4; ++z)
        for (int b = 0; b < 2; ++b, ++i) {
          const double s = static_cast<double>(k * 131 + i);
          const double off = 0.5 * a * b;
          st.add_atom(Atom(Element("Ta"),
                           x * a + off + 0.1 * std::sin(1.3 * s),
                           y * a + off + 0.1 * std::cos(0.7 * s),
                           z * a + off + 0.1 * std::sin(2.1 * s + 0.4),
                           0.2 * std::sin(0.9 * s), -0.15 * std::cos(1.7 * s),
                           0.1 * std::sin(2.9 * s)));
        }
  st.energy = -8.0 * static_cast<double>(st.natoms()) +
              0.05 * std::sin(static_cast<double>(k));
  return st;
}

/* Energies and forces of 50 such structures, 20 coefficients (D2_Blip 20,
 * BIAS false): 50 + 50 x 384 = 19250 rows of 20 columns. */
Context context() {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1);
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  std::string cgrid, sgrid;
  for (int j = 0; j < 20; ++j) {
    cgrid += (j ? " " : "") + std::to_string(2.0 + 0.15 * j);
    sgrid += (j ? " " : "") + std::to_string(1.0 + 0.05 * j);
  }
  c.add("TYPE2B", "D2_Blip 20 20 Ta Ta");
  c.add("CGRID2B", cgrid);
  c.add("SGRID2B", sgrid);
  c.add("TYPEMB", "DM_Dummy");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 5.5);
  c.add("RCUTMB", 5.5);
  c.add("DIMER", "false 0 false");
  c.add("FORCE", "true");
  c.add("STRESS", "false");
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("OALGO", 1);
  c.add("LAMBDA", "1e-3");
  return c;
}

StructureDB data() {
  StructureDB db;
  for (std::size_t k = 0; k < 50; ++k) db.add(displaced_bcc(k));
  db.dbidx = {0, 50};
  return db;
}

constexpr std::size_t kRows = 50 + 50 * 128 * 3;
constexpr std::size_t kCols = 20;
/* The bytes of one copy of the rows of the fit. */
constexpr std::size_t kCopy = kRows * kCols * sizeof(double);

}  // namespace

/* Pins: the balance table of a fit of 19250 rows and 20 coefficients
 * allocates less than half of one copy of its rows (1.54 MB of 3.08 MB):
 * it reads them in blocks from the design matrix the fit keeps.
 * Before the fix: it allocated two copies of the weighted rows (the rows
 * of the observations, then the column-major copy of their SVD), 6.2 MB:
 * beside the fit's own two, a table of a fit that fitted within a memory
 * limit could fail. */
TEST_CASE("Fit balance allocates less than a copy of the rows of the fit",
          "[fit_balance][memory]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  Context c = context();
  Trainer tr(c, ws);
  StructureDB db = data();
  tr.train(db);
  g_bytes = 0;
  g_counting = true;
  const FitBalance b = tr.model->fit_balance();
  g_counting = false;
  INFO(b.none);
  REQUIRE(b.computed());
  REQUIRE(b.coefficients == kCols);
  INFO("allocated " << g_bytes.load() << " bytes; one copy of the rows is "
                    << kCopy);
  CHECK(g_bytes.load() < kCopy / 2);
}

/* Pins: a table that cannot have its memory (every allocation above 64 kB
 * refused while it is computed) is none, saying that the memory ran out,
 * and the model's fit_balance() does not throw: tadah train writes the
 * potential after the table, and tadah hpo the best potential, so an
 * exception would lose the fit.
 * Before the fix: std::bad_alloc left M_BLR::fit_balance (which caught
 * std::runtime_error only), and tadah train ended with "ERROR:
 * std::bad_alloc" without writing the potential. */
TEST_CASE("Fit balance without its memory is none, and the fit is kept",
          "[fit_balance][memory]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  Context c = context();
  Trainer tr(c, ws);
  StructureDB db = data();
  tr.train(db);
  FitBalance b;
  bool threw = false;
  g_refuse_above = 64 * 1024;
  g_counting = true;
  try {
    b = tr.model->fit_balance();
  } catch (const std::bad_alloc &) {
    threw = true;
  }
  g_counting = false;
  g_refuse_above = 0;
  REQUIRE(!threw);
  CHECK(!b.computed());
  CHECK(b.none == "it could not be computed: not enough memory");
  // The fit is the model's still: its table, computed with memory.
  CHECK(tr.model->fit_balance().computed());
}
