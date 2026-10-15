// bench_dphi_dtheta_walk.cpp
//
// Microbench gate for the per-θ walker stack.
//
// Times three call paths that dominate the HPO analytical gradient
// per-θ wall on configs with a DM_mBlip arm + WATOM bias (a production
// blip6 search):
//
//   case=combined_dm_mblip   — assemble_dphi_train_both_rows on a
//                               10-struct DM_mBlip db, one θ
//                               ("CGRIDMB", arg_pos=1), 100 iterations.
//   case=cached_dm_mblip     — same setup driven through
//                               assemble_dphi_train_both_rows_cached,
//                               with the rho-cache build timed
//                               separately. This is the path the live
//                               per-θ loop takes when
//                               TADAH_DISABLE_DM_CACHE is unset.
//   case=full_loop_emulation — synthetic 29-θ sequential loop (NO OMP)
//                               over the cached walker. Reports
//                               un-parallelised throughput so future
//                               kernel/walker regressions show up
//                               independent of OMP scheduling changes.
//
// Output is one parseable line per case, prefixed with [BENCH_DPHI_WALK].
// Reference numbers from the first commit live in
// bench_dphi_dtheta_walk_reference.txt next to this file; future runs
// should diff against that.
//
// This is NOT a Catch2 test — chrono dispatch overhead is smaller than
// the per-call cost we want to time, and a standalone executable
// keeps the bench out of the test-suite pass/fail surface.
//
// Build:    cmake --build build_tadah --target bench_dphi_dtheta_walk -j
// Run:      ./build_tadah/_deps/tadah.mlip-build/bench/bench_dphi_dtheta_walk

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/analytics/dphi_dtheta_walk.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/models/cutoffs.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/descriptors/dm/dm_blip.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::models::DC_Selector;

using clk = std::chrono::high_resolution_clock;

inline double ms_between(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// 4-atom H cluster — same geometry as MLIP's walker tests, just
// replicated with a tiny x-offset per copy. Forces are filler; the
// walker reads only positions and neighbour lists.
Structure make_four_atom_off_axis(double offset_x = 0.0) {
    Structure st;
    st.cell.load(20.0, 0.0, 0.0,
                 0.0, 20.0, 0.0,
                 0.0, 0.0, 20.0);
    st.add_atom(Atom(Element("H"), 0.0 + offset_x, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 1.2 + offset_x, 0.7, 0.4, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 0.3 + offset_x, 1.1, 0.6, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 0.5 + offset_x, 0.6, 1.3, 0.0, 0.0, 0.0));
    return st;
}

StructureDB make_bench_stdb(std::size_t N, double rcut) {
    StructureDB db;
    db.dbidx = {0, N};
    db.structures.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
        db.structures.push_back(
            make_four_atom_off_axis(0.05 * static_cast<double>(i)));
    NNFinder finder;
    for (std::size_t i = 0; i < N; ++i) finder.calc(db.structures[i], rcut);
    return db;
}

// Composite context — DM_mBlip + a no-op D2 arm so DC_Selector::init
// accepts it. The HPO analytical gradient always runs through
// DC_Selector dispatchers; bench against the same path.
Context make_bench_context(double rcut, int Lmax = 1) {
    const double cgrid_mb_val = 0.7;
    const double sgrid_mb_val = 1.5;
    const std::vector<double> cembfunc = {0.10, 0.25};
    const std::vector<double> sembfunc = {3.0, 3.0};
    const std::vector<double> cgrid_2b = {0.5, 1.4};
    const std::vector<double> sgrid_2b = {3.0, 5.0};

    Context c;
    c.add("ATOM",    "H");
    c.add("WATOM",   1.0);
    c.add("VERBOSE", 0);

    c.add("INIT2B",  "true");
    c.add("TYPE2B",  "D2_Blip");
    c.add("TYPE2B",  static_cast<int>(cgrid_2b.size()));
    c.add("TYPE2B",  static_cast<int>(sgrid_2b.size()));
    c.add("TYPE2B",  "H"); c.add("TYPE2B", "H");
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid_2b.size(); ++i) {
            if (i) ss << " "; ss << cgrid_2b[i];
        }
        c.add("CGRID2B", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid_2b.size(); ++i) {
            if (i) ss << " "; ss << sgrid_2b[i];
        }
        c.add("SGRID2B", ss.str());
    }
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B",   rcut);

    c.add("INITMB",   "true");
    c.add("TYPEMB",   "DM_mBlip");
    c.add("TYPEMB",   Lmax);
    c.add("TYPEMB",   1);
    c.add("TYPEMB",   1);
    c.add("TYPEMB",   static_cast<int>(cembfunc.size()));
    c.add("TYPEMB",   static_cast<int>(sembfunc.size()));
    c.add("TYPEMB",   "H"); c.add("TYPEMB", "H");
    c.add("CGRIDMB",  cgrid_mb_val);
    c.add("SGRIDMB",  sgrid_mb_val);
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cembfunc.size(); ++i) {
            if (i) ss << " "; ss << cembfunc[i];
        }
        c.add("CEMBFUNC", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sembfunc.size(); ++i) {
            if (i) ss << " "; ss << sembfunc[i];
        }
        c.add("SEMBFUNC", ss.str());
    }
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB",   rcut);

    c.add("FORCE",    "true");
    c.add("STRESS",   "false");
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);
    return c;
}

struct BenchStats {
    double median_ms;
    double mean_ms;
    double min_ms;
};

BenchStats summarise(std::vector<double>& samples) {
    BenchStats s{};
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    s.min_ms    = samples.front();
    s.median_ms = samples[samples.size() / 2];
    s.mean_ms   =
        std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    PeriodicTable::initialize();

    constexpr std::size_t N_STRUCT     = 10;
    constexpr double      rcut         = 3.5;
    constexpr int         Lmax         = 1;
    constexpr std::size_t ITERS_SINGLE = 100;
    constexpr std::size_t ITERS_LOOP   = 10;
    constexpr std::size_t N_PREFS_LOOP = 29;

    auto stdb = make_bench_stdb(N_STRUCT, rcut);
    Context ctx = make_bench_context(rcut, Lmax);

    DC_Selector dcs(ctx);
    const std::size_t n_cols = dcs.descriptor_size();

    // Build per-struct force-row offsets. Per-struct: 3·natoms force
    // rows, no per-struct energy interleave (walker writes energy rows
    // to a separate dense buffer indexed by s × n_cols).
    std::vector<std::size_t> force_row_start_per_struct(stdb.size(), 0);
    {
        std::size_t cursor = 0;
        for (std::size_t s = 0; s < stdb.size(); ++s) {
            force_row_start_per_struct[s] = cursor;
            cursor += 3 * stdb(s).natoms();
        }
    }
    std::size_t n_force_rows = 0;
    for (std::size_t s = 0; s < stdb.size(); ++s)
        n_force_rows += 3 * stdb(s).natoms();

    std::vector<double> energy_buf(stdb.size() * n_cols, 0.0);
    std::vector<double> force_buf (n_force_rows  * n_cols, 0.0);

    std::printf("[BENCH_DPHI_WALK] config n_struct=%zu n_atoms_per_struct=%zu "
                "n_cols=%zu rcut=%.2f Lmax=%d\n",
                N_STRUCT, stdb(0).natoms(), n_cols, rcut, Lmax);

    // ---- Case 1: combined walker (non-cached) ----
    {
        std::vector<double> samples;
        samples.reserve(ITERS_SINGLE);
        for (std::size_t k = 0; k < ITERS_SINGLE; ++k) {
            std::fill(energy_buf.begin(), energy_buf.end(), 0.0);
            std::fill(force_buf .begin(), force_buf .end(), 0.0);
            const auto t0 = clk::now();
            const auto pr = tadah::mlip::assemble_dphi_train_both_rows(
                stdb, dcs, "CGRIDMB", /*arg_pos=*/1,
                energy_buf.data(), force_buf.data(),
                n_cols, force_row_start_per_struct);
            const auto t1 = clk::now();
            (void)pr;
            samples.push_back(ms_between(t0, t1));
        }
        const auto st = summarise(samples);
        std::printf("[BENCH_DPHI_WALK] case=combined_dm_mblip "
                    "theta=CGRIDMB[1] iters=%zu "
                    "median_ms=%.4f mean_ms=%.4f min_ms=%.4f\n",
                    ITERS_SINGLE, st.median_ms, st.mean_ms, st.min_ms);
    }

    // ---- Case 2: cached walker + separate cache-build timing ----
    {
        std::vector<double> cache_samples;
        std::vector<double> walk_samples;
        cache_samples.reserve(ITERS_SINGLE);
        walk_samples .reserve(ITERS_SINGLE);
        for (std::size_t k = 0; k < ITERS_SINGLE; ++k) {
            tadah::mlip::DMRhoCache cache;
            const auto t0 = clk::now();
            tadah::mlip::build_dm_rho_cache(stdb, *dcs.dmb, cache, n_cols);
            const auto t1 = clk::now();

            std::fill(energy_buf.begin(), energy_buf.end(), 0.0);
            std::fill(force_buf .begin(), force_buf .end(), 0.0);
            const auto t2 = clk::now();
            const auto pr = tadah::mlip::assemble_dphi_train_both_rows_cached(
                stdb, dcs, cache, "CGRIDMB", /*arg_pos=*/1,
                energy_buf.data(), force_buf.data(),
                n_cols, force_row_start_per_struct);
            const auto t3 = clk::now();
            (void)pr;
            cache_samples.push_back(ms_between(t0, t1));
            walk_samples .push_back(ms_between(t2, t3));
        }
        const auto cs = summarise(cache_samples);
        const auto ws = summarise(walk_samples);
        std::printf("[BENCH_DPHI_WALK] case=cached_dm_mblip "
                    "theta=CGRIDMB[1] iters=%zu "
                    "cache_median_ms=%.4f walk_median_ms=%.4f "
                    "walk_mean_ms=%.4f walk_min_ms=%.4f\n",
                    ITERS_SINGLE, cs.median_ms, ws.median_ms,
                    ws.mean_ms, ws.min_ms);
    }

    // ---- Case 3: full per-θ-loop emulation (sequential, no OMP) ----
    //
    // Synthetic 29-θ batch over a mix of DM-arm θ keys: cycle through
    // {CGRIDMB[1], SGRIDMB[1], RCUTMB[1], WATOM[1], CEMBFUNC[i],
    //  SEMBFUNC[i]} to exercise the same kernel paths the live config
    // does (per-pair drho/dθ vs per-atom daed/dθ). The cache is built
    // once per loop call (mirrors the live per-θ-loop dispatcher).
    {
        struct PrefEntry { std::string name; int arg; };
        std::vector<PrefEntry> prefs;
        prefs.reserve(N_PREFS_LOOP);
        const std::vector<PrefEntry> palette = {
            {"CGRIDMB",  1}, {"SGRIDMB",  1}, {"RCUTMB",   1},
            {"WATOM",    1},
            {"CEMBFUNC", 1}, {"CEMBFUNC", 2},
            {"SEMBFUNC", 1}, {"SEMBFUNC", 2},
        };
        for (std::size_t i = 0; i < N_PREFS_LOOP; ++i)
            prefs.push_back(palette[i % palette.size()]);

        std::vector<double> samples;
        samples.reserve(ITERS_LOOP);
        for (std::size_t k = 0; k < ITERS_LOOP; ++k) {
            tadah::mlip::DMRhoCache cache;
            tadah::mlip::build_dm_rho_cache(stdb, *dcs.dmb, cache, n_cols);

            const auto t0 = clk::now();
            for (const auto& p : prefs) {
                std::fill(energy_buf.begin(), energy_buf.end(), 0.0);
                std::fill(force_buf .begin(), force_buf .end(), 0.0);
                const auto pr =
                    tadah::mlip::assemble_dphi_train_both_rows_cached(
                        stdb, dcs, cache, p.name, p.arg,
                        energy_buf.data(), force_buf.data(),
                        n_cols, force_row_start_per_struct);
                (void)pr;
            }
            const auto t1 = clk::now();
            samples.push_back(ms_between(t0, t1));
        }
        const auto st = summarise(samples);
        std::printf("[BENCH_DPHI_WALK] case=full_loop_emulation "
                    "n_theta=%zu iters=%zu "
                    "median_ms=%.4f mean_ms=%.4f min_ms=%.4f "
                    "per_theta_median_ms=%.4f\n",
                    N_PREFS_LOOP, ITERS_LOOP, st.median_ms, st.mean_ms,
                    st.min_ms,
                    st.median_ms / static_cast<double>(N_PREFS_LOOP));
    }

    return 0;
}
