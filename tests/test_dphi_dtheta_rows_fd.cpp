// test_dphi_dtheta_rows_fd.cpp
//
// The rows of the design matrix that the combined walker of the analytical
// gradient of tadah hpo claims, against central differences of the rows.
//
// The walker (assemble_dphi_train_both_rows and its form with the rho cache,
// dphi_dtheta_walk.h) writes the derivative of the energy rows and of the
// force rows of the design matrix with respect to one search parameter, and
// claims the rows it has differentiated: the analytical gradient takes the
// claimed rows for the derivative of the rows. A claimed row that is not
// that derivative is a wrong gradient, which the run uses without a word.
//
// Here the design rows are built from the descriptors of a Trainer as
// compute_analytical_gradient_from_trainer (HPO dloss_dtheta.h) builds them
// (the energy row of a structure: the sum of the descriptors of its atoms;
// the three force rows of an atom: minus the sum over its pairs of
// fij - fji), with the parameter moved by +-h and +-h/2, for every two-body
// descriptor beside every many-body model and every parameter of the
// configuration that the closed forms of a descriptor take. Every row block
// the walker claims must equal the central difference. A row block it does
// not claim is not checked here: the analytical gradient hands the
// parameter to finite differences.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/analytics/dphi_dtheta_walk.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/descriptors/dm/embed_spec.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;

std::string num(double v)
{
    std::ostringstream o;
    o.precision(17);
    o << v;
    return o.str();
}

/* ---- the descriptors ------------------------------------------------- */

/* The two-body descriptor `name` ("none": INIT2B false) with the cutoff
 * Cut_Cos `rc`: grids whose functions reach the pairs of the structures
 * below. */
std::string two_body(const std::string& name, double rc)
{
    const std::string cut = "RCTYPE2B Cut_Cos\nRCUT2B " + num(rc) + "\n";
    const std::string grid = "CGRID2B 2.8 3.6\nSGRID2B 1.0 1.1\n";
    if (name == "none") return "INIT2B false\n";
    if (name == "D2_Blip" || name == "D2_Blip5" || name == "D2_BP")
        return "INIT2B true\nTYPE2B " + name + " 2 2 Ta Ta\n" + grid + cut;
    if (name == "D2_Knot5")
        return "INIT2B true\nTYPE2B D2_Knot5 2 2 Ta Ta\nCGRID2B " +
               num(0.75 * rc) + " " + num(0.97 * rc) + "\nKDEG2B 5 5\n" + cut;
    if (name == "D2_LJ") return "INIT2B true\nTYPE2B D2_LJ Ta Ta\n" + cut;
    if (name == "D2_MIE")
        return "INIT2B true\nTYPE2B D2_MIE 12 6 Ta Ta\n" + cut;
    if (name == "D2_ZBL")
        return "INIT2B true\nTYPE2B D2_ZBL 0.5292 0.88534 0.23 0.23 Ta Ta\n" +
               cut;
    if (name == "D2_mJoin")
        return "INIT2B true\nTYPE2B D2_mJoin\nTYPE2B D2_BP 1 1 Ta Ta\n"
               "TYPE2B D2_Blip 1 1 Ta Ta\nRCTYPE2B Cut_Cos Cut_Cos\nRCUT2B " +
               num(rc) + " " + num(rc) + "\n" + grid;
    FAIL("unknown two-body descriptor " << name);
    return {};
}

/* A DESCRIPTOR block `id` (a DM_Embed component) of the density basis
 * `density` (BASIS, CENTRES, ETA, AMPLITUDES lines) and the embedding
 * basis `embedding`, with the cutoff Cut_Cos `rc`. */
std::string block(const std::string& id, const std::string& density,
                  const std::string& embedding, double rc)
{
    return "DESCRIPTOR " + id + " DM_Embed\nPAIRS Ta Ta\nDENSITY\n" +
           density + "CUTOFF\nTYPE Cut_Cos\nRADIUS " + num(rc) +
           "\nENDCUTOFF\nENDDENSITY\nEMBEDDING\n" + embedding +
           "REFERENCE ZERO\nENDEMBEDDING\nENDDESCRIPTOR\n";
}

const std::string kReamLine = "TYPEMB DM_REAM 3 3 3 3 5 5 0 0 Ta Ta\n";
const std::string kReamGrids =
    "CGRIDMB 2.8 3.6 4.4\nSGRIDMB 3.0 3.0 3.0\nAMPGRIDMB 0.5 0.3 0.1\n"
    "CEMBFUNC -0.2 0.3 0.8 1.3 1.8\nSEMBFUNC 2.0 2.0 2.0 2.0 2.0\n";

/* The many-body model `name` ("none": INITMB false) with the cutoff
 * Cut_Cos `rc`: DM_REAM in the density form after its name
 * ("DM_REAM NONNEG"), "DESCRIPTOR" one DESCRIPTOR block (DM_Embed),
 * "DM_mJoin lines" DM_mBlip and DM_REAM joined, "DM_mJoin blocks" two
 * DESCRIPTOR blocks. */
std::string many_body(const std::string& name, double rc)
{
    const std::string cut = "RCTYPEMB Cut_Cos\nRCUTMB " + num(rc) + "\n";
    const std::string grid = "CGRIDMB 2.8 3.6\nSGRIDMB 1.0 1.1\n";
    const std::string emb = "CEMBFUNC 1.0 2.0\nSEMBFUNC 1.0 1.0\n";
    const std::string blip5 =
        "BASIS Blip5\nFORM SIGNED\nCENTRES 2.8 3.6 4.4\nETA 3 3 3\n"
        "AMPLITUDES 0.5 0.3 0.1\n";
    const std::string gauss = "BASIS Gaussian\nCENTRES 0 0.5 1\nETA 2 2 2\n";
    if (name == "none") return "INITMB false\n";
    if (name == "DM_Blip")
        return "INITMB true\nTYPEMB DM_Blip 0 2 2 Ta Ta\n" + grid + cut;
    if (name == "DM_EAD")
        return "INITMB true\nTYPEMB DM_EAD 1 2 2 Ta Ta\n" + grid + cut;
    if (name == "DM_mBlip")
        return "INITMB true\nTYPEMB DM_mBlip 0 1 1 2 2 Ta Ta\n"
               "CGRIDMB 0.0\nSGRIDMB 0.2\n" + emb + cut;
    if (name == "DM_mBlipEnv")
        return "INITMB true\nTYPEMB DM_mBlipEnv 0 1 1 2 2 Ta Ta\n"
               "CGRIDMB 0.0\nSGRIDMB 0.2\n" + emb + "RCUTENV 4.0\n" + cut;
    if (name == "DM_mRLR" || name == "DM_mSQRT")
        return "INITMB true\nTYPEMB " + name + " 0 2 2 2 2 Ta Ta\n" + grid +
               emb + cut;
    if (name.rfind("DM_REAM ", 0) == 0)
        return "INITMB true\n" + kReamLine + kReamGrids + "RHOFORM " +
               name.substr(8) + "\n" + cut;
    if (name == "DM_PS_SQ")
        return "INITMB true\nTYPEMB DM_PS_SQ 1 2 2 Ta Ta\n" + grid + cut;
    if (name == "DESCRIPTOR")
        return "INITMB true\n" + block("ta", blip5, gauss, rc);
    if (name == "DM_mJoin lines")
        return "INITMB true\nTYPEMB DM_mJoin\n"
               "TYPEMB DM_mBlip 0 1 1 2 2 Ta Ta\n" + kReamLine +
               "RCTYPEMB Cut_Cos Cut_Cos\nRCUTMB " + num(rc) + " " + num(rc) +
               "\nCGRIDMB 0.0 2.8 3.6 4.4\nSGRIDMB 0.2 3.0 3.0 3.0\n"
               "AMPGRIDMB 0.5 0.3 0.1\n"
               "CEMBFUNC 1.0 2.0 -0.2 0.3 0.8 1.3 1.8\n"
               "SEMBFUNC 1.0 1.0 2.0 2.0 2.0 2.0 2.0\nRHOFORM SIGNED\n";
    if (name == "DM_mJoin blocks")
        return "INITMB true\n" + block("ta", blip5, gauss, rc) +
               block("tb",
                     "BASIS Blip\nFORM SIGNED\nCENTRES 3.0 3.8\nETA 1.5 1.5\n"
                     "AMPLITUDES 0.4 0.2\n",
                     "BASIS Blip\nCENTRES 0.2 0.9\nETA 1 1\n", rc);
    FAIL("unknown many-body model " << name);
    return {};
}

/* The training configuration of the two-body descriptor `d2` and the
 * many-body model `dm`, with their cutoffs, FORCE true (the descriptors
 * carry their force parts), WATOM 1.3. */
Context configuration(const std::string& d2, const std::string& dm,
                      double rc2, double rcm)
{
    std::istringstream in(
        "ATOM Ta\nWATOM 1.3\nMODEL M_BLR BF_Linear\nVERBOSE 0\nBIAS false\n"
        "FORCE true\nSTRESS false\n" + two_body(d2, rc2) +
        many_body(dm, rcm));
    return tadah::core::CoreUserConfigParser().parse(in);
}

/* ---- the structures -------------------------------------------------- */

/* Ta atoms at `xyz` in a cubic cell of side `side`, with forces (none
 * read here) and the neighbour list to 5 A, beyond every cutoff below
 * moved by a step. */
Structure ta_structure(double side, const std::vector<std::vector<double>>& xyz)
{
    Structure st;
    st.cell.load(side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side);
    for (const auto& p : xyz)
        st.add_atom(Atom(Element("Ta"), p[0], p[1], p[2], 0.0, 0.0, 0.0));
    NNFinder().calc(st, 5.0);
    return st;
}

/* A two-atom bcc Ta cell (a = 3.3 A), the second atom off its site, and a
 * Ta trimer (2.7, 3.1 and 3.9 A) in a box of 20 A: pairs at many distances
 * inside cutoffs of 4.5 A, forces not 0. */
StructureDB cell_and_trimer()
{
    StructureDB db;
    db.add(ta_structure(3.3, {{0.0, 0.0, 0.0}, {1.72, 1.61, 1.66}}));
    db.add(ta_structure(20.0, {{5.0, 5.0, 5.0},
                               {7.7, 5.0, 5.0},
                               {6.68, 6.94, 5.83}}));
    return db;
}

/* A Ta dimer at 3.7 A along (0.6, 0.48, 0.64) in a box of 20 A: its pair
 * lies inside a cutoff of 4.5 A and beyond one of 3 A. */
StructureDB dimer()
{
    const double r = 3.7, n = std::sqrt(0.6 * 0.6 + 0.48 * 0.48 + 0.64 * 0.64);
    StructureDB db;
    db.add(ta_structure(20.0, {{5.0, 5.0, 5.0},
                               {5.0 + r * 0.6 / n, 5.0 + r * 0.48 / n,
                                5.0 + r * 0.64 / n}}));
    return db;
}

/* The data of a check: the structures and the cutoffs of the two-body and
 * the many-body descriptor. */
struct Data {
    const char* name;
    StructureDB db;
    double rc2, rcm;
};

/* Pairs inside both cutoffs; the pair inside the two-body cutoff only; the
 * pair inside the many-body cutoff only. */
std::vector<Data> data_sets()
{
    return {{"cell and trimer, both cutoffs 4.5 A", cell_and_trimer(), 4.5,
             4.5},
            {"dimer at 3.7 A, two-body cutoff 4.5 A, many-body 3 A", dimer(),
             4.5, 3.0},
            {"dimer at 3.7 A, two-body cutoff 3 A, many-body 4.5 A", dimer(),
             3.0, 4.5}};
}

std::size_t force_rows(const StructureDB& db)
{
    std::size_t n = 0;
    for (std::size_t s = 0; s < db.size(); ++s) n += 3 * db(s).natoms();
    return n;
}

/* ---- the parameters -------------------------------------------------- */

/* The search parameters of `c` that the closed forms of a descriptor take
 * (index 1 of each key the configuration sets, and index 1 of each
 * component path). */
std::vector<std::string> parameters(const Context& c)
{
    std::vector<std::string> out;
    for (const char* key :
         {"WATOM", "CGRID2B", "SGRID2B", "RCUT2B", "CGRIDMB", "SGRIDMB",
          "RCUTMB", "AMPGRIDMB", "CEMBFUNC", "SEMBFUNC", "RCUTENV"})
        if (c.has_stored(key)) out.push_back(key);
    for (const auto& b : tadah::models::embed_bindings(c))
        out.push_back(b.path);
    return out;
}

/* Value `pos` (1-based) of the parameter `key` of `c`. */
double value(const Context& c, const std::string& key, int pos)
{
    if (key.find('.') != std::string::npos)
        return tadah::models::embed_binding(c, key).values.at(pos - 1);
    return c.get<double>(key, static_cast<std::size_t>(pos - 1));
}

/* `c` with value `pos` of the parameter `key` set to `v`. */
Context with_value(Context c, const std::string& key, int pos, double v)
{
    if (key.find('.') != std::string::npos) {
        tadah::models::embed_binding(c, key).write(
            c, static_cast<std::size_t>(pos - 1), v);
        return c;
    }
    std::vector<std::string> tokens(c.size(key));
    c.get(key, tokens);
    tokens.at(static_cast<std::size_t>(pos - 1)) = num(v);
    c.remove(key);
    for (const auto& t : tokens) c.add(key, t);
    return c;
}

/* ---- the rows -------------------------------------------------------- */

/* The energy rows (one per structure) and the force rows (three per atom)
 * of the design matrix of the descriptors of a Trainer of `c`, as
 * compute_analytical_gradient_from_trainer builds them, without row
 * factors: row-major, descriptor_size() columns. */
struct Rows {
    std::vector<double> energy, force;
};

Rows design_rows(Context c, const StructureDB& db)
{
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    Trainer t(c, ws);
    const std::size_t n = t.DCS.descriptor_size();
    Rows r;
    r.energy.assign(db.size() * n, 0.0);
    r.force.assign(force_rows(db) * n, 0.0);
    std::size_t f0 = 0;
    for (std::size_t s = 0; s < db.size(); ++s) {
        const Structure& st = db(s);
        const tadah::mlip::StDescriptors d = t.dc.calc(st);
        for (std::size_t a = 0; a < d.naed(); ++a)
            for (std::size_t k = 0; k < n; ++k)
                r.energy[s * n + k] += d.get_aed(a)[k];
        for (std::size_t a = 0; a < st.natoms(); ++a) {
            for (std::size_t jj = 0; jj < d.fd[a].size(); ++jj) {
                const std::size_t j = st.nn->idx[a][jj];
                const std::size_t aa = st.get_nn_iindex(a, j, jj);
                const auto& fij = d.fd[a][jj];
                const auto& fji = d.fd[j][aa];
                for (std::size_t x = 0; x < 3; ++x)
                    for (std::size_t k = 0; k < n; ++k)
                        r.force[(f0 + 3 * a + x) * n + k] -=
                            fij(k, x) - fji(k, x);
            }
        }
        f0 += 3 * st.natoms();
    }
    return r;
}

/* The walker's derivative rows of the parameter (`key`, `pos`) on `db`
 * with the descriptors of `t`, the force rows asked when `force`, with the
 * rho cache when `cached`, and its claims. */
struct Walk {
    std::pair<bool, bool> claims;
    Rows rows;
};

Walk walk(const Trainer& t, const StructureDB& db, const std::string& key,
          int pos, bool force, bool cached)
{
    const std::size_t n = t.DCS.descriptor_size();
    Walk w;
    w.rows.energy.assign(db.size() * n, 0.0);
    w.rows.force.assign(force_rows(db) * n, 0.0);
    std::vector<std::size_t> starts;
    std::size_t f0 = 0;
    for (std::size_t s = 0; s < db.size(); ++s) {
        starts.push_back(f0);
        f0 += 3 * db(s).natoms();
    }
    double* f = force ? w.rows.force.data() : nullptr;
    if (cached) {
        tadah::mlip::DMRhoCache cache;
        tadah::mlip::build_dm_rho_cache(db, *t.DCS.dmb, cache, n);
        w.claims = tadah::mlip::assemble_dphi_train_both_rows_cached(
            db, t.DCS, cache, key, pos, w.rows.energy.data(), f, n, starts);
    } else {
        w.claims = tadah::mlip::assemble_dphi_train_both_rows(
            db, t.DCS, key, pos, w.rows.energy.data(), f, n, starts);
    }
    return w;
}

/* The central differences of the rows with the steps h and h/2. */
struct Difference {
    std::vector<double> d1, d2;  // steps h and h/2
    double largest_row = 0.0;    // the largest |row entry| evaluated
};

Difference central_difference(const std::vector<double>& p1,
                              const std::vector<double>& m1,
                              const std::vector<double>& p2,
                              const std::vector<double>& m2, double h)
{
    Difference d;
    for (std::size_t k = 0; k < p1.size(); ++k) {
        d.d1.push_back((p1[k] - m1[k]) / (2.0 * h));
        d.d2.push_back((p2[k] - m2[k]) / h);
        for (double v : {p1[k], m1[k], p2[k], m2[k]})
            d.largest_row = std::max(d.largest_row, std::abs(v));
    }
    return d;
}

/* Whether the derivative rows `a` are the central difference `d` of the
 * rows, step `h`; `worst` names the entry furthest from it.
 *
 * Tolerance. The central difference D(h) of a function with three
 * continuous derivatives has the error c h^2 + O(h^4), so D(h) - D(h/2) is
 * three times the error of D(h/2), and R = D(h/2) + (D(h/2) - D(h))/3
 * (Richardson) removes the h^2 term. Each entry is checked against R with
 * the bound |D(h) - D(h/2)|, many times the remainder of R, plus the
 * round-off of the rows (a few ulp of the largest row entry, divided by
 * the step: 64 eps max|row| / h) and 1e-7 of the block's largest
 * derivative for the O(h^4) term of R. A row block without the part of a
 * descriptor misses by that part, a fraction of the rows' derivative far
 * above the bound. */
bool rows_match(const std::vector<double>& a, const Difference& d, double h,
                std::string& worst)
{
    const double eps = std::numeric_limits<double>::epsilon();
    double scale = 0.0;
    std::vector<double> r(a.size());
    for (std::size_t k = 0; k < a.size(); ++k) {
        r[k] = d.d2[k] + (d.d2[k] - d.d1[k]) / 3.0;
        scale = std::max(scale, std::abs(r[k]));
    }
    double ratio = 0.0;
    bool ok = true;
    for (std::size_t k = 0; k < a.size(); ++k) {
        const double tol = std::abs(d.d1[k] - d.d2[k]) + 1e-7 * scale +
                           64.0 * eps * d.largest_row / h;
        const double miss = std::abs(a[k] - r[k]);
        if (miss > tol) ok = false;
        if (miss / tol > ratio) {
            ratio = miss / tol;
            std::ostringstream o;
            o.precision(10);
            o << "entry " << k << ": walker " << a[k] << ", central "
              << "difference " << r[k] << " (h: " << d.d1[k] << ", h/2: "
              << d.d2[k] << "), bound " << tol;
            worst = o.str();
        }
    }
    return ok;
}

/* The count of row blocks a check compared. */
struct Compared {
    std::size_t energy = 0, force = 0;
};

/* Every row block the walker claims for the parameters of the
 * configuration of `d2` and `dm` on each data set, with the force rows
 * asked and not, with and without the rho cache, is the central difference
 * of the design rows. */
Compared check_claimed_rows(const std::string& d2, const std::string& dm)
{
    tadah::core::PeriodicTable::initialize();
    Compared compared;
    for (const Data& data : data_sets()) {
        const Context base = configuration(d2, dm, data.rc2, data.rcm);
        tadah::mlip::memory::MLIPWorkspaceManager ws;
        Context base_copy = base;
        const Trainer t(base_copy, ws);
        for (const std::string& key : parameters(base)) {
            const int pos = 1;
            const double v = value(base, key, pos);
            const double h = 1e-4 * std::max(1.0, std::abs(v));
            const Rows p1 = design_rows(with_value(base, key, pos, v + h),
                                        data.db);
            const Rows m1 = design_rows(with_value(base, key, pos, v - h),
                                        data.db);
            const Rows p2 = design_rows(
                with_value(base, key, pos, v + 0.5 * h), data.db);
            const Rows m2 = design_rows(
                with_value(base, key, pos, v - 0.5 * h), data.db);
            const Difference de =
                central_difference(p1.energy, m1.energy, p2.energy, m2.energy, h);
            const Difference df =
                central_difference(p1.force, m1.force, p2.force, m2.force, h);
            for (const bool force : {true, false}) {
                for (const bool cached : {false, true}) {
                    const Walk w = walk(t, data.db, key, pos, force, cached);
                    INFO(d2 << " + " << dm << ", " << data.name << ", " << key
                            << " (" << pos << "), force rows "
                            << (force ? "asked" : "not asked")
                            << (cached ? ", rho cache" : "")
                            << ": claims (energy " << w.claims.first
                            << ", force " << w.claims.second << ")");
                    std::string worst;
                    if (w.claims.first) {
                        ++compared.energy;
                        const bool ok = rows_match(w.rows.energy, de, h, worst);
                        INFO("energy rows, " << worst);
                        CHECK(ok);
                    }
                    if (w.claims.second) {
                        ++compared.force;
                        const bool ok = rows_match(w.rows.force, df, h, worst);
                        INFO("force rows, " << worst);
                        CHECK(ok);
                    }
                }
            }
        }
    }
    return compared;
}

const std::vector<std::string>& two_body_descriptors()
{
    static const std::vector<std::string> v = {
        "D2_Blip", "D2_Blip5", "D2_BP",    "D2_Knot5", "D2_LJ",
        "D2_MIE",  "D2_ZBL",   "D2_mJoin", "none"};
    return v;
}

/* The checks of the many-body model `dm` beside every two-body descriptor;
 * some row blocks compared. */
void check_beside_every_two_body(const std::string& dm)
{
    Compared all;
    for (const std::string& d2 : two_body_descriptors()) {
        if (d2 == "none" && dm == "none") continue;
        const Compared c = check_claimed_rows(d2, dm);
        all.energy += c.energy;
        all.force += c.force;
    }
    INFO("row blocks compared: energy " << all.energy << ", force "
                                        << all.force);
    CHECK(all.energy > 0);
}

}  // namespace

/* Pins: every row block (the energy rows; the force rows, when they are
 * asked) that the combined walker claims, with and without the rho cache,
 * equals the central difference of the design rows (tolerance:
 * rows_match), for each two-body descriptor that reads WATOM (D2_Blip,
 * D2_Blip5, D2_BP, D2_Knot5, D2_LJ, D2_MIE, D2_ZBL), D2_mJoin (D2_BP and
 * D2_Blip joined) and none, beside the many-body model of the test case,
 * for index 1 of each parameter of the configuration (WATOM, CGRID2B,
 * SGRID2B, RCUT2B, CGRIDMB, SGRIDMB, RCUTMB, AMPGRIDMB, CEMBFUNC, SEMBFUNC,
 * RCUTENV, the component paths), on a cell and a trimer with pairs inside
 * both cutoffs (4.5 A), and on a dimer inside the two-body or the
 * many-body cutoff only (the other 3 A). Some row blocks are compared.
 * Before the fix: the WATOM row blocks of a two-body descriptor beside
 * DM_REAM (each form), a DESCRIPTOR block, DM_mJoin (lines or blocks) and
 * DM_PS_SQ, and of D2_mJoin beside DM_Blip, DM_EAD, DM_mBlip, DM_mBlipEnv,
 * DM_mRLR and DM_mSQRT, fail: the walker claimed them while they lacked the
 * part of the descriptor without a closed form for WATOM (its claims were
 * joined by an OR with the other descriptor's). The other test cases pass
 * as well (pins). */
TEST_CASE("walker rows: no many-body descriptor, beside each two-body "
          "descriptor, equal the central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("none");
}

TEST_CASE("walker rows: DM_Blip beside each two-body descriptor equal the "
          "central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_Blip");
}

TEST_CASE("walker rows: DM_EAD beside each two-body descriptor equal the "
          "central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_EAD");
}

TEST_CASE("walker rows: DM_mBlip beside each two-body descriptor equal the "
          "central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_mBlip");
}

TEST_CASE("walker rows: DM_mBlipEnv beside each two-body descriptor equal "
          "the central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_mBlipEnv");
}

TEST_CASE("walker rows: DM_mRLR beside each two-body descriptor equal the "
          "central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_mRLR");
}

TEST_CASE("walker rows: DM_mSQRT beside each two-body descriptor equal the "
          "central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_mSQRT");
}

TEST_CASE("walker rows: DM_REAM RHOFORM SIGNED beside each two-body "
          "descriptor equal the central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_REAM SIGNED");
}

TEST_CASE("walker rows: DM_REAM RHOFORM NONNEG beside each two-body "
          "descriptor equal the central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_REAM NONNEG");
}

TEST_CASE("walker rows: DM_REAM RHOFORM MONOTONE beside each two-body "
          "descriptor equal the central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_REAM MONOTONE");
}

TEST_CASE("walker rows: DM_PS_SQ beside each two-body descriptor equal the "
          "central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_PS_SQ");
}

TEST_CASE("walker rows: a DESCRIPTOR block beside each two-body descriptor "
          "equal the central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DESCRIPTOR");
}

TEST_CASE("walker rows: DM_mJoin of DM_mBlip and DM_REAM beside each "
          "two-body descriptor equal the central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_mJoin lines");
}

TEST_CASE("walker rows: DM_mJoin of two DESCRIPTOR blocks beside each "
          "two-body descriptor equal the central difference where claimed",
          "[dphi_dtheta_walk][rows_fd][watom]")
{
    check_beside_every_two_body("DM_mJoin blocks");
}
