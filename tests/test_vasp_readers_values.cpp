// Value tests for the VASP readers: the stored energy, the stored stress
// (sign and unit), the element names of vasprun.xml and the species line of
// POSCAR/CONTCAR files.
//
// Stored convention:
// - Energy: the free energy F, the energy whose gradient the forces are.
//   OUTCAR: the ionic-step line 'free  energy   TOTEN  =' (two spaces
//   between the words; the per-SCF lines read 'free energy    TOTEN').
//   vasprun.xml: the calculation-level <i name="e_fr_energy">.
// - Stress: Cauchy stress, positive under tension, in eV/A^3,
//   sigma = (1/V) dE/d(strain). VASP prints the pressure-signed tensor in kB,
//   so stored = -(kB value) * 6.241509074e-4 (1 kB = 0.1 GPa; e from
//   CODATA 2018). This is what ASE's get_stress() returns.
//
// Reference values: the fixture files' own numbers, read by hand (or with
// python's ElementTree for vasprun.xml), and ASE 3.25
// (ase.io.read(f, index=':'); get_potential_energy(force_consistent=True),
// get_stress(voigt=False)), which agrees with them. ASE converts with
// 0.1 * ase.units.GPa (CODATA 2014 e), which differs from 6.241509074e-4
// by 8.3e-9 relative; the ASE checks below allow 2e-8 relative.
//
// Fixtures: ./tests_data/valid_outcars/OUTCAR_{1,2} and
// ./tests_data/valid_vaspruns/vasprun_{1,2,3}.xml; the POSCAR/CONTCAR
// inputs and the padded vasprun copy are written to a temporary directory
// that each test case removes.

#include "catch2/catch.hpp"

#include "tadah/core/periodic_table.h"

#include <tadah/mlip/dataset_readers/vasp_outcar_reader.h>
#include <tadah/mlip/dataset_readers/vasp_vasprun_reader.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>
#include <tadah/mlip/structure_readers/vasp_poscar_reader.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <tuple>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

using Catch::Matchers::Contains;
using tadah::core::PeriodicTable;
using tadah::mlip::IStructureReader;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::StructureReaderSelector;
using tadah::mlip::VaspOutcarReader;
using tadah::mlip::VaspPoscarReader;
using tadah::mlip::VaspVasprunReader;

namespace {

namespace fs = std::filesystem;

using M3 = std::array<std::array<double, 3>, 3>;

constexpr double kKbToEvA3 = 6.241509074e-4;  // eV/A^3 per kB (CODATA 2018)

const std::string kOutcarDir = "./tests_data/valid_outcars/";
const std::string kVasprunDir = "./tests_data/valid_vaspruns/";

/* The energies are printed with 8 decimals and are read exactly; the
 * candidates the readers could store differ from F by at least 9.9e-4 eV
 * (vasprun_1 calc 1: energy(sigma->0) - F), and the MD bookkeeping line
 * '% ion-electron TOTEN = -555.753322' of OUTCAR_1 by 4.4e-7 eV. */
constexpr double kEnergyMargin = 1e-9;  // eV

/* A scratch directory, unique to this process and test case, removed on
 * exit. */
struct ScratchDir {
  fs::path dir;
  explicit ScratchDir(const std::string &stem)
      : dir(fs::temp_directory_path() /
            ("tadah_vasp_values_" + stem + "_" + std::to_string(::getpid()))) {
    fs::remove_all(dir);
    fs::create_directories(dir);
  }
  ~ScratchDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string write(const std::string &name, const std::string &body) const {
    const fs::path p = dir / name;
    std::ofstream os(p, std::ios::binary);
    os << body;
    return p.string();
  }
  ScratchDir(const ScratchDir &) = delete;
  ScratchDir &operator=(const ScratchDir &) = delete;
};

std::string slurp(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
}

/* -(VASP tensor in kB) in eV/A^3: the stored Cauchy stress. */
M3 cauchy_from_kb(const M3 &kb) {
  M3 s{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) s[i][j] = -kb[i][j] * kKbToEvA3;
  return s;
}

/* The OUTCAR 'in kB' line lists XX YY ZZ XY YZ ZX. */
M3 outcar_kb(double xx, double yy, double zz, double xy, double yz,
             double zx) {
  return {{{xx, xy, zx}, {xy, yy, yz}, {zx, yz, zz}}};
}

void check_stress(const Structure &st, const M3 &expected, double rel,
                  double margin) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      INFO("stress(" << i << "," << j << ")");
      CHECK(st.stress(i, j) ==
            Approx(expected[i][j]).epsilon(rel).margin(margin));
    }
}

/* The six 'in kB' lines of the ionic steps 1-3, shared by OUTCAR_1 and
 * OUTCAR_2 (OUTCAR_1:1879,2790,3649), and the ionic-step free energies
 * (OUTCAR_1:2026,2937,3796). */
const std::vector<M3> kOutcarKb = {
    outcar_kb(1138.75304, 1138.74392, 1138.75347, -0.00095, -0.00086, -0.00003),
    outcar_kb(1138.73855, 1138.73010, 1138.71646, -0.00268, -0.01121, 0.00831),
    outcar_kb(1138.72754, 1138.71193, 1138.59681, 0.01023, -0.00121, 0.02110)};
const std::vector<double> kOutcarF = {-555.75332244, -555.75369711,
                                      -555.75471737};

/* Reads an OUTCAR fixture. OUTCAR_1 has a 4th ionic step cut off after its
 * energy lines, OUTCAR_2 one cut off inside its force block; an MD step is
 * complete at its EKIN_LAT line, so both give 3 structures (ASE reads 4
 * from OUTCAR_1, its last one without a temperature). */
StructureDB read_outcar(const std::string &name) {
  PeriodicTable::initialize();
  StructureDB db;
  VaspOutcarReader reader(db, kOutcarDir + name);
  reader.parse_data();
  return db;
}

StructureDB read_vasprun(const std::string &path) {
  PeriodicTable::initialize();
  StructureDB db;
  VaspVasprunReader reader(db, path);
  reader.parse_data();
  return db;
}

/* Reads a POSCAR/CONTCAR through StructureReaderSelector, the entry point
 * of 'tadah predict -s', 'data print -s', 'data write -s' and 'analysis
 * dataset_stats -s'. */
Structure read_poscar(const std::string &path) {
  PeriodicTable::initialize();
  auto reader = StructureReaderSelector::getReader(path);
  REQUIRE(reader != nullptr);
  REQUIRE(dynamic_cast<VaspPoscarReader *>(reader.get()) != nullptr);
  reader->read(path);
  return reader->getStructure();
}

/* The message of the std::runtime_error that `f` throws; nullopt when it
 * throws nothing (any other exception propagates and fails the test). */
template <typename F>
std::optional<std::string> runtime_error_of(F &&f) {
  try {
    f();
  } catch (const std::runtime_error &e) {
    return std::string(e.what());
  }
  return std::nullopt;
}

/* True when `s` names "line <n>" (and not, for n = 6, "line 60"). */
bool names_line(const std::string &s, int n) {
  const std::string what = "line " + std::to_string(n);
  for (auto p = s.find(what); p != std::string::npos; p = s.find(what, p + 1)) {
    const auto after = p + what.size();
    if (after == s.size() || !std::isdigit(static_cast<unsigned char>(s[after])))
      return true;
  }
  return false;
}

std::vector<std::string> symbols_of(const Structure &st) {
  std::vector<std::string> v;
  for (const auto &a : st.atoms) v.push_back(a.symbol);
  return v;
}

std::vector<std::string> split_lines(const std::string &text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  for (std::string l; std::getline(in, l);) lines.push_back(l);
  return lines;
}

std::string join_lines(const std::vector<std::string> &lines) {
  std::string out;
  for (const auto &l : lines) out += l + "\n";
  return out;
}

/* Replaces `from` by `to` in line `n` (1-based) of `lines`; REQUIREs that
 * the line holds `from`, so a changed fixture fails here, not later. */
void edit_line(std::vector<std::string> &lines, size_t n,
               const std::string &from, const std::string &to) {
  INFO("line " << n << ": '" << lines.at(n - 1) << "'");
  const auto p = lines.at(n - 1).find(from);
  REQUIRE(p != std::string::npos);
  lines[n - 1].replace(p, from.size(), to);
}

/* OUTCAR_1 up to the end of ionic step 3 (line 3825, after its EKIN_LAT
 * block), without the step 4 that the file is cut off inside. */
std::vector<std::string> outcar1_three_steps() {
  std::vector<std::string> l = split_lines(slurp(kOutcarDir + "OUTCAR_1"));
  REQUIRE(l.size() == 4657);
  REQUIRE(l[3819].find("EKIN_LAT=") != std::string::npos);
  l.resize(3825);
  return l;
}

/* The three-step OUTCAR_1 with ionic step 2 (lines 2053-2961) turned into a step of
 * VASP's machine-learned force field (ML_MODE = train), with the markers
 * VASP 6.3+ prints for such a step (as dpdata reads them):
 * 'ML FORCE on cell', 'TOTAL-FORCE (eV/Angst) (ML)', 'ML FREE ENERGIE',
 * 'free  energy ML TOTEN' and 'ML energy  without entropy'. The numbers
 * are left as they are. */
std::vector<std::string> outcar1_with_ml_step2() {
  std::vector<std::string> l = outcar1_three_steps();
  edit_line(l, 2776, "  FORCE on cell", "  ML FORCE on cell");
  edit_line(l, 2871, "TOTAL-FORCE (eV/Angst)", "TOTAL-FORCE (eV/Angst) (ML)");
  edit_line(l, 2935, "  FREE ENERGIE", "  ML FREE ENERGIE");
  edit_line(l, 2937, "free  energy   TOTEN", "free  energy ML TOTEN");
  edit_line(l, 2939, "  energy  without entropy", "  ML energy  without entropy");
  return l;
}

/* `lines` without the MD marker (line 492, 'molecular dynamics for
 * ions') and the EKIN_LAT lines: the layout of a relaxation, whose ionic
 * steps end at their 'free  energy   TOTEN' line. */
std::vector<std::string> as_relaxation(const std::vector<std::string> &lines) {
  std::vector<std::string> out;
  size_t md = 0, ekin = 0;
  for (const auto &l : lines) {
    if (l.find("molecular dynamics") != std::string::npos) { ++md; continue; }
    if (l.find("EKIN_LAT=") != std::string::npos) { ++ekin; continue; }
    out.push_back(l);
  }
  REQUIRE(md == 1);
  REQUIRE(ekin == 3);
  return out;
}

/* Parses an OUTCAR with std::cout and std::cerr captured; returns what was
 * printed. */
std::string parse_outcar_capturing(StructureDB &db, const std::string &path) {
  PeriodicTable::initialize();
  std::ostringstream out, err;
  std::streambuf *old_out = std::cout.rdbuf(out.rdbuf());
  std::streambuf *old_err = std::cerr.rdbuf(err.rdbuf());
  try {
    VaspOutcarReader reader(db, path);
    reader.parse_data();
  } catch (...) {
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    throw;
  }
  std::cout.rdbuf(old_out);
  std::cerr.rdbuf(old_err);
  return out.str() + err.str();
}

size_t count_word(const std::string &text, const std::string &word) {
  size_t n = 0;
  for (auto p = text.find(word); p != std::string::npos;
       p = text.find(word, p + word.size()))
    ++n;
  return n;
}

/* Two structures hold the same data: energy, temperature, cell, stress and
 * atoms (exactly: both come from the same printed numbers). */
void check_same_structure(const Structure &a, const Structure &b) {
  CHECK(a.energy == b.energy);
  CHECK(a.T == b.T);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      INFO("(" << i << "," << j << ")");
      CHECK(a.cell(i, j) == b.cell(i, j));
      CHECK(a.stress(i, j) == b.stress(i, j));
    }
  REQUIRE(a.natoms() == b.natoms());
  for (size_t k = 0; k < a.natoms(); ++k) {
    INFO("atom " << k);
    CHECK(a.atoms[k].symbol == b.atoms[k].symbol);
    for (int c = 0; c < 3; ++c) {
      CHECK(a.atoms[k].position[c] == b.atoms[k].position[c]);
      CHECK(a.atoms[k].force[c] == b.atoms[k].force[c]);
    }
  }
}

}  // namespace

// --------------------------------------------------------------------
// OUTCAR
// --------------------------------------------------------------------

/* Pins: the OUTCAR reader stores the free energy F of each ionic step, read
 * from 'free  energy   TOTEN  =': OUTCAR_1 and OUTCAR_2 give 3 structures
 * with F = -555.75332244, -555.75369711 and -555.75471737 eV (= ASE's
 * force-consistent energy).
 * Before the fix: the reader stored the 'energy  without entropy' value,
 * -554.74525404 for step 1 (1.008 eV above F), so every CHECK on the energy
 * failed. */
TEST_CASE("VASP OUTCAR reader: stores the free energy F of each ionic step",
          "[vasp][outcar][energy]") {
  for (const std::string name : {"OUTCAR_1", "OUTCAR_2"}) {
    INFO(name);
    const StructureDB db = read_outcar(name);
    REQUIRE(db.size() == kOutcarF.size());
    for (size_t k = 0; k < kOutcarF.size(); ++k) {
      INFO("ionic step " << k + 1);
      CHECK(db(k).natoms() == 54);
      CHECK(db(k).energy == Approx(kOutcarF[k]).margin(kEnergyMargin));
    }
  }
}

/* Pins: the OUTCAR reader stores the 'in kB' tensor (XX YY ZZ XY YZ ZX)
 * negated and converted, -(kB) * 6.241509074e-4 eV/A^3, in all six
 * components of every ionic step, symmetric; OUTCAR_1 step 1 gives
 * sigma_xx = -1138.75304 * 6.241509074e-4 = -0.710754 eV/A^3 under the
 * 113.9 GPa compression, ASE's get_stress() to 2e-8 relative.
 * Tolerance: the expectation is the same product of two doubles the reader
 * forms, so 1e-12 relative (margin 1e-15 eV/A^3 for the 1.9e-8 off-diagonal).
 * Before the fix: the stored stress was pressure-signed, +0.710754 for
 * sigma_xx of step 1, so every non-zero component failed with the opposite
 * sign. */
TEST_CASE("VASP OUTCAR reader: stress is minus the in-kB tensor in eV/A^3 "
          "(Cauchy, as ASE)",
          "[vasp][outcar][stress]") {
  for (const std::string name : {"OUTCAR_1", "OUTCAR_2"}) {
    INFO(name);
    const StructureDB db = read_outcar(name);
    REQUIRE(db.size() == kOutcarKb.size());
    for (size_t k = 0; k < kOutcarKb.size(); ++k) {
      INFO("ionic step " << k + 1);
      check_stress(db(k), cauchy_from_kb(kOutcarKb[k]), 1e-12, 1e-15);
    }
  }

  /* ASE 3.25 on OUTCAR_1, image 0, get_stress(voigt=False). */
  const M3 ase_step1 = {{{-0.7107537491287303, 5.929433669589095e-07, 1.8724527377649774e-08},
                         {5.929433669589095e-07, -0.7107480568724075, 5.367697848259602e-07},
                         {1.8724527377649774e-08, 5.367697848259602e-07, -0.7107540175136228}}};
  const StructureDB db = read_outcar("OUTCAR_1");
  REQUIRE(db.size() >= 1);
  INFO("OUTCAR_1 step 1 against ASE");
  check_stress(db(0), ase_step1, 2e-8, 0.0);
}

/* Pins: in an OUTCAR that is not MD (a relaxation, where a step is complete
 * at its ionic-step energy lines), each structure gets the F of its own
 * ionic step and minus its 'in kB' tensor in eV/A^3. The synthetic two-step
 * file has F, energy without entropy and energy(sigma->0) all different
 * (step 1: -42.5, -42.0, -42.25; step 2: -43.125, -42.75, -42.9375), and
 * per-SCF lines 'free energy    TOTEN  = ...' with other values (-41.9,
 * -42.9) before each step's stress, which must not be taken.
 * Before the fix: the reader stored the energy without entropy (-42.0,
 * -42.75) and the pressure-signed stress (+10 kB -> +6.24e-3 eV/A^3), so
 * the energy and stress CHECKs failed. */
TEST_CASE("VASP OUTCAR reader: a relaxation stores F and minus the in-kB "
          "tensor of each ionic step",
          "[vasp][outcar][energy][stress]") {
  ScratchDir tmp("outcar_relax");
  auto step = [](const std::string &scf_f, const std::string &kb,
                 const std::string &x, const std::string &f,
                 const std::string &ewo, const std::string &e0) {
    return "  free energy    TOTEN  =       " + scf_f + " eV\n"
           "  energy without entropy =      " + ewo + "  energy(sigma->0) =      " + e0 + "\n"
           "  FORCE on cell =-STRESS in cart. coord.  units (eV):\n"
           "  in kB  " + kb + "\n\n"
           "      direct lattice vectors                 reciprocal lattice vectors\n"
           "     2.500000000  0.000000000  0.000000000     0.400000000 -0.100000000  0.000000000\n"
           "     0.500000000  2.000000000  0.000000000     0.000000000  0.500000000  0.000000000\n"
           "     0.000000000  0.000000000  3.000000000     0.000000000  0.000000000  0.333333333\n\n"
           " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
           " -----------------------------------------------------------------------------------\n"
           "      " + x + "      0.00000      0.00000        -0.010000      0.020000     -0.030000\n"
           " -----------------------------------------------------------------------------------\n\n"
           "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n"
           "  ---------------------------------------------------\n"
           "  free  energy   TOTEN  =      " + f + " eV\n\n"
           "  energy  without entropy=     " + ewo + "  energy(sigma->0) =     " + e0 + "\n\n";
  };
  const std::string body =
      "  Synthetic two-step relaxation OUTCAR (not MD).\n"
      " POTCAR:    PAW_PBE Ta_pv 07Sep2000\n"
      "   VRHFIN =Ta:  5p6s5d\n"
      "   number of dos      NEDOS =    301   number of ions     NIONS =      1\n"
      "   ions per type =               1\n" +
      step("-41.900000", "10.00000  20.00000  30.00000  1.00000  2.00000  3.00000",
           "0.00000", "-42.500000", "-42.000000", "-42.250000") +
      step("-42.900000", "-5.00000  -6.00000  -7.00000  -0.50000  0.25000  -0.12500",
           "0.10000", "-43.125000", "-42.750000", "-42.937500");
  const std::string path = tmp.write("OUTCAR_relax", body);

  PeriodicTable::initialize();
  StructureDB db;
  VaspOutcarReader reader(db, path);
  reader.parse_data();
  REQUIRE(db.size() == 2);
  CHECK(db(0).energy == Approx(-42.5).margin(kEnergyMargin));
  CHECK(db(1).energy == Approx(-43.125).margin(kEnergyMargin));
  check_stress(db(0), cauchy_from_kb(outcar_kb(10, 20, 30, 1, 2, 3)), 1e-12, 1e-15);
  check_stress(db(1), cauchy_from_kb(outcar_kb(-5, -6, -7, -0.5, 0.25, -0.125)), 1e-12, 1e-15);
  REQUIRE(db(1).natoms() == 1);
  CHECK(db(1).atoms[0].position[0] == Approx(0.1).margin(1e-12));
}

/* Pins: an OUTCAR from an on-the-fly training run (ML_MODE = train)
 * holds ionic steps computed by VASP's machine-learned force field, marked
 * 'ML'. They are not first-principles data: the reader stores only the
 * first-principles steps, exactly as it stores them from the file without
 * ML steps, and prints one WARNING naming the file and the skipped step.
 * - MD layout (OUTCAR_1's first three steps, step 2 marked ML): steps 1
 *   and 3.
 * - Relaxation layout (the same file without its MD marker and EKIN_LAT
 *   lines, so each step ends at its energy line): steps 1 and 3.
 * Each label ends with 'Structure <n>', n the ionic step.
 * Before the fix: the MD layout stored step 2 with energy 0 and the ML
 * forces and stress, silently (3 structures); the relaxation layout added
 * step 2's 54 ML atoms to step 3 (a 108-atom structure 2 with step 3's
 * energy), silently. */
TEST_CASE("VASP OUTCAR reader: steps of the machine-learned force field are "
          "skipped with one WARNING",
          "[vasp][outcar][ml_ff]") {
  ScratchDir tmp("outcar_ml");
  SECTION("MD layout") {
    StructureDB ref;
    parse_outcar_capturing(
        ref, tmp.write("OUTCAR_md_ref", join_lines(outcar1_three_steps())));
    REQUIRE(ref.size() == 3);
    const std::string path =
        tmp.write("OUTCAR_ml_md", join_lines(outcar1_with_ml_step2()));
    StructureDB db;
    const std::string printed = parse_outcar_capturing(db, path);
    INFO("printed: '" << printed << "'");
    CHECK(count_word(printed, "WARNING") == 1);
    CHECK_THAT(printed, Contains(path));
    CHECK_THAT(printed, Contains("ionic step 2"));
    REQUIRE(db.size() == 2);
    {
      INFO("stored 1 = step 1");
      check_same_structure(db(0), ref(0));
    }
    {
      INFO("stored 2 = step 3");
      check_same_structure(db(1), ref(2));
    }
    CHECK(db(1).energy == Approx(kOutcarF[2]).margin(kEnergyMargin));
    // The label numbers the ionic step, as the WARNING does.
    CHECK_THAT(db(0).label, Catch::Matchers::EndsWith("Structure 1"));
    CHECK_THAT(db(1).label, Catch::Matchers::EndsWith("Structure 3"));
  }
  SECTION("relaxation layout") {
    const std::string ref_path = tmp.write(
        "OUTCAR_relax_ref", join_lines(as_relaxation(outcar1_three_steps())));
    StructureDB ref;
    const std::string ref_printed = parse_outcar_capturing(ref, ref_path);
    INFO("reference printed: '" << ref_printed << "'");
    CHECK(count_word(ref_printed, "WARNING") == 0);
    REQUIRE(ref.size() == 3);
    CHECK(ref(1).energy == Approx(kOutcarF[1]).margin(kEnergyMargin));

    const std::string path = tmp.write(
        "OUTCAR_ml_relax", join_lines(as_relaxation(outcar1_with_ml_step2())));
    StructureDB db;
    const std::string printed = parse_outcar_capturing(db, path);
    INFO("printed: '" << printed << "'");
    CHECK(count_word(printed, "WARNING") == 1);
    CHECK_THAT(printed, Contains(path));
    CHECK_THAT(printed, Contains("ionic step 2"));
    REQUIRE(db.size() == 2);
    const size_t steps[2] = {0, 2};
    for (size_t k = 0; k < 2; ++k) {
      INFO("stored " << k + 1 << " = step " << steps[k] + 1);
      check_same_structure(db(k), ref(steps[k]));
      CHECK_THAT(db(k).label, Catch::Matchers::EndsWith(
                                  "Structure " + std::to_string(steps[k] + 1)));
    }
  }
}

/* Pins: an MD file cut off inside its last ionic step -- after the step's
 * free energy, before its EKIN_LAT line -- gives the complete steps and
 * one WARNING naming the file and the skipped step: OUTCAR_1 (cut off after
 * step 4's energy lines) gives steps 1-3 and 'ionic step 4'. A relaxation
 * step ends at its energy line, so the same cut gives no warning there.
 * Before the fix: step 4 was dropped silently. */
TEST_CASE("VASP OUTCAR reader: a last MD step the file ends inside is skipped "
          "with one WARNING",
          "[vasp][outcar][truncated]") {
  const std::string path = kOutcarDir + "OUTCAR_1";
  StructureDB db;
  const std::string printed = parse_outcar_capturing(db, path);
  INFO("printed: '" << printed << "'");
  CHECK(count_word(printed, "WARNING") == 1);
  CHECK_THAT(printed, Contains(path));
  CHECK_THAT(printed, Contains("ionic step 4"));
  REQUIRE(db.size() == 3);
  CHECK(db(2).energy == Approx(kOutcarF[2]).margin(kEnergyMargin));
}

/* Pins: an ionic step without an 'in kB' line is stored with zero stress
 * and stress priority 0, so a STRESS fit ignores it; a step with one keeps
 * weight 1: the first three steps of OUTCAR_1 give sweight 1, the same
 * file without its 'in kB' lines sweight 0 (and the same energies).
 * Before the fix: zero stress with sweight 1. */
TEST_CASE("VASP OUTCAR reader: a step without a stress tensor has stress "
          "priority 0",
          "[vasp][outcar][stress]") {
  ScratchDir tmp("outcar_no_stress");
  const std::vector<std::string> l = outcar1_three_steps();
  std::vector<std::string> no_kb;
  for (const auto &line : l)
    if (line.find("in kB") == std::string::npos) no_kb.push_back(line);
  REQUIRE(no_kb.size() == l.size() - 3);
  StructureDB with, without;
  parse_outcar_capturing(with, tmp.write("OUTCAR_with", join_lines(l)));
  parse_outcar_capturing(without, tmp.write("OUTCAR_without", join_lines(no_kb)));
  REQUIRE(with.size() == 3);
  REQUIRE(without.size() == 3);
  for (size_t k = 0; k < 3; ++k) {
    INFO("step " << k + 1);
    CHECK(with(k).sweight == 1.0);
    CHECK(without(k).sweight == 0.0);
    CHECK(without(k).energy == with(k).energy);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) CHECK(without(k).stress(i, j) == 0.0);
  }
}

/* Pins: an 'in kB' line that cannot be read does not count as a stress
 * tensor: the first three steps of OUTCAR_1 with step 1's line turned into
 * '  in kB  ****** ...' (VASP's overflow) give step 1 zero stress with stress
 * weight 0; steps 2 and 3 keep theirs.
 * Before the fix: step 1 was stored with zero stress and weight 1. */
TEST_CASE("VASP OUTCAR reader: an unreadable stress line gives stress priority 0",
          "[vasp][outcar][stress]") {
  ScratchDir tmp("outcar_bad_kb");
  std::vector<std::string> l = outcar1_three_steps();
  edit_line(l, 1879, "1138.75304  1138.74392", "**********  **********");
  StructureDB db;
  parse_outcar_capturing(db, tmp.write("OUTCAR_bad_kb", join_lines(l)));
  REQUIRE(db.size() == 3);
  CHECK(db(0).sweight == 0.0);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) CHECK(db(0).stress(i, j) == 0.0);
  CHECK(db(1).sweight == 1.0);
  CHECK(db(2).sweight == 1.0);
}

/* Pins: an MD step that reaches its EKIN_LAT line with no
 * 'free  energy   TOTEN' line, and is not an ML step, is refused: the
 * error names the file and the ionic step (OUTCAR_1 without line 2937,
 * step 2's energy).
 * Before the fix: the step was stored with energy 0, silently. */
TEST_CASE("VASP OUTCAR reader: an MD step without its free energy is refused",
          "[vasp][outcar][ml_ff]") {
  ScratchDir tmp("outcar_no_energy");
  std::vector<std::string> l = split_lines(slurp(kOutcarDir + "OUTCAR_1"));
  REQUIRE(l.size() == 4657);
  REQUIRE(l[2936].find("free  energy   TOTEN") != std::string::npos);
  l.erase(l.begin() + 2936);
  const std::string path = tmp.write("OUTCAR_no_energy", join_lines(l));
  StructureDB db;
  const auto what = runtime_error_of([&] { parse_outcar_capturing(db, path); });
  REQUIRE(what.has_value());
  CHECK_THAT(*what, Contains(path));
  CHECK_THAT(*what, Contains("ionic step 2"));
  CHECK_THAT(*what, Contains("free  energy   TOTEN"));
}

// --------------------------------------------------------------------
// vasprun.xml
// --------------------------------------------------------------------

/* Pins: the vasprun.xml reader stores the calculation-level e_fr_energy:
 * vasprun_2 calcs 1-4, vasprun_1 calcs 1-25 and vasprun_3 (= ASE's
 * force-consistent energy).
 * Before the fix: the reader stored the calculation-level e_wo_entrp, which
 * VASP 5.4.4 fills with energy(sigma->0): -620.27193646 for vasprun_2
 * calc 1 (0.071 eV above F), -406.06282522 for vasprun_1 calc 1 and
 * -620.05426408 for vasprun_3, so every CHECK on the energy failed. */
TEST_CASE("VASP vasprun.xml reader: stores the calculation-level e_fr_energy",
          "[vasp][vasprun][energy]") {
  SECTION("vasprun_2.xml") {
    const std::vector<double> F = {-620.34293557, -620.34271140, -620.34063177,
                                   -620.33662535};
    const StructureDB db = read_vasprun(kVasprunDir + "vasprun_2.xml");
    REQUIRE(db.size() == F.size());
    for (size_t k = 0; k < F.size(); ++k) {
      INFO("calculation " << k + 1);
      CHECK(db(k).natoms() == 53);
      CHECK(db(k).energy == Approx(F[k]).margin(kEnergyMargin));
    }
  }
  SECTION("vasprun_1.xml") {
    const std::vector<double> F = {
        -406.06381148, -406.24823354, -406.40599766, -406.52738658, -406.60501012,
        -406.63313325, -406.60893328, -406.53078805, -406.40003557, -406.22162279,
        -406.00300096, -405.75511774, -405.49228440, -405.23022935, -404.98604891,
        -404.77708674, -404.61804730, -404.51971852, -404.48778220, -404.52174048,
        -404.61554997, -404.75824364, -404.93472093, -405.12765379, -405.31996675};
    const StructureDB db = read_vasprun(kVasprunDir + "vasprun_1.xml");
    REQUIRE(db.size() == F.size());
    for (size_t k = 0; k < F.size(); ++k) {
      INFO("calculation " << k + 1);
      CHECK(db(k).natoms() == 49);
      CHECK(db(k).energy == Approx(F[k]).margin(kEnergyMargin));
    }
  }
  SECTION("vasprun_3.xml") {
    const StructureDB db = read_vasprun(kVasprunDir + "vasprun_3.xml");
    REQUIRE(db.size() == 1);
    CHECK(db(0).natoms() == 53);
    CHECK(db(0).energy == Approx(-620.05621075).margin(kEnergyMargin));
  }
}

/* Pins: the vasprun.xml reader stores the <varray name="stress"> tensor
 * negated and converted, -(kB) * 6.241509074e-4 eV/A^3, for every
 * calculation of vasprun_2 and for vasprun_3; vasprun_2 calc 1 gives
 * sigma_xx = -33.62613502 * 6.241509074e-4 = -0.0209878 eV/A^3, ASE's
 * get_stress() to 2e-8 relative. vasprun_1 has no stress varray and its
 * stress stays zero (that part held before the fix; it fails if the reader
 * stopped zeroing the stress of a calculation without one), with stress
 * weight 0 so that a STRESS fit ignores it (vasprun_2's calculations keep
 * weight 1; before an earlier fix the weight was 1).
 * Tolerance: VASP prints the tensor slightly asymmetric (vasprun_3:
 * 0.82409354 against 0.82409317 kB, 2.3e-10 eV/A^3); the absolute margin
 * of 1e-9 eV/A^3 accepts storing the tensor as printed, its upper triangle
 * (as ASE) or its symmetric part, and is 2600 times smaller than the
 * smallest off-diagonal here (2.6e-6 eV/A^3), so a sign error still fails.
 * Before the fix: the stored stress was pressure-signed (+0.0209878 for
 * vasprun_2 calc 1 sigma_xx), so every component failed with the opposite
 * sign. */
TEST_CASE("VASP vasprun.xml reader: stress is minus the stress varray in "
          "eV/A^3 (Cauchy, as ASE)",
          "[vasp][vasprun][stress]") {
  SECTION("vasprun_2.xml, every calculation") {
    const std::vector<M3> kb = {
        {{{33.62613502, -0.00413452, 0.01904932},
          {-0.00413448, 33.63453289, -0.00929155},
          {0.01904936, -0.00929140, 33.63908924}}},
        {{{34.20764133, -0.03260938, -0.05518298},
          {-0.03260944, 34.38693047, 0.03986634},
          {-0.05518284, 0.03986629, 34.27071972}}},
        {{{34.19853757, -0.08497141, -0.09564246},
          {-0.08497146, 34.57822695, 0.08245903},
          {-0.09564233, 0.08245888, 34.36570133}}},
        {{{34.05434970, -0.13126358, -0.13244993},
          {-0.13126350, 34.62456569, 0.12366972},
          {-0.13244991, 0.12366955, 34.30672781}}}};
    const StructureDB db = read_vasprun(kVasprunDir + "vasprun_2.xml");
    REQUIRE(db.size() == kb.size());
    for (size_t k = 0; k < kb.size(); ++k) {
      INFO("calculation " << k + 1);
      check_stress(db(k), cauchy_from_kb(kb[k]), 0.0, 1e-9);
    }
    /* ASE 3.25, image 0, get_stress(voigt=False) (upper triangle). */
    const M3 ase_calc1 = {{{-0.020987782859551263, 2.580564431114685e-06, -1.1889650462187049e-05},
                           {2.580564431114685e-06, -0.02099302439777556, 5.799329411860059e-06},
                           {-1.1889650462187049e-05, 5.799329411860059e-06, -0.02099586824778613}}};
    INFO("vasprun_2 calc 1 against ASE");
    check_stress(db(0), ase_calc1, 2e-8, 1e-9);
  }
  SECTION("vasprun_3.xml") {
    const M3 kb = {{{3.43877172, -1.12842186, 0.82409354},
                    {-1.12842213, 3.96178418, 1.28067635},
                    {0.82409317, 1.28067632, 4.03328211}}};
    const StructureDB db = read_vasprun(kVasprunDir + "vasprun_3.xml");
    REQUIRE(db.size() == 1);
    check_stress(db(0), cauchy_from_kb(kb), 0.0, 1e-9);
    const M3 ase = {{{-0.0021463125072209268, 0.000704305533703616, -0.000514358735049144},
                     {0.000704305533703616, -0.002472751211424992, -0.0007993353125827862},
                     {-0.000514358735049144, -0.0007993353125827862, -0.0025173767096826685}}};
    INFO("vasprun_3 against ASE");
    check_stress(db(0), ase, 2e-8, 1e-9);
  }
  SECTION("vasprun_1.xml has no stress varray: zero stress, priority 0") {
    const StructureDB db = read_vasprun(kVasprunDir + "vasprun_1.xml");
    REQUIRE(db.size() == 25);
    for (size_t k = 0; k < db.size(); ++k) {
      INFO("calculation " << k + 1);
      check_stress(db(k), M3{}, 0.0, 0.0);
      CHECK(db(k).sweight == 0.0);
    }
  }
  SECTION("vasprun_2.xml has a stress varray: weight 1") {
    const StructureDB db = read_vasprun(kVasprunDir + "vasprun_2.xml");
    REQUIRE(db.size() == 4);
    for (size_t k = 0; k < db.size(); ++k) {
      INFO("calculation " << k + 1);
      CHECK(db(k).sweight == 1.0);
    }
  }
}

/* Pins: element names in <atominfo> are trimmed. VASP pads them to two
 * characters ('<c>O </c>'), so a copy of vasprun_3.xml whose 53 Ta are
 * written '<c>W </c>' (atoms and atomtypes arrays) reads as 53 W atoms
 * (Z = 74), as ASE reads it.
 * Before the fix: parse_data() threw 'Unknown symbol: W ' (the name with
 * its trailing blank), so REQUIRE_NOTHROW failed. */
TEST_CASE("VASP vasprun.xml reader: padded element names are trimmed",
          "[vasp][vasprun][species]") {
  ScratchDir tmp("padded_w");
  std::string xml = slurp(kVasprunDir + "vasprun_3.xml");
  const std::string from = "<c>Ta</c>", to = "<c>W </c>";
  size_t n = 0;
  for (auto p = xml.find(from); p != std::string::npos;
       p = xml.find(from, p + to.size())) {
    xml.replace(p, from.size(), to);
    ++n;
  }
  REQUIRE(n == 54);  // 53 in the atoms array, 1 in the atomtypes array
  const std::string path = tmp.write("vasprun_W_padded.xml", xml);

  PeriodicTable::initialize();
  StructureDB db;
  VaspVasprunReader reader(db, path);
  REQUIRE_NOTHROW(reader.parse_data());
  REQUIRE(db.size() == 1);
  REQUIRE(db(0).natoms() == 53);
  for (size_t i = 0; i < db(0).natoms(); ++i) {
    INFO("atom " << i);
    CHECK(db(0).atoms[i].symbol == "W");
    CHECK(db(0).atoms[i].Z == 74);
  }
}

/* Pins: a <calculation> without a readable calculation-level
 * <i name="e_fr_energy"> is refused, and the error names the file and the
 * calculation (1-based): vasprun_3.xml with that line removed, and with
 * its value replaced by 'abc'. The per-SCF e_fr_energy lines inside
 * <scstep> are not taken instead.
 * Before the fix: the refusal named neither the file nor the calculation
 * (and the unreadable value was reported on std::cerr only). */
TEST_CASE("VASP vasprun.xml reader: a calculation without e_fr_energy is "
          "refused, naming the file and the calculation",
          "[vasp][vasprun][energy]") {
  ScratchDir tmp("vasprun_no_efr");
  const std::string xml = slurp(kVasprunDir + "vasprun_3.xml");
  // The calculation-level line (indented 3 spaces; the last SCF step prints
  // the same value indented 4).
  const std::string line = "\n   <i name=\"e_fr_energy\">   -620.05621075 </i>";
  const auto p = xml.find(line);
  REQUIRE(p != std::string::npos);
  REQUIRE(xml.find(line, p + 1) == std::string::npos);
  SECTION("missing") {
    std::string bad = xml;
    bad.erase(p, line.size());
    const std::string path = tmp.write("vasprun_missing_efr.xml", bad);
    const auto what = runtime_error_of([&] { read_vasprun(path); });
    REQUIRE(what.has_value());
    CHECK_THAT(*what, Contains(path));
    CHECK_THAT(*what, Contains("calculation 1"));
    CHECK_THAT(*what, Contains("e_fr_energy"));
  }
  SECTION("not a number") {
    std::string bad = xml;
    bad.replace(p, line.size(), "\n   <i name=\"e_fr_energy\">   abc </i>");
    const std::string path = tmp.write("vasprun_abc_efr.xml", bad);
    const auto what = runtime_error_of([&] { read_vasprun(path); });
    REQUIRE(what.has_value());
    CHECK_THAT(*what, Contains(path));
    CHECK_THAT(*what, Contains("calculation 1"));
    CHECK_THAT(*what, Contains("abc"));
  }
}

// --------------------------------------------------------------------
// POSCAR / CONTCAR species line
// --------------------------------------------------------------------

/* Pins: line 6 of a POSCAR/CONTCAR may hold POTCAR names, as VASP 6 writes
 * them: each token is cut at '/', then at '_', and what is left must be an
 * element. 'Ta_pv/7a3e21f0' -> Ta, 'Ti_pv O_s' -> Ti O,
 * 'Ti_pv/8c1b2d3e  O/a4f1c2d9' -> Ti O, 'Si/b0d4a2' -> Si (pymatgen reads
 * them so). The plain 'Ta' and 'Ti O' files read as before (they passed
 * before the fix). Positions are checked on one atom of each file, by
 * hand: r = f0 a0 + f1 a1 + f2 a2.
 * Before the fix: every POTCAR-name file was refused with 'Unknown symbol:
 * Ta_pv/7a3e21f0' (resp. 'Ti_pv', 'Ti_pv/8c1b2d3e', 'Si/b0d4a2'), so the
 * read threw and the test failed at the first such file. */
TEST_CASE("VASP POSCAR reader: VASP 6 POTCAR names on line 6 are read as "
          "elements",
          "[vasp][poscar][species]") {
  ScratchDir tmp("potnames");
  const std::string bcc_ta_tail =
      "2\nDirect\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n";
  const std::string bcc_ta_head =
      "bcc Ta\n3.3\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n";
  const std::string rutile_head =
      "TiO2 rutile\n1.0\n   4.594000   0.000000   0.000000\n"
      "   0.000000   4.594000   0.000000\n   0.000000   0.000000   2.959000\n";
  const std::string rutile_tail =
      "2 4\nDirect\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n 0.3053 0.3053 0.0\n"
      " 0.6947 0.6947 0.0\n 0.8053 0.1947 0.5\n 0.1947 0.8053 0.5\n";
  const std::vector<std::string> rutile = {"Ti", "Ti", "O", "O", "O", "O"};

  struct Case {
    std::string name, body;
    std::vector<std::string> symbols;
    size_t atom;
    std::array<double, 3> position;
  };
  const std::vector<Case> cases = {
      {"POSCAR_plain_Ta", bcc_ta_head + "Ta\n" + bcc_ta_tail, {"Ta", "Ta"}, 1,
       {1.65, 1.65, 1.65}},
      {"CONTCAR_Ta_pv_hash", bcc_ta_head + "Ta_pv/7a3e21f0\n" + bcc_ta_tail,
       {"Ta", "Ta"}, 1, {1.65, 1.65, 1.65}},
      {"POSCAR_plain_TiO", rutile_head + "Ti O\n" + rutile_tail, rutile, 4,
       {0.8053 * 4.594, 0.1947 * 4.594, 0.5 * 2.959}},
      {"POSCAR_potnames", rutile_head + "Ti_pv O_s\n" + rutile_tail, rutile, 4,
       {0.8053 * 4.594, 0.1947 * 4.594, 0.5 * 2.959}},
      {"CONTCAR_vasp6", rutile_head + "Ti_pv/8c1b2d3e  O/a4f1c2d9\n" + rutile_tail,
       rutile, 2, {0.3053 * 4.594, 0.3053 * 4.594, 0.0}},
      {"POSCAR_si6",
       "Si diamond, VASP 6 CONTCAR style\n   5.43\n     0.0 0.5 0.5\n"
       "     0.5 0.0 0.5\n     0.5 0.5 0.0\n   Si/b0d4a2\n     2\nDirect\n"
       "  0.00 0.00 0.00\n  0.25 0.25 0.25\n",
       {"Si", "Si"}, 1, {1.3575, 1.3575, 1.3575}}};

  for (const auto &c : cases) {
    INFO(c.name << ":\n" << c.body);
    const std::string path = tmp.write(c.name, c.body);
    Structure st;
    REQUIRE_NOTHROW(st = read_poscar(path));
    CHECK(symbols_of(st) == c.symbols);
    for (const auto &a : st.atoms) {
      INFO("atom " << a.symbol);
      CHECK(a.Z == PeriodicTable::find_by_symbol(a.symbol).Z);
    }
    REQUIRE(st.natoms() > c.atom);
    for (int k = 0; k < 3; ++k)
      CHECK(st.atoms[c.atom].position[k] == Approx(c.position[k]).margin(1e-12));
  }
}

/* Pins: a line-6 token that is still no element after the POTCAR suffixes
 * are cut ('H.75/abc', a fractional-valence pseudo-hydrogen, cut to 'H.75';
 * 'Xx'; 'Xx_pv' in the second column) is refused: the read throws a
 * std::runtime_error that names the token and says it is on line 6.
 * Before the fix: the read threw 'Unknown symbol: H.75/abc' (resp. 'Xx'),
 * which names the token but not the line, so the "line 6" CHECKs failed;
 * for 'Ta_pv Xx_pv' it threw 'Unknown symbol: Ta_pv', refusing the valid
 * first token, so the token CHECK failed as well. */
TEST_CASE("VASP POSCAR reader: a line-6 token that is no element is refused, "
          "naming the token and line 6",
          "[vasp][poscar][species]") {
  ScratchDir tmp("bad_species");
  struct Case {
    std::string name, species, counts, token;
  };
  const std::vector<Case> cases = {{"POSCAR_H75", "H.75/abc", "2", "H.75"},
                                   {"POSCAR_Xx", "Xx", "2", "Xx"},
                                   {"POSCAR_Ta_Xx", "Ta_pv Xx_pv", "1 1", "Xx"}};
  for (const auto &c : cases) {
    const std::string body = "refused species\n3.3\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n"
                             " 0.0 0.0 1.0\n" + c.species + "\n" + c.counts +
                             "\nDirect\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n";
    INFO(c.name << ":\n" << body);
    const std::string path = tmp.write(c.name, body);
    const auto msg = runtime_error_of([&] { (void)read_poscar(path); });
    REQUIRE(msg);
    INFO("message: " << *msg);
    CHECK_THAT(*msg, Contains(c.token));
    CHECK(names_line(*msg, 6));
  }
}

/* Pins: an error of the POSCAR reader names the file, as
 *   VaspPoscarReader: POSCAR file <path>: <what>
 * for a line-6 token that is no element ('Xx': exactly "... <path>: 'Xx'
 * on line 6 is not an element symbol (or a VASP POTCAR name such as Ti_pv
 * or Si/b0d4a2)."), a scaling factor 0, an atom count 'two' on line 7
 * ("... the atom count 'two' on line 7 is not a whole number"), and a file
 * of 5 lines. A reader that read a good POSCAR (2 Ta atoms) keeps its
 * structure (cell and atoms) after each refused file, and reading the good
 * file again gives its 2 atoms, not 4.
 * Before the fix: the messages began "VaspPoscarReader: " or
 * "VaspPoscarReader::read: " and named no file, and the count was refused
 * by std::stoi ('stoi', a std::invalid_argument, which ends the case); the
 * file refused on line 6 had already replaced the reader's cell (4 A), so
 * every later cell check failed too. */
TEST_CASE("VASP POSCAR reader: an error names the file, and a refused file "
          "leaves the structure",
          "[vasp][poscar][species]") {
  PeriodicTable::initialize();
  ScratchDir tmp("poscar_source");
  const std::string good = tmp.write(
      "POSCAR_good", "bcc Ta\n3.3\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n"
                     "Ta\n2\nDirect\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n");
  struct Bad {
    std::string name, body, what;
  };
  const std::vector<Bad> bad = {
      {"POSCAR_Xx",
       "unknown element\n4.0\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n"
       "Xx\n2\nDirect\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n",
       "'Xx' on line 6 is not an element symbol (or a VASP POTCAR name such "
       "as Ti_pv or Si/b0d4a2)."},
      {"POSCAR_scale0",
       "zero scale\n0.0\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n"
       "Ta\n2\nDirect\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n",
       "scaling factor on line 2"},
      {"POSCAR_count",
       "bad count\n4.0\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n"
       "Ta\ntwo\nDirect\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n",
       "the atom count 'two' on line 7 is not a whole number"},
      {"POSCAR_short", "short\n4.0\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n",
       "not enough lines"}};
  auto reader = StructureReaderSelector::getReader(good);
  REQUIRE(dynamic_cast<VaspPoscarReader *>(reader.get()) != nullptr);
  REQUIRE_NOTHROW(reader->read(good));
  const Structure before = reader->getStructure();
  REQUIRE(before.natoms() == 2);
  for (const auto &b : bad) {
    const std::string path = tmp.write(b.name, b.body);
    INFO(b.name << ":\n" << b.body);
    const auto msg = runtime_error_of([&] { reader->read(path); });
    REQUIRE(msg);
    INFO("message: " << *msg);
    const std::string head = "VaspPoscarReader: POSCAR file " + path + ": ";
    CHECK(msg->rfind(head, 0) == 0);
    CHECK_THAT(*msg, Contains(b.what));
    if (b.name == "POSCAR_Xx") CHECK(*msg == head + b.what);
    const Structure after = reader->getStructure();
    CHECK(after.label == before.label);
    CHECK(after.natoms() == 2);
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k) CHECK(after.cell(i, k) == before.cell(i, k));
  }
  REQUIRE_NOTHROW(reader->read(good));
  CHECK(reader->getStructure().natoms() == 2);
}

/* Pins: a POSCAR atom count is a whole number written with digits, after at
 * most one '+' (as VASP's integer read, ASE and pymatgen accept it):
 *  - "+2" on line 7 (after the element line "Ta") reads 2 Ta atoms, and
 *    "+1 +1" after "Ta W" one of each;
 *  - "+2" on line 6, where the element symbols belong, is taken for the
 *    counts of the VASP 4 layout and refused as such ("line 6 holds the atom
 *    counts, not the element symbols (the VASP 4 layout)"), as "2" is;
 *  - "-2", "+", "++2", "+-2" and "2.0" on line 7 are refused, "the atom
 *    count '<token>' on line 7 is not a whole number", and "4294967296" as
 *    "the atom count '4294967296' on line 7 is too large".
 * Before the fix: std::from_chars took no '+', so "+2" and "+1 +1" were
 * refused ("the atom count '+2' on line 7 is not a whole number": std::stoi
 * read them in the released code); "+2" on line 6 was taken for the element
 * symbols, so "Direct" on line 7 was read as the counts ("the atom count
 * 'Direct' on line 7 is not a whole number"; std::stoi's 'stoi' in the
 * released code); "4294967296" was "not a whole number". */
TEST_CASE("VASP POSCAR reader: an atom count may carry a '+' sign",
          "[vasp][poscar]") {
  PeriodicTable::initialize();
  ScratchDir tmp("poscar_plus_count");
  const std::string cell = "4.0\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n";
  const std::string coords = "Direct\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n";
  SECTION("+2 after Ta") {
    const std::string path = tmp.write("POSCAR_plus", "plus\n" + cell + "Ta\n+2\n" + coords);
    VaspPoscarReader reader;
    REQUIRE_NOTHROW(reader.read(path));
    const Structure st = reader.getStructure();
    REQUIRE(st.natoms() == 2);
    CHECK(st.atoms[0].symbol == "Ta");
    CHECK(st.atoms[1].symbol == "Ta");
  }
  SECTION("+1 +1 after Ta W") {
    const std::string path =
        tmp.write("POSCAR_plus_two", "plus two\n" + cell + "Ta W\n+1 +1\n" + coords);
    VaspPoscarReader reader;
    REQUIRE_NOTHROW(reader.read(path));
    const Structure st = reader.getStructure();
    REQUIRE(st.natoms() == 2);
    CHECK(st.atoms[0].symbol == "Ta");
    CHECK(st.atoms[1].symbol == "W");
  }
  SECTION("counts on line 6") {
    for (const std::string counts : {"2", "+2"}) {
      INFO("line 6: " << counts);
      const std::string path =
          tmp.write("POSCAR_vasp4", "vasp 4\n" + cell + counts + "\n" + coords);
      VaspPoscarReader reader;
      const auto msg = runtime_error_of([&] { reader.read(path); });
      REQUIRE(msg);
      CHECK_THAT(*msg, Contains("line 6 holds the atom counts, not the element symbols (the "
                                "VASP 4 layout)"));
    }
  }
  SECTION("counts refused on line 7") {
    const std::vector<std::pair<std::string, std::string>> refused = {
        {"-2", "is not a whole number"},  {"+", "is not a whole number"},
        {"++2", "is not a whole number"}, {"+-2", "is not a whole number"},
        {"2.0", "is not a whole number"}, {"4294967296", "is too large"}};
    for (const auto &r : refused) {
      INFO("line 7: " << r.first);
      const std::string path =
          tmp.write("POSCAR_count", "count\n" + cell + "Ta\n" + r.first + "\n" + coords);
      VaspPoscarReader reader;
      const auto msg = runtime_error_of([&] { reader.read(path); });
      REQUIRE(msg);
      CHECK_THAT(*msg, Contains("the atom count '" + r.first + "' on line 7 " + r.second));
    }
  }
}

// --------------------------------------------------------------------
// Every error names the file (and the line, step or calculation)
// --------------------------------------------------------------------

namespace {

/* The message of the std::runtime_error parsing the OUTCAR `lines` throws
 * (written to `name` in `tmp`), or "no exception"; `path` receives the
 * file's path. */
std::string outcar_refusal(const ScratchDir &tmp, const std::string &name,
                           const std::vector<std::string> &lines,
                           std::string &path) {
  path = tmp.write(name, join_lines(lines));
  StructureDB db;
  const auto what = runtime_error_of([&] { parse_outcar_capturing(db, path); });
  return what ? *what : "no exception";
}

/* Parses a vasprun.xml with std::cout and std::cerr captured; returns what
 * was printed. */
std::string parse_vasprun_capturing(StructureDB &db, const std::string &path) {
  PeriodicTable::initialize();
  std::ostringstream out, err;
  std::streambuf *old_out = std::cout.rdbuf(out.rdbuf());
  std::streambuf *old_err = std::cerr.rdbuf(err.rdbuf());
  try {
    VaspVasprunReader reader(db, path);
    reader.parse_data();
  } catch (...) {
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    throw;
  }
  std::cout.rdbuf(old_out);
  std::cerr.rdbuf(old_err);
  return out.str() + err.str();
}

std::string vasprun_refusal(const ScratchDir &tmp, const std::string &name,
                            const std::vector<std::string> &lines,
                            std::string &path) {
  path = tmp.write(name, join_lines(lines));
  StructureDB db;
  const auto what = runtime_error_of([&] { parse_vasprun_capturing(db, path); });
  return what ? *what : "no exception";
}

}  // namespace

/* Pins: the OUTCAR reader refuses a value it cannot read, naming the file,
 * the line and what it reads there, and stores nothing:
 *  - VRHFIN '=Qq:' (line 41): "VASP OUTCAR <path>: line 41: VRHFIN names
 *    'Qq', which is not an element symbol";
 *  - NIONS 'x4' or one too large for any count (line 341): "... line 341:
 *    cannot read the number of ions NIONS from '<line>'", and the same for a
 *    count of 'ions per type' too large;
 *  - 'ions per type' 'abc' (line 348): "... line 348: 'ions per type' must
 *    give one count for each of the 1 VRHFIN elements, got '<line>'";
 *  - 'ions per type' 53 (line 348): "... line 348: the ions per type add up
 *    to 53, but NIONS is 54";
 *  - no VRHFIN line (line 41 renamed): "... line 348: 'ions per type' comes
 *    before any VRHFIN line, which names the elements" (an OUTCAR cut down
 *    without its POTCAR headers, as ASE's test OUTCAR_example_1);
 *  - no 'ions per type' line: "... line 1960: a TOTAL-FORCE block before the
 *    'ions per type' line";
 *  - lattice vector 1 of step 1 'abc' (line 1891): "... line 1891: lattice
 *    vector 1 of ionic step 1, '<line>', is not three numbers";
 *  - atom 1 of step 1's TOTAL-FORCE block 'abc' (line 1962): "... line
 *    1962: the TOTAL-FORCE line of atom 1 of ionic step 1, '<line>', is not
 *    'x y z fx fy fz'";
 *  - step 1's free energy 'abc' (line 2026): "... line 2026: cannot read the
 *    free energy of ionic step 1 from '<line>'";
 *  - step 1's EKIN_LAT temperature 'abc' (line 2052): "... line 2052:
 *    cannot read the temperature of ionic step 1 from '<line>'";
 *  - a missing file: "VASP OUTCAR <path>: cannot open the file".
 * Before the fix (v1-dev): none named the file: the element was refused as "Unknown
 * symbol: Qq"; the free energy as "... ionic step 1 has no 'free  energy
 * TOTEN' line before its EKIN_LAT line"; the missing file as "Could not open
 * the file: <path>"; every other defect was a line on standard error
 * ('Error: ...', 'Warning: Unexpected end of data ...') and the steps were
 * stored, with a partial cell, 53 or no atoms, or merged with the next. */
TEST_CASE("VASP OUTCAR reader: a value it cannot read is refused, naming the "
          "file and the line",
          "[vasp][outcar][source]") {
  ScratchDir tmp("outcar_source");
  struct Case {
    std::string name;
    size_t line;
    std::string from, to, what;
  };
  const std::vector<Case> cases = {
      {"vrhfin", 41, "=Ta:", "=Qq:",
       "line 41: VRHFIN names 'Qq', which is not an element symbol"},
      {"nions", 341, "NIONS =     54", "NIONS =     x4",
       "line 341: cannot read the number of ions NIONS from 'number of dos "
       "     NEDOS =    301   number of ions     NIONS =     x4'"},
      {"nions_huge", 341, "NIONS =     54", "NIONS =     99999999999999999999",
       "line 341: cannot read the number of ions NIONS from 'number of dos "
       "     NEDOS =    301   number of ions     NIONS =     "
       "99999999999999999999'"},
      {"per_type_huge", 348, "54", "99999999999999999999",
       "line 348: 'ions per type' must give one count for each of the 1 "
       "VRHFIN elements, got 'ions per type =              "
       "99999999999999999999'"},
      {"per_type", 348, "54", "abc",
       "line 348: 'ions per type' must give one count for each of the 1 "
       "VRHFIN elements, got 'ions per type =              abc'"},
      {"per_type_sum", 348, "54", "53",
       "line 348: the ions per type add up to 53, but NIONS is 54"},
      {"no_vrhfin", 41, "VRHFIN =Ta:", "XXXXXX =Ta:",
       "line 348: 'ions per type' comes before any VRHFIN line, which names "
       "the elements"},
      {"lattice", 1891, "9.027612000  0.000000000", "abc  0.000000000",
       "line 1891: lattice vector 1 of ionic step 1, 'abc  0.000000000  "
       "0.000000000     0.110771265  0.000000000  0.000000000', is not three "
       "numbers"},
      {"force_line", 1962, "-0.000036", "abc",
       "line 1962: the TOTAL-FORCE line of atom 1 of ionic step 1, '0.00000 "
       "     0.00000      0.00000        abc      0.000537      0.000015', is "
       "not 'x y z fx fy fz'"},
      {"free_energy", 2026, "-555.75332244", "abc",
       "line 2026: cannot read the free energy of ionic step 1 from 'free  "
       "energy   TOTEN  =      abc eV'"},
      {"ekin_lat", 2052, "300.00", "abc",
       "line 2052: cannot read the temperature of ionic step 1 from 'kin. "
       "lattice  EKIN_LAT=         0.000000  (temperature  abc K)'"},
  };
  for (const Case &c : cases) {
    INFO(c.name);
    std::vector<std::string> l = outcar1_three_steps();
    edit_line(l, c.line, c.from, c.to);
    std::string path;
    const std::string got = outcar_refusal(tmp, "OUTCAR_" + c.name, l, path);
    CHECK(got == "VASP OUTCAR " + path + ": " + c.what);
  }
  {
    std::vector<std::string> l = outcar1_three_steps();
    REQUIRE(l[347].find("ions per type") != std::string::npos);
    l.erase(l.begin() + 347);
    std::string path;
    const std::string got = outcar_refusal(tmp, "OUTCAR_no_per_type", l, path);
    CHECK(got == "VASP OUTCAR " + path +
              ": line 1959: a TOTAL-FORCE block before the 'ions per type' "
              "line");
  }
  const std::string missing = (tmp.dir / "OUTCAR_missing").string();
  StructureDB db;
  CHECK(runtime_error_of([&] { parse_outcar_capturing(db, missing); })
            .value_or("no exception") ==
        "VASP OUTCAR " + missing + ": cannot open the file");
}

/* Pins: an OUTCAR that ends inside a block (a run stopped while writing it)
 * keeps the steps before and skips the step it ends inside with one
 * WARNING naming the file, the block, the step and the line: OUTCAR_1 cut
 * after line 1980 (inside step 1's TOTAL-FORCE block) gives no structure
 * and "WARNING: VASP OUTCAR <path>: the file ends inside the TOTAL-FORCE
 * block of ionic step 1 (line 1980), which was skipped; 0 structure(s)
 * read."; cut after line 2892 (inside step 2's block) it gives step 1 and
 * the warning for step 2 with 1 structure.
 * Before the fix: 'Warning: Unexpected end of data when reading atom
 * information ...' lines that named neither the file nor the step. */
TEST_CASE("VASP OUTCAR reader: a file that ends inside a block warns, naming "
          "the file, the block and the step",
          "[vasp][outcar][source]") {
  ScratchDir tmp("outcar_ends");
  for (const auto &[cut, step, stored] :
       {std::tuple<size_t, int, size_t>{1980, 1, 0}, {2892, 2, 1}}) {
    INFO("cut after line " << cut);
    std::vector<std::string> l = outcar1_three_steps();
    l.resize(cut);
    const std::string path =
        tmp.write("OUTCAR_cut_" + std::to_string(cut), join_lines(l));
    StructureDB db;
    const std::string printed = parse_outcar_capturing(db, path);
    CHECK(db.size() == stored);
    CHECK_THAT(printed,
               Contains("WARNING: VASP OUTCAR " + path +
                        ": the file ends inside the TOTAL-FORCE block of ionic "
                        "step " + std::to_string(step) + " (line " +
                        std::to_string(cut) + "), which was skipped; " +
                        std::to_string(stored) + " structure(s) read."));
    CHECK_THAT(printed, !Contains("Unexpected end of data"));
  }
}

/* Pins: an OUTCAR that ends inside an ionic step outside its blocks (a run
 * stopped during the step's SCF iterations, or between its blocks) keeps
 * the steps before and skips that step with one WARNING naming the file,
 * the step and the line: the three-step OUTCAR_1 cut after line 2930 (step
 * 2's stress, lattice and TOTAL-FORCE block read, before its free energy)
 * gives step 1 and "WARNING: VASP OUTCAR <path>: the file ends inside ionic
 * step 2 (line 2930), which was skipped (it has no 'free  energy   TOTEN'
 * line); 1 structure(s) read."; cut after line 3100 (inside step 3's SCF
 * iterations) it gives steps 1 and 2 and the warning for step 3 at line
 * 3100; as a relaxation, cut after its line 2928 (OUTCAR_1's 2930), step 1
 * and the warning for step 2. A cut after a step's last lines (after line
 * 2975: step 2's EKIN_LAT, before step 3's first SCF iteration) gives steps
 * 1 and 2 and no warning, as the whole three-step file and its relaxation
 * give their 3 steps without one. (Pins as well.)
 * Before the fix: the same structures, and no warning for any cut. */
TEST_CASE("VASP OUTCAR reader: a file that ends inside a step outside its "
          "blocks warns, naming the file and the step",
          "[vasp][outcar][truncated]") {
  ScratchDir tmp("outcar_ends_step");
  const std::vector<std::string> md = outcar1_three_steps();
  const std::vector<std::string> relax = as_relaxation(md);
  struct Cut {
    bool relaxation;
    size_t after;  // the last line kept
    size_t stored;
    int step;      // the step warned about, 0 for none
  };
  for (const Cut &c : {Cut{false, 2930, 1, 2}, Cut{false, 3100, 2, 3},
                       Cut{true, 2928, 1, 2}, Cut{false, 2975, 2, 0},
                       Cut{false, md.size(), 3, 0},
                       Cut{true, relax.size(), 3, 0}}) {
    INFO((c.relaxation ? "relaxation" : "MD") << " cut after line " << c.after);
    std::vector<std::string> l = c.relaxation ? relax : md;
    l.resize(c.after);
    const std::string path = tmp.write(
        std::string("OUTCAR_") + (c.relaxation ? "relax_" : "md_") +
            std::to_string(c.after),
        join_lines(l));
    StructureDB db;
    const std::string printed = parse_outcar_capturing(db, path);
    INFO("printed: '" << printed << "'");
    CHECK(db.size() == c.stored);
    if (c.step == 0) {
      CHECK(count_word(printed, "WARNING") == 0);
    } else {
      CHECK(count_word(printed, "WARNING") == 1);
      CHECK_THAT(printed,
                 Contains("WARNING: VASP OUTCAR " + path +
                          ": the file ends inside ionic step " +
                          std::to_string(c.step) + " (line " +
                          std::to_string(c.after) +
                          "), which was skipped (it has no 'free  energy   "
                          "TOTEN' line); " + std::to_string(c.stored) +
                          " structure(s) read."));
    }
  }
}

/* Pins: the vasprun.xml reader refuses what it cannot read, naming the file
 * (and the calculation, the varray and the row):
 *  - atom 1 of <atominfo> 'Qq' (line 483): "VASP vasprun.xml <path>: atom 1
 *    of <atominfo>, 'Qq', is not an element symbol";
 *  - calculation 1's basis vector 1 'abc' (line 841): "... calculation 1:
 *    basis vector 1, '<v>', is not three numbers";
 *  - the position of atom 1 of calculation 1 'abc' (line 853): "...
 *    calculation 1: the position of atom 1, '<v>', is not three numbers";
 *  - calculation 1's forces varray with a row less (line 909 removed): "...
 *    calculation 1: the forces varray has 52 rows for the 53 atoms of
 *    <atominfo>";
 *  - calculation 1's stress varray with a fourth row: "... calculation 1:
 *    the stress varray has 4 rows, not 3";
 *  - a file that is not well-formed XML (a closing tag '</struct>' for
 *    '</structure>', line 907): "VASP vasprun.xml <path>: not well-formed
 *    XML (...)";
 *  - a missing file: "VASP vasprun.xml <path>: cannot open the file".
 * Before the fix: the element was refused as "Unknown symbol: Qq", naming no
 * file; the other defects printed a line on standard error ('Error parsing
 * ...', 'Error: File ... cannot be opened or read.', 'Error reading file:
 * ...', 'Root node not found.') and stored the calculations (an atom less,
 * a zero force) or nothing, and the 4th stress row was written past the 3x3
 * tensor (a SIGSEGV). */
TEST_CASE("VASP vasprun.xml reader: a value it cannot read is refused, naming "
          "the file and the calculation",
          "[vasp][vasprun][source]") {
  ScratchDir tmp("vasprun_source");
  const std::vector<std::string> base =
      split_lines(slurp(kVasprunDir + "vasprun_2.xml"));
  REQUIRE(base.at(482).find("<rc><c>Ta</c>") != std::string::npos);
  struct Case {
    std::string name;
    size_t line;
    std::string from, to, what;
  };
  const std::vector<Case> cases = {
      {"element", 483, "<c>Ta</c>", "<c>Qq</c>",
       "atom 1 of <atominfo>, 'Qq', is not an element symbol"},
      {"basis", 841, "9.96025800", "abc",
       "calculation 1: basis vector 1, 'abc       0.00000000       "
       "0.00000000', is not three numbers"},
      {"position", 853, "0.00000000       0.00000000       0.33333300",
       "abc 0 0",
       "calculation 1: the position of atom 1, 'abc 0 0', is not three "
       "numbers"},
  };
  for (const Case &c : cases) {
    INFO(c.name);
    std::vector<std::string> l = base;
    edit_line(l, c.line, c.from, c.to);
    std::string path;
    const std::string got =
        vasprun_refusal(tmp, "vasprun_" + c.name + ".xml", l, path);
    CHECK(got == "VASP vasprun.xml " + path + ": " + c.what);
  }
  {
    std::vector<std::string> l = base;
    REQUIRE(l.at(908).find("<v>") != std::string::npos);
    l.erase(l.begin() + 908);
    std::string path;
    const std::string got = vasprun_refusal(tmp, "vasprun_forces.xml", l, path);
    CHECK(got == "VASP vasprun.xml " + path +
              ": calculation 1: the forces varray has 52 rows for the 53 "
              "atoms of <atominfo>");
  }
  {
    // A mismatched closing tag (a file that ends early is read up to its
    // last complete calculation instead: see the next test case).
    std::vector<std::string> l = base;
    edit_line(l, 907, "</structure>", "</struct>");
    std::string path;
    const std::string msg = vasprun_refusal(tmp, "vasprun_bad_xml.xml", l, path);
    CHECK_THAT(msg, Catch::StartsWith("VASP vasprun.xml " + path +
                                      ": not well-formed XML ("));
  }
  const std::string missing = (tmp.dir / "vasprun_missing.xml").string();
  StructureDB db;
  CHECK(runtime_error_of([&] { parse_vasprun_capturing(db, missing); })
            .value_or("no exception") ==
        "VASP vasprun.xml " + missing + ": cannot open the file");
  // Last: before the fix this one wrote past the 3x3 stress (SIGSEGV).
  {
    std::vector<std::string> l = base;
    REQUIRE(l.at(965).find("<v>") != std::string::npos);
    l.insert(l.begin() + 966, l.at(965));
    std::string path;
    const std::string got = vasprun_refusal(tmp, "vasprun_stress4.xml", l, path);
    CHECK(got == "VASP vasprun.xml " + path +
              ": calculation 1: the stress varray has 4 rows, not 3");
  }
}

/* Pins: an unreadable stress or force row of a vasprun.xml calculation (a
 * value VASP could not print, '*****') is no stress tensor, or no forces: the
 * calculation is stored with stress priority 0 (and zero stress), or force
 * weight 0 (and zero forces), with one WARNING naming the file, the
 * calculation and the row; a calculation without a forces varray has force
 * weight 0 and zero forces as well. The other calculations keep weight 1.
 * Before the fix: 'Error parsing stress tensor components.' / 'Error
 * parsing force components.' on standard error, and the calculation was
 * stored with weight 1 and that row 0; without a forces varray, zero forces
 * with force weight 1. */
TEST_CASE("VASP vasprun.xml reader: an unreadable stress or force row gives "
          "that priority 0",
          "[vasp][vasprun][source]") {
  ScratchDir tmp("vasprun_weights");
  const std::vector<std::string> base =
      split_lines(slurp(kVasprunDir + "vasprun_2.xml"));
  {
    std::vector<std::string> l = base;
    edit_line(l, 964, "33.62613502", "**********");
    const std::string path = tmp.write("vasprun_bad_stress.xml", join_lines(l));
    StructureDB db;
    const std::string printed = parse_vasprun_capturing(db, path);
    REQUIRE(db.size() == 4);
    CHECK(db(0).sweight == 0.0);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) CHECK(db(0).stress(i, j) == 0.0);
    CHECK(db(1).sweight == 1.0);
    CHECK_THAT(printed, Contains("WARNING: VASP vasprun.xml " + path +
                                 ": calculation 1: the stress row 1, '"));
    CHECK_THAT(printed, Contains("', cannot be read; the calculation is "
                                 "stored without stress"));
  }
  {
    std::vector<std::string> l = base;
    edit_line(l, 909, "0.00025635", "**********");
    const std::string path = tmp.write("vasprun_bad_force.xml", join_lines(l));
    StructureDB db;
    const std::string printed = parse_vasprun_capturing(db, path);
    REQUIRE(db.size() == 4);
    CHECK(db(0).fweight == 0.0);
    for (const auto &a : db(0).atoms)
      for (int k = 0; k < 3; ++k) CHECK(a.force[k] == 0.0);
    CHECK(db(1).fweight == 1.0);
    CHECK_THAT(printed, Contains("WARNING: VASP vasprun.xml " + path +
                                 ": calculation 1: the force of atom 1, '"));
    CHECK_THAT(printed, Contains("', cannot be read; the calculation is "
                                 "stored without forces"));
  }
  {
    std::vector<std::string> l = base;
    REQUIRE(l.at(907).find("name=\"forces\"") != std::string::npos);
    REQUIRE(l.at(961).find("</varray>") != std::string::npos);
    l.erase(l.begin() + 907, l.begin() + 962);
    const std::string path = tmp.write("vasprun_no_forces.xml", join_lines(l));
    StructureDB db;
    parse_vasprun_capturing(db, path);
    REQUIRE(db.size() == 4);
    CHECK(db(0).fweight == 0.0);
    for (const auto &a : db(0).atoms)
      for (int k = 0; k < 3; ++k) CHECK(a.force[k] == 0.0);
    CHECK(db(1).fweight == 1.0);
  }
}

/* Pins: the EKIN_LAT temperature of an OUTCAR is read where VASP's fixed
 * width joins it to the word before ('(temperature10147.82 K)', from 10000
 * K up, as in the 6580 K MD runs of the data store): OUTCAR_1 with step 1's
 * temperature 10147.82 gives its 3 steps, step 1 with T 10147.82.
 * Before the fix: refused, "... line 2052: cannot read the temperature of
 * ionic step 1 from '...'"; v1-dev printed 'Warning: Unexpected end of data
 * when reading temperature' and did not end the step there, so steps 1 and
 * 2 were stored as one structure of 108 atoms (2 structures). */
TEST_CASE("VASP OUTCAR reader: a temperature of 10000 K and more is read",
          "[vasp][outcar][source]") {
  ScratchDir tmp("outcar_hot");
  std::vector<std::string> l = outcar1_three_steps();
  edit_line(l, 2052, "(temperature  300.00 K)", "(temperature10147.82 K)");
  const std::string path = tmp.write("OUTCAR_hot", join_lines(l));
  StructureDB db;
  parse_outcar_capturing(db, path);
  REQUIRE(db.size() == 3);
  CHECK(db(0).T == 10147.82);
  CHECK(db(0).natoms() == 54);
  CHECK(db(1).T == 300.04);
}

/* Pins: a vasprun.xml that ends after its last complete calculation,
 * without its closing </modeling> tag (a run stopped, or still running,
 * after it), is read whole, and one WARNING says that nothing was skipped:
 * vasprun_2 without its last line, and the same with the start of a
 * trailing <structure name="finalpos"> (VASP writes it after the last
 * calculation), give its 4 structures (the energies, forces and stress of
 * the whole file) and "WARNING: VASP vasprun.xml <path>: the file ends
 * after calculation 4, without its closing </modeling> tag (a run stopped,
 * or still running); no calculation was skipped: 4 structure(s) read.".
 * Before the fix: the same structures with "WARNING: ...: the file ends
 * inside calculation 5 (a run stopped while writing it), which was skipped;
 * 4 structure(s) read." (the file has no calculation 5). */
TEST_CASE("VASP vasprun.xml reader: a file that ends after its last "
          "calculation says that nothing was skipped",
          "[vasp][vasprun][source]") {
  ScratchDir tmp("vasprun_after_last");
  const std::vector<std::string> base =
      split_lines(slurp(kVasprunDir + "vasprun_2.xml"));
  REQUIRE(base.size() == 1589);
  REQUIRE(base.at(1587).find("</calculation>") != std::string::npos);
  REQUIRE(base.at(1588).find("</modeling>") != std::string::npos);
  StructureDB whole = read_vasprun(kVasprunDir + "vasprun_2.xml");
  REQUIRE(whole.size() == 4);
  std::vector<std::string> nomodeling = base;
  nomodeling.resize(1588);
  std::vector<std::string> finalpos = nomodeling;
  for (const char *l : {" <structure name=\"finalpos\" >", "  <crystal>",
                        "   <varray name=\"basis\" >"})
    finalpos.push_back(l);
  for (const auto &[name, lines] :
       {std::pair<std::string, std::vector<std::string>>{"nomodeling",
                                                          nomodeling},
        {"finalpos", finalpos}}) {
    INFO(name);
    const std::string path =
        tmp.write("vasprun_" + name + ".xml", join_lines(lines));
    StructureDB db;
    const std::string printed = parse_vasprun_capturing(db, path);
    INFO("printed: '" << printed << "'");
    REQUIRE(db.size() == 4);
    for (size_t k = 0; k < 4; ++k) {
      CHECK(db(k).energy == whole(k).energy);
      CHECK(db(k).natoms() == whole(k).natoms());
      CHECK(db(k).atoms[52].force[2] == whole(k).atoms[52].force[2]);
      CHECK(db(k).stress(2, 2) == whole(k).stress(2, 2));
    }
    CHECK(count_word(printed, "WARNING") == 1);
    CHECK_THAT(printed, Contains("WARNING: VASP vasprun.xml " + path +
                                 ": the file ends after calculation 4, "
                                 "without its closing </modeling> tag (a run "
                                 "stopped, or still running); no calculation "
                                 "was skipped: 4 structure(s) read."));
  }
}

/* Pins: a vasprun.xml that ends early (VASP writes it as it runs; a run
 * stopped, or still running) is read up to its last complete calculation,
 * and the calculation it ends inside is skipped with one WARNING naming the
 * file and the calculation: vasprun_2 cut inside calculation 2 (after line
 * 1100) gives calculation 1 and "WARNING: VASP vasprun.xml <path>: the
 * file ends inside calculation 2 (a run stopped while writing it), which
 * was skipped; 1 structure(s) read."; cut inside calculation 1 (after line
 * 900), no structure and the same warning for calculation 1 with 0; cut
 * before the first calculation (after line 300), no structure and "...:
 * the file ends before its first calculation (a run stopped while writing
 * it); 0 structure(s) read.". The data store's 24 truncated vasprun.xml
 * files end so.
 * Before the fix: each was refused, "VASP vasprun.xml <path>: not
 * well-formed XML (unexpected end of data)"; v1-dev printed 'Error reading
 * file: unexpected end of data' and 'Root node not found.' and stored
 * nothing. */
TEST_CASE("VASP vasprun.xml reader: a file that ends early is read up to its "
          "last complete calculation",
          "[vasp][vasprun][source]") {
  ScratchDir tmp("vasprun_cut");
  const std::vector<std::string> base =
      split_lines(slurp(kVasprunDir + "vasprun_2.xml"));
  REQUIRE(base.size() == 1589);
  REQUIRE(base.at(978).find("</calculation>") != std::string::npos);
  StructureDB whole = read_vasprun(kVasprunDir + "vasprun_2.xml");
  REQUIRE(whole.size() == 4);
  for (const auto &[cut, calc, stored] :
       {std::tuple<size_t, int, size_t>{1100, 2, 1}, {900, 1, 0}}) {
    INFO("cut after line " << cut);
    std::vector<std::string> l = base;
    l.resize(cut);
    const std::string path =
        tmp.write("vasprun_cut_" + std::to_string(cut) + ".xml", join_lines(l));
    StructureDB db;
    const std::string printed = parse_vasprun_capturing(db, path);
    REQUIRE(db.size() == stored);
    if (stored == 1) {
      CHECK(db(0).energy == whole(0).energy);
      CHECK(db(0).natoms() == whole(0).natoms());
      CHECK(db(0).atoms[52].force[2] == whole(0).atoms[52].force[2]);
      CHECK(db(0).stress(2, 2) == whole(0).stress(2, 2));
    }
    CHECK_THAT(printed, Contains("WARNING: VASP vasprun.xml " + path +
                                 ": the file ends inside calculation " +
                                 std::to_string(calc) +
                                 " (a run stopped while writing it), which "
                                 "was skipped; " + std::to_string(stored) +
                                 " structure(s) read."));
  }
  std::vector<std::string> l = base;
  l.resize(300);
  const std::string path = tmp.write("vasprun_cut_300.xml", join_lines(l));
  StructureDB db;
  const std::string printed = parse_vasprun_capturing(db, path);
  CHECK(db.size() == 0);
  CHECK_THAT(printed, Contains("WARNING: VASP vasprun.xml " + path +
                               ": the file ends before its first calculation "
                               "(a run stopped while writing it); 0 "
                               "structure(s) read."));
}
