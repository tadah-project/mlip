// Dataset files written by `tadah data convert|merge|split|sample|dedup|
// balance` (Structure::dump_to_file via StructureDB::dump_to_file) follow
// NUMERIC as SIGNIFICANT digits, default 18 = bit-exact round trip, like
// potential files. The LAMMPS data and VASP POSCAR exporters of `tadah data
// write` follow the same rule, and the POSCAR reader accepts what the
// exporter writes (subnormals included).
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/number_format.h>
#include <tadah/mlip/engines/data_write_engine.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_readers/vasp_poscar_reader.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/structure_writers/lammps_structure_writer.h>
#include <tadah/mlip/structure_writers/vasp_poscar_writer.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using tadah::core::Element;
using tadah::mlip::Atom;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
namespace fs = std::filesystem;

namespace {

std::uint64_t ds_bits(double v) {
  std::uint64_t u;
  std::memcpy(&u, &v, sizeof u);
  return u;
}

// Values the old fixed-DECIMAL writer could not keep: computed weights,
// unit-converted stresses, tiny forces, a subnormal, -0 and DBL_MAX.
Structure make_structure() {
  Structure st;
  st.label = "precision probe";
  st.eweight = 8.100000072900001e-05;  // 1/|E| from `data balance --rescale`
  st.fweight = 1.0 / 3.0;
  st.sweight = 7.65e-9;
  st.energy = -12345.678901234567;
  st.T = 300.00000000000006;
  st.cell.load(9.9321641234567, 1e-7, -0.1,
               0.30000000000000004, 14.87106012345678, 2.0 / 3.0,
               -1e-20, -0.0, 17.525592123456789);
  st.stress.load(3.3333333333333335e-05, 1e-20, 1e-310,
                 -2.4966036296e-05 * 6.241509074e-4, 257.0893807653,
                 DBL_MAX,
                 -0.1, 1.0 / 7.0, -123456.789012345);
  st.add_atom(Atom(Element("Ta"), 9.9321641234567, -0.004522, 1e-7,
                   1.2345678901234566e-07, 0.5, -0.3));
  st.add_atom(Atom(Element("H"), 1.0 / 3.0, 2.0 / 3.0, 0.1,
                   -1.2345678901234567e-05, 0.25, 4.9406564584124654e-324));
  return st;
}

void require_bit_identical(const Structure &a, const Structure &b) {
  CHECK(a.label == b.label);
  CHECK(ds_bits(a.eweight) == ds_bits(b.eweight));
  CHECK(ds_bits(a.fweight) == ds_bits(b.fweight));
  CHECK(ds_bits(a.sweight) == ds_bits(b.sweight));
  CHECK(ds_bits(a.energy) == ds_bits(b.energy));
  CHECK(ds_bits(a.T) == ds_bits(b.T));
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      INFO("i=" << i << " j=" << j);
      CHECK(ds_bits(a.cell(i, j)) == ds_bits(b.cell(i, j)));
      CHECK(ds_bits(a.stress(i, j)) == ds_bits(b.stress(i, j)));
    }
  REQUIRE(a.natoms() == b.natoms());
  for (size_t n = 0; n < a.natoms(); ++n) {
    INFO("atom " << n);
    CHECK(a.atoms[n].symbol == b.atoms[n].symbol);
    for (int k = 0; k < 3; ++k) {
      INFO("k=" << k);
      CHECK(ds_bits(a.atoms[n].position(k)) == ds_bits(b.atoms[n].position(k)));
      CHECK(ds_bits(a.atoms[n].force(k)) == ds_bits(b.atoms[n].force(k)));
    }
  }
}

struct DatasetTempDir {
  fs::path dir;
  DatasetTempDir() {
    static int counter = 0;
    dir = fs::temp_directory_path() /
          ("tadah_dsprec_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
  }
  ~DatasetTempDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string path(const std::string &name) const {
    return (dir / name).string();
  }
};

std::vector<std::string> lines_of(const std::string &text) {
  std::vector<std::string> out;
  std::istringstream is(text);
  std::string line;
  while (std::getline(is, line)) out.push_back(line);
  return out;
}

}  // namespace

TEST_CASE("Structure::dump_to_file: default NUMERIC 18 round-trips every "
          "double bit-exactly", "[dataset][numeric][lossless]") {
  const Structure st = make_structure();
  DatasetTempDir tmp;
  const std::string file = tmp.path("db.tadah");
  {
    std::ofstream out(file);
    st.dump_to_file(out);  // default prec: 18
  }
  Structure rt;
  std::ifstream in(file);
  REQUIRE(rt.read(in) == 0);
  require_bit_identical(st, rt);

  // Significant digits, not decimals: tiny values keep all their digits.
  std::ifstream again(file);
  const std::string text((std::istreambuf_iterator<char>(again)),
                         std::istreambuf_iterator<char>());
  CHECK(text.find("8.10000007290000076e-05") != std::string::npos);
  CHECK(text.find("9.99999999999999945e-21") != std::string::npos);
  CHECK(text.find("0.000000000000000000") == std::string::npos);
}

TEST_CASE("StructureDB::dump_to_file -> StructureDB read-back is "
          "bit-identical", "[dataset][numeric][lossless]") {
  StructureDB db;
  db.add(make_structure());
  Structure second = make_structure();
  second.label = "second";
  second.energy = -0.1;
  db.add(second);

  DatasetTempDir tmp;
  const std::string file = tmp.path("db2.tadah");
  db.dump_to_file(file, /*append=*/false, 18);
  StructureDB rt;
  rt.add(file);
  REQUIRE(rt.size() == 2);
  require_bit_identical(db(0), rt(0));
  require_bit_identical(db(1), rt(1));
}

TEST_CASE("Structure::dump_to_file: an explicit NUMERIC below 17 gives "
          "that many significant digits, aligned columns, finite text",
          "[dataset][numeric][lossless]") {
  const Structure st = make_structure();
  for (int n : {1, 3, 12, 16}) {
    INFO("NUMERIC " << n);
    std::ostringstream os;
    st.dump_to_file(os, static_cast<size_t>(n));
    const std::vector<std::string> ln = lines_of(os.str());
    REQUIRE(ln.size() >= 11);
    // energy line: "<energy> <T>" at n significant digits.
    std::istringstream el(ln[2]);
    std::string e_tok;
    el >> e_tok;
    CHECK(e_tok == tadah::core::format_sig(st.energy, n));
    // cell + stress rows (lines 3..8) all have the same width; atom rows
    // (9, 10) too.
    for (int r = 4; r <= 8; ++r) CHECK(ln[r].size() == ln[3].size());
    CHECK(ln[10].size() == ln[9].size());
    // Every number reads back finite (DBL_MAX must not round past it).
    Structure rt;
    DatasetTempDir tmp;  // read() takes an ifstream: go through a file
    const std::string file = tmp.path("n.tadah");
    { std::ofstream out(file); out << os.str(); }
    std::ifstream in(file);
    REQUIRE(rt.read(in) == 0);
    CHECK(std::isfinite(rt.stress(1, 2)));
    CHECK(rt.stress(1, 2) <= DBL_MAX);
  }
}

TEST_CASE("StructureDB::dataset_digits resolves NUMERIC without inserting "
          "it", "[dataset][numeric][lossless]") {
  tadah::core::Context c;
  CHECK(StructureDB::dataset_digits(c) == 18);
  CHECK_FALSE(c.has_stored("NUMERIC"));
  c.add("NUMERIC", 12);
  CHECK(StructureDB::dataset_digits(c) == 12);
  tadah::core::Context c25;
  c25.add("NUMERIC", 25);
  CHECK(StructureDB::dataset_digits(c25) == 25);
}

TEST_CASE("Structure::read accepts a subnormal energy on the energy-only "
          "line", "[dataset][numeric][lossless]") {
  DatasetTempDir tmp;
  const std::string file = tmp.path("sub.tadah");
  {
    std::ofstream out(file);
    out << "subnormal energy\n1e-310  \n"
        << "1 0 0\n0 1 0\n0 0 1\n0 0 0\n0 0 0\n0 0 0\n"
        << "Ta 0 0 0 0 0 0\n\n";
  }
  Structure st;
  std::ifstream in(file);
  REQUIRE(st.read(in) == 0);
  CHECK(st.energy == 1e-310);
}

TEST_CASE("LAMMPS data / VASP POSCAR exporters write NUMERIC significant "
          "digits", "[dataset][numeric][lossless]") {
  StructureDB db;
  Structure st;
  st.label = "export";
  st.cell.load(3.30247123456789012, 0.0, 0.0,
               0.0, 3.3000000000000003, 0.0,
               0.0, 0.0, 3.29999999999999982);
  st.add_atom(Atom(Element("Ta"), 1.2345678901234568e-05, 2.5e-06, 0.0,
                   0.0, 0.0, 0.0));
  st.add_atom(Atom(Element("Ta"), 1.6512356172839451, 1.65,
                   1.6499999999999999, 0.0, 0.0, 0.0));
  db.add(st);
  DatasetTempDir tmp;

  SECTION("VASP POSCAR re-reads bit-identically at the default precision") {
    tadah::mlip::VaspPoscarWriter w(db);
    w.set_precision(18);
    const std::string file = tmp.path("POSCAR");
    w.write_data(file, 0);
    tadah::mlip::VaspPoscarReader r;
    r.read(file);
    const Structure back = r.getStructure();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        CHECK(ds_bits(back.cell(i, j)) == ds_bits(st.cell(i, j)));
    REQUIRE(back.natoms() == 2);
    for (size_t n = 0; n < 2; ++n)
      for (int k = 0; k < 3; ++k)
        CHECK(ds_bits(back.atoms[n].position(k)) ==
              ds_bits(st.atoms[n].position(k)));
  }
  SECTION("LAMMPS data: every real parses back exactly, columns separated") {
    tadah::mlip::LammpsStructureWriter w(db);
    w.set_precision(18);
    const std::string file = tmp.path("data.lmp");
    w.write_data(file, 0);
    std::ifstream in(file);
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    CHECK(text.find("1.23456789012345678e-05") != std::string::npos);
    CHECK(text.find("3.30247123456789016") != std::string::npos);
    // Atoms section: id type x y z with exact reals.
    const std::vector<std::string> ln = lines_of(text);
    std::vector<std::vector<double>> atoms;
    bool in_atoms = false;
    for (const auto &l : ln) {
      if (l.rfind("Atoms", 0) == 0) { in_atoms = true; continue; }
      if (!in_atoms || l.empty()) continue;
      std::istringstream ls(l);
      double id, type, x, y, z;
      REQUIRE(static_cast<bool>(ls >> id >> type >> x >> y >> z));
      atoms.push_back({x, y, z});
    }
    REQUIRE(atoms.size() == 2);
    for (size_t n = 0; n < 2; ++n)
      for (int k = 0; k < 3; ++k)
        CHECK(ds_bits(atoms[n][k]) == ds_bits(st.atoms[n].position(k)));
  }
}

TEST_CASE("VASP POSCAR export -> read round-trips subnormal and signed-zero "
          "coordinates", "[dataset][numeric][lossless]") {
  // The %.{N}g exporter writes a subnormal coordinate in exponent form
  // (the old fixed-decimal export wrote 0.000000000000); the reader must
  // take it back (std::stod threw out_of_range on the underflow).
  StructureDB db;
  Structure st;
  st.label = "subnormal";
  st.cell.load(3.29999999999999982, 0.0, 0.0,
               0.0, 3.29999999999999982, 0.0,
               0.0, 0.0, 3.29999999999999982);
  st.add_atom(Atom(Element("Ta"), 1e-310, -0.0, 4.9406564584124654e-324,
                   0.0, 0.0, 0.0));
  st.add_atom(Atom(Element("Ta"), 1.65, -1e-310, 1.6499999999999999,
                   0.0, 0.0, 0.0));
  db.add(st);
  DatasetTempDir tmp;
  tadah::mlip::VaspPoscarWriter w(db);
  w.set_precision(18);
  const std::string file = tmp.path("POSCAR");
  w.write_data(file, 0);
  {
    std::ifstream in(file);
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    CHECK(text.find("9.99999999999996945e-311") != std::string::npos);
  }
  tadah::mlip::VaspPoscarReader r;
  REQUIRE_NOTHROW(r.read(file));
  const Structure back = r.getStructure();
  REQUIRE(back.natoms() == 2);
  for (size_t n = 0; n < 2; ++n)
    for (int k = 0; k < 3; ++k) {
      INFO("atom " << n << " k=" << k);
      CHECK(ds_bits(back.atoms[n].position(k)) ==
            ds_bits(st.atoms[n].position(k)));
    }
}

namespace {

// Runs `tadah data write` (DataWriteEngine) with WARNING logging on and
// std::cout (the Logger's console) captured; returns what was logged.
std::string run_data_write(const tadah::core::Context &c) {
  auto &log = tadah::core::Logger::instance();
  const auto old_level = log.getLogLevel();
  log.setLogLevel(tadah::core::LogLevel::WARNING);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  try {
    tadah::mlip::Task task("data.write", c);
    tadah::mlip::engines::DataWriteEngine engine;
    engine.run(task);
  } catch (...) {
    std::cout.rdbuf(old_buf);
    log.setLogLevel(old_level);
    throw;
  }
  std::cout.rdbuf(old_buf);
  log.setLogLevel(old_level);
  return captured.str();
}

}  // namespace

TEST_CASE("tadah data write: the LAMMPS and POSCAR exports warn for an "
          "explicit NUMERIC below 17; CASTEP (decimal places) does not",
          "[dataset][numeric][lossless]") {
  DatasetTempDir tmp;
  const std::string db_file = tmp.path("db.tadah");
  {
    StructureDB db;
    Structure st;
    st.label = "export";
    st.cell.load(3.30247123456789012, 0.0, 0.0,
                 0.0, 3.30247123456789012, 0.0,
                 0.0, 0.0, 3.30247123456789012);
    st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("Ta"), 1.6512356172839451, 1.65,
                     1.6499999999999999, 0.0, 0.0, 0.0));
    db.add(st);
    db.dump_to_file(db_file, /*append=*/false, 18);
  }
  auto context = [&](const std::string &format, const std::string &out,
                     int numeric) {
    tadah::core::Context c;
    c.add("DBFILE", db_file);
    c.add("INDEX", "1");
    c.add("OUTFILE", out);
    c.add("FORMAT", format);
    if (numeric > 0) c.add("NUMERIC", numeric);
    return c;
  };
  auto slurp = [](const std::string &f) {
    std::ifstream in(f);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
  };
  struct Export { const char *format, *noun; };
  for (const Export &e : {Export{"lammps", "LAMMPS data file"},
                          Export{"vasp", "VASP POSCAR file"}}) {
    INFO("FORMAT " << e.format);
    const std::string out5 = tmp.path(std::string(e.format) + "_5");
    const std::string log5 = run_data_write(context(e.format, out5, 5));
    CHECK(log5.find("WARNING") != std::string::npos);
    CHECK(log5.find(std::string("NUMERIC 5 limits every real number in the "
                                "written ") + e.noun) != std::string::npos);
    CHECK(log5.find("dataset") == std::string::npos);
    const std::string text5 = slurp(out5);
    CHECK(text5.find("3.3025") != std::string::npos);        // 5 digits
    CHECK(text5.find("3.30247123456789016") == std::string::npos);

    for (int n : {0, 17}) {                  // absent (18, exact) and 17
      INFO("NUMERIC " << n);
      const std::string out = tmp.path(std::string(e.format) + "_" +
                                       std::to_string(n));
      const std::string log = run_data_write(context(e.format, out, n));
      CHECK(log.find("WARNING") == std::string::npos);
      CHECK(slurp(out).find(n == 0 ? "3.30247123456789016"
                                   : "3.3024712345678902") !=
            std::string::npos);
    }
  }
  // CASTEP cell: NUMERIC is decimal places (fixed notation), no warning.
  const std::string cell = tmp.path("castep_5.cell");
  const std::string logc = run_data_write(context("castep", cell, 5));
  CHECK(logc.find("WARNING") == std::string::npos);
  CHECK(slurp(cell).find("3.30247") != std::string::npos);
}
