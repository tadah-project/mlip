// The messages of the structure readers and data providers name the file,
// the database or the id they are about, never the class of the code.
#include "catch2/catch.hpp"

#include <tadah/mlip/data_providers/aflow_data_provider.h>
#include <tadah/mlip/data_providers/cod_data_provider.h>
#include <tadah/mlip/data_providers/nomad_data_provider.h>
#include <tadah/mlip/structure_readers/castep_cell_reader.h>
#include <tadah/mlip/structure_readers/cif_reader.h>
#include <tadah/mlip/structure_readers/cod_reader.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>
#include <tadah/mlip/structure_readers/vasp_poscar_reader.h>

#include <functional>
#include <stdexcept>
#include <string>

namespace {

/* The message @p f throws, or "no exception". */
std::string thrown(const std::function<void()> &f) {
  try {
    f();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "no exception";
}

}  // namespace

/* Pins: a structure file that does not open is named as such by the CIF,
 * POSCAR and CASTEP .cell readers: "the structure file nosuch.cif cannot be
 * opened".
 * Before the fix: "CifReader::read: Cannot open nosuch.cif",
 * "VaspPoscarReader::read: Cannot open nosuch_POSCAR" and "CastepCellReader:
 * Cannot open file nosuch.cell". */
TEST_CASE("A structure file that does not open is named by its reader",
          "[structure_readers][messages]") {
  CHECK(thrown([] { tadah::mlip::CifReader().read("nosuch.cif"); }) ==
        "the structure file nosuch.cif cannot be opened");
  CHECK(thrown([] { tadah::mlip::VaspPoscarReader().read("nosuch_POSCAR"); }) ==
        "the structure file nosuch_POSCAR cannot be opened");
  CHECK(thrown([] { tadah::mlip::CastepCellReader().read("nosuch.cell"); }) ==
        "the structure file nosuch.cell cannot be opened");
}

/* Pins: -s of something that is neither a structure file of a format tadah
 * reads nor an online id is refused, naming it and what is read:
 * "nosuch.xyz is neither a structure file tadah reads (CIF, POSCAR or
 * CASTEP .cell) nor an online id (mp-..., cod-..., aflow:... or a NOMAD
 * entry id)".
 * Before the fix: "StructureReaderSelector::getReader - Unknown format for
 * nosuch.xyz". */
TEST_CASE("A structure that no reader takes is refused, naming what is read",
          "[structure_readers][messages]") {
  CHECK(thrown([] {
          (void)tadah::mlip::StructureReaderSelector::getReader("nosuch.xyz");
        }) ==
        "nosuch.xyz is neither a structure file tadah reads (CIF, POSCAR or "
        "CASTEP .cell) nor an online id (mp-..., cod-..., aflow:... or a "
        "NOMAD entry id)");
}

/* Pins: -s of a path that names a structure file by its name (a .cif or
 * .cell extension in any letter case, a file name that begins with POSCAR
 * or CONTCAR) and that does not exist is refused as such: "the structure
 * file missing.cif does not exist", "... sub/POSCAR does not exist",
 * "... missing.CELL does not exist"; a path of no such name stays "... is
 * neither a structure file tadah reads ... nor an online id ...".
 * Before the fix: "missing.cif is neither a structure file tadah reads
 * (CIF, POSCAR or CASTEP .cell) nor an online id (...)": the format was
 * guessed from the name of an existing file only, so a mistyped path read
 * as a format tadah does not read. */
TEST_CASE("A structure file that does not exist is refused as such",
          "[structure_readers][messages]") {
  for (const std::string path :
       {"missing.cif", "sub/POSCAR", "sub/CONTCAR_relaxed", "missing.CELL"}) {
    INFO(path);
    CHECK(thrown([&] {
            (void)tadah::mlip::StructureReaderSelector::getReader(path);
          }) == "the structure file " + path + " does not exist");
  }
  CHECK(thrown([] {
          (void)tadah::mlip::StructureReaderSelector::getReader(
              "missing.xyz");
        }) ==
        "missing.xyz is neither a structure file tadah reads (CIF, POSCAR or "
        "CASTEP .cell) nor an online id (mp-..., cod-..., aflow:... or a "
        "NOMAD entry id)");
}

/* Pins: a COD URL without a COD ID is refused before any request: "COD:
 * the COD ID cannot be read from the URL
 * https://www.crystallography.net/cod/".
 * Before the fix: "CODReader::read - Cannot parse COD ID from URL: ...". */
TEST_CASE("COD reader: a URL without a COD ID is refused",
          "[cod][structure_readers][messages]") {
  CHECK(thrown([] {
          tadah::mlip::CODReader().read("https://www.crystallography.net/cod/");
        }) ==
        "COD: the COD ID cannot be read from the URL "
        "https://www.crystallography.net/cod/");
}

/* Pins: the AFLOW, NOMAD and COD providers answer a formula query 501 with
 * "<database>: formula queries are not supported." ("AFLOW", "NOMAD",
 * "COD").
 * Before the fix: "AFLOWDataProvider does not implement formula queries.",
 * "NomadDataProvider does not implement formula queries." and
 * "CODDataProvider does not support formula-based queries yet.". */
TEST_CASE("The data providers name their database in a formula query's "
          "answer",
          "[data_providers][messages]") {
  tadah::mlip::AFLOWDataProvider aflow;
  tadah::mlip::NomadDataProvider nomad;
  tadah::mlip::CODDataProvider cod;
  for (auto &[provider, name] :
       {std::pair<tadah::mlip::IMaterialsDataProvider *, std::string>{
            &aflow, "AFLOW"},
        {&nomad, "NOMAD"},
        {&cod, "COD"}}) {
    const tadah::mlip::RawResponse rr = provider->QueryByFormula("Ta");
    CAPTURE(name);
    CHECK(rr.httpStatus == 501);
    CHECK(rr.response == name + ": formula queries are not supported.");
  }
}
