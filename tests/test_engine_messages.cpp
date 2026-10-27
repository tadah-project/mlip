// The refusals of the commands' engines (IEngine::validate) and of their
// structure writers: each names the command and the key or option, never
// the class of the code that checks it.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/mlip/engines/analysis_bfunc_engine.h>
#include <tadah/mlip/engines/analysis_cutoff_engine.h>
#include <tadah/mlip/engines/analysis_descriptor_engine.h>
#include <tadah/mlip/engines/explain_engine.h>
#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/mlip/engines/properties_pairwise_engine.h>
#include <tadah/mlip/engines/refit_engine.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_writers/structure_writer_selector.h>
#include <tadah/mlip/task.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using tadah::core::Context;
using tadah::mlip::Task;
namespace engines = tadah::mlip::engines;

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

/* A context of @p keys (each "KEY v1 v2 ...", one value per word). */
Context context_of(const std::vector<std::string> &keys) {
  Context c;
  for (const std::string &line : keys) {
    const std::size_t sp = line.find(' ');
    const std::string key = line.substr(0, sp);
    std::string rest = sp == std::string::npos ? "" : line.substr(sp + 1);
    std::size_t at = 0;
    while (at < rest.size()) {
      const std::size_t next = rest.find(' ', at);
      c.add(key, rest.substr(at, next == std::string::npos ? std::string::npos
                                                           : next - at));
      if (next == std::string::npos) break;
      at = next + 1;
    }
  }
  return c;
}

template <class Engine>
std::string refusal_of(const std::string &command,
                       const std::vector<std::string> &keys) {
  const Task task(command, context_of(keys));
  return thrown([&] { Engine().validate(task); });
}

}  // namespace

/* Pins: each engine refuses what its command cannot run, naming the
 * command and the key or option ("tadah predict needs a potential file
 * (--potential)", "tadah analysis cutoff takes three RANGE values: start,
 * stop and the number of points", ...); tadah train names INIT2B and
 * INITMB ("tadah train: INIT2B and INITMB are both false, so the potential
 * has no term; tadah train needs INIT2B true or INITMB true"); tadah refit
 * says which line names the EAM file. Most of these the command line
 * refuses first (an option's required values and the options it needs:
 * tadah predict -d without -p is "--dbfile requires --potential"), so
 * validate() gives them to a Task that a C++ program builds; the refusals
 * of tadah train, tadah refit and the OUTFILE counts of tadah predict are
 * reached from the command line too.
 * Before the fix: every refusal began with the engine's class
 * ("PredictEngine requires a potential file to be set.", "TrainEngine
 * requires INIT2B or/and INITMB parameters to be set", "RefitEngine requires
 * an input EAM file: declare `EAM <file>` in the config file passed with -c
 * ...", "ExplainEngine requires OPTION parameter to be set", ...). */
TEST_CASE("The refusals of a command's engine name the command",
          "[engines][messages]") {
  CHECK(refusal_of<engines::TrainEngine>(
            "train", {"INIT2B false", "INITMB false"}) ==
        "tadah train: INIT2B and INITMB are both false, so the potential has "
        "no term; tadah train needs INIT2B true or INITMB true");
  CHECK(refusal_of<engines::PredictEngine>("predict", {}) ==
        "tadah predict needs a potential file (--potential)");
  CHECK(refusal_of<engines::PropertiesPairwiseEngine>(
            "properties.pairwise", {}) ==
        "tadah properties pairwise needs a potential file (--potential)");
  CHECK(refusal_of<engines::PropertiesPairwiseEngine>(
            "properties.pairwise", {"POTENTIAL pot.tadah"}) ==
        "tadah properties pairwise needs a range of distances (--range)");
  CHECK(refusal_of<engines::PropertiesPairwiseEngine>(
            "properties.pairwise", {"POTENTIAL pot.tadah", "RANGE 1 5 10"}) ==
        "tadah properties pairwise needs a pair of elements (--atompair)");
  CHECK(refusal_of<engines::AnalysisDescriptorEngine>("analysis.descriptor",
                                                      {}) ==
        "tadah analysis descriptor needs a potential file (--potential)");
  CHECK(refusal_of<engines::AnalysisDescriptorEngine>(
            "analysis.descriptor",
            {"POTENTIAL pot.tadah", "DBFILE a.tadah b.tadah", "INDEX 1"}) ==
        "tadah analysis descriptor takes one DBFILE with INDEX");
  CHECK(refusal_of<engines::AnalysisDescriptorEngine>(
            "analysis.descriptor", {"POTENTIAL pot.tadah", "DBFILE a.tadah",
                                    "INDEX 1", "STRUCTURE s.cif"}) ==
        "tadah analysis descriptor takes INDEX or STRUCTURE, not both");
  CHECK(refusal_of<engines::AnalysisDescriptorEngine>(
            "analysis.descriptor",
            {"POTENTIAL pot.tadah", "DBFILE a.tadah", "MERGE true",
             "OUTFILE x.txt y.txt"}) ==
        "tadah analysis descriptor writes one OUTFILE with MERGE");
  CHECK(refusal_of<engines::AnalysisDescriptorEngine>(
            "analysis.descriptor",
            {"POTENTIAL pot.tadah", "DBFILE a.tadah b.tadah",
             "OUTFILE x.txt"}) ==
        "tadah analysis descriptor writes one OUTFILE for each DBFILE "
        "without MERGE");
  CHECK(refusal_of<engines::AnalysisCutoffEngine>(
            "analysis.cutoff", {"OUTFILE a.txt b.txt"}) ==
        "tadah analysis cutoff writes one OUTFILE or none");
  CHECK(refusal_of<engines::AnalysisCutoffEngine>(
            "analysis.cutoff", {"TYPE Cut_Cos Cut_Tanh", "RANGE 0 5 10"}) ==
        "tadah analysis cutoff takes one TYPE value");
  CHECK(refusal_of<engines::AnalysisCutoffEngine>(
            "analysis.cutoff", {"TYPE Cut_Cos", "RANGE 0 5"}) ==
        "tadah analysis cutoff takes three RANGE values: start, stop and "
        "the number of points");
  CHECK(refusal_of<engines::AnalysisBfuncEngine>(
            "analysis.bfunc", {"OUTFILE a.txt b.txt"}) ==
        "tadah analysis bfunc writes one OUTFILE or none");
  CHECK(refusal_of<engines::AnalysisBfuncEngine>(
            "analysis.bfunc", {"TYPE 2b Y Z", "RANGE 0 5 10"}) ==
        "tadah analysis bfunc takes one or two TYPE values");
  CHECK(refusal_of<engines::AnalysisBfuncEngine>(
            "analysis.bfunc", {"TYPE 2b", "RANGE 0 5"}) ==
        "tadah analysis bfunc takes three RANGE values: start, stop and the "
        "number of points");
  CHECK(refusal_of<engines::ExplainEngine>("explain", {}) ==
        "tadah explain needs an OPTION");
  CHECK(refusal_of<engines::RefitEngine>("refit", {"REFIT_MODE dense"}) ==
        "tadah refit: no EAM line in the configuration (-c) names the input "
        "EAM file, a single-element setfl file (eam/alloy or eam/fs); --edit "
        "and --retarget take the outputs of an earlier refit");
}

/* Pins: tadah predict refuses an --outfile that does not name a file for
 * each output, before anything is read, in one message: the energy file,
 * then the force file with --force and the stress file with --stress, or
 * the three files of the default layout (energy, force, stress):
 * "tadah predict writes the energies and the forces (--force): --outfile
 * names 1 file, e.dat, and takes 2, the energy file and the force file, or
 * 3, the energy, force and stress files", and the forms of --stress, of
 * both and of neither; 3 files are accepted with --force alone.
 * Before the fix: validate() logged an ERROR, "Provide 2 output files: one
 * for energy and one for forces.", "Provide 3 output files for energy
 * forces and stress." or "Provide single output file for energy.", and
 * returned; the run was refused later, when the output was written
 * ("Invalid OUTFILE size 1. Expected either 2 for the requested outputs or
 * 3 for the default [energy, force, stress] layout."). */
TEST_CASE("tadah predict refuses an --outfile without a file for each "
          "output",
          "[engines][messages][predict]") {
  const auto refused = [](const std::vector<std::string> &keys) {
    std::vector<std::string> all = {"POTENTIAL pot.tadah",
                                    "DBFILE db.tadah"};
    all.insert(all.end(), keys.begin(), keys.end());
    return refusal_of<engines::PredictEngine>("predict", all);
  };
  CHECK(refused({"FORCE true", "OUTFILE e.dat"}) ==
        "tadah predict writes the energies and the forces (--force): "
        "--outfile names 1 file, e.dat, and takes 2, the energy file and the "
        "force file, or 3, the energy, force and stress files");
  CHECK(refused({"STRESS true", "OUTFILE e.dat"}) ==
        "tadah predict writes the energies and the stresses (--stress): "
        "--outfile names 1 file, e.dat, and takes 2, the energy file and the "
        "stress file, or 3, the energy, force and stress files");
  CHECK(refused({"FORCE true", "STRESS true", "OUTFILE e.dat f.dat"}) ==
        "tadah predict writes the energies, the forces and the stresses "
        "(--force, --stress): --outfile names 2 files, e.dat f.dat, and "
        "takes 3, the energy, force and stress files");
  CHECK(refused({"OUTFILE e.dat f.dat"}) ==
        "tadah predict writes the energies: --outfile names 2 files, e.dat "
        "f.dat, and takes 1, the energy file, or 3, the energy, force and "
        "stress files");
  CHECK(refused({"FORCE true", "OUTFILE e.dat f.dat s.dat"}) ==
        "no exception");
  CHECK(refused({"OUTFILE e.dat"}) == "no exception");
}

/* Pins: the structure writer of an unknown format is refused, naming the
 * format and the formats written: "unknown structure format 'CELL': the
 * formats written are CASTEP, LAMMPS and VASP" (std::invalid_argument).
 * Before the fix: get_writer printed "Unknown type! Returning nullptr." to
 * standard error and returned a null writer, which tadah data write
 * dereferenced (a segmentation fault). */
TEST_CASE("The structure writer of an unknown format is refused",
          "[engines][messages][write]") {
  tadah::mlip::StructureDB db;
  CHECK_THROWS_AS(
      tadah::mlip::StructureWriterSelector::get_writer("CELL", db),
      std::invalid_argument);
  CHECK(thrown([&] {
          (void)tadah::mlip::StructureWriterSelector::get_writer("CELL", db);
        }) ==
        "unknown structure format 'CELL': the formats written are CASTEP, "
        "LAMMPS and VASP");
}
