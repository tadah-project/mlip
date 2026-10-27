#include <tadah/core/logger.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/engines/data_write_engine.h>
#include <tadah/mlip/output/writable_outputs.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>
#include <tadah/mlip/structure_writers/structure_writer_selector.h>


#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

namespace {
/* The 1-based INDEX list of the DBFILE structures, checked against their
 * number, @p total (core::parse_dataset_indices). */
std::vector<size_t> dbfile_indices(const core::Context &taskContext, size_t total) {
  std::vector<std::string> indicesStr(taskContext.size("INDEX"));
  taskContext.get<>("INDEX", indicesStr);
  return core::parse_dataset_indices(indicesStr, total, "tadah data write");
}
}  // namespace

int DataWriteEngine::run(tadah::mlip::Task &task) {
  core::Context taskContext = task.getContext();


  // The 0-based places in stdb of the structures to write: the DBFILE
  // structures INDEX selects (1-based, checked against their number), then
  // every -s structure (STRUCTURE excludes INDEX).
  StructureDB stdb;
  std::vector<size_t> indices;
  if(taskContext.contains("DBFILE")) {
    stdb.add(taskContext);
    for (size_t i : dbfile_indices(taskContext, stdb.size())) indices.push_back(i - 1);
  }
  if(taskContext.contains("STRUCTURE")) {
    std::vector<std::string> structures(taskContext.size("STRUCTURE"));
    taskContext.get<>("STRUCTURE", structures);
    for (std::string &struc : structures) {
      auto reader = StructureReaderSelector::getReader(struc);
      reader->read(struc);
      Structure s = reader->getStructure();
      stdb.add(s);
      indices.push_back(stdb.size() - 1);
    }
  }
  if (taskContext.size("OUTFILE") != indices.size())
    throw std::runtime_error("Number of OUTFILE must be equal to the number of INDEX");

  const std::string format = core::to_upper(taskContext.get<std::string>("FORMAT"));
  auto writer = StructureWriterSelector::get_writer(format,stdb);

  if (format == "LAMMPS" || format == "VASP") {
    // Reals are written with NUMERIC significant digits (%.{N}g): default
    // 18 = exact; an explicit NUMERIC below 17 is honoured with a WARNING,
    // as for dataset files.
    writer->set_precision(StructureDB::export_digits(taskContext,
        format == "LAMMPS" ? "LAMMPS data file" : "VASP POSCAR file"));
  } else if (taskContext.contains("NUMERIC")) {
    // CASTEP cell: NUMERIC is the number of decimal places (fixed).
    writer->set_precision(taskContext.get<size_t>("NUMERIC"));
  }

  // Every structure and every output is checked before any file is written.
  std::vector<std::string> outFiles;
  for (size_t i=0; i<indices.size(); ++i) {
    writer->check(indices[i]);
    outFiles.push_back(taskContext.get<std::string>("OUTFILE",i));
  }
  require_writable_outputs(outFiles, "tadah data write");
  for (size_t i=0; i<indices.size(); ++i) {
    writer->write_data(outFiles[i], indices[i]);
  }

  return EXIT_SUCCESS;
}

void DataWriteEngine::validate(const tadah::mlip::Task &task) {
  core::Context taskContext = task.getContext();
  // The formats StructureWriterSelector writes, refused before anything is
  // read. FORMAT is a command_only key: the command line gives it with -f
  // (--format), so the refusal names the option.
  if (taskContext.contains("FORMAT")) {
    const std::string format = taskContext.get<std::string>("FORMAT");
    const std::string up = core::to_upper(format);
    if (up != "CASTEP" && up != "LAMMPS" && up != "VASP")
      throw std::runtime_error("tadah data write: --format " + format +
                               " is not a format it writes: castep, lammps "
                               "or vasp");
  }
  // if dbfile is provided than we check if number of indices is equal to
  // the number of outfiles
  if(taskContext.contains("DBFILE")) {
    const std::vector<size_t> indices =
        dbfile_indices(taskContext, StructureDB::count(taskContext).first);
    if (taskContext.size("OUTFILE") != indices.size()) {
      throw std::runtime_error("Number of OUTFILE must be equal to the number of INDEX");
    }
  }
  // if STRUCTURE we check if number of structures is equal to outfile
  if(taskContext.contains("STRUCTURE")) {
    if (taskContext.size("OUTFILE") != taskContext.size("STRUCTURE")) {
      throw std::runtime_error("Number of OUTFILE must be equal to the number of STRUCTURE");
    }
  }
}


} // namespace engines
} // namespace mlip
} // namespace tadah
