#include <tadah/core/logger.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/engines/data_write_engine.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>
#include <tadah/mlip/structure_writers/structure_writer_selector.h>


#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

int DataWriteEngine::run(tadah::mlip::Task &task) {
  core::Context taskContext = task.getContext();

  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "DataWriteEngine started...");

  std::vector<std::string> indicesStr;
  StructureDB stdb;
  if(taskContext.contains("DBFILE")) {
    stdb.add(taskContext);
    indicesStr.resize(taskContext.size("INDEX"));
    taskContext.get<>("INDEX", indicesStr);
  }
  if(taskContext.contains("STRUCTURE")) {
    std::vector<std::string> structures(taskContext.size("STRUCTURE"));
    taskContext.get<>("STRUCTURE", structures);
    size_t index = 0;
    for (std::string &struc : structures) {
      auto reader = StructureReaderSelector::getReader(struc);
      reader->read(struc);
      Structure s = reader->getStructure();
      stdb.add(s);
      // generate full list of indices as structure excludes INDEX key
      indicesStr.push_back(core::to_string(index++)+",");
    }
  }
  std::vector<size_t> indices =  core::parse_indices(indicesStr);
  // Adjust for 0-based indexing
  for (auto &i : indices) i--;

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

  for (size_t i=0; i<indices.size(); ++i) {
    std::string outFile = taskContext.get<std::string>("OUTFILE",i);
    writer->write_data(outFile, indices[i]);
  }

  return EXIT_SUCCESS;
}

void DataWriteEngine::validate(const tadah::mlip::Task &task) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Validating DataWriteEngine...");
  core::Context taskContext = task.getContext();
  // if dbfile is provided than we check if number of indices is equal to
  // the number of outfiles
  if(taskContext.contains("DBFILE")) {
    std::vector<std::string> indicesStr;
    indicesStr.resize(taskContext.size("INDEX"));
    taskContext.get<>("INDEX", indicesStr);
    std::vector<size_t> indices =  core::parse_indices(indicesStr);
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
