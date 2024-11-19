#ifndef CASTEP_GEOM_READER_H
#define CASTEP_GEOM_READER_H

#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/castep_md_reader.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

/**
 * @class CastepGeomReader
 * @brief A class for reading and parsing CASTEP .geom files.
 *
 * Implements data extraction and processing from geometry optimisation runs.
 * Data are converted from atomic units to common units:
 * eV for energy, Ångström for distance, eV/Å for force, and eV/Å³
 * for pressure.
 *
 * Example usage:
 * @code
 * StructureDB my_db;
 * // Using the basic constructor
 * CastepGeomReader reader1(my_db);
 * reader1.read_data("test.geom");
 * reader1.parse_data();
 * reader1.print_summary();
 * 
 * // Using the constructor with filename
 * CastepGeomReader reader2(my_db, "test.geom");
 * reader2.print_summary();
 * @endcode
 */
class CastepGeomReader : public CastepMDReader {
public:
  /**
   * @brief Constructs a CastepGeomReader with a StructureDB reference.
   * @param db Reference to a StructureDB object for storing parsed data.
   */
  CastepGeomReader(StructureDB& db);

  /**
   * @brief Constructs a CastepGeomReader and reads the specified file.
   * @param db Reference to a StructureDB object for storing parsed data.
   * @param filename Name of the .geom file to read.
   */
  CastepGeomReader(StructureDB& db, const std::string& filename);

private:

  /**
   * @brief Dummy ideal gas pressure.
   * @return zero.
   */
  double calc_P_ideal(Structure &, double ) override;

  std::string get_first_label(std::string &) override;
  std::string get_label(std::string &) override;

};

#endif // CASTEP_GEOM_READER_H

