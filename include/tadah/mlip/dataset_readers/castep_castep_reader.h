#ifndef CASTEP_CASTEP_READER_H
#define CASTEP_CASTEP_READER_H

#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/dataset_reader.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
/**
 * @class CastepCastepReader
 * @brief A class for reading and parsing CASTEP .castep files.
 *
 * Implements data extraction and processing for molecular dynamics
 * simulations (both constant and non-constant volume),
 * geometry optimization, and single-point energy calculations.
 *
 * Supports concatenated .castep files, such as an MD run followed
 * by a geometry optimization.
 *
 * The output units are:
 * - eV for energy
 * - Ångström for distance
 * - eV/Å for force
 * - eV/Å³ for pressure (converted from GPa)
 * - ps for time
 * 
 * If there is no stress tensor, it is set to zero.
 *
 * Example usage:
 * @code
 * StructureDB my_db;
 * // Using the basic constructor
 * CastepCastepReader reader1(my_db);
 * reader1.read_data("test.castep");
 * reader1.parse_data();
 * reader1.print_summary();
 * 
 * // Using the constructor with filename
 * CastepCastepReader reader2(my_db, "test.castep");
 * reader2.print_summary();
 * @endcode
 */
class CastepCastepReader : public DatasetReader {
public:
  /**
   * @brief Constructs a CastepCastepReader with a StructureDB reference.
   * @param db Reference to a StructureDB object for storing parsed data.
   */
  CastepCastepReader(StructureDB& db);

  /**
   * @brief Constructs a CastepCastepReader and reads the specified file.
   * @param db Reference to a StructureDB object for storing parsed data.
   * @param filename Name of the .castep file to read.
   */
  CastepCastepReader(StructureDB& db, const std::string& filename);

  /**
   * @brief Reads data from a specified .castep file.
   * @param filename The name of the .castep file to read data from.
   */
  virtual void read_data(const std::string& filename) override;

  /**
   * @brief Parses the data read from the .castep file.
   */
  virtual void parse_data() override;

  /**
   * @brief Prints a summary of the parsed .castep data.
   */
  virtual void print_summary() const override;
  virtual std::string get_summary() const override;

protected:
  std::string raw_data_;  /**< Stores raw file data */
  std::string filename_; /**< Filename of the .castep file */

  // Unit conversion factors
  double p_conv = 0.00624150913; /**< Conversion factor from GPa to eV/A^3 */

};

#endif // CASTEP_CASTEP_READER_H


