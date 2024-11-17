#ifndef CASTEP_MD_READER_H
#define CASTEP_MD_READER_H

#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/dataset_reader.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

/**
 * @class CastepMDReader
 * @brief A class for reading and parsing CASTEP .md files.
 *
 * Implements data extraction and processing for molecular dynamics
 * simulations by converting data from atomic units to common units:
 * eV for energy, Ångström for distance, eV/Å for force, and eV/Å³
 * for pressure.
 *
 * Example usage:
 * @code
 * StructureDB my_db;
 * // Using the basic constructor
 * CastepMDReader reader1(my_db);
 * reader1.read_data("test.md");
 * reader1.parse_data();
 * reader1.print_summary();
 * 
 * // Using the constructor with filename
 * CastepMDReader reader2(my_db, "test.md");
 * reader2.print_summary();
 * @endcode
 */
class CastepMDReader : public DatasetReader {
public:
  /**
   * @brief Constructs a CastepMDReader with a StructureDB reference.
   * @param db Reference to a StructureDB object for storing parsed data.
   */
  CastepMDReader(StructureDB& db);

  /**
   * @brief Constructs a CastepMDReader and reads the specified file.
   * @param db Reference to a StructureDB object for storing parsed data.
   * @param filename Name of the .md file to read.
   */
  CastepMDReader(StructureDB& db, const std::string& filename);

  /**
   * @brief Reads data from a specified .md file.
   * @param filename The name of the .md file to read data from.
   */
  void read_data(const std::string& filename) override;

  /**
   * @brief Parses the data read from the .md file.
   */
  void parse_data() override;

  /**
   * @brief Prints a summary of the parsed .md data.
   */
  void print_summary() const override;

private:
  std::string raw_data_;  /**< Stores raw file data */
  std::string filename_; /**< Filename of the .md file */

  // Unit conversion factors
  double e_conv = 27.211386245988; /**< Conversion factor from Hartree to eV */
  double d_conv = 0.529177210903; /**< Conversion factor from Bohr to Ångström */
  double f_conv = e_conv / d_conv;  /**< Conversion factor from Hartree/Bohr to eV/Å */
  double s_conv = e_conv / (d_conv * d_conv * d_conv);  /**< Conversion factor from Hartree/Bohr³ to eV/Å³ */
  double t_conv = 2.418884326505e-5; /**< Time conversion to ps */ 
  double k_b = 8.617333262145e-5 / e_conv; /**< Boltzmann constant in atomic units (Hartree/K) */

  // Helper methods
  /**
   * @brief Checks if a string ends with a given suffix.
   * @param str The string to check.
   * @param suffix The suffix to check for.
   * @return True if str ends with suffix, otherwise false.
   */
  bool ends_with(const std::string& str, const std::string& suffix);

  /**
   * @brief Calculates the ideal gas pressure.
   * @param s Reference to a Structure object.
   * @param T Temperature in atomic units.
   * @return Calculated ideal gas pressure.
   */
  double calc_P_ideal(Structure &s, double T);

  /**
   * @brief Performs post-processing on the given structure.
   * @param s Reference to a Structure object.
   */
  void postproc_structure(Structure &s);

  double T = 0; /**< Temperature in atomic units (Hartree/k_B) */
  bool stress_tensor_bool = false; /**< Indicates presence of a stress tensor */
};

#endif // CASTEP_MD_READER_H

