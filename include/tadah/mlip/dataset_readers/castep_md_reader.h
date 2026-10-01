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
namespace tadah {
namespace mlip {

/**
 * @class CastepMDReader
 * @brief A class for reading and parsing CASTEP .md files.
 *
 * Implements data extraction and processing for molecular dynamics
 * simulations by converting data from atomic units to common units:
 * eV for energy, Ångström for distance, eV/Å for force, eV/Å³
 * for stress and ps for time.
 * 
 * The stored stress is the Cauchy stress (positive under tension) without
 * its kinetic part: the '<-- S' tensor of a .md frame includes
 * -(1/V) sum m v (x) v, which is removed with the frame's velocities and
 * the periodic-table masses, scaled to the frame's kinetic energy (the third
 * '<-- E' value). A frame with a '<-- S' tensor whose kinetic energy is
 * more than 1e-3 off those masses (a custom SPECIES_MASS) is refused; a
 * frame without one needs no masses and is read, with zero stress and
 * stress weight 0 (sweight), so that a STRESS fit ignores it. Every frame is read, the first
 * one included. A malformed frame is an error naming the file, the frame
 * (1-based) and the line; a last frame that the file ends inside (a run
 * stopped while writing it) is skipped with one WARNING.
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
  virtual void read_data(const std::string& filename) override;

  /**
   * @brief Parses the data read from the .md file.
   */
  virtual void parse_data() override;

  /**
   * @brief Prints a summary of the parsed .md data.
   */
  virtual void print_summary() const override;
  virtual std::string get_summary() const override;

  /**
   * @brief Checks if a string ends with a given suffix.
   * @param str The string to check.
   * @param suffix The suffix to check for.
   * @return True if str ends with suffix, otherwise false.
   */
  bool ends_with(const std::string& str, const std::string& suffix);

protected:
  std::string raw_data_;  /**< Stores raw file data */
  std::string filename_; /**< Filename of the .md file */

  // Unit conversion factors
  double e_conv = 27.211386245988; /**< Conversion factor from Hartree to eV */
  double d_conv = 0.529177210903; /**< Conversion factor from Bohr to Ångström */
  double f_conv = e_conv / d_conv;  /**< Conversion factor from Hartree/Bohr to eV/Å */
  double s_conv = e_conv / (d_conv * d_conv * d_conv);  /**< Conversion factor from Hartree/Bohr³ to eV/Å³ */
  double t_conv = 2.418884326505e-5; /**< Time conversion to ps */ 
  double k_b = 8.617333262145e-5 / e_conv; /**< Boltzmann constant in atomic units (Hartree/K) */

public:
  /** One line of the file and its 1-based number. */
  struct FrameLine {
    size_t number;
    std::string text;
  };

protected:
  /**
   * @brief Whether frames carry velocities ('<-- V') and a kinetic energy
   *        whose stress part is removed: true for .md, false for .geom.
   */
  virtual bool has_velocities() const { return true; }

  /**
   * @brief The structure of one frame (its non-blank lines); frame_index is
   *        0 for the file's first frame, which gets get_first_label().
   */
  Structure parse_frame(const std::vector<FrameLine> &frame, size_t frame_index);

private:
  virtual std::string get_first_label(std::string &) override;
  virtual std::string get_label(std::string &) override;

};

} // namespace mlip
} // namespace tadah
#endif // CASTEP_MD_READER_H

