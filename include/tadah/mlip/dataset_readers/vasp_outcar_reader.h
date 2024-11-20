#ifndef VASP_OUTCAR_READER_H
#define VASP_OUTCAR_READER_H

#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/dataset_reader.h>
#include <string>

/**
 * @class VaspOutcarReader
 * @brief Concrete class for reading and parsing VASP OUTCAR files.
 *
 * This class implements the DatasetReader interface for handling
 * VASP OUTCAR files.
 *
 * Example usage:
 * @code
 * StructureDB my_db;
 * // Using the basic constructor
 * VaspOutcarReader reader1(my_db);
 * reader1.read_data("OUTCAR");
 * reader1.parse_data();
 * reader1.print_summary();
 * 
 * // Using the constructor with filename
 * VaspOutcarReader reader2(my_db, "OUTCAR");
 * reader2.print_summary();
 * @endcode
 */
class VaspOutcarReader : public DatasetReader {
public:
  /**
   * @brief Constructor initializing base class reference.
   *
   * @param db Reference to a StructureDB object to store parsed data.
   */
  VaspOutcarReader(StructureDB& db);

  /**
   * @brief Constructor that initializes and reads from a file.
   *
   * Reads and parses data immediately upon construction.
   *
   * @param db Reference to a StructureDB object to store parsed data.
   * @param filename The name of the OUTCAR file to read data from.
   */
  VaspOutcarReader(StructureDB& db, const std::string& filename);

  /**
   * @brief Reads data from the specified OUTCAR file.
   *
   * Implements the logic to read data specific to OUTCAR files.
   *
   * @param filename The name of the file to read data from.
   */
  void read_data(const std::string& filename) override;

  /**
   * @brief Parses the read OUTCAR data.
   *
   * Implements the parsing logic for the OUTCAR file data.
   */
  void parse_data() override;

  /**
   * @brief Prints or returns a summary of the OUTCAR data.
   *
   * Provides a summary using the parsed StructureDB data.
   *
   * @param stdb Reference to a StructureDB object for data summary.
   */
  void print_summary() const override;

private:
  std::string raw_data_;  // Stores raw file data
  double s_conv = 6.241509074e-4; // kbar -> eV/A^3
};

#endif // VASP_OUTCAR_READER_H
