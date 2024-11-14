#ifndef DATASET_READER_H
#define DATASET_READER_H

#include <tadah/mlip/structure_db.h>
#include <string>
#include <vector>

/**
 * @class DatasetReader
 * @brief Abstract base class for reading and parsing dataset files.
 *
 * This class provides an interface for reading, parsing, and summarizing
 * dataset files. Derived classes must implement the pure virtual functions
 * to handle specific file formats.
 */
class DatasetReader {
public:
  /**
   * @brief Virtual destructor.
   *
   * Ensures proper cleanup of derived class objects.
   */
  virtual ~DatasetReader() = default;

  /**
   * @brief Reads data from a specified file.
   *
   * This pure virtual function must be implemented by derived classes
   * to define how data is read from files.
   *
   * @param filename The name of the file to read data from.
   */
  virtual void read_data(const std::string& filename) = 0;

  /**
   * @brief Parses the read data.
   *
   * This pure virtual function must be implemented by derived classes
   * to define how the data is parsed.
   */
  virtual void parse_data() = 0;

  /**
   * @brief Prints or returns a summary of the data.
   *
   * This pure virtual function must be implemented by derived classes
   * to provide a summary of the dataset.
   *
   * @param stdb Reference to a StructureDB object for data summary.
   */
  virtual void print_summary() const = 0;

  /**
   * @brief Constructor initializing the reference.
   *
   * @param db Reference to a StructureDB object to store parsed data.
   */
  DatasetReader(StructureDB& db) : stdb(db) {}

  /**
   * @brief Constructor initializing the reference and filename.
   *
   * @param db Reference to a StructureDB object to store parsed data.
   * @param filename The name of the file to read data from.
   */
  DatasetReader(StructureDB& db, const std::string& filename) 
  : stdb(db), filename(filename) {}

protected:
  StructureDB& stdb; /**< Reference to store parsed data. */
  std::string filename; /**< Filename for data reading. */
};
#endif // DATASET_READER_H
