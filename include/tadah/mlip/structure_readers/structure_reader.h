#ifndef STRUCTURE_READER_H
#define STRUCTURE_READER_H

#include <string>
#include <tadah/mlip/structure.h>

/**
 * @brief Abstract base class for reading a Structure from a file or an online
 * source.
 *
 * Provides a unified interface for various file formats or REST API sources.
 */
class StructureReader {
public:
  /// Virtual destructor.
  virtual ~StructureReader() = default;

  /**
   * @brief Reads structural information from the given path or identifier.
   *
   * @param path A file path (e.g., "myfile.cif") or an online ID (e.g.,
   * "mp-149").
   */
  virtual void read(const std::string &path) = 0;

  /**
   * @brief Retrieves the last read Structure.
   *        All coordinates should be in absolute (Cartesian) Angstrom units.
   */
  virtual Structure getStructure() const = 0;
};

#endif // STRUCTURE_READER_H
