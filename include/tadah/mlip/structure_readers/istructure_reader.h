#ifndef TADAH_MLIP_STRUCTURE_READERS_ISTRUCTURE_READER_H
#define TADAH_MLIP_STRUCTURE_READERS_ISTRUCTURE_READER_H

#include <string>
#include <tadah/mlip/structure.h>

/**
 * @brief Abstract base class for reading a Structure from a file or remote source.
 *
 * Provides a unified interface for converting raw local/remote data
 * into a Structure object. Parsing routines belong in classes
 * derived from this interface.
 */
class IStructureReader {
public:
  /// Virtual destructor for proper cleanup.
  virtual ~IStructureReader() = default;

  /**
   * @brief Reads structure data from a path or identifier.
   *
   * @param source  Could be a file path (e.g., "myfile.cif") or a recognized
   *                online ID (e.g., "mp-149").
   */
  virtual void read(const std::string &source) = 0;

  /**
   * @brief Retrieves the last successfully read Structure.
   *        Coordinates are returned in absolute Cartesian units.
   *
   * @return  A Structure object containing lattice and atomic positions.
   */
  virtual Structure getStructure() const = 0;
};

#endif // TADAH_MLIP_STRUCTURE_READERS_ISTRUCTURE_READER_H
