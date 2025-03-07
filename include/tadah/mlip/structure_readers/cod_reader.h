#ifndef COD_READER_H
#define COD_READER_H

#include <tadah/mlip/structure_readers/structure_reader.h>
#include <tadah/mlip/structure_readers/cif_reader.h>
#include <tadah/mlip/structure.h>

#include <string>

/**
 * @brief Reads a CIF file from the Crystallography Open Database (COD)
 *        in memory and parses its content using an existing CifReader.
 *
 * Inherits from StructureReader to provide a common interface
 * for reading structures from online sources.
 */
class codReader : public StructureReader {
public:
  /// Default constructor
  codReader() = default;

  /**
   * @brief Retrieves a CIF file from the COD database by constructing
   *        a URL or using a direct URL and loads it into a Structure object.
   *
   * If @p path does not start with "http://" or "https://", the URL is built
   * from a COD ID (e.g., "1534932") by appending ".cif". If ".cif" is already 
   * in @p path, it is appended as is. The downloaded content is parsed with CifReader.
   *
   * @param path A COD entry ID or a fully qualified URL.
   */
  void read(const std::string &path) override;

  /**
   * @brief Accesses the last successfully read Structure.
   */
  Structure getStructure() const override;

private:
  /**
   * @brief A helper callback function that libcurl uses to write received bytes
   *        into a std::string in memory.
   */
  static size_t writeDataCallback(void* ptr, size_t size, size_t nmemb, void* userData);

  /**
   * @brief Downloads data from the given URL and returns it as a single string.
   *
   * Uses libcurl's easy interface for making a GET request.
   *
   * @param url A valid URL that points to a CIF resource on COD.
   * @return The complete content fetched from @p url.
   */
  std::string fetchDataFromURL(const std::string &url) const;

  /**
   * @brief Internal buffer for storing the last read structure.
   */
  Structure m_structure;
};

#endif // COD_READER_H
