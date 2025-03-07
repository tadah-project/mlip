#ifndef STRUCTURE_READER_SELECTOR_H
#define STRUCTURE_READER_SELECTOR_H

#include <memory>
#include <string>
#include <tadah/mlip/structure_readers/istructure_reader.h>

/**
 * @brief A factory that provides an appropriate IStructureReader
 *        based on file extension or recognized online ID prefixes.
 */
class StructureReaderSelector {
public:
  /**
   * @brief Creates a reader by analyzing the input path or ID.
   *
   * @param pathOrId File path or an online ID (e.g., "mp-149")
   * @return A suitable IStructureReader capable of parsing the structure.
   * @throws std::runtime_error if no suitable reader is found.
   */
  static std::unique_ptr<IStructureReader>
  getReader(const std::string &pathOrId);

private:
  static std::string guessFormat(const std::string &pathOrId);
  static std::string guessFormatFromContent(const std::string &filePath);
};

#endif // STRUCTURE_READER_SELECTOR_H
