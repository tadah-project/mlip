#ifndef DATASET_READER_SELECTOR_H
#define DATASET_READER_SELECTOR_H

#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/dataset_reader.h>
#include <string>
#include <memory>

/**
 * @class DatasetReaderSelector
 * @brief Selects the appropriate DatasetReader based on file type.
 */
class DatasetReaderSelector {
public:
  /**
   * @brief Factory method to create specific DatasetReader objects.
   *
   * @param type The type of the dataset reader to create.
   * @param db Reference to a StructureDB object to store parsed data.
   * @return A unique pointer to a DatasetReader object.
   */
  static std::unique_ptr<DatasetReader> get_reader(const std::string& type, StructureDB& db);

  /**
   * @brief Determines the file type based on its content.
   *
   * @param filepath File path to check the content.
   * @return A string representing the detected file type.
   */
  static std::string determine_file_type_by_content(const std::string& filepath);

};

#endif // DATASET_READER_SELECTOR_H
