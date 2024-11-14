#ifndef VASPRUN_READER_H
#define VASPRUN_READER_H

#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/dataset_reader.h>
#include <RapidXml/rapidxml.hpp>
#include <RapidXml/rapidxml_utils.hpp>
#include <string>
#include <vector>
#include <iostream>

namespace rx = rapidxml;

/**
 * @class VasprunReader
 * @brief Concrete class for reading and parsing VASP vasprun.xml files.
 *
 * This class implements the DatasetReader interface for handling
 * VASP vasprun.xml files and stores the parsed data in a StructureDB.
 *
 * Example usage:
 * @code
 * StructureDB my_db;
 * // Using the basic constructor
 * VasprunReader reader1(my_db);
 * reader1.read_data("vasprun.xml");
 * reader1.parse_data();
 * reader1.print_summary();
 * 
 * // Using the constructor with filename
 * VasprunReader reader2(my_db, "vasprun.xml");
 * reader2.print_summary();
 * @endcode
 */
class VasprunReader : public DatasetReader {
public:
  /**
     * @brief Constructor initializing with a StructureDB reference.
     *
     * @param stdb Reference to a StructureDB object for storing parsed data.
     */
  VasprunReader(StructureDB& stdb);

  /**
     * @brief Constructor that initializes and reads from a file.
     *
     * Automatically reads and parses data upon construction using the given file.
     *
     * @param stdb Reference to a StructureDB object.
     * @param filename Name of the vasprun.xml file to read.
     */
  VasprunReader(StructureDB& stdb, const std::string& filename);

  /**
     * @brief Destructor for VasprunReader.
     *
     * Cleans up dynamically allocated resources, if any.
     */

  // Delete copy constructor
  VasprunReader(const VasprunReader& other) = delete;

  // Delete copy assignment operator
  VasprunReader& operator=(const VasprunReader& other) = delete;   ~VasprunReader();

  /**
     * @brief Reads data from the specified vasprun.xml file.
     *
     * Loads the file content into memory for parsing.
     *
     * @param filename Name of the file to read data from.
     */
  void read_data(const std::string& filename) override;

  /**
     * @brief Parses the vasprun.xml data.
     *
     * Processes the XML document loaded into memory.
     */
  void parse_data() override;

  /**
     * @brief Prints a summary of the vasprun.xml data.
     *
     * Uses the parsed data stored in the StructureDB to provide a summary.
     */
  void print_summary() const override;

protected:
  /**
     * @brief Extracts the number of atoms in the current structure.
     *
     * Parses the XML document to count the number of atoms.
     * 
     * @return Number of atoms in the structure.
     */
  int get_number_of_atoms() const;

  /**
     * @brief Parses atom types from the XML.
     *
     * Extracts atom type information from the atominfo section.
     *
     * @param root_node Pointer to the root XML node.
     */
  void extract_atom_types(rx::xml_node<> *root_node);

  /**
     * @brief Parses calculation information from the XML.
     *
     * Extracts data for each calculation block in the XML.
     *
     * @param root_node Pointer to the root XML node.
     */
  void extract_calculations(rx::xml_node<> *root_node);

  /**
     * @brief Extracts total energy from a calculation node.
     *
     * Retrieves the total energy value from each calculation.
     *
     * @param calculation_node Pointer to the calculation XML node.
     */
  void extract_total_energy(rx::xml_node<> *calculation_node);

  /**
     * @brief Extracts stress tensor from a calculation node.
     *
     * Outputs the stress tensor for each calculation.
     *
     * @param calculation_node Pointer to the calculation XML node.
     */
  void extract_stress_tensor(rx::xml_node<> *calculation_node);

  /**
     * @brief Extracts basis vectors and atomic positions from a structure node.
     *
     * Outputs basis vectors and matches atom positions to their types.
     *
     * @param structure_node Pointer to the structure XML node.
     */
  void extract_basis_vectors_and_positions(rx::xml_node<> *structure_node);

  /**
     * @brief Extracts forces from a calculation node.
     *
     * Retrieves atomic forces from each calculation.
     *
     * @param calculation_node Pointer to the calculation XML node.
     */
  void extract_forces(rx::xml_node<> *calculation_node);

protected:
  rx::xml_document<> doc; ///< XML document object for parsing.
  rx::file<> *xmlFile = nullptr; ///< Optional pointer for file management.
  std::vector<std::string> atom_types; ///< List of atom types; length is the number of atoms.
  StructureDB& stdb; ///< Reference to the StructureDB for storing data.

private:
  Structure _s; ///< Internal structure representation.
  bool stress_tensor_bool = false; ///< Flag indicating stress tensor presence.
};

#endif // VASPRUN_READER_H
