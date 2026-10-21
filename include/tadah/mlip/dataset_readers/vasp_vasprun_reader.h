#ifndef VASP_VASPRUN_READER_H
#define VASP_VASPRUN_READER_H

#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/dataset_reader.h>
#include <RapidXml/rapidxml.hpp>
#include <RapidXml/rapidxml_utils.hpp>
#include <string>
#include <vector>
#include <iostream>
namespace tadah {
namespace mlip {

namespace rx = rapidxml;

/**
 * @class VaspVasprunReader
 * @brief Concrete class for reading and parsing VASP vasprun.xml files.
 *
 * This class implements the DatasetReader interface for handling
 * VASP vasprun.xml files and stores the parsed data in a StructureDB.
 *
 * Each <calculation> gives a structure with the free energy F (the
 * calculation-level e_fr_energy, whose gradient the forces and the stress
 * are; e_wo_entrp at that level holds energy(sigma->0) in VASP 5.4.4) and
 * the Cauchy stress in eV/A^3, minus the 'stress' varray (pressure sign).
 * A calculation without a 'stress' varray is stored with zero stress and
 * stress priority 0 (sweight), so that a STRESS fit ignores it.
 *
 * Example usage:
 * @code
 * StructureDB my_db;
 * // Using the basic constructor
 * VaspVasprunReader reader1(my_db);
 * reader1.read_data("vasprun.xml");
 * reader1.parse_data();
 * reader1.print_summary();
 * 
 * // Using the constructor with filename
 * VaspVasprunReader reader2(my_db, "vasprun.xml");
 * reader2.print_summary();
 * @endcode
 */
class VaspVasprunReader : public DatasetReader {
public:
  /**
     * @brief Constructor initializing with a StructureDB reference.
     *
     * @param stdb Reference to a StructureDB object for storing parsed data.
     */
  VaspVasprunReader(StructureDB& stdb);

  /**
     * @brief Constructor that initializes and reads from a file.
     *
     * Automatically reads and parses data upon construction using the given file.
     *
     * @param stdb Reference to a StructureDB object.
     * @param filename Name of the vasprun.xml file to read.
     */
  VaspVasprunReader(StructureDB& stdb, const std::string& filename);

  /**
     * @brief Destructor for VaspVasprunReader.
     *
     * Cleans up dynamically allocated resources, if any.
     */

  // Delete copy constructor
  VaspVasprunReader(const VaspVasprunReader& other) = delete;

  // Delete copy assignment operator
  VaspVasprunReader& operator=(const VaspVasprunReader& other) = delete;   ~VaspVasprunReader();

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
  virtual void print_summary() const override;
  virtual std::string get_summary() const override;

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
     * @brief Extracts the free energy (e_fr_energy) from a calculation node.
     *
     * Throws std::runtime_error, naming the file and the calculation
     * (1-based), when the calculation has none or its value is not a
     * number.
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
  std::vector<char> buffer_; ///< The text doc is parsed from (in place); empty when the file ends before any complete calculation.
  std::vector<std::string> atom_types; ///< List of atom types; length is the number of atoms.
  StructureDB& stdb; ///< Reference to the StructureDB for storing data.

private:
  /// "VASP vasprun.xml <file>: ", the start of every message.
  std::string where() const;
  /// where() and "calculation <n>: ".
  std::string where_calculation() const;

  Structure _s; ///< Internal structure representation.
  bool stress_tensor_bool = false; ///< Flag indicating stress tensor presence.
  bool forces_found_ = false; ///< The calculation's forces were read.
  std::size_t ends_inside_ = 0; ///< The calculation the file ends inside (1-based), 0 for a complete file.
  std::size_t ends_after_ = 0; ///< The last calculation of a file that ends after it, without its closing modeling tag (no calculation open), 0 otherwise.
  bool ends_before_first_ = false; ///< The file ends before its first calculation.
  std::vector<tadah::core::Element> elements_; ///< The elements of atom_types.
  std::string filename_; ///< The file read, for messages.
  size_t calculation_ = 0; ///< The calculation being read, 1-based, for messages.
  double s_conv = 6.241509074e-4; // kbar -> eV/A^3
};

} // namespace mlip
} // namespace tadah
#endif // VASP_VASPRUN_READER_H
