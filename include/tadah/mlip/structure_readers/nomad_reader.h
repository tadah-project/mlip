#ifndef NOMAD_READER_H
#define NOMAD_READER_H

#include <string>
#include <tadah/mlip/structure_readers/structure_reader.h>
#include <tadah/mlip/structure.h>

/**
 * @class NomadReader
 * @brief Implements the StructureReader interface to fetch structural data from NOMAD.
 *
 * Uses a POST request:
 *   https://nomad-lab.eu/prod/v1/api/v1/entries/<entryID>/archive/query
 * requesting topology, space group, cartesian positions, and chemical symbols.
 */
class NomadReader : public StructureReader {
public:
  /**
   * @brief Default constructor. No special authentication is required.
   */
  NomadReader();

  /**
   * @brief Reads structural data from the provided NOMAD entry ID.
   *
   * @param entryID A unique identifier in NOMAD, e.g. "zyXabc123..."
   */
  void read(const std::string &entryID) override;

  /**
   * @brief Returns the Structure, including cell, atoms, and metadata label.
   *
   * @return The last retrieved structure from NOMAD.
   */
  Structure getStructure() const override;

private:
  /// Internal storage of the retrieved structure.
  Structure m_structure;

  /**
   * @brief fetchAndParseNomad() forms the POST request to NOMAD
   *        and calls parseNomadJson() to interpret the response.
   *
   * @param entryID The NOMAD entry identifier.
   */
  void fetchAndParseNomad(const std::string &entryID);

  /**
   * @brief httpPost() sends a POST request with JSON body to @p url.
   *        Sets standard JSON headers.
   *
   * @param url The target endpoint.
   * @param jsonBody The JSON payload to post.
   * @return The response body as a std::string.
   */
  std::string httpPost(const std::string &url, const std::string &jsonBody);

  /**
   * @brief parseNomadJson() extracts cell parameters, space group, 
   *        cartesian positions, and species from the JSON response.
   *
   * @param jsonContent Raw JSON response string from NOMAD.
   */
  void parseNomadJson(const std::string &jsonContent);

  /**
   * @brief Builds a 3×3 lattice from edges (a,b,c) and angles (α,β,γ).
   *        Angles are in radians. Orthorhombic or monoclinic or general triclinic
   *        cells can be handled.
   *
   * @param a Edge length a.
   * @param b Edge length b.
   * @param c Edge length c.
   * @param alpha Angle α in radians.
   * @param beta Angle β in radians.
   * @param gamma Angle γ in radians.
   * @param cell Output 3×3 array for storing the resulting lattice vectors.
   */
  void makeCellMatrix(double a, double b, double c,
                      double alpha, double beta, double gamma,
                      double (&cell)[3][3]);
};

#endif // NOMAD_READER_H

