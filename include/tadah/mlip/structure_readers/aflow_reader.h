#ifndef AFLOW_READER_H
#define AFLOW_READER_H

#include <string>
#include <tadah/mlip/structure_readers/structure_reader.h>
#include <tadah/mlip/structure.h>

/**
 * @class AflowReader
 * @brief Implements the StructureReader interface to fetch structural data from the AFLOW REST API
 *        using only an AUID (AFLOW Unique Identifier).
 *
 * The AUID is a string like: "aflow:e9c6d914c4b8d9ca".
 *
 * Example usage:
 *   AflowReader reader;
 *   reader.read("aflow:e9c6d914c4b8d9ca");
 *   Structure st = reader.getStructure();
 *   // Now st.cell and st.atoms are populated if geometry info was found.
 *
 * The AFLOW search endpoint is constructed as:
 *   "https://aflowlib.duke.edu/search/?auid=<AUID>&format=json"
 */
class AflowReader : public StructureReader {
public:
  /**
   * @brief Default constructor. No special authentication is required.
   */
  AflowReader();

  /**
   * @brief Reads structural data by AUID.
   *
   * @param auid Should be of the form "aflow:XXXXXXXXXXXXXXXX".
   */
  void read(const std::string &auid) override;

  /**
   * @brief Returns the last retrieved structure from AFLOW.
   *
   * @return The structure, including cell and atoms.
   */
  Structure getStructure() const override;

private:
  /// Internal storage of the retrieved structure.
  Structure m_structure;

  /**
   * @brief fetchAndParseAflowByAuid() builds the AFLOW search URL using the given AUID
   *        and calls parseAflowJson().
   *
   * @param auid The AUID string: "aflow:XXXXXXXXXXXX...".
   */
  void fetchAndParseAflowByAuid(const std::string &auid);

  /**
   * @brief httpGet() sends a GET request to the provided URL
   *        and collects the response body into a std::string.
   *
   * @param url The target endpoint.
   * @return The full response body as a std::string.
   */
  std::string httpGet(const std::string &url);

  /**
   * @brief parseAflowJson() interprets JSON from AFLOW, looking for
   *        geometry, positions_cartesian, species, stoichiometry, etc.
   *
   * @param jsonContent Raw JSON response from AFLOW.
   */
  void parseAflowJson(const std::string &jsonContent);

  /**
   * @brief makeCellMatrix() constructs a 3×3 lattice from the provided
   *        cell parameters (a,b,c,α,β,γ in degrees).
   *
   * The standard crystallographic formula is used:
   *   a1 = (  a,   0,   0 )
   *   a2 = (  b cos γ,  b sin γ,           0 )
   *   a3 = (  c cos β,  c( cos α - cos β cos γ ) / sin γ,
   *           c sqrt(1 - cos^2 β - [ (cos α - cos β cos γ)/ sin γ]^2 ) )
   *
   * @param a Length a in Å.
   * @param b Length b in Å.
   * @param c Length c in Å.
   * @param alpha Angle α in degrees.
   * @param beta Angle β in degrees.
   * @param gamma Angle γ in degrees.
   * @param cell Output 3×3 array for storing the resulting lattice vectors.
   */
  void makeCellMatrix(double a, double b, double c,
                      double alpha, double beta, double gamma,
                      double (&cell)[3][3]);
};

#endif // AFLOW_READER_H

