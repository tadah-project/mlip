#ifndef TADAH_MLIP_DATA_PROVIDERS_FETCH_FAILURE_H
#define TADAH_MLIP_DATA_PROVIDERS_FETCH_FAILURE_H

#include <tadah/mlip/data_providers/imaterials_data_provider.h>

#include <cstddef>
#include <string>

namespace tadah {
namespace mlip {

/// The longest provider reason failed_fetch_message() quotes in full.
constexpr std::size_t kMaxFetchReason = 300;

/**
 * @brief The message of a structure reader whose provider answered a status
 *        other than 200.
 *
 * The message names the database, the status, what the ID is, the ID and
 * the provider's reason, e.g.
 *
 *     COD: HTTP 500 for COD ID '1534932': COD: cURL error: Could not connect to server
 *
 * The reason is @p rr's response, as the provider gives it: its own text
 * for a cURL error (status 500, as above), else the server's body. A
 * reason longer than kMaxFetchReason characters is cut to its first
 * kMaxFetchReason characters followed by " ... (4107 characters)" (the
 * length of the whole reason; a body may be a whole HTML page). Without a
 * reason the message ends after the ID with a full stop. Nothing else is
 * added: no request URL, no header, no API key.
 *
 * Used by AflowReader (`AUID`), CODReader (`COD ID`),
 * MaterialsProjectReader (`material`) and NomadReader (`entry`).
 *
 * @param reader  the database the reader reads, such as `COD`
 * @param id_kind what @p id is, e.g. `COD ID`
 * @param id      the ID the provider was asked for
 * @param rr      the provider's answer
 */
std::string failed_fetch_message(const std::string &reader,
                                 const std::string &id_kind,
                                 const std::string &id, const RawResponse &rr);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_DATA_PROVIDERS_FETCH_FAILURE_H
