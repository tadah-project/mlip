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
 *        other than 200:
 *        `<reader>: HTTP <status> for <id_kind> '<id>': <reason>`.
 *
 * The reason is @p rr's response, as the provider gives it: its own text
 * for a cURL error (e.g. `CODDataProvider: cURL error: Could not connect to
 * server`, status 500), else the server's body. A reason longer than
 * kMaxFetchReason characters is cut to its first kMaxFetchReason characters
 * followed by ` ... (<n> characters)` (a body may be a whole HTML page);
 * without a reason the message ends `'<id>'.`. Nothing else is added: no
 * request URL, no header, no API key.
 *
 * Used by AflowReader (`AUID`), CODReader (`COD ID`),
 * MaterialsProjectReader (`material`) and NomadReader (`entry`).
 *
 * @param reader  the reader's name, e.g. `CODReader`
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
