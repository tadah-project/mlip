#include <tadah/mlip/data_providers/fetch_failure.h>

namespace tadah {
namespace mlip {

std::string failed_fetch_message(const std::string &reader,
                                 const std::string &id_kind,
                                 const std::string &id, const RawResponse &rr) {
  std::string msg = reader + ": HTTP " + std::to_string(rr.httpStatus) +
                    " for " + id_kind + " '" + id + "'";
  if (rr.response.empty()) return msg + ".";
  // The provider's reason (its own message, or the server's page, cut
  // short: it may be a whole HTML page).
  if (rr.response.size() > kMaxFetchReason)
    return msg + ": " + rr.response.substr(0, kMaxFetchReason) + " ... (" +
           std::to_string(rr.response.size()) + " characters)";
  return msg + ": " + rr.response;
}

}  // namespace mlip
}  // namespace tadah
