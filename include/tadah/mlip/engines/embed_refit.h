#pragma once
#include <tadah/models/eam/eam_refit.h>

namespace tadah {
namespace mlip {
namespace engines {
void validate_embed_refit(const core::Context &);
void prepare_embed_refit(const core::Context &, const models::EAMSetflIO &,
                         models::EAMRefitConfig &);
} // namespace engines
} // namespace mlip
} // namespace tadah
