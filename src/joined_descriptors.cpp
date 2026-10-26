#include <tadah/mlip/joined_descriptors.h>

#include <memory>
#include <set>

#include <tadah/core/registry.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/d_mjoin.h>

namespace tadah {
namespace mlip {

bool is_joined_line(const tadah::core::Context &c,
                    const std::string &type_key) {
  if (!c.contains(type_key) || c.size(type_key) == 0) return false;
  if (type_key == "TYPEMB" && !c.blocks("DESCRIPTOR").empty()) return false;
  return c.get<std::string>(type_key, 0).find("mJoin") != std::string::npos;
}

std::vector<JoinedDescriptor> joined_descriptors(
    const tadah::core::Context &c, const std::string &type_key) {
  std::vector<JoinedDescriptor> out;
  if (!is_joined_line(c, type_key)) return out;
  const std::vector<tadah::core::Context> contexts =
      models::D_mJoin::parse_context(c, type_key);
  // The keys the line splits: one RCUT and one RCTYPE value per
  // descriptor, and the list keys its descriptors take.
  const std::string suffix = type_key.substr(type_key.size() - 2);
  std::set<std::string> split = {"RCUT" + suffix, "RCTYPE" + suffix};
  std::vector<std::string> labels;
  for (const auto &ctx : contexts) {
    labels.push_back(ctx.get<std::string>(type_key, 0));
    std::unique_ptr<models::D_Base> d(
        tadah::core::factory<models::D_Base>(labels.back()));
    split.insert(d->keys.begin(), d->keys.end());
    split.insert(d->data_keys.begin(), d->data_keys.end());
  }
  std::map<std::string, std::size_t> before;
  for (std::size_t k = 0; k < contexts.size(); ++k) {
    JoinedDescriptor jd;
    jd.label = labels[k];
    jd.position = k;
    jd.context = contexts[k];
    for (const auto &key : split) {
      jd.first_entries[key] = before[key];
      if (contexts[k].contains(key)) before[key] += contexts[k].size(key);
    }
    out.push_back(std::move(jd));
  }
  return out;
}

}  // namespace mlip
}  // namespace tadah
