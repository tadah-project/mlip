#include <tadah/mlip/joined_descriptors.h>

#include <memory>
#include <set>

#include <tadah/core/number_format.h>
#include <tadah/core/registry.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/d_mjoin.h>
#include <tadah/models/descriptors/dm/embed_spec.h>

namespace tadah {
namespace mlip {

bool is_joined_line(const tadah::core::Context &c,
                    const std::string &type_key) {
  if (!c.contains(type_key) || c.size(type_key) == 0) return false;
  if (type_key == "TYPEMB" && !c.blocks("DESCRIPTOR").empty()) return false;
  return c.get<std::string>(type_key, 0).find("mJoin") != std::string::npos;
}

namespace {

/* The descriptors of the contexts @p contexts, each a descriptor's own of
 * the @p type_key key in the order of the key, with the first entries of
 * each key that the descriptors split (joined_descriptors()). */
std::vector<JoinedDescriptor> with_first_entries(
    const std::vector<tadah::core::Context> &contexts,
    const std::string &type_key) {
  std::vector<JoinedDescriptor> out;
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
  // The numbers of the TYPE line before a descriptor's own line: an OPTIM
  // line numbers the values of a key by the numbers among them.
  std::size_t numbers_before = 0;
  for (std::size_t k = 0; k < contexts.size(); ++k) {
    JoinedDescriptor jd;
    jd.label = labels[k];
    jd.position = k;
    jd.context = contexts[k];
    for (const auto &key : split) {
      jd.first_entries[key] = before[key];
      if (contexts[k].contains(key)) before[key] += contexts[k].size(key);
    }
    jd.first_entries[type_key] = numbers_before;
    for (std::size_t i = 0; i < contexts[k].size(type_key); ++i) {
      double v = 0.0;
      if (tadah::core::parse_number(
              contexts[k].get<std::string>(type_key, i), v))
        ++numbers_before;
    }
    out.push_back(std::move(jd));
  }
  return out;
}

}  // namespace

std::vector<JoinedDescriptor> joined_descriptors(
    const tadah::core::Context &c, const std::string &type_key) {
  if (!is_joined_line(c, type_key)) return {};
  return with_first_entries(models::D_mJoin::parse_context(c, type_key),
                            type_key);
}

std::vector<JoinedDescriptor> block_line_descriptors(
    const tadah::core::Context &c) {
  if (c.blocks("DESCRIPTOR").empty()) return {};
  // The contexts the model builds the descriptors of TYPEMB from, in the
  // order of the key: a block's carries the block, a descriptor's of the
  // line its TYPEMB line and its values.
  std::vector<tadah::core::Context> contexts;
  for (auto &child : models::embed_contexts(c))
    if (child.blocks("DESCRIPTOR").empty())
      contexts.push_back(std::move(child));
  return with_first_entries(contexts, "TYPEMB");
}

}  // namespace mlip
}  // namespace tadah
