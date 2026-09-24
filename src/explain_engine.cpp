#include <tadah/mlip/engines/explain_engine.h>
#include <tadah/core/context.h>
#include <tadah/core/blueprint_parser.h>
#include <iostream>
#include <boost/algorithm/string.hpp> // for boost::split

namespace tadah {
namespace mlip {
namespace engines {

int ExplainEngine::run(mlip::Task &task) {
  const auto &taskContext = task.getContext();
  std::string option = taskContext.get<std::string>("OPTION", 0);
  std::vector<std::string> tokens;
  boost::split(tokens, option, boost::is_any_of("."));
  boost::to_upper(tokens[0]);
  std::string option2 =  boost::join(tokens, ".");
  const auto &taskMap = core::BlueprintParser::getTasksMapRef();

  try {
    // deal with plain commands and subcommands (without options)
    std::cout << taskMap.at(option2).longDescription << std::endl;
  } catch (const std::out_of_range &e) {}

  try {
    if (tokens.size() == 2) {
      auto it = taskMap.find(tokens[0]);
      boost::to_lower(tokens[1]);
      for (const auto &key : it->second.blueprintKeys) {
        std::vector<std::string> tokens2;
        boost::split(tokens2, key.keyName, boost::is_any_of("."));
        if (tokens2.back() == tokens[1]) {
          std::cout << key.longDescription << std::endl;
          if (!key.examples.empty()) {
            std::cout << "\nEXAMPLES: " << std::endl;
            for (const auto &example : key.examples) {
              std::cout << "  " << example << std::endl;
            }
          }
          return EXIT_SUCCESS;
        }
      }
      for (const auto &group : it->second.groups) {
        std::vector<std::string> tokens2;
        boost::split(tokens2, group.groupName, boost::is_any_of("."));
        for (const auto &key : group.blueprintKeys) {
          std::vector<std::string> tokens3;
          boost::split(tokens3, key.keyName, boost::is_any_of("."));
          if (tokens3.back() == tokens[1]) {
            std::cout << key.longDescription << std::endl;
            if (!key.examples.empty()) {
              std::cout << "\nEXAMPLES: " << std::endl;
              for (const auto &example : key.examples) {
                std::cout << "  " << example << std::endl;
              }
            }
            return EXIT_SUCCESS;
          }
        }
      }
    } else if (tokens.size() == 3) {
      auto it = taskMap.find(tokens[0]+"." + tokens[1]);
      for (const auto &key : it->second.blueprintKeys) {
        std::vector<std::string> tokens2;
        boost::split(tokens2, key.keyName, boost::is_any_of("."));
        if (tokens2.back() == tokens[2]) {
          std::cout << key.longDescription << std::endl;
          if (!key.examples.empty()) {
            std::cout << "\nEXAMPLES: " << std::endl;
            for (const auto &example : key.examples) {
              std::cout << "  " << example << std::endl;
            }
          }
          return EXIT_SUCCESS;
        }
      }
      for (const auto &group : it->second.groups) {
        std::vector<std::string> tokens2;
        boost::split(tokens2, group.groupName, boost::is_any_of("."));
        for (const auto &key : group.blueprintKeys) {
          std::vector<std::string> tokens3;
          boost::split(tokens3, key.keyName, boost::is_any_of("."));
          if (tokens3.back() == tokens[2]) {
            std::cout << key.longDescription << std::endl;
            if (!key.examples.empty()) {
              std::cout << "\nEXAMPLES: " << std::endl;
              for (const auto &example : key.examples) {
                std::cout << "  " << example << std::endl;
              }
            }
            return EXIT_SUCCESS;
          }
        }
      }
    }
  } catch (const std::out_of_range &e) {}

  return EXIT_SUCCESS;
}
void ExplainEngine::validate(const tadah::mlip::Task &task) {
  const auto &context = task.getContext();
  if (!context.contains("OPTION")) {
    throw std::runtime_error("ExplainEngine requires OPTION parameter to be set");
  }
  if (context.size("OPTION") != 1) {
    throw std::runtime_error("ExplainEngine requires exactly one OPTION parameter in a dot-separated format. For example: data.sample.index");
  }
}

} // namespace engines
} // namespace mlip
} // namespace tadah

