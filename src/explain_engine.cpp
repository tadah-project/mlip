#include <tadah/mlip/engines/explain_engine.h>
#include <tadah/mlip/engines/engine_factory.h>
#include <tadah/core/context.h>
#include <tadah/core/blueprint_parser.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <boost/algorithm/string.hpp> // boost::split, iequals

namespace tadah {
namespace mlip {
namespace engines {

namespace {

bool is_space(char ch) {
  return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

std::string rstrip(const std::string &s) {
  std::size_t end = s.size();
  while (end > 0 && is_space(s[end - 1])) --end;
  return s.substr(0, end);
}

std::string trim(const std::string &s) {
  const std::string r = rstrip(s);
  std::size_t b = 0;
  while (b < r.size() && is_space(r[b])) ++b;
  return r.substr(b);
}

bool ends_with(const std::string &s, const std::string &tail) {
  return s.size() >= tail.size() &&
         s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// The leading whitespace of a line.
std::string margin_of(const std::string &line) {
  std::size_t n = 0;
  while (n < line.size() && (line[n] == ' ' || line[n] == '\t')) ++n;
  return line.substr(0, n);
}

// The lines of `text` without the whitespace prefix all its non-blank
// lines share, blank lines empty (Python's textwrap.dedent, then rstrip).
std::vector<std::string> dedented_lines(const std::string &text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  for (std::string line; std::getline(in, line);) lines.push_back(rstrip(line));
  bool first = true;
  std::string common;
  for (const auto &l : lines) {
    if (l.empty()) continue;
    const std::string m = margin_of(l);
    if (first) {
      common = m;
      first = false;
    } else {
      std::size_t k = 0;
      while (k < common.size() && k < m.size() && common[k] == m[k]) ++k;
      common.resize(k);
    }
  }
  for (auto &l : lines)
    if (!l.empty()) l.erase(0, common.size());
  return lines;
}

// Whether @p ch may be part of the name of an RST role (':ref:',
// ':hpo:key:').
bool role_name_char(char ch) {
  return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_' ||
         ch == '-' || ch == '+' || ch == '.';
}

// @p line with each inline role ':name:`content`' replaced by the text
// the documentation shows: the title of 'Title <target>', else the
// content without a leading '!' or '~' (the no-link and short forms). A
// role starts a word: a ':' at the start of the line or after a character
// that cannot end a name.
std::string render_roles(const std::string &line) {
  std::string out;
  std::size_t i = 0;
  while (i < line.size()) {
    if (line[i] == ':' && (i == 0 || !role_name_char(line[i - 1]))) {
      // ':' name (':' name)* ':' '`' content '`'
      std::size_t j = i + 1;
      bool named = false;
      while (j < line.size() && role_name_char(line[j])) {
        while (j < line.size() && role_name_char(line[j])) ++j;
        if (j < line.size() && line[j] == ':') {
          named = true;
          ++j;
          if (j < line.size() && line[j] == '`') break;
        } else {
          named = false;
          break;
        }
      }
      if (named && j < line.size() && line[j] == '`') {
        const std::size_t close = line.find('`', j + 1);
        if (close != std::string::npos && close > j + 1) {
          std::string content = line.substr(j + 1, close - j - 1);
          const std::size_t lt = content.rfind('<');
          if (!content.empty() && content.back() == '>' &&
              lt != std::string::npos && lt > 0)
            content = rstrip(content.substr(0, lt));
          else if (content[0] == '!' || content[0] == '~')
            content.erase(0, 1);
          out += content;
          i = close + 1;
          continue;
        }
      }
    }
    out += line[i];
    ++i;
  }
  return out;
}

// The key named `name` (lower case, the last part of its blueprint name)
// among a command's options and its groups' options; nullptr if none.
const core::BlueprintKey *find_option(const core::BlueprintTask &task,
                                      const std::string &name) {
  const auto named = [&](const core::BlueprintKey &key) {
    std::vector<std::string> parts;
    boost::split(parts, key.keyName, boost::is_any_of("."));
    return parts.back() == name;
  };
  for (const auto &key : task.blueprintKeys)
    if (named(key)) return &key;
  for (const auto &group : task.groups)
    for (const auto &key : group.blueprintKeys)
      if (named(key)) return &key;
  return nullptr;
}

// The configuration key named `name` (ALLOWED_KEYS, any letter case);
// nullptr if none.
const core::BlueprintKey *find_config_key(const std::string &name) {
  const auto &keys = core::BlueprintParser::getKeysMapRef();
  const auto it = keys.find(boost::to_upper_copy(name));
  return it == keys.end() ? nullptr : &it->second;
}

void print_key(const core::BlueprintKey &key) {
  std::cout << explain_key_text(key);
  std::cout.flush();
}

void print_text(const std::string &text) {
  for (const auto &line : explain_text_lines(text)) std::cout << line << "\n";
  std::cout.flush();
}

// The application's source of further texts, and what its names are
// (set_explain_hook).
struct RegisteredHook {
  ExplainHook hook;
  std::string what;
};
RegisteredHook &explain_hook() {
  static RegisteredHook hook;
  return hook;
}

}  // namespace

void set_explain_hook(ExplainHook hook, std::string what) {
  explain_hook() = RegisteredHook{std::move(hook), std::move(what)};
}

std::vector<std::string> explain_text_lines(const std::string &text) {
  std::vector<std::string> lines = dedented_lines(text);
  while (!lines.empty() && lines.front().empty()) lines.erase(lines.begin());
  while (!lines.empty() && lines.back().empty()) lines.pop_back();
  std::vector<std::string> out;
  bool in_literal = false;
  std::size_t literal_indent = 0;  // of the paragraph that opened one
  bool skip_blank = false;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    std::string ln = lines[i];
    if (skip_blank && ln.empty()) {
      skip_blank = false;
      continue;
    }
    skip_blank = false;
    const std::size_t indent = margin_of(ln).size();
    if (in_literal) {
      if (ln.empty() || indent > literal_indent) {
        out.push_back(ln);
        continue;
      }
      in_literal = false;
    }
    const bool closes_paragraph = i + 1 == lines.size() || lines[i + 1].empty();
    if (closes_paragraph && ends_with(ln, "::")) {
      in_literal = true;
      literal_indent = indent;
      if (trim(ln) == "::") {
        skip_blank = true;
        continue;
      }
      ln = ends_with(ln, " ::") ? rstrip(ln.substr(0, ln.size() - 3))
                                : ln.substr(0, ln.size() - 1);
    }
    out.push_back(render_roles(ln));
  }
  return out;
}

std::string explain_key_text(const core::BlueprintKey &key) {
  std::string out;
  for (const auto &line : explain_text_lines(
           key.longDescription.empty() ? key.description : key.longDescription))
    out += line + "\n";
  // A retired key: why, what replaces it and how a value of it converts
  // (a fit refuses it, with the same conversion).
  if (key.isRetired()) {
    out += "\nRETIRED: " + key.retired.reason + "\n";
    out += "REPLACED BY: " + key.retired.replacement + "\n";
    out += "CONVERSION: " + key.retired.conversion + "\n";
  }
  if (!key.examples.empty()) {
    out += "\nEXAMPLES: \n";
    for (const auto &example : key.examples) {
      std::istringstream in(example);
      for (std::string line; std::getline(in, line);) {
        line = trim(line);
        if (!line.empty()) out += "  " + line + "\n";
      }
    }
  }
  return out;
}

int ExplainEngine::run(mlip::Task &task) {
  const auto &taskContext = task.getContext();
  const std::string option = taskContext.get<std::string>("OPTION", 0);
  std::vector<std::string> tokens;
  boost::split(tokens, option, boost::is_any_of("."));
  if (tokens.size() > 3)
    throw std::runtime_error("tadah explain: '" + option +
                             "' is not <command>, <command>.<option> or "
                             "<command>.<subcommand>.<option>");
  const auto &taskMap = core::BlueprintParser::getTasksMapRef();
  // The command or subcommand the first n tokens name, in any letter case
  // (as an option's name), or taskMap.end(): the tasks map writes a command
  // in upper case and a subcommand in lower case ("DATA.merge"), and a
  // subcommand matched in one case only let a configuration key of its name
  // (MERGE) answer for data.MERGE.
  const auto command = [&](std::size_t n) {
    std::string name = tokens[0];
    for (std::size_t i = 1; i < n; ++i) name += "." + tokens[i];
    return std::find_if(taskMap.begin(), taskMap.end(), [&](const auto &t) {
      return boost::iequals(t.first, name);
    });
  };
  // The command or subcommand all the tokens name, if one.
  bool printed = false;
  const auto named = command(tokens.size());
  if (named != taskMap.end()) {
    print_text(named->second.longDescription);
    // A command this program was built without says so after its text.
    const std::string not_built = EngineFactory::notBuiltMessage(named->first);
    if (!not_built.empty()) std::cout << "\n" << not_built << ".\n";
    std::cout.flush();
    printed = true;
  }
  if (tokens.size() == 1) {
    // a configuration key, which no command need have as an option
    if (!printed) {
      if (const core::BlueprintKey *key = find_config_key(tokens[0])) {
        print_key(*key);
        printed = true;
      }
    }
    // a name the application explains (an optimiser of tadah hpo), after
    // a command's or key's text of the same name: a name with two meanings
    // prints both (RANDOM, the data key and TADAH:RANDOM)
    const RegisteredHook &registered = explain_hook();
    if (registered.hook) {
      std::ostringstream text;
      if (registered.hook(tokens[0], text)) {
        if (printed)
          std::cout << "\n" << tokens[0] << " also names " << registered.what
                    << ":\n";
        std::cout << text.str();
        std::cout.flush();
        return EXIT_SUCCESS;
      }
    }
    if (printed) return EXIT_SUCCESS;
    throw std::runtime_error("tadah explain: no command or configuration "
                             "key '" + tokens[0] + "'");
  }
  // An option of the command the tokens before the last name.
  const auto owner = command(tokens.size() - 1);
  if (owner == taskMap.end()) {
    if (printed) return EXIT_SUCCESS;
    if (command(1) == taskMap.end())
      throw std::runtime_error("tadah explain: no command '" + tokens[0] +
                               "'");
    throw std::runtime_error("tadah explain: " + tokens[0] +
                             " has no subcommand '" + tokens[1] + "'");
  }
  if (const core::BlueprintKey *key =
          find_option(owner->second, boost::to_lower_copy(tokens.back()))) {
    print_key(*key);
    return EXIT_SUCCESS;
  }
  if (printed) return EXIT_SUCCESS;
  // a configuration key that is no option of the command
  if (const core::BlueprintKey *key = find_config_key(tokens.back())) {
    print_key(*key);
    return EXIT_SUCCESS;
  }
  std::string where = tokens[0];
  for (std::size_t i = 1; i + 1 < tokens.size(); ++i) where += " " + tokens[i];
  // <command>.<name> could have named a subcommand of a command that has one
  const std::string sub_prefix = owner->first + ".";
  const bool has_subcommands =
      tokens.size() == 2 &&
      std::any_of(taskMap.begin(), taskMap.end(), [&](const auto &t) {
        return t.first.compare(0, sub_prefix.size(), sub_prefix) == 0;
      });
  throw std::runtime_error("tadah explain: " + where + " has no " +
                           (has_subcommands ? "subcommand or option"
                                            : "option") +
                           " '" + tokens.back() +
                           "', and it is no configuration key");
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
