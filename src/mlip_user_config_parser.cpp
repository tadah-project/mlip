#include "tadah/mlip/mlip_user_config_parser.h"
#include "tadah/mlip/retired_options.h"
#include "tadah/core/core_user_config_parser.h"
#include "tadah/core/blueprint_parser.h"
#include "tadah/core/logger.h"
#include "tadah/core/utils.h"

#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace tadah {
namespace mlip {

namespace {

/* "TASK data balance": the TASK line of the blueprint task @p task
 * ("DATA.balance"). */
std::string task_line(const std::string &task) {
  std::string line = core::to_lower(task);
  std::replace(line.begin(), line.end(), '.', ' ');
  return "TASK " + line;
}

/* "line 4 of b.task" ("line 4" for lines of no file). */
std::string line_of(std::size_t line, const std::string &source) {
  return "line " + std::to_string(line) +
         (source.empty() ? std::string() : " of " + source);
}

/* True when the blueprint task @p task has the option @p key (any letter
 * case) and does not retire it (RESCALE of tadah analysis bfunc). */
bool takes_option(const std::string &task, const std::string &key) {
  const auto &tasks = core::BlueprintParser::getTasksMapRef();
  const auto it = tasks.find(task);
  if (it == tasks.end()) return false;
  const std::string wanted = core::to_lower(key);
  auto is = [&](const core::BlueprintKey &k) {
    const auto dot = k.keyName.rfind('.');
    const std::string name = core::to_lower(
        dot == std::string::npos ? k.keyName : k.keyName.substr(dot + 1));
    return name == wanted && !k.isRetired();
  };
  for (const core::BlueprintKey &k : it->second.blueprintKeys)
    if (is(k)) return true;
  for (const auto &group : it->second.groups)
    for (const core::BlueprintKey &k : group.blueprintKeys)
      if (is(k)) return true;
  return false;
}

}  // namespace

MLIPUserConfigParser::MLIPUserConfigParser(const tadah::core::IUserConfigParser* backend)
  : backendParser_(
      backend
        // if caller supplied a parser, wrap it in a shared_ptr with a no-op deleter
        ? std::shared_ptr<core::IUserConfigParser>(const_cast<core::IUserConfigParser*>(backend),
            [](core::IUserConfigParser*){})
        // otherwise build our own default parser with normal deleter
        : std::make_shared<core::CoreUserConfigParser>()
    )
{}
// MLIPUserConfigParser::~MLIPUserConfigParser() {
//   // Only delete if this object created the backend.
//   if (ownsBackend_) {
//     delete backendParser_;
//   }
//   backendParser_ = nullptr;
//   ownsBackend_    = false;
// }

/**
 * @brief Parses a user config from file, then merges sections if mergeFlag is true.
 */
std::vector<std::pair<std::string, core::Context>>
MLIPUserConfigParser::parse(const std::string& filename, bool mergeFlag) const
{
  std::ifstream infile(filename.c_str());
  if (!infile.is_open()) {
    throw std::runtime_error(
      "MLIPUserConfigParser::parse(file): Cannot open file: " + filename
    );
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(infile, line)) {
    lines.push_back(line);
  }

  return parseNamed(lines, mergeFlag, filename);
}

std::vector<std::pair<std::string, core::Context>>
MLIPUserConfigParser::parse(
    const std::string& filename, bool mergeFlag, const core::KeyLineRule& skip,
    std::vector<std::vector<core::SkippedKeyLine>>& skipped) const
{
  std::ifstream infile(filename.c_str());
  if (!infile.is_open()) {
    throw std::runtime_error(
      "MLIPUserConfigParser::parse(file): Cannot open file: " + filename
    );
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(infile, line)) {
    lines.push_back(line);
  }

  return parseNamed(lines, mergeFlag, filename, &skip, &skipped);
}

/**
 * @brief Parses a user config from lines, ensuring:
 *        - The first section is "GLOBAL" if it's non-empty (lines before any TASK).
 *        - Every TASK section, with or without key lines, must match an
 *          entry in the blueprint’s getTasksMapRef().
 *        - If mergeFlag is true, merges each section with the global context.
 */
std::vector<std::pair<std::string, core::Context>>
MLIPUserConfigParser::parse(const std::vector<std::string>& lines, bool mergeFlag) const
{
  return parseNamed(lines, mergeFlag, "");
}

std::vector<std::pair<std::string, core::Context>>
MLIPUserConfigParser::parseNamed(
    const std::vector<std::string>& lines, bool mergeFlag,
    const std::string& source, const core::KeyLineRule* skip,
    std::vector<std::vector<core::SkippedKeyLine>>* skipped) const
{
  if (skipped) skipped->clear();
  // Split into named sections: "GLOBAL" + each "TASK...." 
  const std::vector<Section> namedSections = splitSections(lines);

  // The final result: vec of <sectionName, Context>
  std::vector<std::pair<std::string, core::Context>> resultVec;

  // The blueprint tasks, for validation
  const auto &taskMap = tadah::core::BlueprintParser::getTasksMapRef();

  // A line refused for an option a task retires (RESCALE of tadah data
  // balance): the task and the option, or no option.
  struct Refusal {
    std::string task;
    const core::BlueprintKey *option = nullptr;
  };
  // The line of @p key in section @p s is refused when its task retires
  // the key (the lines of a TASK section, and of the files it includes
  // with CONFIG, are that task's), and, in the GLOBAL section, which sets
  // the keys of every task of the file, when a TASK section of the file
  // retires it, unless the option acted only with another (only_with:
  // tadah data balance --force acted only with --rescale, and a GLOBAL
  // FORCE line is for the other tasks).
  auto refused = [&namedSections](const Section &s, const std::string &key) {
    if (s.task) return Refusal{s.name, retired_task_option(s.name, key)};
    for (const Section &t : namedSections) {
      if (!t.task) continue;
      const core::BlueprintKey *option = retired_task_option(t.name, key);
      if (option && option->retired.only_with.empty())
        return Refusal{t.name, option};
    }
    return Refusal{};
  };

  // A key that sets a fit (row_weights.h: the error scales, ENORM, ENERGY
  // and the dataset priorities WDBFILE) changes nothing in a task that
  // fits nothing, TASK predict: its line is refused there, and in the
  // GLOBAL section when the file has a TASK predict section and no TASK
  // section that fits (train, hpo), for which the GLOBAL line would be
  // meant. The task it reaches, or "".
  const bool file_fits = std::any_of(
      namedSections.begin(), namedSections.end(), [](const Section &t) {
        return t.task && (t.name == "TRAIN" || t.name == "HPO");
      });
  auto fit_key_refused = [&namedSections, file_fits](
                             const Section &s,
                             const std::string &key) -> std::string {
    static const std::set<std::string> fit_keys = {
        "ESIGMA", "FSIGMA", "SSIGMA", "ENORM", "ENERGY", "WDBFILE"};
    if (!fit_keys.count(key)) return "";
    if (s.task) return s.name == "PREDICT" ? s.name : "";
    if (file_fits) return "";
    for (const Section &t : namedSections)
      if (t.task && t.name == "PREDICT") return t.name;
    return "";
  };

  // Parse each section. If it's not "GLOBAL", validate it using blueprint tasks.
  for (std::size_t i = 0; i < namedSections.size(); ++i) {
    const std::string &sectionName = namedSections[i].name;
    const auto &sectionLines       = namedSections[i].lines;

    // A TASK section must name a blueprint task, whatever its name ("TASK
    // global" too: only the lines before the first TASK line are GLOBAL).
    if (namedSections[i].task) {
      auto found = taskMap.find(sectionName);
      if (found == taskMap.end()) {
        // The user specified a "TASK..." that does not exist in the blueprint
        throw std::runtime_error(
          "MLIPUserConfigParser::parse: Unknown task section \"" + sectionName +
          "\". Must be defined in blueprint tasks."
        );
      }
      // A group of tasks (a blueprint task with tasks of its own, such as
      // DATA) has no engine: only its tasks run, as TaskFactory requires of a
      // command line. Refused here, before any task of the file runs.
      const std::string prefix = sectionName + ".";
      std::vector<std::string> subtasks;
      for (const auto &kv : taskMap) {
        if (kv.first.compare(0, prefix.size(), prefix) == 0) {
          std::string sub = kv.first.substr(prefix.size());
          std::replace(sub.begin(), sub.end(), '.', ' ');
          subtasks.push_back(sub);
        }
      }
      if (!subtasks.empty()) {
        std::sort(subtasks.begin(), subtasks.end());
        std::string line = core::to_lower(sectionName);
        std::replace(line.begin(), line.end(), '.', ' ');
        std::string list;
        for (const auto &sub : subtasks) list += (list.empty() ? "" : ", ") + sub;
        throw std::runtime_error(
          "MLIPUserConfigParser::parse: TASK section \"" + sectionName +
          "\" is a group of tasks, not a task: write 'TASK " + line +
          " <task>', <task> one of: " + list + ".");
      }
      // A task: the backend parser will parse the lines themselves.
    }

    // Parse the lines for this section to produce a Context. Its errors
    // name the line of the input (and the input's file). Under a rule, the
    // section is one file of it: the keys of its own lines decide. The
    // lines of an option a task retires (refused) are read under a rule
    // too, so that the line, in the section or in a file it includes with
    // CONFIG, is named below.
    const Section &section = namedSections[i];
    const core::KeyLineRule rule =
        [&](const std::set<std::string> &file_keys, const std::string &key) {
          return refused(section, key).option != nullptr ||
                 !fit_key_refused(section, key).empty() ||
                 (skip && (*skip)(file_keys, key));
        };
    std::vector<core::SkippedKeyLine> lines;
    core::Context ctx = backendParser_->parse(
        sectionLines, section.first_line, source, rule, lines);

    // A line that gives an option a task the section reaches retires (TASK
    // data balance with RESCALE, or a GLOBAL RESCALE with a TASK data
    // balance section) is refused, with why.
    for (const core::SkippedKeyLine &line : lines) {
      // A key that sets a fit, in a task that fits nothing.
      if (const std::string task = fit_key_refused(section, line.key);
          !task.empty()) {
        const std::string why =
            line.key + " sets the fit of tadah train and tadah hpo, and "
            "tadah predict fits nothing: it predicts with the potential as "
            "it was fitted.";
        throw std::runtime_error(
            line.key + " (" + line_of(line.line, line.source) + ") in " +
            (section.task
                 ? task_line(task) + ": " + why
                 : "the GLOBAL section reaches " + task_line(task) +
                       " (the GLOBAL section sets the keys of every task of "
                       "the file), and no TASK section of the file fits: " +
                       why));
      }
      const Refusal r = refused(section, line.key);
      if (!r.option) continue;
      std::string msg = line.key + " (" + line_of(line.line, line.source) +
                        ") in ";
      if (section.task) {
        msg += task_line(r.task) + ": " +
               retired_task_option_text(r.task, *r.option);
      } else {
        msg += "the GLOBAL section reaches " + task_line(r.task) +
               " (the GLOBAL section sets the keys of every task of the "
               "file): " + retired_task_option_text(r.task, *r.option);
        // The TASK sections of the file that take the key: the line may
        // be meant for them.
        std::string meant;
        for (const Section &t : namedSections) {
          if (!t.task || !takes_option(t.name, line.key)) continue;
          meant += (meant.empty() ? "" : " or ") + task_line(t.name) + " (" +
                   line_of(t.first_line - 1, source) + ")";
        }
        if (!meant.empty())
          msg += " A " + line.key + " line meant for " + meant +
                 " belongs in that TASK section.";
      }
      throw std::runtime_error(msg);
    }
    // The lines of the caller's rule.
    if (skip) skipped->push_back(std::move(lines));

    // Put it in the vec
    resultVec.push_back(std::make_pair(sectionName, ctx));
  }

  // If mergeFlag is true, we do the final merges.
  if (mergeFlag) {
    mergeAllSections(resultVec);
    // The lines of GLOBAL not stored belong to every TASK section too, as
    // its keys do.
    if (skipped && !resultVec.empty() && resultVec[0].first == "GLOBAL") {
      const std::vector<core::SkippedKeyLine> global = (*skipped)[0];
      for (std::size_t i = 1; i < skipped->size(); ++i)
        (*skipped)[i].insert((*skipped)[i].begin(), global.begin(),
                             global.end());
    }
  }

  return resultVec;
}

/**
 * @brief Splits lines into named sections:
 *        - The first section is "GLOBAL" if it has key lines before "TASK..."
 *        - Each "TASK..." line forms a new name like "TRAIN" or "DATA.convert",
 *          from the tokens before its in-line comment.
 *
 * Only returns "GLOBAL" if it has a key line. Every TASK section is
 * returned, with or without key lines: its name is checked by the caller,
 * and a keyless task runs on the GLOBAL keys.
 */
std::vector<MLIPUserConfigParser::Section>
MLIPUserConfigParser::splitSections(const std::vector<std::string>& lines) const
{
  std::vector<Section> sections;
  sections.reserve(8); // optional

  // The section being read. Its blank and comment lines are kept, so that
  // its lines keep the input's line numbers. A TASK section is always
  // returned; the GLOBAL section only when it has another line
  // (hasContent).
  Section current{"GLOBAL", false, 1, {}};
  bool hasContent = false;
  auto finish = [&]() {
    if (hasContent || current.task) {
      sections.push_back(std::move(current));
    }
  };

  for (std::size_t n = 0; n < lines.size(); ++n) {
    // Trim using core::trim(...)
    std::string line = core::trim(lines[n]);

    // Empty or comment lines: kept for the numbering only
    if (line.empty() || line[0] == '#') {
      current.lines.push_back(line);
      continue;
    }

    // Convert to uppercase for checking prefix:
    std::string uppercaseLine = core::to_upper(line);
    // Check if starts with "TASK"
    if (uppercaseLine.size() >= 4 && uppercaseLine.compare(0, 4, "TASK") == 0) {
      // A new section starts:
      // finalize the current one if it has lines
      finish();

      // Parse tokens from this line, without its in-line comment: the task
      // name is the text before the '#', as the value of a key line is
      // (CoreUserConfigParser drops the comment and the blanks before it).
      std::istringstream iss(core::trim(line.substr(0, line.find('#'))));
      std::vector<std::string> tokens;
      {
        std::string token;
        while (iss >> token) {
          tokens.push_back(token);
        }
      }
      // tokens[0] should be "TASK" (uppercaseLine), we need at least 1 more token
      if (tokens.size() < 2) {
        throw std::runtime_error(
          "MLIPUserConfigParser::splitSections: 'TASK' keyword not followed by any name:\n" + line
        );
      }

      std::string newSectionName;
      newSectionName = core::to_upper(tokens[1]);

      for (std::size_t i = 2; i < tokens.size(); ++i) {
        newSectionName += "." + core::to_lower(tokens[i]);
      }

      // Switch to the new section, which starts on the next line (line n
      // is line n + 1 of the input).
      current = Section{newSectionName, true, n + 2, {}};
      hasContent = false;
    }
    else {
      // This line belongs to the current section (GLOBAL or a previous TASK).
      current.lines.push_back(line);
      hasContent = true;
    }
  }

  // If there's anything left, push it
  finish();

  return sections;
}

/**
 * @brief Merges all sections in the vec with the "GLOBAL" if it exists and is non-empty.
 *
 * We skip merging the global context into itself.
 */
void MLIPUserConfigParser::mergeAllSections(
    std::vector<std::pair<std::string, core::Context>> &resultVec) const
{
  if (resultVec.empty() || resultVec[0].first != "GLOBAL") {
    // No global section present, nothing to merge
    return;
  }

  // Possibly empty global context
  const core::Context &globalCtx = resultVec[0].second;
  if (globalCtx.size()==0) {
    // If global is empty, merging does nothing
    return;
  }

  // Merge global into each other section
  for (auto &pair : resultVec) {
    // skip the global section
    if (pair.first == "GLOBAL") {
      continue;
    }
    // add keys from global to pair.second
    // pair.second takes precedence over global
    pair.second = core::Context::merge(globalCtx, pair.second);
  }
}

} // namespace mlip
} // namespace tadah
