// `tadah explain`: what it prints for a command or one of its options, and
// its refusal of an option that names neither.
#include "catch2/catch.hpp"

#include <tadah/core/blueprint_parser.h>
#include <tadah/core/context.h>
#include <tadah/mlip/engines/explain_engine.h>
#include <tadah/mlip/task.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using tadah::core::Context;
using tadah::mlip::Task;
using tadah::mlip::engines::ExplainEngine;
using tadah::mlip::engines::explain_key_text;
using tadah::mlip::engines::explain_text_lines;

namespace {
// What `tadah explain <option>` writes to standard output.
std::string explain(const std::string &option) {
  Context c;
  c.add("OPTION", option);
  Task task("explain", c);
  std::ostringstream captured;
  std::streambuf *old = std::cout.rdbuf(captured.rdbuf());
  try {
    ExplainEngine().run(task);
  } catch (...) {
    std::cout.rdbuf(old);
    throw;
  }
  std::cout.rdbuf(old);
  return captured.str();
}

std::vector<std::string> lines_of(const std::string &text) {
  std::vector<std::string> out;
  std::istringstream in(text);
  for (std::string line; std::getline(in, line);) out.push_back(line);
  return out;
}

bool has_line(const std::vector<std::string> &lines, const std::string &l) {
  for (const auto &x : lines)
    if (x == l) return true;
  return false;
}

// The index of line `l` in `lines`, or lines.size().
std::size_t line_at(const std::vector<std::string> &lines,
                    const std::string &l) {
  for (std::size_t i = 0; i < lines.size(); ++i)
    if (lines[i] == l) return i;
  return lines.size();
}
}  // namespace

TEST_CASE("tadah explain: a long description is printed as text, a "
          "paragraph's closing '::' as ':' and its literal block indented",
          "[explain]") {
  /* Pins: `tadah explain refit.optimizer` prints the key's long description
   * (RST) as text: "Written inside a stage block, for example:" and "or,
   * with an INNER block in EXPLORE:" (each paragraph's closing '::' printed
   * as ':'), each followed by a blank line and its literal block with the
   * indentation it has in the text ("    REFINE", "      OPTIMIZER",
   * "        LIB CERES  ALGO LM  MAXEVAL 1000", ...), and no line ends with
   * "::"; `tadah explain refit.init` likewise ("Written as a block, one
   * placement per line, for example:", blank, "    INIT",
   * "      PHI XMIN PIN", "    ENDINIT").
   * Before the fix: the text was printed as written, so the lines were
   * "Written inside a stage block, for example::" and "or, with an INNER
   * block in EXPLORE::" (RST's literal-block marker), and the ':'-ending
   * lines are not found. */
  {
    const auto lines = lines_of(explain("refit.optimizer"));
    INFO(explain("refit.optimizer"));
    const std::size_t a =
        line_at(lines, "Written inside a stage block, for example:");
    REQUIRE(a + 7 < lines.size());
    CHECK(lines[a + 1].empty());
    CHECK(lines[a + 2] == "    REFINE");
    CHECK(lines[a + 3] == "      OPTIMIZER");
    CHECK(lines[a + 4] == "        LIB CERES  ALGO LM  MAXEVAL 1000");
    CHECK(lines[a + 5] == "      ENDOPTIMIZER");
    CHECK(lines[a + 6] == "    ENDREFINE");
    const std::size_t b =
        line_at(lines, "or, with an INNER block in EXPLORE:");
    REQUIRE(b + 2 < lines.size());
    CHECK(b > a);
    CHECK(lines[b + 1].empty());
    CHECK(lines[b + 2] == "    EXPLORE");
    CHECK(has_line(lines, "          LIB NLOPT  ALGO LN_SBPLX  MAXEVAL 20000"));
    for (const auto &l : lines) {
      INFO(l);
      CHECK_FALSE((l.size() >= 2 && l.compare(l.size() - 2, 2, "::") == 0));
    }
  }
  {
    const auto lines = lines_of(explain("refit.init"));
    INFO(explain("refit.init"));
    const std::size_t a = line_at(
        lines, "Written as a block, one placement per line, for example:");
    REQUIRE(a + 4 < lines.size());
    CHECK(lines[a + 1].empty());
    CHECK(lines[a + 2] == "    INIT");
    CHECK(lines[a + 3] == "      PHI XMIN PIN");
    CHECK(lines[a + 4] == "    ENDINIT");
  }
}

TEST_CASE("tadah explain: a command, a subcommand and an option print as "
          "before",
          "[explain]") {
  /* Pins: `tadah explain data.split` prints the subcommand's description
   * ("Split a dataset into parts." is in it), `tadah explain
   * data.split.even` the option's ("Split dataset into equal-size
   * partitions."), `tadah explain refit` the command's ("Convert a
   * tabulated EAM potential"), `tadah explain refit.add` the option's.
   * Before the fix: passes as well (a pin of the lookups that printed). */
  CHECK(explain("data.split").find("Split a dataset into parts") !=
        std::string::npos);
  CHECK(explain("data.split.even")
            .find("Split dataset into equal-size partitions.") !=
        std::string::npos);
  CHECK(explain("refit").find("Convert a tabulated EAM potential") !=
        std::string::npos);
  CHECK(explain("refit.add").find("Insert one basis function into the curve "
                                  "selected by --func") != std::string::npos);
}

TEST_CASE("tadah explain: an option that names no command or no option of "
          "its command is an error",
          "[explain]") {
  /* Pins: ExplainEngine::run throws, naming what it did not find, for an
   * option of an existing command that the command does not have and that
   * is no configuration key ("tadah explain: refit has no option
   * 'nosuchkey', and it is no configuration key", "tadah explain: data
   * split has no option 'nokey', and it is no configuration key", "tadah
   * explain: data has no subcommand or option 'nosuch', and it is no
   * configuration key"), a single name that is no command and no
   * configuration key ("tadah explain: no command or configuration key
   * 'bogus'") and more than three names ("tadah explain: 'a.b.c.d' is not
   * <command>, <command>.<option> or <command>.<subcommand>.<option>"); it
   * prints nothing. (Until `tadah explain` read the configuration keys, the
   * messages ended at the option, "... has no option 'nosuchkey'", and the
   * single name's was "tadah explain: no command 'bogus'".)
   * Before the fix: each returned EXIT_SUCCESS and printed nothing, so no
   * exception is thrown. (An unknown command with an option: the next test
   * case.) */
  for (const auto &[option, message] :
       std::vector<std::pair<std::string, std::string>>{
           {"refit.nosuchkey", "tadah explain: refit has no option "
                               "'nosuchkey', and it is no configuration key"},
           {"data.split.nokey", "tadah explain: data split has no option "
                                "'nokey', and it is no configuration key"},
           {"data.nosuch", "tadah explain: data has no subcommand or option "
                           "'nosuch', and it is no configuration key"},
           {"bogus", "tadah explain: no command or configuration key "
                     "'bogus'"},
           {"a.b.c.d", "tadah explain: 'a.b.c.d' is not <command>, "
                       "<command>.<option> or <command>.<subcommand>.<option>"}}) {
    INFO(option);
    std::string printed;
    CHECK_THROWS_WITH(printed = explain(option), message);
    CHECK(printed.empty());
  }
}

TEST_CASE("tadah explain: an unknown command or subcommand with an option "
          "is an error, not a crash",
          "[explain]") {
  /* Pins: `tadah explain bogus.key` and `bogus.sub.key` throw "tadah
   * explain: no command 'bogus'", and `data.nosub.key` "tadah explain: data
   * has no subcommand 'nosub'".
   * Before the fix: the lookup dereferenced the end of the command map
   * (taskMap.find of an absent command), and the test process died of
   * SIGSEGV. */
  CHECK_THROWS_WITH(explain("bogus.key"), "tadah explain: no command 'bogus'");
  CHECK_THROWS_WITH(explain("bogus.sub.key"),
                    "tadah explain: no command 'bogus'");
  CHECK_THROWS_WITH(explain("data.nosub.key"),
                    "tadah explain: data has no subcommand 'nosub'");
}

TEST_CASE("tadah explain: explain_text_lines renders a long description as "
          "the CLI page does",
          "[explain]") {
  /* Pins: explain_text_lines gives the lines generate_cli.py's
   * literal_text_lines gives the CLI page (Tadah.MLIP docs): the
   * indentation all lines share removed and the rest kept, the blank lines
   * around the text left out, a paragraph's closing 'text::' as 'text:',
   * 'text ::' as 'text', a paragraph of '::' alone and the blank line after
   * it left out, and '::' inside a literal block (the lines indented more
   * than the paragraph that opened it, and blank lines) or inside a line
   * kept: the lines below are literal_text_lines' for the same text.
   * SETFL's long description (an option of no command, which `tadah
   * explain SETFL` prints: the test case "a configuration key that is no
   * option of a command ..."): "A configuration with both descriptors and
   * one setfl file:", a blank line, then its literal block indented by 4
   * ("    INIT2B    true" ... "    SETFL     Ta.eam.alloy", the last line);
   * OPTIMIZER's: "Written inside a stage block, for example:" and "or, with
   * an INNER block in EXPLORE:" with their blocks; no line of either ends
   * with "::".
   * Before the fix: explain_text_lines does not exist (the test does not
   * compile); the long description was printed as written, with RST's
   * '::'. */
  const std::string text =
      "\n  Intro paragraph\n  ends here::\n\n      literal :: kept\n"
      "      line two::\n\n  Next ::\n\n      block\n\n  ::\n\n"
      "      lone block\n  Inline std::move stays.\n\n";
  CHECK(explain_text_lines(text) ==
        std::vector<std::string>{"Intro paragraph", "ends here:", "",
                                 "    literal :: kept", "    line two::", "",
                                 "Next", "", "    block", "", "    lone block",
                                 "Inline std::move stays."});
  const auto &keys = tadah::core::BlueprintParser::getKeysMapRef();
  {
    const auto lines = explain_text_lines(keys.at("SETFL").longDescription);
    REQUIRE_FALSE(lines.empty());
    CHECK(lines.front().rfind("A single-element setfl file", 0) == 0);
    const std::size_t a = line_at(
        lines, "A configuration with both descriptors and one setfl file:");
    REQUIRE(a + 2 < lines.size());
    CHECK(lines[a + 1].empty());
    CHECK(lines[a + 2] == "    INIT2B    true");
    CHECK(lines.back() == "    SETFL     Ta.eam.alloy");
    for (const auto &l : lines) {
      INFO(l);
      CHECK_FALSE((l.size() >= 2 && l.compare(l.size() - 2, 2, "::") == 0));
    }
  }
  {
    const auto lines =
        explain_text_lines(keys.at("OPTIMIZER").longDescription);
    REQUIRE_FALSE(lines.empty());
    CHECK(lines.front() == "Written inside a stage block, for example:");
    CHECK(lines[1].empty());
    CHECK(lines[2] == "    REFINE");
    CHECK(has_line(lines, "or, with an INNER block in EXPLORE:"));
    CHECK(lines.back() == "    ENDEXPLORE");
  }
}

TEST_CASE("tadah explain: each line of an example of several lines is "
          "indented",
          "[explain]") {
  /* Pins: explain_key_text, what `tadah explain <command>.<key>` prints,
   * ends with the key's examples, each line of each example on its own
   * line indented by two spaces: TYPE2B's (its long description first)
   * "\nEXAMPLES: \n  D2_LJ Kr Kr\n  D2_mJoin\n  D2_MIE 11 6 Ti Ti\n
   *   D2_Blip 6 6 Ti Nb Nb Nb\n"; a key without a long description prints
   * its description (refit.refit_compact_seed, as `tadah explain` printed
   * it), and `tadah explain refit.optimizer` prints explain_key_text of
   * the key.
   * Before the fix: explain_key_text does not exist (the test does not
   * compile); `tadah explain` printed "  " before an example only, so the
   * later lines of TYPE2B's D2_mJoin example had no indentation. */
  const auto &keys = tadah::core::BlueprintParser::getKeysMapRef();
  const std::string type2b = explain_key_text(keys.at("TYPE2B"));
  const std::string tail =
      "\nEXAMPLES: \n  D2_LJ Kr Kr\n  D2_mJoin\n  D2_MIE 11 6 Ti Ti\n"
      "  D2_Blip 6 6 Ti Nb Nb Nb\n";
  INFO(type2b);
  REQUIRE(type2b.size() > tail.size());
  CHECK(type2b.compare(type2b.size() - tail.size(), tail.size(), tail) == 0);
  CHECK(type2b.rfind(explain_text_lines(keys.at("TYPE2B").longDescription)
                         .front(),
                     0) == 0);
  const auto &refit = tadah::core::BlueprintParser::getTasksMapRef().at("REFIT");
  int found = 0;
  for (const auto &key : refit.blueprintKeys) {
    if (key.keyName.size() < 10 ||
        key.keyName.compare(key.keyName.size() - 10, 10, ".optimizer") != 0)
      continue;
    ++found;
    CHECK(explain("refit.optimizer") == explain_key_text(key));
  }
  CHECK(found == 1);
  const std::string seed = explain("refit.refit_compact_seed");
  CHECK(seed.rfind("`tadah refit` compact mode: RNG seed", 0) == 0);
}

TEST_CASE("tadah explain: a configuration key that is no option of a "
          "command is explained by its name, alone or after a command",
          "[explain]") {
  /* Pins: `tadah explain <KEY>`, a name that is no command, and `tadah
   * explain <command>[.<subcommand>].<KEY>`, a name that is no option of
   * the command, print the text of the configuration key (ALLOWED_KEYS, any
   * letter case), explain_key_text of it: SETFL (as `setfl`, `SETFL`,
   * `refit.setfl` and `data.split.setfl`), whose text rendering starts "A
   * single-element setfl file" and has "A configuration with both
   * descriptors and one setfl file:", a blank line and "    INIT2B    true";
   * TYPE2B, ending with its examples of several lines, each line indented
   * ("  D2_mJoin", "  D2_MIE 11 6 Ti Ti"); RHOFORM, EMBF0, NUMERIC,
   * RCUT2B, RCUTMB and AMPGRIDMB after refit. A command, a subcommand and an
   * option of the command come first: `refit.hpofileout` prints refit's own
   * HPOFILEOUT text ("Output file name for the generated `tadah hpo`
   * starter target file ..."), not the configuration key's ("Output file
   * for the `tadah hpo` starter target ..."), and `data.merge` the
   * subcommand, not the MERGE key.
   * Before the fix: only the commands and their options were read, so
   * SETFL threw "tadah explain: no command 'SETFL'", refit.setfl "tadah
   * explain: refit has no option 'setfl'", data.split.setfl "tadah
   * explain: data split has no option 'setfl'", and so on. */
  const auto &keys = tadah::core::BlueprintParser::getKeysMapRef();
  const std::string setfl = explain_key_text(keys.at("SETFL"));
  for (const char *name :
       {"setfl", "SETFL", "refit.setfl", "data.split.setfl", "refit.SETFL"}) {
    INFO(name);
    std::string out;
    CHECK_NOTHROW(out = explain(name));
    CHECK(out == setfl);
  }
  {
    const auto lines = lines_of(setfl);
    REQUIRE_FALSE(lines.empty());
    CHECK(lines.front().rfind("A single-element setfl file", 0) == 0);
    const std::size_t a = line_at(
        lines, "A configuration with both descriptors and one setfl file:");
    REQUIRE(a + 2 < lines.size());
    CHECK(lines[a + 1].empty());
    CHECK(lines[a + 2] == "    INIT2B    true");
  }
  {
    std::string out;
    CHECK_NOTHROW(out = explain("TYPE2B"));
    CHECK(out == explain_key_text(keys.at("TYPE2B")));
    const std::string tail =
        "\nEXAMPLES: \n  D2_LJ Kr Kr\n  D2_mJoin\n  D2_MIE 11 6 Ti Ti\n"
        "  D2_Blip 6 6 Ti Nb Nb Nb\n";
    CHECK(out.size() > tail.size());
    if (out.size() > tail.size())
      CHECK(out.compare(out.size() - tail.size(), tail.size(), tail) == 0);
  }
  for (const char *key :
       {"RHOFORM", "EMBF0", "NUMERIC", "RCUT2B", "RCUTMB", "AMPGRIDMB"}) {
    INFO(key);
    std::string lower = key;
    for (auto &ch : lower)
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    std::string out;
    CHECK_NOTHROW(out = explain("refit." + lower));
    CHECK(out == explain_key_text(keys.at(key)));
  }
  CHECK(explain("refit.hpofileout")
            .rfind("Output file name for the generated `tadah hpo` starter "
                   "target file", 0) == 0);
  CHECK(explain_key_text(keys.at("HPOFILEOUT"))
            .rfind("Output file for the `tadah hpo` starter target", 0) == 0);
  CHECK(explain("data.merge") ==
        [] {
          std::string out;
          for (const auto &l : explain_text_lines(
                   tadah::core::BlueprintParser::getTasksMapRef()
                       .at("DATA.merge")
                       .longDescription))
            out += l + "\n";
          return out;
        }());
}

TEST_CASE("tadah explain: every configuration key `tadah refit --help` "
          "names is explained after refit",
          "[explain]") {
  /* Pins: the help of tadah refit says `tadah explain refit.<key>` gives
   * the full description of a key it names: for every word of its long
   * description that is a configuration key (ALLOWED_KEYS: EAM, RHOFORM,
   * REFIT_MODE, EMBF0, NUMERIC, RCUT2B, ... in upper case), `tadah explain
   * refit.<word in lower case>` prints a text and throws nothing.
   * Before the fix: RHOFORM, EMBF0, NUMERIC, RCUT2B, RCUTMB and AMPGRIDMB,
   * which are no options of tadah refit, threw "tadah explain: refit has no
   * option 'rhoform'" (and so on). */
  const auto &keys = tadah::core::BlueprintParser::getKeysMapRef();
  const std::string help =
      tadah::core::BlueprintParser::getTasksMapRef().at("REFIT").longDescription;
  std::vector<std::string> named;
  std::string word;
  const auto flush = [&] {
    if (!word.empty() && keys.count(word) &&
        std::find(named.begin(), named.end(), word) == named.end())
      named.push_back(word);
    word.clear();
  };
  for (const char ch : help) {
    if (std::isupper(static_cast<unsigned char>(ch)) ||
        std::isdigit(static_cast<unsigned char>(ch)) || ch == '_')
      word += ch;
    else
      flush();
  }
  flush();
  INFO(named.size() << " keys named");
  CHECK(named.size() >= 30);
  for (const char *must : {"RHOFORM", "EMBF0", "NUMERIC", "RCUT2B", "RCUTMB",
                           "AMPGRIDMB", "REFIT_MODE", "EAM"})
    CHECK(std::find(named.begin(), named.end(), std::string(must)) !=
          named.end());
  for (const auto &key : named) {
    std::string lower = key;
    for (auto &ch : lower)
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    INFO("refit." << lower);
    std::string out;
    CHECK_NOTHROW(out = explain("refit." + lower));
    CHECK_FALSE(out.empty());
  }
}
