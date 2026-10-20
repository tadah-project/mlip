// `tadah explain`: what it prints for a command or one of its options, and
// its refusal of an option that names neither.
#include "catch2/catch.hpp"

#include <tadah/core/blueprint_parser.h>
#include <tadah/core/context.h>
#include <tadah/mlip/engines/engine_factory.h>
#include <tadah/mlip/engines/explain_engine.h>
#include <tadah/mlip/task.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
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
   * ("Splits a dataset into parts." is in it), `tadah explain
   * data.split.even` the option's ("Splits the dataset into equal-size
   * partitions."), `tadah explain refit` the command's ("Converts a
   * tabulated EAM potential"), `tadah explain refit.add` the option's.
   * Before the fix: passes as well (a pin of the lookups that printed). */
  CHECK(explain("data.split").find("Splits a dataset into parts") !=
        std::string::npos);
  CHECK(explain("data.split.even")
            .find("Splits the dataset into equal-size partitions.") !=
        std::string::npos);
  CHECK(explain("refit").find("Converts a tabulated EAM potential") !=
        std::string::npos);
  CHECK(explain("refit.add").find("Inserts one basis function into the curve "
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
   * OPTIMIZER's: its text, then "Written inside a stage block, for
   * example:" and "or, with an INNER block in EXPLORE:" with their blocks;
   * no line of either ends with "::". (Until OPTIMIZER's long description
   * held its text, "Written inside a stage block, for example:" was its
   * first line.)
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
    const std::size_t a =
        line_at(lines, "Written inside a stage block, for example:");
    REQUIRE(a + 2 < lines.size());
    CHECK(a > 0);
    CHECK(lines[a + 1].empty());
    CHECK(lines[a + 2] == "    REFINE");
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
   * the full description of a configuration key it names. Every upper-case
   * word of its long description that is a configuration key (ALLOWED_KEYS:
   * EAM, RHOFORM, REFIT_MODE, EMBF0, NUMERIC, RCUT2B, ... in upper case;
   * at least 30) is explained after refit as the key the help means:
   * `tadah explain refit.<word in lower case>` prints refit's own option of
   * that name (explain_key_text of it), and a word that is no option of
   * tadah refit is one of the six the help names as such, RHOFORM, EMBF0,
   * NUMERIC, RCUT2B, RCUTMB and AMPGRIDMB, printed as the configuration key.
   * Every upper-case word with an underscore is a configuration key, the
   * prefix of some written with '*' (REFIT_COMPACT_*, REFIT_CURVE_*,
   * REFIT_*), or one of the names the help gives that are no keys (DM_REAM,
   * a descriptor; TADAH_ENABLE_NFP and TADAH_ENABLE_CERES, build options).
   * The keys inside a stage block are no configuration keys: the help names
   * `tadah explain refit.optimizer` for ALGO and LIB (the next test case
   * checks what it prints).
   * Before the fix: the help wrote "OPT_EVERY, INSERT_TOPK and MERGE with a
   * knot basis only" for REFIT_COMPACT_OPT_EVERY, REFIT_COMPACT_INSERT_TOPK
   * and REFIT_COMPACT_MERGE: MERGE, no option of tadah refit, was explained
   * after refit as the MERGE key of data dedup ("Merge deduplication
   * results into one file."), OPT_EVERY and INSERT_TOPK are no keys, and
   * the help did not name `tadah explain refit.optimizer`. (Before `tadah
   * explain` read the configuration keys, RHOFORM, EMBF0, NUMERIC, RCUT2B,
   * RCUTMB and AMPGRIDMB threw "tadah explain: refit has no option
   * 'rhoform'" and so on.) */
  const auto &keys = tadah::core::BlueprintParser::getKeysMapRef();
  const auto &refit =
      tadah::core::BlueprintParser::getTasksMapRef().at("REFIT");
  const std::string &help = refit.longDescription;
  // refit's own option whose name (the last part of its blueprint name) is
  // `lower`, or nullptr
  const auto option_of_refit =
      [&](const std::string &lower) -> const tadah::core::BlueprintKey * {
    const auto named = [&](const tadah::core::BlueprintKey &key) {
      const std::size_t dot = key.keyName.rfind('.');
      return key.keyName.substr(dot == std::string::npos ? 0 : dot + 1) ==
             lower;
    };
    for (const auto &key : refit.blueprintKeys)
      if (named(key)) return &key;
    for (const auto &group : refit.groups)
      for (const auto &key : group.blueprintKeys)
        if (named(key)) return &key;
    return nullptr;
  };
  // the upper-case words of the help (runs of letters, digits and '_' with
  // a letter and no lower-case one), each once, with the character after it
  std::vector<std::pair<std::string, char>> words;
  for (std::size_t i = 0; i < help.size();) {
    const auto word_char = [](char ch) {
      return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
    };
    if (!word_char(help[i])) {
      ++i;
      continue;
    }
    std::size_t j = i;
    while (j < help.size() && word_char(help[j])) ++j;
    const std::string w = help.substr(i, j - i);
    const bool upper =
        std::any_of(w.begin(), w.end(),
                    [](char ch) {
                      return std::isupper(static_cast<unsigned char>(ch));
                    }) &&
        std::none_of(w.begin(), w.end(), [](char ch) {
          return std::islower(static_cast<unsigned char>(ch));
        });
    const char next = j < help.size() ? help[j] : '\0';
    if (upper && std::find(words.begin(), words.end(),
                           std::make_pair(w, next)) == words.end())
      words.emplace_back(w, next);
    i = j;
  }
  const std::vector<std::string> not_refit_options = {
      "RHOFORM", "EMBF0", "NUMERIC", "RCUT2B", "RCUTMB", "AMPGRIDMB"};
  const std::vector<std::string> no_keys = {"DM_REAM", "TADAH_ENABLE_NFP",
                                            "TADAH_ENABLE_CERES"};
  std::vector<std::string> named;
  for (const auto &[w, next] : words) {
    INFO("the help names " << w);
    if (keys.count(w)) {
      if (std::find(named.begin(), named.end(), w) != named.end()) continue;
      named.push_back(w);
      std::string lower = w;
      for (auto &ch : lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      std::string out;
      CHECK_NOTHROW(out = explain("refit." + lower));
      if (const tadah::core::BlueprintKey *own = option_of_refit(lower)) {
        CHECK(out == explain_key_text(*own));
      } else {
        CHECK(std::find(not_refit_options.begin(), not_refit_options.end(),
                        w) != not_refit_options.end());
        CHECK(out == explain_key_text(keys.at(w)));
      }
    } else if (w.find('_') != std::string::npos) {
      if (next == '*') {
        CHECK(w.back() == '_');
        CHECK(std::any_of(keys.begin(), keys.end(), [&](const auto &k) {
          return k.first.compare(0, w.size(), w) == 0;
        }));
      } else {
        CHECK(std::find(no_keys.begin(), no_keys.end(), w) != no_keys.end());
      }
    }
  }
  INFO(named.size() << " keys named");
  CHECK(named.size() >= 30);
  for (const char *must : {"RHOFORM", "EMBF0", "NUMERIC", "RCUT2B", "RCUTMB",
                           "AMPGRIDMB", "REFIT_MODE", "EAM",
                           "REFIT_COMPACT_MERGE", "REFIT_COMPACT_OPT_EVERY",
                           "REFIT_COMPACT_INSERT_TOPK", "OPTIMIZER"})
    CHECK(std::find(named.begin(), named.end(), std::string(must)) !=
          named.end());
  CHECK(help.find("`tadah explain refit.optimizer`") != std::string::npos);
}

namespace {
// `text` with every run of whitespace one space, none at either end.
std::string squeezed(const std::string &text) {
  std::string out;
  bool space = false;
  for (const char ch : text) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      space = !out.empty();
      continue;
    }
    if (space) out += ' ';
    space = false;
    out += ch;
  }
  return out;
}
}  // namespace

TEST_CASE("tadah explain: a stage block's text describes the keys of the "
          "block before its layout",
          "[explain]") {
  /* Pins: `tadah explain refit.<block>` for the stage blocks OPTIMIZER,
   * INIT, EXPLORE, REFINE and FUSE prints the block's text and then its
   * layout: with whitespace squeezed the output contains the key's
   * description (the summary the Configuration File page shows above the
   * long description) and what the block takes: OPTIMIZER "`ALGO <name>`
   * (required", "`LIB <name>` (required", "MAXEVAL, an integer from 1 to
   * 2147483647", "A stopping criterion the block's ALGO does not read is an
   * error", "FTOL_REL, FTOL_ABS, XTOL_REL and GTOL are finite numbers >=
   * 0"; INIT "`<PHI|RHO|F> <MANUAL <mu> [<eta>] | XMIN | XMAX | YMIN | YMAX
   * | SMAX | CMAX> [PIN]`" and "PIN freezes that position"; EXPLORE "Takes
   * `MAX_ADD <n>` (max basis added per function)"; REFINE "per-rung
   * high-precision local polish"; FUSE "Under REFIT_OBJECTIVE energy a joint
   * refinement of all three functions"; and the line that opens the layout
   * ("Written inside a stage block, for example:", "Written as a block,
   * ...") comes after them. `tadah refit --help` sends ALGO and LIB here.
   * Before the fix: the long description of each block held its layout
   * only, which is what `tadah explain` printed, and the block's keys were
   * in the description, which `tadah explain` does not print when a long
   * description exists: the description and the key texts are not found,
   * and the layout's line is the first. */
  const auto &keys = tadah::core::BlueprintParser::getKeysMapRef();
  const std::vector<std::pair<std::string, std::vector<std::string>>> blocks =
      {{"OPTIMIZER",
        {"`ALGO <name>` (required", "`LIB <name>` (required",
         "MAXEVAL, an integer from 1 to 2147483647",
         "A stopping criterion the block's ALGO does not read is an error",
         "FTOL_REL, FTOL_ABS, XTOL_REL and GTOL are finite numbers >= 0",
         "Written inside a stage block, for example:"}},
       {"INIT",
        {"`<PHI|RHO|F> <MANUAL <mu> [<eta>] | XMIN | XMAX | YMIN | YMAX | "
         "SMAX | CMAX> [PIN]`",
         "PIN freezes that position",
         "Written as a block, one placement per line, for example:"}},
       {"EXPLORE",
        {"Takes `MAX_ADD <n>` (max basis added per function)",
         "Written as a block, with an OPTIMIZER block inside"}},
       {"REFINE",
        {"per-rung high-precision local polish",
         "Written as a block, with an OPTIMIZER block inside"}},
       {"FUSE",
        {"Under REFIT_OBJECTIVE energy a joint refinement of all three "
         "functions",
         "Written as a block, with an OPTIMIZER block inside"}}};
  for (const auto &[key, texts] : blocks) {
    std::string lower = key;
    for (auto &ch : lower)
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    std::string out;
    CHECK_NOTHROW(out = explain("refit." + lower));
    const std::string flat = squeezed(out);
    INFO(key << ":\n" << out);
    const std::string description = squeezed(keys.at(key).description);
    REQUIRE_FALSE(description.empty());
    const std::size_t d = flat.find(description);
    CHECK(d != std::string::npos);
    std::size_t last = d == std::string::npos ? 0 : d;
    for (const auto &text : texts) {
      INFO(text);
      const std::size_t at = flat.find(text);
      CHECK(at != std::string::npos);
      if (at == std::string::npos) continue;
      CHECK(at >= last);  // in this order, the layout's line last
      last = at;
    }
    const auto ls = lines_of(out);
    REQUIRE_FALSE(ls.empty());
    CHECK(ls.front().rfind("Written ", 0) != 0);
  }
}

TEST_CASE("tadah explain: a subcommand is found in any letter case, before "
          "a configuration key",
          "[explain]") {
  /* Pins: a command, a subcommand and an option are found in any letter
   * case, the subcommand before a configuration key of the same name:
   * `tadah explain data.MERGE`, `data.Merge` and `DATA.MERGE` print what
   * `data.merge` prints, the merge subcommand ("Merges multiple dataset files
   * into a single file."), not the MERGE configuration key ("Merges
   * deduplication results into one file."); `data.SPLIT`,
   * `DATA.SPLIT.EVEN` and `ANALYSIS.DATASET_STATS` print what `data.split`,
   * `data.split.even` and `analysis.dataset_stats` print.
   * Before the fix: a subcommand was matched in lower case only, so
   * data.MERGE, data.Merge and DATA.MERGE printed the MERGE key, data.SPLIT
   * threw "tadah explain: data has no subcommand or option 'SPLIT', and it
   * is no configuration key", DATA.SPLIT.EVEN "tadah explain: DATA has no
   * subcommand 'SPLIT'" and ANALYSIS.DATASET_STATS "tadah explain: ANALYSIS
   * has no subcommand or option 'DATASET_STATS', and it is no configuration
   * key". */
  const std::string merge = explain("data.merge");
  REQUIRE(merge.find("Merges multiple dataset files into a single file") !=
          std::string::npos);
  for (const char *name : {"data.MERGE", "data.Merge", "DATA.MERGE"}) {
    INFO(name);
    std::string out;
    CHECK_NOTHROW(out = explain(name));
    CHECK(out == merge);
  }
  for (const auto &[name, lower] :
       std::vector<std::pair<std::string, std::string>>{
           {"data.SPLIT", "data.split"},
           {"DATA.SPLIT.EVEN", "data.split.even"},
           {"ANALYSIS.DATASET_STATS", "analysis.dataset_stats"}}) {
    INFO(name);
    const std::string expected = explain(lower);
    REQUIRE_FALSE(expected.empty());
    std::string out;
    CHECK_NOTHROW(out = explain(name));
    CHECK(out == expected);
  }
}

TEST_CASE("tadah explain: a name that is no command and no configuration key "
          "is asked of the application's explain hook",
          "[explain][explain_hook]") {
  /* Pins: an application sets one source of texts beyond MLIP's commands
   * and configuration keys (set_explain_hook, with what its names are;
   * tadah sets HPO's optimiser help, which MLIP does not depend on).
   * `tadah explain <NAME>` asks it for a single name, as the user wrote
   * it: for one that is no command and no configuration key what it writes
   * is the text and the command succeeds ("WIDGET" and "widget", which this
   * hook reads in any letter case); a name it does not know (false) is the
   * error it was ("tadah explain: no command or configuration key
   * 'bogus'"); an error it throws is the command's error; a command (train)
   * or a configuration key (SETFL) that is not one of its own prints its
   * text alone -- the hook is asked and adds nothing --; a dotted name is
   * not asked; set_explain_hook({}, "") removes it.
   * Before the fix: none compiles (set_explain_hook takes one argument, the
   * hook), and the hook was not asked for train and SETFL (a command or a
   * key answered first and alone, which hid an optimiser of the same name);
   * before the hook, no text from outside MLIP could be reached. */
  using tadah::mlip::engines::set_explain_hook;
  const std::string train_alone = explain("train");
  const std::string setfl_alone = explain("SETFL");
  std::vector<std::string> asked;
  set_explain_hook(
      [&](const std::string &name, std::ostream &out) {
        asked.push_back(name);
        std::string upper = name;
        for (auto &ch : upper)
          ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        if (upper == "BROKEN") throw std::runtime_error("tadah explain: broken");
        if (upper != "WIDGET") return false;
        out << "widget text\n";
        return true;
      },
      "a widget of the test");
  CHECK(explain("WIDGET") == "widget text\n");
  CHECK(explain("widget") == "widget text\n");
  CHECK_THROWS_WITH(explain("bogus"),
                    "tadah explain: no command or configuration key 'bogus'");
  CHECK_THROWS_WITH(explain("BROKEN"), "tadah explain: broken");
  CHECK(asked == std::vector<std::string>{"WIDGET", "widget", "bogus",
                                          "BROKEN"});
  asked.clear();
  CHECK(explain("train") == train_alone);
  CHECK(explain("SETFL") == setfl_alone);
  CHECK(asked == std::vector<std::string>{"train", "SETFL"});
  asked.clear();
  CHECK_THROWS(explain("refit.widget"));
  CHECK(asked.empty());
  set_explain_hook({}, "");
  CHECK_THROWS_WITH(explain("WIDGET"),
                    "tadah explain: no command or configuration key 'WIDGET'");
}

TEST_CASE("tadah explain: a name that is a command or a configuration key "
          "and one of the explain hook's prints both texts",
          "[explain][explain_hook]") {
  /* Pins: a single name with two meanings -- a configuration key (SETFL)
   * or a command (train) that the application's hook also explains, as
   * RANDOM is the data key and the ALGO of TADAH:RANDOM -- prints the
   * command's or key's text, then a blank line, "<NAME> also names <what
   * the hook's names are>:" and the hook's text: "SETFL also names a widget
   * of the test:" and "widget SETFL"; "train also names a widget of the
   * test:" and "widget train", the name as the user wrote it.
   * Before the fix: does not compile (set_explain_hook takes one argument);
   * the key's or command's text alone was printed and the hook was not
   * asked, so `tadah explain RANDOM` printed the data key alone, although
   * the optimiser table says `tadah explain <ALGO>` describes TADAH:RANDOM. */
  using tadah::mlip::engines::set_explain_hook;
  const std::string setfl_alone = explain("SETFL");
  const std::string train_alone = explain("train");
  set_explain_hook(
      [](const std::string &name, std::ostream &out) {
        std::string upper = name;
        for (auto &ch : upper)
          ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        if (upper != "SETFL" && upper != "TRAIN") return false;
        out << "widget " << name << "\n";
        return true;
      },
      "a widget of the test");
  CHECK(explain("SETFL") ==
        setfl_alone + "\nSETFL also names a widget of the test:\nwidget SETFL\n");
  CHECK(explain("train") ==
        train_alone + "\ntrain also names a widget of the test:\nwidget train\n");
  set_explain_hook({}, "");
  CHECK(explain("SETFL") == setfl_alone);
}

TEST_CASE("tadah explain: the text of a command this program was built "
          "without ends with why",
          "[explain]") {
  /* Pins: after EngineFactory::registerNotBuilt("HPO", "tadah hpo needs a
   * part this program was built without") (test_mlip has no HPO engine in
   * any build), `tadah explain hpo` prints the command's long description
   * (its first line, "Refines a model's architecture and hyperparameters
   * with Tadah!'s nested", first), then a blank line and the message with
   * a full stop as its last two lines; the text of `tadah explain
   * data.split`, whose engine is never registered as not built, and of
   * `tadah explain hpo.verbose`, an option of the command, do not have the
   * message.
   * Before the fix: does not compile (EngineFactory has no
   * registerNotBuilt). */
  const std::string msg = "tadah hpo needs a part this program was built without";
  tadah::mlip::engines::EngineFactory::registerNotBuilt("HPO", msg);
  const std::string out = explain("hpo");
  INFO(out);
  const auto lines = lines_of(out);
  REQUIRE(lines.size() >= 3);
  CHECK(lines.front() ==
        "Refines a model's architecture and hyperparameters with Tadah!'s nested");
  CHECK(lines[lines.size() - 2].empty());
  CHECK(lines.back() == msg + ".");
  CHECK(explain("data.split").find(msg) == std::string::npos);
  CHECK(explain("hpo.verbose").find(msg) == std::string::npos);
}

TEST_CASE("tadah explain: a retired key's text ends with why it is retired, "
          "what replaces it and its conversion",
          "[explain][retired]") {
  /* Pins: `tadah explain EWEIGHT` and `tadah explain WDBFILE_AUTO` print the
   * key's description, then a blank line and the lines "RETIRED: <reason>",
   * "REPLACED BY: <replacement>" and "CONVERSION: <conversion>" of the
   * blueprint's retired table, in this order (EWEIGHT's replacement names
   * ESIGMA and ENERGY false, its conversion 1000/(N sqrt(EWEIGHT)); that of
   * WDBFILE_AUTO names WDBFILE and n_d^-a); a retired key has no
   * EXAMPLES; the text of ESIGMA, a key that is not retired, has no
   * RETIRED line.
   * Before the fix: the text of a retired key was its description alone. */
  const std::string out = explain("EWEIGHT");
  INFO(out);
  const auto lines = lines_of(out);
  const std::size_t r = std::find_if(lines.begin(), lines.end(),
                                     [](const std::string &l) {
                                       return l.rfind("RETIRED: ", 0) == 0;
                                     }) -
                        lines.begin();
  REQUIRE(r < lines.size());
  REQUIRE(r >= 2);
  CHECK(lines[r - 1].empty());
  CHECK_THAT(lines[r], Catch::Matchers::Contains("total energy"));
  REQUIRE(r + 2 < lines.size());
  CHECK_THAT(lines[r + 1], Catch::Matchers::StartsWith("REPLACED BY: ESIGMA") &&
                               Catch::Matchers::Contains("ENERGY false"));
  CHECK_THAT(lines[r + 2], Catch::Matchers::StartsWith("CONVERSION: ") &&
                               Catch::Matchers::Contains("1000/(N sqrt(EWEIGHT))"));
  CHECK(out.find("EXAMPLES:") == std::string::npos);

  const std::string wa = explain("WDBFILE_AUTO");
  INFO(wa);
  CHECK_THAT(wa, Catch::Matchers::Contains("\nREPLACED BY: WDBFILE") &&
                     Catch::Matchers::Contains("n_d^-a"));
  CHECK(explain("ESIGMA").find("RETIRED:") == std::string::npos);
}

TEST_CASE("tadah explain: an inline role of a help text is shown as the text "
          "the documentation shows",
          "[explain]") {
  /* Pins: the help texts are reStructuredText (the configuration-keys page
   * renders them, and the CLI page shows an option's text in a
   * parsed-literal block, which renders its roles), so explain_text_lines
   * shows an inline role as its text: ':ref:`Title <label>`' as 'Title',
   * ':ref:`LAMBDA`' as 'LAMBDA', ':code:`M`' as 'M', ':math:`x^2`' as 'x^2',
   * two roles of one line each so, and a text with a back-quote but no
   * role as it is; a line of a literal block is kept as written (RST does
   * not read roles there). `tadah explain OALGO` (whose text cites LAMBDA,
   * WCOV and WNOISE with :ref:) prints no ':ref:`' and the line "- The
   * regularisation parameter is controlled by the LAMBDA key, which should
   * be set separately to apply regularisation to the model.".
   * Before the fix: the roles were printed as written (":ref:`LAMBDA`"). */
  CHECK(explain_text_lines(
            "The page :ref:`What a Fit Minimises <fit_objective>` and "
            ":ref:`LAMBDA`.\n") ==
        std::vector<std::string>{"The page What a Fit Minimises and LAMBDA."});
  CHECK(explain_text_lines(":code:`M` = :math:`x^2`, `not a role`\n") ==
        std::vector<std::string>{"M = x^2, `not a role`"});
  CHECK(explain_text_lines("Literal::\n\n    :ref:`kept` here\n\nAfter "
                           ":ref:`LAMBDA`\n") ==
        std::vector<std::string>{"Literal:", "", "    :ref:`kept` here", "",
                                 "After LAMBDA"});
  const std::string oalgo = explain("OALGO");
  INFO(oalgo);
  CHECK(oalgo.find(":ref:`") == std::string::npos);
  CHECK(has_line(lines_of(oalgo),
                 "- The regularisation parameter is controlled by the LAMBDA "
                 "key, which should be set separately to apply "
                 "regularisation to the model."));
}
