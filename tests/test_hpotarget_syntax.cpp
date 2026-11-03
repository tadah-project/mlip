// Catch2 v2 tests of the syntax of a target file as tadah hpo reads it
// (tadah/mlip/hpotarget_syntax.h): its logical lines, the index list of an
// OPTIM line and its box, without a training configuration. tadah hpo,
// tadah analysis check_bounds and tadah refit --retarget read their OPTIM
// lines with it; HPO's tests hold its words to the target-file registry and
// its refusals to tadah hpo's (the registry conformance cases).
//
// Tag [hpotarget_syntax]. The file compiles with or without the header
// (__has_include): without it it holds one case, which fails.

#include "catch2/catch.hpp"

#if __has_include(<tadah/mlip/hpotarget_syntax.h>)

#include <tadah/mlip/hpotarget_syntax.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace tadah::mlip;

namespace {

std::vector<TargetLine> logical(const std::string &file) {
  std::istringstream in(file);
  return target_logical_lines(in);
}

/* The refusal of the OPTIM line whose text after the word OPTIM is
 * @p rest, "" when the syntax reads it. */
std::string syntax_error(const std::string &rest) {
  std::istringstream in(rest);
  try {
    (void)read_optim_syntax(in);
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

OptimSyntax syntax(const std::string &rest) {
  std::istringstream in(rest);
  return read_optim_syntax(in);
}

const std::string kGrammar =
    " -- the grammar is OPTIM <KEY> (indices) <low> <high> [LIN | LOG | "
    "ASINH <c>] or OPTIM <KEY> (indices) AUTO [RATIO <f> | FRAC <p>] "
    "[FLOOR <a>] [GROUP_FLOOR <r>] [LIN | LOG | ASINH <c>]";

}  // namespace

/* Pins: target_logical_lines reads a file as tadah hpo's read() does: a
 * line is cut at '#' and its trailing blanks removed, an empty one
 * skipped; a line ending with a backslash takes the next physical line
 * (cut and trimmed, the backslash removed), a blank or comment line
 * included, which ends the line with the blank before the backslash kept;
 * first and last are the physical lines; the key is the first token in the
 * case written; an unfinished last line keeps its backslash and is marked.
 * Before the fix: the header does not exist. */
TEST_CASE("target file syntax: the logical lines are those tadah hpo reads",
          "[hpotarget_syntax]") {
  const auto l = logical(
      "# a comment line\n"
      "\n"
      "OPTIMIZER   # opens the block\n"
      "  optim CGRID2B (1) 1 2   \n"
      "OPTIM SGRID2B (1) \\\n"
      "   0.5 \\\n"
      "2.0  # the high bound\n"
      "OPTIM CGRID2B (2) 1 \\\n"
      "# a comment ends the line\n"
      "ENDOPTIMIZER\n"
      "OPTIM CGRID2B (3) 1 \\");
  REQUIRE(l.size() == 6);
  CHECK(l[0].first == 3);
  CHECK(l[0].last == 3);
  CHECK(l[0].text == "OPTIMIZER");
  CHECK(l[0].key == "OPTIMIZER");
  CHECK(l[1].text == "  optim CGRID2B (1) 1 2");
  CHECK(l[1].key == "optim");
  CHECK(l[2].first == 5);
  CHECK(l[2].last == 7);
  CHECK(l[2].text == "OPTIM SGRID2B (1)    0.5 2.0");
  CHECK_FALSE(l[2].unfinished);
  CHECK(l[3].first == 8);
  CHECK(l[3].last == 9);
  CHECK(l[3].text == "OPTIM CGRID2B (2) 1 ");
  CHECK(l[4].first == 10);
  CHECK(l[4].key == "ENDOPTIMIZER");
  CHECK(l[5].first == 11);
  CHECK(l[5].last == 11);
  CHECK(l[5].text == "OPTIM CGRID2B (3) 1 \\");
  CHECK(l[5].unfinished);
  CHECK(std::string(kTargetUnfinishedLineMsg) ==
        "the line ends with \\, which continues it on the next line, and "
        "the file has no next line");
}

/* Pins: the index list of an OPTIM line: (2) and ( 2 ) select 2, (1,3) 1
 * and 3, (1-3,5) 1, 2, 3 and 5, (1-7:2) 1, 3, 5 and 7, in the order
 * written, with the tokens after the list; each refusal is tadah hpo's
 * text: no parenthesis, no closing one, a second opening one, empty
 * parentheses, a character right after the closing one, a second list, no
 * token after the list, an index list core::parse_indices_strict refuses
 * and the index 0. Before the fix: the header does not exist. */
TEST_CASE("target file syntax: the index list of an OPTIM line",
          "[hpotarget_syntax]") {
  auto sel = [](const std::string &rest) {
    std::istringstream in(rest);
    return read_optim_selector(in, "CGRID2B");
  };
  CHECK(sel("(2) 1 2").indices == std::vector<std::size_t>{2});
  CHECK(sel("( 2 ) 1 2").indices == std::vector<std::size_t>{2});
  CHECK(sel("(1,3) 1 2").indices == std::vector<std::size_t>{1, 3});
  CHECK(sel("(1-3,5) AUTO").indices ==
        std::vector<std::size_t>{1, 2, 3, 5});
  CHECK(sel("(1-7:2) AUTO").indices == std::vector<std::size_t>{1, 3, 5, 7});
  const OptimSelector s = sel("(1-3) 1 2 LOG");
  CHECK(s.index_text == "1-3");
  CHECK(s.tokens == std::vector<std::string>{"1", "2", "LOG"});

  auto refusal = [&](const std::string &rest) -> std::string {
    try {
      (void)sel(rest);
    } catch (const std::exception &e) {
      return e.what();
    }
    return "";
  };
  CHECK(refusal("1 1 2") == "OPTIM CGRID2B: the key is followed by its "
                            "indices in parentheses, such as OPTIM CGRID2B "
                            "(1)");
  CHECK(refusal("(1 1 2") ==
        "OPTIM CGRID2B: the indices have no closing parenthesis");
  CHECK(refusal("((1) 1 2") ==
        "OPTIM CGRID2B: the indices hold another opening parenthesis");
  CHECK(refusal("( ) 1 2") == "OPTIM CGRID2B: the parentheses hold no index");
  CHECK(refusal("(1)1 2") == "OPTIM CGRID2B: a space follows the closing "
                             "parenthesis of the indices");
  CHECK(refusal("(1) 1 (2)") ==
        "OPTIM CGRID2B: the line takes one list of indices in parentheses "
        "(unexpected '(2)')");
  CHECK(refusal("(1)") == "OPTIM CGRID2B: expected `(indices) <low> <high>` "
                          "or `(indices) AUTO ...` (got 0 token(s))");
  CHECK(refusal("(a) 1 2") ==
        "OPTIM CGRID2B: bad index list (a): Invalid index number 'a'");
  CHECK(refusal("(3-1) 1 2").rfind("OPTIM CGRID2B: bad index list (3-1): ",
                                   0) == 0);
  CHECK(refusal("(0) 1 2") ==
        "OPTIM CGRID2B: bad index list (0): OPTIM indices start at 1");
}

/* Pins: the box of an OPTIM line read without a training configuration:
 * an explicit box with its bounds as written and its tag (LIN, LOG, ASINH
 * <c>, in any case; none: LIN, untagged); AUTO in any case with each
 * option in any case (RATIO: geometric; FRAC; FLOOR and GROUP_FLOOR) and a
 * tag. Before the fix: the header does not exist. */
TEST_CASE("target file syntax: the box of an OPTIM line",
          "[hpotarget_syntax]") {
  const OptimSyntax e = syntax("cgrid2b (2) 1.5 2.5");
  CHECK(e.key == "CGRID2B");
  CHECK(e.key_as_written == "cgrid2b");
  CHECK_FALSE(e.box.auto_box);
  CHECK(e.box.low == 1.5);
  CHECK(e.box.high == 2.5);
  CHECK(e.box.low_tok == "1.5");
  CHECK(e.box.high_tok == "2.5");
  CHECK(e.box.shape == OptimShape::Linear);
  CHECK_FALSE(e.box.tagged);

  CHECK(syntax("CGRID2B (2) 1.5 2.5 lin").box.tagged);
  CHECK(syntax("CGRID2B (2) 1.5 2.5 Log").box.shape == OptimShape::Log);
  const OptimSyntax a = syntax("CGRID2B (2) -1 1 asinh 0.25");
  CHECK(a.box.shape == OptimShape::Asinh);
  CHECK(a.box.c == 0.25);
  CHECK(a.box.c_tok == "0.25");
  CHECK(a.box.tagged);

  const OptimSyntax r = syntax("SGRID2B (1-4) auto ratio 3 LOG");
  CHECK(r.box.auto_box);
  CHECK(r.box.mode_set);
  CHECK(r.box.geometric);
  CHECK(r.box.factor == 3.0);
  CHECK(r.box.factor_tok == "3");
  CHECK(r.box.shape == OptimShape::Log);
  const OptimSyntax f =
      syntax("CGRID2B (1-4) AUTO Frac 0 floor 0.1 Group_Floor 0.01");
  CHECK(f.box.mode_set);
  CHECK_FALSE(f.box.geometric);
  CHECK(f.box.factor == 0.0);
  CHECK(f.box.floor == 0.1);
  CHECK(f.box.group_floor == 0.01);
  CHECK(f.box.group_floor_tok == "0.01");
  CHECK(f.box.floor_set);
  const OptimSyntax d = syntax("CGRID2B (1-4) AUTO");
  CHECK(d.box.auto_box);
  CHECK_FALSE(d.box.mode_set);
  CHECK_FALSE(d.box.floor_set);
  CHECK_FALSE(d.box.tagged);
}

/* Pins: each refusal of the box syntax is tadah hpo's text: an AUTO option
 * without a value, a word that is no AUTO option, an option given twice,
 * RATIO with FRAC, RATIO <= 1, FRAC < 0, FLOOR < 0, ASINH c <= 0, ASINH
 * without c, a number below the smallest normal double (with the note),
 * tokens past the explicit box, a high bound not above the low one, and a
 * line without a key. Before the fix: the header does not exist. */
TEST_CASE("target file syntax: the refusals of the box of an OPTIM line",
          "[hpotarget_syntax]") {
  CHECK(syntax_error("CGRID2B (1) AUTO FRAC") ==
        "OPTIM CGRID2B: AUTO option 'FRAC' needs a value" + kGrammar);
  CHECK(syntax_error("CGRID2B (1) AUTO WIDTH 2") ==
        "OPTIM CGRID2B: 'WIDTH' is not an AUTO option (RATIO, FRAC, FLOOR "
        "or GROUP_FLOOR)" + kGrammar);
  CHECK(syntax_error("CGRID2B (1) AUTO FLOOR 1 floor 2") ==
        "OPTIM CGRID2B: AUTO FLOOR is given twice on this line; each option "
        "is given once");
  CHECK(syntax_error("CGRID2B (1) AUTO RATIO 2 FRAC 0.1") ==
        "OPTIM CGRID2B: AUTO takes RATIO or FRAC, not both");
  CHECK(syntax_error("CGRID2B (1) AUTO RATIO 1") ==
        "OPTIM CGRID2B: AUTO RATIO must be > 1 (got 1)");
  CHECK(syntax_error("CGRID2B (1) AUTO FRAC -0.1") ==
        "OPTIM CGRID2B: AUTO FRAC must be >= 0 (got -0.1); FRAC 0 with "
        "FLOOR or GROUP_FLOOR gives a constant half-width");
  CHECK(syntax_error("CGRID2B (1) AUTO FLOOR -1") ==
        "OPTIM CGRID2B: AUTO FLOOR must be >= 0 (got -1)");
  CHECK(syntax_error("CGRID2B (1) 1 2 ASINH 0") ==
        "OPTIM CGRID2B: ASINH requires a positive linear scale c (got 0)");
  CHECK(syntax_error("CGRID2B (1) 1 2 ASINH") ==
        "OPTIM CGRID2B: ASINH needs one argument <c> (the linear scale)");
  CHECK(syntax_error("CGRID2B (1) 1e-400 2") ==
        "OPTIM CGRID2B: <low> must be a finite number (got '1e-400') (too "
        "small: its magnitude is below the smallest normal double, "
        "2.2250738585072014e-308)");
  CHECK(syntax_error("CGRID2B (1) 1 x") ==
        "OPTIM CGRID2B: <high> must be a finite number (got 'x')");
  CHECK(syntax_error("CGRID2B (2) 3.4 3.6 x") ==
        "OPTIM CGRID2B: unexpected token(s) '3.4 3.6 x'" + kGrammar);
  CHECK(syntax_error("SGRIDMB (1) 0.2 0.2") ==
        "OPTIM SGRIDMB: <high> 0.2 is not greater than <low> 0.2");
  CHECK(syntax_error("") == "Failed to read key from input.");
  CHECK(number_underflow_note("1e-320") ==
        " (too small: its magnitude is below the smallest normal double, "
        "2.2250738585072014e-308)");
  CHECK(number_underflow_note("1e400").empty());
  CHECK(number_underflow_note("x").empty());
  CHECK(number_underflow_note("0").empty());
}

/* Pins: the words of the OPTIM lines, optim_auto_options() (RATIO, FRAC,
 * FLOOR, GROUP_FLOOR), optim_shape_tags() (LIN, LOG, ASINH),
 * kOptimAutoWord (AUTO) and the lexing characters, are those the syntax
 * reads: each AUTO option is taken with a value, each tag on an explicit
 * box (ASINH with its c), AUTO opens an AUTO box, and the keys of
 * kOptimAutoExcludedKeys, which take no AUTO box, are the distance cutoffs
 * RCUT2B and RCUTMB (it was the prefix RCUT, which took RCUTENV, a width in
 * density units, for a cutoff). HPO's tests hold the same words to the
 * target-file registry. Before the fix: the header does not exist. */
TEST_CASE("target file syntax: the words are those the syntax reads",
          "[hpotarget_syntax]") {
  for (std::string_view o : optim_auto_options()) {
    CAPTURE(o);
    CHECK(syntax_error("CGRID2B (1) AUTO " + std::string(o) + " 2") == "");
  }
  CHECK(optim_auto_options().size() == 4);
  for (std::string_view t : optim_shape_tags()) {
    CAPTURE(t);
    const std::string tag =
        t == "ASINH" ? std::string(t) + " 1" : std::string(t);
    CHECK(syntax("CGRID2B (1) 1 2 " + tag).box.tagged);
  }
  CHECK(optim_shape_tags().size() == 3);
  CHECK(syntax("CGRID2B (1) " + std::string(kOptimAutoWord)).box.auto_box);
  CHECK(kOptimAutoExcludedKeys ==
        std::array<std::string_view, 2>{"RCUT2B", "RCUTMB"});
  CHECK(kTargetComment == '#');
  CHECK(kTargetContinuation == '\\');
}

#else

TEST_CASE("target file syntax: the header exists", "[hpotarget_syntax]") {
  FAIL("tadah/mlip/hpotarget_syntax.h does not exist: MLIP has no reader of "
       "the target file's syntax");
}

#endif
