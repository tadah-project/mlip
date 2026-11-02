// Catch2 v2 tests of the reader of the OPTIM lines of a target file against
// a training configuration (tadah/mlip/hpotarget_optim.h): the boxes of the
// values a line selects (explicit, or the AUTO box tadah hpo derives), the
// slot of each value, a DM_Embed component path, FIXWEIGHT coefficients,
// the order of the refusals and the cross-line rule. tadah hpo reads its
// OPTIM lines with it and tadah analysis check_bounds its target file; HPO's
// registry conformance cases compare it with tadah hpo on every registry
// case.
//
// Tag [hpotarget_optim]. The file compiles with or without the header
// (__has_include): without it it holds one case, which fails.

#include "catch2/catch.hpp"

#if __has_include(<tadah/mlip/hpotarget_optim.h>)

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/hpotarget_optim.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

using tadah::core::Context;
using namespace tadah::mlip;

namespace {

/* D2_Blip 4 4 Ta Ta, CGRID2B 2 3 4 5, SGRID2B 1 1 1 1 (a radial bump: its
 * widths > 0), RCUT2B 5.5. */
Context blip_seed() {
  tadah::core::PeriodicTable::initialize();
  Context c;
  c.add("ATOM", "Ta");
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_Blip 4 4 Ta Ta");
  c.add("CGRID2B", "2.0 3.0 4.0 5.0");
  c.add("SGRID2B", "1.0 1.0 1.0 1.0");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCUT2B", 5.5);
  c.add("VERBOSE", 0);
  return c;
}

/* D2_Blip5 + DM_REAM (Blip5 density) under RHOFORM NONNEG, AMPGRIDMB
 * 0.5 0 0.1. */
Context ream_nonneg_seed() {
  tadah::core::PeriodicTable::initialize();
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 73);
  c.add("INIT2B", "true");
  c.add("INITMB", "true");
  c.add("TYPE2B", "D2_Blip5 4 4 Ta Ta");
  c.add("CGRID2B", "2.5 3.2 3.9 4.5");
  c.add("SGRID2B", "3.0 3.0 3.0 3.0");
  c.add("TYPEMB", "DM_REAM 3 3 3 3 5 5 0 0 Ta Ta");
  c.add("CGRIDMB", "2.8 3.6 4.4");
  c.add("SGRIDMB", "3.0 3.0 3.0");
  c.add("AMPGRIDMB", "0.5 0 0.1");
  c.add("RHOFORM", "NONNEG");
  c.add("CEMBFUNC", "-0.2 0.3 0.8 1.3 1.8");
  c.add("SEMBFUNC", "2.0 2.0 2.0 2.0 2.0");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 5.5);
  c.add("RCUTMB", 5.5);
  c.add("VERBOSE", 0);
  return c;
}

/* One DM_Embed component, ta (PAIRS Ta Ta), density centres 1 2.5 4. */
Context embed_seed() {
  tadah::core::PeriodicTable::initialize();
  std::istringstream text(
      "ATOM Ta\nWATOM 73\nINIT2B false\nINITMB true\nVERBOSE 0\n"
      "DESCRIPTOR ta DM_Embed\nPAIRS Ta Ta\nDENSITY\nBASIS Blip5\n"
      "FORM NONNEG\nCENTRES 1.0 2.5 4.0\nETA 1.5 1.5 1.5\n"
      "AMPLITUDES 0.4 0.2 0.1\nCUTOFF\nTYPE Cut_Poly\nRADIUS 5.0\n"
      "ENDCUTOFF\nENDDENSITY\nEMBEDDING\nBASIS Gaussian\n"
      "CENTRES 0.0 0.5 1.0 1.5\nWIDTHS 0.5 0.5 0.5 0.5\nREFERENCE ZERO\n"
      "ENDEMBEDDING\nENDDESCRIPTOR\n");
  return tadah::core::CoreUserConfigParser().parse(text);
}

/* The OPTIM lines of @p file read against @p c, prepared as tadah hpo
 * prepares it. */
std::vector<OptimTargetLine> read_file(Context &c, const std::string &file) {
  const OptimSeed seed = prepare_optim_seed(c, "tadah hpo");
  std::istringstream in(file);
  return read_optim_lines(in, c, seed);
}

/* The boxes of the one OPTIM line @p line, read against @p c. */
OptimLine read_line(Context c, const std::string &line) {
  const auto l = read_file(c, line + "\n");
  REQUIRE(l.size() == 1);
  return l[0].optim;
}

/* "<line>: <what>" of the refusal of @p file read against @p c, "" when
 * it is read. */
std::string refusal(Context c, const std::string &file) {
  try {
    (void)read_file(c, file);
  } catch (const TargetLineError &e) {
    return std::to_string(e.line) + ": " + e.what();
  } catch (const std::exception &e) {
    return std::string("not a TargetLineError: ") + e.what();
  }
  return "";
}

std::vector<std::pair<double, double>> boxes_of(const OptimLine &l) {
  std::vector<std::pair<double, double>> out;
  for (const auto &b : l.boxes) out.emplace_back(b.lo, b.hi);
  return out;
}

}  // namespace

/* Pins: the AUTO boxes of a line, those tadah hpo derives: CGRID2B 2 3 4 5
 * (sign-free) (1-4) AUTO is the linear FRAC 0.5 box of each seed, [1, 3],
 * [1.5, 4.5], [2, 6], [2.5, 7.5], AutoLinear, slots 0 to 3; SGRID2B 1 1 1 1
 * (positive-only) the geometric RATIO 2 box [0.5, 2] each, AutoGeometric;
 * FRAC 0 FLOOR 0.1 the half-width 0.1; FRAC 0 GROUP_FLOOR 0.1 the
 * half-width 0.1 x 5, the largest seed of the line. Before the fix: the
 * header does not exist. */
TEST_CASE("OPTIM reader: the AUTO boxes are those tadah hpo derives",
          "[hpotarget_optim]") {
  const OptimLine c = read_line(blip_seed(), "OPTIM CGRID2B (1-4) AUTO");
  CHECK(c.key == "CGRID2B");
  CHECK(c.auto_box);
  CHECK(c.line == 1);
  CHECK(c.config == std::vector<std::string>{"2", "3", "4", "5"});
  CHECK(boxes_of(c) == std::vector<std::pair<double, double>>{
                           {1.0, 3.0}, {1.5, 4.5}, {2.0, 6.0}, {2.5, 7.5}});
  for (std::size_t i = 0; i < c.boxes.size(); ++i) {
    CHECK(c.boxes[i].slot == i);
    CHECK(c.boxes[i].index == i + 1);
    CHECK(c.boxes[i].origin == OptimBoxOrigin::AutoLinear);
  }
  const OptimLine s = read_line(blip_seed(), "OPTIM SGRID2B (1-4) AUTO");
  CHECK(boxes_of(s) == std::vector<std::pair<double, double>>(
                           4, std::pair<double, double>{0.5, 2.0}));
  for (const auto &b : s.boxes)
    CHECK(b.origin == OptimBoxOrigin::AutoGeometric);
  const OptimLine f =
      read_line(blip_seed(), "OPTIM CGRID2B (1,4) AUTO FRAC 0 FLOOR 0.1");
  CHECK(boxes_of(f) == std::vector<std::pair<double, double>>{
                           {2.0 - 0.1, 2.0 + 0.1}, {5.0 - 0.1, 5.0 + 0.1}});
  const OptimLine g =
      read_line(blip_seed(), "OPTIM CGRID2B (1,4) AUTO FRAC 0 GROUP_FLOOR 0.1");
  CHECK(boxes_of(g) == std::vector<std::pair<double, double>>{
                           {2.0 - 0.1 * 5.0, 2.0 + 0.1 * 5.0},
                           {5.0 - 0.1 * 5.0, 5.0 + 0.1 * 5.0}});
}

/* Pins: the AUTO boxes clamped at 0: AMPGRIDMB 0.5 0 0.1 under RHOFORM
 * NONNEG takes the group floor 1e-3 of the largest seed, so the 0 seed's
 * box is [0, 5e-4], the 0.5 seed's [0.25, 0.75], AutoLinearNonneg; FSIGMA
 * 0.1 0.05 (1-2) AUTO: the absolute part a (> 0) the geometric box [0.05,
 * 0.2], the relative part b (>= 0) the linear box [0.025, 0.075] clamped
 * at 0, AutoLinearNonneg. Before the fix: the header does not exist. */
TEST_CASE("OPTIM reader: the AUTO boxes of non-negative values",
          "[hpotarget_optim]") {
  const OptimLine a = read_line(ream_nonneg_seed(), "OPTIM AMPGRIDMB (1-2) AUTO");
  REQUIRE(a.boxes.size() == 2);
  CHECK(a.boxes[0].lo == 0.5 - 0.25);
  CHECK(a.boxes[0].hi == 0.5 + 0.25);
  CHECK(a.boxes[1].lo == 0.0);
  CHECK(a.boxes[1].hi == 1e-3 * 0.5);
  for (const auto &b : a.boxes)
    CHECK(b.origin == OptimBoxOrigin::AutoLinearNonneg);
  Context sig = blip_seed();
  sig.add("FSIGMA", "0.1 0.05");
  const OptimLine s = read_line(sig, "OPTIM FSIGMA (1-2) AUTO");
  REQUIRE(s.boxes.size() == 2);
  CHECK(s.boxes[0].lo == 0.1 / 2.0);
  CHECK(s.boxes[0].hi == 0.1 * 2.0);
  CHECK(s.boxes[0].origin == OptimBoxOrigin::AutoGeometric);
  CHECK(s.boxes[1].lo == 0.05 - 0.5 * 0.05);
  CHECK(s.boxes[1].hi == 0.05 + 0.5 * 0.05);
  CHECK(s.boxes[1].origin == OptimBoxOrigin::AutoLinearNonneg);
}

/* Pins: an explicit box applies to each value its index list selects:
 * (1,3) gives slots 0 and 2, the box and its tag (LOG) on each, Explicit;
 * a FIXWEIGHT index names a coefficient, the slot of its FIXINDEX entry
 * (FIXINDEX 3 7: OPTIM FIXWEIGHT (7) is slot 1, index 7); a DM_Embed
 * component path, in any case, is upper-cased and reads the component's
 * values. Before the fix: the header does not exist. */
TEST_CASE("OPTIM reader: index lists, FIXWEIGHT coefficients and component "
          "paths",
          "[hpotarget_optim]") {
  const OptimLine l = read_line(blip_seed(), "OPTIM cgrid2b (1,3) 1.5 4.5 LOG");
  CHECK(l.key == "CGRID2B");
  REQUIRE(l.boxes.size() == 2);
  CHECK(l.boxes[0].slot == 0);
  CHECK(l.boxes[1].slot == 2);
  CHECK(l.boxes[1].index == 3);
  CHECK(l.boxes[1].lo == 1.5);
  CHECK(l.boxes[1].hi == 4.5);
  CHECK(l.boxes[1].origin == OptimBoxOrigin::Explicit);
  CHECK(l.shape == OptimShape::Log);
  CHECK(l.tagged);

  Context fw = blip_seed();
  fw.add("FIXINDEX", "3 7");
  fw.add("FIXWEIGHT", "0.3 0.4");
  const OptimLine w = read_line(fw, "OPTIM FIXWEIGHT (7) 0 1");
  REQUIRE(w.boxes.size() == 1);
  CHECK(w.boxes[0].index == 7);
  CHECK(w.boxes[0].slot == 1);
  CHECK(refusal(fw, "OPTIM FIXWEIGHT (5) 0 1") ==
        "1: OPTIM FIXWEIGHT (5): coefficient 5 is not in FIXINDEX (3 7); "
        "the indices of OPTIM FIXWEIGHT name coefficients, those FIXINDEX "
        "lists, not positions among the FIXWEIGHT values");

  const OptimLine e = read_line(embed_seed(), "OPTIM ta.Density.Centres (2) 2.4 2.6");
  CHECK(e.key == "TA.DENSITY.CENTRES");
  CHECK(e.named);
  CHECK(e.config == std::vector<std::string>{"1", "2.5", "4"});
  REQUIRE(e.boxes.size() == 1);
  CHECK(e.boxes[0].slot == 1);
}

/* Pins: the refusals come in tadah hpo's order: a key that is no key of
 * the training configuration before its index list (OPTIM FOO 2 1 2), the
 * discrete index OALGO (1) before its bounds, RATIO <= 1 before the refusal
 * of AUTO on a cutoff key; a value two lines select is refused at the
 * second, naming the first ("on line 1"), the keys matched in any case; a
 * refusal of an entry names the key and the entry once; an index past the
 * key's values names their count; an unfinished last line is refused at
 * its line. Each is a TargetLineError at the line's last physical line.
 * Before the fix: the header does not exist. */
TEST_CASE("OPTIM reader: the refusals and their order", "[hpotarget_optim]") {
  CHECK(refusal(blip_seed(), "OPTIM FOO 2 1 2").rfind(
            "1: OPTIM FOO: FOO is not a key of the training configuration",
            0) == 0);
  Context oalgo = blip_seed();
  oalgo.add("OALGO", "1 0.5");
  CHECK(refusal(oalgo, "OPTIM OALGO (1) x 2").rfind(
            "1: OPTIM OALGO (1) is not a valid search target: the first "
            "OALGO value selects the training algorithm",
            0) == 0);
  CHECK(refusal(blip_seed(), "OPTIM RCUT2B (1) AUTO RATIO 0.5") ==
        "1: OPTIM RCUT2B: AUTO RATIO must be > 1 (got 0.5)");
  CHECK(refusal(blip_seed(), "OPTIM RCUT2B (1) AUTO") ==
        "1: OPTIM RCUT2B: a cutoff key takes explicit bounds, OPTIM RCUT2B "
        "(indices) <low> <high>, and no AUTO box: the neighbour list is "
        "sized from the explicit high bound");
  CHECK(refusal(blip_seed(),
                "OPTIM CGRID2B (1-2) 1 6\n# a comment\nOPTIM cgrid2b (2) \\\n"
                "  2 4\n") ==
        "4: OPTIM CGRID2B (2): the OPTIM CGRID2B line on line 1 selects "
        "this index as well; one OPTIM line selects each index (OPTIM keys "
        "match in any case)");
  const std::string w = refusal(blip_seed(), "OPTIM SGRID2B (2) 0 3");
  CHECK(w.rfind("1: OPTIM SGRID2B (2): the low bound 0 is not > 0, but the "
                "D2_Blip pair basis requires every SGRID2B width > 0",
                0) == 0);
  CHECK(w.find("OPTIM SGRID2B (2)", w.find("OPTIM SGRID2B (2)") + 1) ==
        std::string::npos);
  CHECK(refusal(blip_seed(), "OPTIM CGRID2B (1-5) AUTO") ==
        "1: OPTIM CGRID2B (5): CGRID2B has 4 values in the training "
        "configuration, so the indices of OPTIM CGRID2B run from 1 to 4");
  CHECK(refusal(blip_seed(), "OPTIM CGRID2B (1) 1 2\nOPTIM CGRID2B (2) 1 \\") ==
        "2: the line ends with \\, which continues it on the next line, and "
        "the file has no next line");
}

/* Pins: read_optim_lines reads only the OPTIM lines, in order, those inside
 * a block too (OPTIMIZER ... ENDOPTIMIZER: tadah hpo --check refuses such a
 * file, and tadah analysis check_bounds audits the boxes), and keeps each
 * line's source (first and last physical line, text). Before the fix: the
 * header does not exist. */
TEST_CASE("OPTIM reader: the OPTIM lines of a file", "[hpotarget_optim]") {
  Context c = blip_seed();
  const auto l = read_file(
      c, "MAXEVAL 10\nOPTIMIZER\nLIB NLOPT\nOPTIM CGRID2B (1) 1.5 2.5\n"
         "ENDOPTIMIZER\nOPTIM SGRID2B (1,2) \\\n 0.5 2  # widths\n");
  REQUIRE(l.size() == 2);
  CHECK(l[0].source.first == 4);
  CHECK(l[0].optim.key == "CGRID2B");
  CHECK(l[1].source.first == 6);
  CHECK(l[1].source.last == 7);
  CHECK(l[1].source.text == "OPTIM SGRID2B (1,2)  0.5 2");
  CHECK(l[1].optim.line == 7);
  CHECK(l[1].optim.boxes.size() == 2);
}

#else

TEST_CASE("OPTIM reader: the header exists", "[hpotarget_optim]") {
  FAIL("tadah/mlip/hpotarget_optim.h does not exist: MLIP has no reader of "
       "the OPTIM lines");
}

#endif
