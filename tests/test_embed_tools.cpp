#include "catch2/catch.hpp"
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/refit_modes.h>
#include <tadah/mlip/engines/hpotarget_retarget.h>
#include <tadah/mlip/engines/analysis_bfunc_engine.h>
#include <tadah/mlip/task.h>
#include <tadah/models/descriptors/dm/embed_spec.h>
#include <tadah/models/cutoffs.h>
#include <tadah/models/descriptors/basis_dispatch.h>
#include <tadah/models/descriptors/dm/embed_kernel.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <iomanip>
#include <iterator>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace tadah;
namespace {
struct Temp {
  std::filesystem::path old=std::filesystem::current_path(),dir;
  Temp():dir(std::filesystem::temp_directory_path()/
    ("tadah-embed-tools-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
    std::filesystem::create_directories(dir);std::filesystem::current_path(dir);
    core::PeriodicTable::initialize();
  }
  ~Temp(){std::filesystem::current_path(old);std::error_code e;std::filesystem::remove_all(dir,e);}
};
core::Context potential() {
  std::istringstream in(
    "ATOM Ta\nWATOM 1\nVERBOSE 0\nINIT2B false\nINITMB true\nBIAS false\n"
    "MODEL M_BLR BF_Linear\nNORM false\nWEIGHTS 1 2 3\n"
    "DESCRIPTOR ta DM_Embed\nPAIRS Ta Ta\nDENSITY\nBASIS WendlandC2\n"
    "FORM NONNEG\nCENTRES 1 2\nWIDTHS 0.5 0.75\nAMPLITUDES 0.4 0.2\n"
    "CUTOFF\nTYPE Cut_Poly\nRADIUS 6\nENDCUTOFF\nENDDENSITY\n"
    "EMBEDDING\nBASIS Gaussian\nCENTRES 0 0.5 1\nWIDTHS 0.5 0.75 1\n"
    "REFERENCE ZERO\nENVELOPE\nTYPE Smoothstep4\nCUTOFF 2\nENDENVELOPE\n"
    "ENDEMBEDDING\nENDDESCRIPTOR\n");
  return core::CoreUserConfigParser().parse(in);
}
}
TEST_CASE("DM_Embed edit preserves independent components envelopes and width coordinates",
          "[unified-dm][edit]") {
  Temp temp;auto original=potential();
  {std::ofstream f("input.tadah");original.write_potential(f,18);}
  core::Context command;command.add("EDIT","input.tadah");command.add("FUNC","F");
  command.add("ADD",1.5);command.add("ETA",2);command.add("OUTFILE","edited.tadah");
  command.add("HPOFILEOUT","edited.hpotarget");
  REQUIRE(mlip::engines::run_refit_edit(command)==0);
  auto saved=core::CoreUserConfigParser().parse("edited.tadah");
  auto spec=models::embed_specs(saved).at(0),before=models::embed_specs(original).at(0);
  REQUIRE(spec.density.centres==before.density.centres);
  REQUIRE(spec.embedding.scales[0]==before.embedding.scales[0]);
  REQUIRE(spec.embedding.coordinates==models::EmbedCoordinates::Widths);
  REQUIRE(spec.envelope==models::EmbedEnvelope::Smoothstep4);
  REQUIRE(spec.embedding.centres.size()==4);
  REQUIRE(saved.get<double>("WEIGHTS",3)==0);
}
TEST_CASE("DM_Embed basis analysis selects the embedding component", "[unified-dm][analysis]") {
  Temp temp;auto c=potential();
  c.add("TYPE","MB EMBEDDING");c.add("RANGE","0 3 31");c.add("OUTFILE","basis.dat");
  mlip::Task task("analysis.bfunc",c);mlip::engines::AnalysisBfuncEngine engine;
  REQUIRE(engine.run(task)==0);
  std::ifstream f("basis.dat");double q,v;std::size_t count=0;
  while(f>>q>>v)++count;
  REQUIRE(count==3*31);
}
TEST_CASE("DM_Embed retarget recognizes named paths and component nonnegative amplitudes",
          "[unified-dm][retarget]") {
  std::vector<std::string> lines{
    "# Bounds are +-10%",
    "# COMPONENT_PARAMETER TA.DENSITY.AMPLITUDES NONNEG",
    "# OPTIM TA.DENSITY.AMPLITUDES (1) 0.0 0.2",
    "# OPTIM TA.EMBEDDING.WIDTHS (1) 0.45 0.55"};
  mlip::engines::RetargetOptions options;options.pct=200;
  auto report=mlip::engines::retarget_hpotarget_lines(lines,options);
  REQUIRE(report.n_rewritten==2);
  for(const auto &line:lines)if(line.find("# OPTIM")==0) {
    std::istringstream in(line.substr(2));std::string optim,key,index;double lo,hi;
    in>>optim>>key>>index>>lo>>hi;
    REQUIRE(lo>=0);if(key.find("WIDTHS")!=std::string::npos)REQUIRE(lo>0);
  }
}

TEST_CASE("Knot basis analysis accepts absent scales mixed degrees and cutoff product rule", "[unified-dm][analysis][legacy]") {
  Temp temp;core::Context c;
  c.add("ATOM","Ta");c.add("VERBOSE",0);c.add("INITMB",false);c.add("INIT2B",true);
  c.add("TYPE2B","D2_Knot5 2 2 Ta Ta");c.add("CGRID2B","3 4");c.add("KDEG2B","3 5");
  c.add("RCTYPE2B","Cut_Poly");c.add("RCUT2B",6);c.add("TYPE","2B");
  c.add("RANGE","2 2.5 2");c.add("DERIVATIVE",true);c.add("RESCALE",true);c.add("NUMERIC",17);
  c.add("OUTFILE","knot.dat");mlip::Task task("analysis.bfunc",c);
  REQUIRE(mlip::engines::AnalysisBfuncEngine().run(task)==0);
  double radius=6;models::Cut_Poly cutoff(radius);std::ifstream in("knot.dat");
  for(int degree:{3,5})for(double x:{2.0,2.5}) {
    const double centre=degree==3?3:4,t=centre-x;
    double q,v;REQUIRE(bool(in>>q>>v));REQUIRE(q==x);
    REQUIRE(v==Approx(-degree*std::pow(t,degree-1)*cutoff.calc(x)+std::pow(t,degree)*cutoff.calc_prime(x)).margin(1e-12));
  }
}

TEST_CASE("DM_Embed embedding removal supports negative right-active knots and envelopes", "[unified-dm][edit]") {
  Temp temp;auto c=potential();auto s=models::embed_specs(c).at(0);
  s.embedding.basis=models::EmbedBasis::Knot5;s.embedding.centres={-5,-4,-3};s.embedding.scales.clear();s.embedding.degrees={5,5,5};
  c.replace_blocks({models::embed_block(s)});
  {std::ofstream out("negative.tadah");c.write_potential(out,18);}
  core::Context command;command.add("EDIT","negative.tadah");command.add("FUNC","F");command.add("REMOVE",-3);
  command.add("OUTFILE","removed.tadah");command.add("HPOFILEOUT","removed.hpo");
  REQUIRE(mlip::engines::run_refit_edit(command)==0);
  auto saved=core::CoreUserConfigParser().parse("removed.tadah");
  REQUIRE(models::embed_specs(saved).at(0).embedding.centres.size()==2);
}

// ---- analysis bfunc: Y appends the cutoff, --rescale tapers ---------------

namespace {
// One curve block as analysis bfunc writes it: `q value` per point in the
// fixed format of NUMERIC digits, then a blank line.
std::string bfunc_block(const std::vector<double> &q,
                        const std::vector<double> &v, int numeric) {
  std::ostringstream o;
  for (std::size_t i = 0; i < q.size(); ++i)
    o << std::fixed << std::setprecision(numeric) << std::setw(numeric + 5)
      << q[i] << "    " << std::setw(numeric + 5) << v[i] << std::endl;
  o << std::endl;
  return o.str();
}
std::string slurp_file(const std::string &path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}
// A legacy DM_REAM potential (Blip, one density function at 2 of eta 2,
// Cut_Poly at 5).
core::Context ream_potential() {
  core::Context c;
  c.add("ATOM","Ta");c.add("VERBOSE",0);c.add("INIT2B",false);c.add("INITMB",true);
  c.add("TYPEMB","DM_REAM 0 1 1 1 2 2 0 0 Ta Ta");c.add("RHOFORM","SIGNED");c.add("EMBF0","FREE");
  c.add("CGRIDMB",2);c.add("SGRIDMB",2);c.add("AMPGRIDMB",0.3);
  c.add("CEMBFUNC","0 1");c.add("SEMBFUNC","2 2");c.add("RCUTMB",5);c.add("RCTYPEMB","Cut_Poly");
  return c;
}
// The analysis bfunc task of potential `c` with these options.
mlip::Task bfunc_task(core::Context c, const std::string &type, bool rescale,
                      bool derivative) {
  c.add("TYPE",type);c.add("RANGE","2.2 2.8 3");c.add("NUMERIC",6);
  c.add("RESCALE",rescale);c.add("DERIVATIVE",derivative);c.add("OUTFILE","basis.dat");
  return mlip::Task("analysis.bfunc",c);
}
std::string run_bfunc(core::Context c, const std::string &type, bool rescale,
                      bool derivative) {
  auto task=bfunc_task(std::move(c),type,rescale,derivative);
  REQUIRE(mlip::engines::AnalysisBfuncEngine().run(task)==0);
  return slurp_file("basis.dat");
}
}  // namespace

/* Pins: analysis bfunc --type MB Y of a legacy DM_REAM density prints the
 * bare basis, then a block with the cutoff function (its derivative under
 * --derivative), and --rescale tapers the basis by the cutoff (product rule
 * under --derivative): the meaning of Y and --rescale for 2B, in the fixed
 * NUMERIC format of the 2B curves. Before the fix Y tapered the basis like
 * --rescale, the cutoff block was gone, and the numbers were printed in
 * another format (no setw/fixed, two blank lines between functions). */
TEST_CASE("Basis analysis of a DM_REAM density keeps the meaning of Y and --rescale",
          "[unified-dm][analysis][legacy]") {
  Temp temp;
  const std::vector<double> q{2.2,2.5,2.8};
  double radius=5;models::Cut_Poly fc(radius);
  std::vector<double> g,dg,c,dc,tapered,dtapered;
  for(double x:q) {
    g.push_back(models::radial_value_bare(models::BasisKind::Blip,x,2,2,5));
    dg.push_back(models::radial_deriv_bare(models::BasisKind::Blip,x,2,2,5));
    c.push_back(fc.calc(x));dc.push_back(fc.calc_prime(x));
    tapered.push_back(g.back()*c.back());
    dtapered.push_back(dg.back()*c.back()+g.back()*dc.back());
  }
  REQUIRE(run_bfunc(ream_potential(),"MB Y",false,false)==
          bfunc_block(q,g,6)+bfunc_block(q,c,6)+"\n");
  REQUIRE(run_bfunc(ream_potential(),"MB Y",false,true)==
          bfunc_block(q,dg,6)+bfunc_block(q,dc,6)+"\n");
  REQUIRE(run_bfunc(ream_potential(),"MB N",true,false)==bfunc_block(q,tapered,6)+"\n");
  REQUIRE(run_bfunc(ream_potential(),"MB N",true,true)==bfunc_block(q,dtapered,6)+"\n");
}

/* Pins: for a DM_Embed density, --type MB Y prints the bare density
 * functions and then the component's spatial cutoff, MB DENSITY with
 * --rescale the tapered functions, and MB EMBEDDING with --rescale is
 * refused: the embedding has no spatial cutoff to taper with. Before the
 * fix Y tapered the density functions (no cutoff block) and --rescale was
 * ignored for the embedding. */
TEST_CASE("DM_Embed basis analysis reads Y and --rescale as for radial curves",
          "[unified-dm][analysis]") {
  Temp temp;
  const auto spec=models::embed_specs(potential()).at(0);
  models::EmbedEvaluator evaluator(spec);
  const std::vector<double> q{2.2,2.5,2.8};
  double radius=spec.radius;models::Cut_Poly fc(radius);
  std::string bare,tapered;
  for(std::size_t i=0;i<spec.density.centres.size();++i) {
    std::vector<double> b,t;
    for(double x:q) {
      b.push_back(evaluator.density_atom(i,x).value);
      t.push_back(evaluator.density(x,'a',i).value);
    }
    bare+=bfunc_block(q,b,6);tapered+=bfunc_block(q,t,6);
  }
  std::vector<double> c;for(double x:q)c.push_back(fc.calc(x));
  REQUIRE(run_bfunc(potential(),"MB Y",false,false)==bare+bfunc_block(q,c,6)+"\n");
  REQUIRE(run_bfunc(potential(),"MB DENSITY",true,false)==tapered+"\n");
  auto embedding=bfunc_task(potential(),"MB EMBEDDING",true,false);
  REQUIRE_THROWS_WITH(mlip::engines::AnalysisBfuncEngine().run(embedding),
                      Catch::Contains("--rescale") && Catch::Contains("embedding"));
}

// ---- refit.hpotarget of a DM_Embed potential ------------------------------

namespace {
core::Context knot_potential() {
  std::istringstream in(
    "ATOM Ta\nWATOM 1\nVERBOSE 0\nINIT2B false\nINITMB true\nBIAS false\n"
    "MODEL M_BLR BF_Linear\nNORM false\nWEIGHTS 1 2 3\n"
    "DESCRIPTOR ta DM_Embed\nPAIRS Ta Ta\nDENSITY\nBASIS Knot5\n"
    "FORM SIGNED\nCENTRES 2 3 4\nDEGREES 5 5 5\nAMPLITUDES 0.4 0.2 0.1\n"
    "CUTOFF\nTYPE NONE\nRADIUS 4\nENDCUTOFF\nENDDENSITY\n"
    "EMBEDDING\nBASIS Gaussian\nCENTRES 0 0.5 1\nWIDTHS 0.5 0.75 1\n"
    "REFERENCE ZERO\nENDEMBEDDING\nENDDESCRIPTOR\n");
  return core::CoreUserConfigParser().parse(in);
}
std::vector<std::string> file_lines(const std::string &path) {
  std::vector<std::string> out;std::ifstream in(path);std::string l;
  while(std::getline(in,l))out.push_back(l);
  return out;
}
struct OptimBox { std::string key; std::size_t index; double lo, hi; };
std::vector<OptimBox> optim_boxes(const std::vector<std::string> &lines,
                                  const std::string &key) {
  std::vector<OptimBox> out;
  for(const auto &l:lines) {
    std::istringstream in(l);std::string w,optim,k,index;double lo,hi;
    if(l.rfind("#",0)==0)in>>w;
    if(!(in>>optim>>k>>index>>lo>>hi) || optim!="OPTIM" || k!=key)continue;
    out.push_back({k,std::stoul(index.substr(1)),lo,hi});
  }
  return out;
}
}  // namespace

/* Pins: the starter target of a DM_Embed potential lists the component
 * parameters under a header of their own (not the pair function's), and
 * its zero-reference note names the component REFERENCE ZERO, not the
 * DM_REAM key EMBF0 (which a potential with blocks refuses). Before the fix
 * the parameters followed '# ---- pair function (two-body) shape ----' and
 * the note read '# EMBF0 ZERO: ...'. */
TEST_CASE("DM_Embed starter target names its component section and reference",
          "[unified-dm][hpotarget]") {
  Temp temp;
  mlip::engines::write_hpotarget("refit.hpotarget",potential(),5);
  const auto text=slurp_file("refit.hpotarget");
  const auto header=text.find("# ---- TA component (DM_Embed) density and embedding shape ----");
  const auto marker=text.find("# COMPONENT_PARAMETER");
  REQUIRE(header!=std::string::npos);
  REQUIRE(marker!=std::string::npos);
  REQUIRE(header<marker);
  REQUIRE(text.substr(header+1,marker-header-1).find("# ----")==std::string::npos);
  REQUIRE(text.find("# REFERENCE ZERO: F(0) = 0 exactly")!=std::string::npos);
  REQUIRE(text.find("EMBF0")==std::string::npos);
}

/* Pins: for a knot density with CUTOFF TYPE NONE, whose functions must end
 * within the fixed RADIUS, the starter boxes of the knot centres end at
 * RADIUS (a parsed comment says so), and --retarget keeps that cap and
 * recovers the capped centre from its lower bound. Before the fix the box
 * of the largest knot crossed RADIUS (half of it infeasible by
 * construction: tadah hpo's support gate rejected every candidate there),
 * and --retarget had nothing to keep. */
TEST_CASE("DM_Embed starter boxes of knot centres end at a TYPE NONE radius",
          "[unified-dm][hpotarget][retarget]") {
  Temp temp;
  mlip::engines::write_hpotarget("refit.hpotarget",knot_potential(),5);
  auto lines=file_lines("refit.hpotarget");
  const auto text=slurp_file("refit.hpotarget");
  REQUIRE(text.find("# TA.DENSITY.CENTRES knot-support cap: RADIUS = 4")!=std::string::npos);
  auto boxes=optim_boxes(lines,"TA.DENSITY.CENTRES");
  REQUIRE(boxes.size()==3);
  for(const auto &b:boxes) {CAPTURE(b.index,b.lo,b.hi);REQUIRE(b.hi<=4.0);REQUIRE(b.lo<b.hi);}
  REQUIRE(boxes[2].hi==4.0);
  mlip::engines::RetargetOptions options;options.pct=10;
  mlip::engines::retarget_hpotarget_lines(lines,options);
  boxes=optim_boxes(lines,"TA.DENSITY.CENTRES");
  REQUIRE(boxes.size()==3);
  for(const auto &b:boxes) {CAPTURE(b.index,b.lo,b.hi);REQUIRE(b.hi<=4.0);}
  REQUIRE(boxes[2].hi==4.0);
  REQUIRE(boxes[2].lo==Approx(3.6).epsilon(1e-12));
  REQUIRE(boxes[1].lo==Approx(2.7).epsilon(1e-12));
  REQUIRE(boxes[1].hi==Approx(3.3).epsilon(1e-12));
}

/* Pins: --retarget rewrites the percentage of a component group's header
 * comment ('# TA.DENSITY.CENTRES  (+-5%, ...)') with its bounds, as it does
 * for a legacy key's group. Before the fix the group-header pattern allowed
 * no '.', so the bounds moved to +-10% under a header that still said
 * +-5%. */
TEST_CASE("DM_Embed retarget rewrites the percentage of component group headers",
          "[unified-dm][retarget]") {
  Temp temp;
  mlip::engines::write_hpotarget("refit.hpotarget",potential(),5);
  auto lines=file_lines("refit.hpotarget");
  mlip::engines::RetargetOptions options;options.pct=10;
  mlip::engines::retarget_hpotarget_lines(lines,options);
  bool seen=false;
  for(const auto &l:lines) if(l.rfind("# TA.",0)==0 && l.find("(+-")!=std::string::npos) {
    CAPTURE(l);seen=true;REQUIRE(l.find("(+-10%")!=std::string::npos);
  }
  REQUIRE(seen);
}

/* Pins: --retarget --rhoform and --rcutmb on a starter target of DM_Embed
 * components (COMPONENT_PARAMETER lines, no AMPGRIDMB or CGRIDMB group)
 * are refused, naming where the component keeps the setting (its FORM, the
 * <ID>.DENSITY.CUTOFF.RADIUS path). Before the fix --rhoform NONNEG added a
 * '# RHOFORM NONNEG: AMPGRIDMB bounds are non-negative' header over FREE
 * amplitude boxes with negative bounds, and --rcutmb printed the legacy
 * RCUTMB advice, both exiting 0. */
TEST_CASE("DM_Embed retarget refuses the density form and radius options of legacy keys",
          "[unified-dm][retarget]") {
  Temp temp;
  mlip::engines::write_hpotarget("refit.hpotarget",potential(),5);
  const auto lines=file_lines("refit.hpotarget");
  {
    auto copy=lines;mlip::engines::RetargetOptions options;options.pct=10;
    options.rhoform=models::RhoForm::NonNeg;
    REQUIRE_THROWS_WITH(mlip::engines::retarget_hpotarget_lines(copy,options),
                        Catch::Contains("--rhoform") && Catch::Contains("FORM") &&
                        Catch::Contains("TA.DENSITY.AMPLITUDES"));
  }
  {
    auto copy=lines;mlip::engines::RetargetOptions options;options.pct=10;
    options.rcut_override["MB"]=6.0;
    REQUIRE_THROWS_WITH(mlip::engines::retarget_hpotarget_lines(copy,options),
                        Catch::Contains("--rcutmb") &&
                        Catch::Contains("TA.DENSITY.CUTOFF.RADIUS"));
  }
}

/* Pins: --retarget --set addresses a component path of the starter target
 * ('TA.DENSITY.CENTRES(2)=1.8,2.2', the OPTIM key the writer emits for a
 * DM_Embed component) and sets that box. Before the fix the --set pattern
 * allowed no '.', so every component box was refused: "cannot parse --set
 * token". */
TEST_CASE("DM_Embed retarget --set addresses a component path",
          "[unified-dm][retarget]") {
  Temp temp;
  mlip::engines::write_hpotarget("refit.hpotarget",potential(),5);
  core::Context command;command.add("RETARGET","refit.hpotarget");
  command.add("SET","TA.DENSITY.CENTRES(2)=1.8,2.2");command.add("OUTFILE","set.hpotarget");
  REQUIRE(mlip::engines::run_refit_retarget(command)==0);
  const auto boxes=optim_boxes(file_lines("set.hpotarget"),"TA.DENSITY.CENTRES");
  REQUIRE(boxes.size()==2);
  REQUIRE(boxes[1].lo==1.8);
  REQUIRE(boxes[1].hi==2.2);
}
