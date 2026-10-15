#include "catch2/catch.hpp"
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/refit_modes.h>
#include <tadah/mlip/engines/hpotarget_retarget.h>
#include <tadah/mlip/engines/analysis_bfunc_engine.h>
#include <tadah/mlip/task.h>
#include <tadah/models/descriptors/dm/embed_spec.h>
#include <tadah/models/cutoffs.h>
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
