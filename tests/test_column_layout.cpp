// Catch2 v2 unit tests for ColumnLayout / ColumnBlock (Phase 2).
//
// We do not exercise a full DesignMatrix build here. DesignMatrix::build()
// checks the descriptor layout (dc.layout(context)) against DSIZE and the
// function's layout (f.phi_layout(), see column_block.h) against the
// matrix width; test_nonlinear_model_training.cpp tests both, with linear
// and non-linear functions. (Phase 2 compared the descriptor layout with
// the matrix width, which only a linear function's equals: every
// non-linear model was refused, and no test trained one.)
//
// The unit tests below verify the layout structure itself:
//   * total_cols == sum of block.col_count
//   * block_for_col(i) returns the correct block for every i
//   * block_index(label) finds blocks by name
//   * validate() catches gaps, overlaps, and width mismatches

#include "catch2/catch.hpp"
#include <tadah/mlip/design_matrix/column_block.h>

#include <stdexcept>

using namespace tadah::mlip;

static ColumnLayout make_blip_layout() {
  // Mirrors what DescriptorsCalc::layout() produces for a typical
  // 2-body + many-body blip configuration with BIAS=false:
  //   cols [0..6]   D2_Blip       (7 cols)
  //   cols [7..13]  DM_mBlipEnv   (7 cols)
  ColumnLayout L;
  ColumnBlock d2;
  d2.descriptor_label = "D2_Blip";
  d2.family           = "D2";
  d2.col_offset       = 0;
  d2.col_count        = 7;
  ColumnBlock dm;
  dm.descriptor_label = "DM_mBlipEnv";
  dm.family           = "DM";
  dm.col_offset       = 7;
  dm.col_count        = 7;
  L.blocks.push_back(d2);
  L.blocks.push_back(dm);
  return L;
}

TEST_CASE("ColumnLayout: total_cols sums block.col_count") {
  auto L = make_blip_layout();
  REQUIRE(L.total_cols() == 14);
}

TEST_CASE("ColumnLayout: block_for_col attributes every column to the right block") {
  auto L = make_blip_layout();
  for (std::size_t i = 0; i < 7; ++i) {
    REQUIRE(L.block_for_col(i).descriptor_label == "D2_Blip");
    REQUIRE(L.block_for_col(i).family == "D2");
  }
  for (std::size_t i = 7; i < 14; ++i) {
    REQUIRE(L.block_for_col(i).descriptor_label == "DM_mBlipEnv");
    REQUIRE(L.block_for_col(i).family == "DM");
  }
  REQUIRE_THROWS_AS(L.block_for_col(14), std::out_of_range);
}

TEST_CASE("ColumnLayout: block_index finds labels and returns -1 on miss") {
  auto L = make_blip_layout();
  REQUIRE(L.block_index("D2_Blip") == 0);
  REQUIRE(L.block_index("DM_mBlipEnv") == 1);
  REQUIRE(L.block_index("does_not_exist") == static_cast<std::size_t>(-1));
}

TEST_CASE("ColumnLayout::validate reports gaps and overlaps") {
  ColumnLayout L;
  // Gap between blocks: [0,5) then [7,9)
  ColumnBlock a;
  a.descriptor_label = "A"; a.col_offset = 0; a.col_count = 5;
  ColumnBlock b;
  b.descriptor_label = "B"; b.col_offset = 7; b.col_count = 2;
  L.blocks = {a, b};
  const std::string err = L.validate(9);
  REQUIRE_FALSE(err.empty());
  REQUIRE(err.find("offset") != std::string::npos);
}

TEST_CASE("ColumnLayout::validate detects width mismatch with materialised matrix") {
  auto L = make_blip_layout();
  // Layout total_cols == 14; pretend the matrix says 13.
  const std::string err = L.validate(13);
  REQUIRE_FALSE(err.empty());
  REQUIRE(err.find("disagrees") != std::string::npos);
}

TEST_CASE("ColumnLayout::validate accepts a contiguous, exact layout") {
  auto L = make_blip_layout();
  REQUIRE(L.validate(14).empty());
}

TEST_CASE("ColumnLayout: bias-leading layout matches DescriptorsCalc convention") {
  // BIAS=true case: bias column at offset 0, then descriptors.
  ColumnLayout L;
  ColumnBlock bias;
  bias.descriptor_label = "bias"; bias.family = "bias";
  bias.col_offset = 0; bias.col_count = 1;
  ColumnBlock d2;
  d2.descriptor_label = "D2_Blip"; d2.family = "D2";
  d2.col_offset = 1; d2.col_count = 7;
  L.blocks = {bias, d2};
  REQUIRE(L.total_cols() == 8);
  REQUIRE(L.validate(8).empty());
  REQUIRE(L.block_for_col(0).descriptor_label == "bias");
  REQUIRE(L.block_for_col(7).descriptor_label == "D2_Blip");
}
