#ifndef TADAH_MLIP_DESIGN_MATRIX_COLUMN_BLOCK_H
#define TADAH_MLIP_DESIGN_MATRIX_COLUMN_BLOCK_H

// ColumnBlock / ColumnLayout — record which columns of a matrix belong to
// which part of the model. Two layouts describe a design-matrix build:
//
//  - the descriptor layout (DescriptorsCalc::layout()): the bias column and
//    the D2 and DM blocks of the descriptor, DSIZE columns in all;
//  - the design-matrix layout (DM_Function_Base::phi_layout(), which
//    DesignMatrix::build() stores): for a linear function (BF_Linear,
//    Kern_Linear) the descriptor layout itself, one column per descriptor
//    component; for a non-linear function one block that it owns over its
//    whole width (a kernel's SBASIS basis-vector columns, family "kernel";
//    BF_Polynomial2's D(D+1)/2 products of descriptor components, family
//    "BF"), since its columns do not belong to single descriptor blocks.
//
// Which columns the regression leaves out of the ridge penalty (the
// intercepts) is declared by the function, Function_Base::
// intercept_columns() in MODELS.
//
// HPO's per-iteration health monitor (HEALTH_LOG) slices the weights of
// every evaluation's model by the design-matrix layout, which it computes
// as build() does: phi_layout() of the descriptor layout.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tadah {
namespace mlip {

/** \brief A contiguous range of columns in the design matrix that belong
 *  to a single descriptor (or to the bias column).
 */
struct ColumnBlock {
  std::string descriptor_label{};   ///< e.g. "D2_Blip", "DM_mBlipEnv", "bias", "Kern_RBF"
  std::string family{};             ///< "D2" | "DM" | "bias" | "kernel" | "BF"
  std::size_t col_offset{0};
  std::size_t col_count{0};
  /** Optional: per-element-pair sub-blocks, for multi-element systems
   *  where a parameter has PairScope::PerElementPair. Nothing fills it
   *  yet: it is empty. */
  std::vector<std::pair<std::string, std::pair<std::size_t, std::size_t>>>
      sub_blocks{};
};

/** \brief Whole-matrix column layout: a list of ColumnBlocks in
 *  ascending col_offset order, plus convenience queries.
 */
struct ColumnLayout {
  std::vector<ColumnBlock> blocks{};

  /** Sum of `col_count` across all blocks. Equal to the design matrix
   *  column count when the layout has been correctly built. */
  std::size_t total_cols() const {
    std::size_t s = 0;
    for (const auto& b : blocks) s += b.col_count;
    return s;
  }

  /** Locate the block that owns column index `c`. Throws if no block
   *  contains it (e.g. layout out of date or `c` past total_cols).
   */
  const ColumnBlock& block_for_col(std::size_t c) const {
    for (const auto& b : blocks) {
      if (c >= b.col_offset && c < b.col_offset + b.col_count) {
        return b;
      }
    }
    throw std::out_of_range(
        "ColumnLayout::block_for_col: index " + std::to_string(c) +
        " is not covered by any block");
  }

  /** Return the index of the block with the given label, or
   *  size_t(-1) if not present.
   */
  std::size_t block_index(const std::string& label) const {
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      if (blocks[i].descriptor_label == label) return i;
    }
    return static_cast<std::size_t>(-1);
  }

  /** Validate basic invariants — used by DesignMatrix::build() to fail
   *  fast if the descriptor layout disagrees with DSIZE, or the
   *  design-matrix layout with the materialised matrix width. `what`
   *  names the expected width in the message ("DSIZE", "design-matrix
   *  width"). Returns an empty string on success, or a diagnostic
   *  message on failure (so the caller can format with file:line
   *  context).
   */
  std::string validate(std::size_t expected_total_cols,
                       const std::string& what = "design-matrix width") const {
    std::size_t expected = 0;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      const auto& b = blocks[i];
      if (b.col_offset != expected) {
        return "ColumnLayout: block[" + std::to_string(i) + "] (\"" +
               b.descriptor_label + "\") has offset " +
               std::to_string(b.col_offset) + ", expected " +
               std::to_string(expected);
      }
      expected = b.col_offset + b.col_count;
    }
    if (expected != expected_total_cols) {
      return "ColumnLayout: total_cols=" + std::to_string(expected) +
             " disagrees with " + what + "=" +
             std::to_string(expected_total_cols);
    }
    return {};
  }
};

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_DESIGN_MATRIX_COLUMN_BLOCK_H
