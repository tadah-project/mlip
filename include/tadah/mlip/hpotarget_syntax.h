#ifndef TADAH_MLIP_HPOTARGET_SYNTAX_H
#define TADAH_MLIP_HPOTARGET_SYNTAX_H

// The syntax of a target file (the --hpotarget file of tadah hpo) as tadah
// hpo reads it: its logical lines (comments cut, continued lines joined)
// and the grammar of an OPTIM line,
//     OPTIM <KEY> (indices) <low> <high> [LIN | LOG | ASINH <c>]
//     OPTIM <KEY> (indices) AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>]
//                                [GROUP_FLOOR <r>] [LIN | LOG | ASINH <c>]
// without the training configuration. Every refusal is tadah hpo's text.
// The reader of the OPTIM lines against a training configuration, which
// tadah hpo and tadah analysis check_bounds use, is hpotarget_optim.h;
// tadah refit --retarget reads its lines with read_optim_syntax().
//
// The words below are those of the target-file registry of HPO (its
// lexing table and the options of the OPTIM lines); HPO's tests hold them
// to it.

#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tadah {
namespace mlip {

/** The comment character of a target file: a line is cut there. */
inline constexpr char kTargetComment = '#';
/** A line that ends with it continues on the next line. */
inline constexpr char kTargetContinuation = '\\';
/** The refusal of a file whose last line ends with kTargetContinuation. */
extern const char* const kTargetUnfinishedLineMsg;

/** \brief A logical line of a target file, as tadah hpo reads it. */
struct TargetLine {
  std::size_t first = 0;  ///< its first physical line, 1-based
  std::size_t last = 0;   ///< its last physical line, 1-based
  /** The text: cut at kTargetComment, trailing blanks removed, continued
   *  lines joined (each cut and trimmed, the backslash removed). */
  std::string text;
  std::string key;        ///< the first token, in the case written
  /** The file ends after a backslash: @c text keeps it, and tadah hpo
   *  refuses the line (kTargetUnfinishedLineMsg). */
  bool unfinished = false;
};

/** \brief The logical lines of @p in, in order, as tadah hpo reads them: a
 *  line empty once cut and trimmed is skipped; a line that ends with a
 *  backslash takes the next physical line (a blank or comment line
 *  included, which ends it); an unfinished last line is the last entry. */
std::vector<TargetLine> target_logical_lines(std::istream& in);

/** \brief A refusal of a line of a target file: what() is the reason (tadah
 *  hpo's text), @c line the last physical line of the logical line and
 *  @c text its text. */
class TargetLineError : public std::runtime_error {
 public:
  TargetLineError(std::size_t line, std::string text, const std::string& what);
  std::size_t line;
  std::string text;
};

/** \brief " (too small: ...)" when @p tok is in the number grammar and its
 *  magnitude is below the smallest normal double (a finite decimal that
 *  core::parse_decimal_normal refuses, as tadah hpo refuses it); "" else. */
std::string number_underflow_note(const std::string& tok);

/** \brief The text between the parentheses that follow the key of an OPTIM
 *  line or of an indexed metric line of tadah hpo (blanks before the
 *  opening parenthesis skipped); refuses a missing opening or closing
 *  parenthesis, a second opening one, empty parentheses and a character
 *  other than a blank right after the closing one. @p what names the line
 *  in a refusal ("OPTIM <KEY>" or the metric key). */
std::string read_index_selector(std::istringstream& iss,
                                const std::string& what);

/** \brief The index list of an OPTIM line and the tokens after it. */
struct OptimSelector {
  /** core::parse_indices_strict of the list: 1-based, in the order
   *  written. */
  std::vector<std::size_t> indices;
  std::string index_text;            ///< the text between the parentheses
  std::vector<std::string> tokens;   ///< the tokens after the list
};

/** \brief Reads the index list of an OPTIM line and the tokens after it
 *  from @p rest (the line after its key): a token holding a parenthesis,
 *  fewer than one token after the list, an index list core::
 *  parse_indices_strict refuses and the index 0 are refused, as tadah hpo
 *  refuses them. @p key is the key, upper-cased. */
OptimSelector read_optim_selector(std::istringstream& rest,
                                  const std::string& key);

/** \brief The shape tag of an OPTIM line (the search coordinate). */
enum class OptimShape : std::uint8_t { Linear, Log, Asinh };

/** \brief The box of an OPTIM line as written: an explicit box or AUTO with
 *  its options, and the shape tag. */
struct OptimBoxSyntax {
  bool auto_box = false;      ///< AUTO
  // An explicit box: the bounds, and as written.
  double low = 0.0, high = 0.0;
  std::string low_tok, high_tok;
  // AUTO: RATIO <f> (geometric) or FRAC <p> (linear), FLOOR, GROUP_FLOOR.
  bool mode_set = false;      ///< RATIO or FRAC given
  bool geometric = false;     ///< RATIO given
  double factor = 0.0;        ///< the RATIO or FRAC value
  std::string factor_tok;     ///< as written
  double floor = 0.0;         ///< FLOOR
  double group_floor = 0.0;   ///< GROUP_FLOOR
  std::string group_floor_tok;
  bool floor_set = false;     ///< FLOOR or GROUP_FLOOR given
  // The shape tag: LIN (the default), LOG, ASINH <c>.
  OptimShape shape = OptimShape::Linear;
  double c = 1.0;             ///< the c of ASINH <c>
  std::string c_tok;          ///< as written
  bool tagged = false;        ///< the line writes LIN, LOG or ASINH <c>
};

/** \brief The box of an OPTIM line from @p tokens, the tokens after its
 *  index list (OptimSelector::tokens), without the training configuration:
 *  AUTO with its options and tag (each option once, a value each, RATIO or
 *  FRAC, RATIO > 1, FRAC, FLOOR and GROUP_FLOOR >= 0), or two bounds with
 *  a tag, the high one above the low one; ASINH takes c > 0. Every number
 *  is in the number grammar (core::parse_decimal_normal). Throws tadah
 *  hpo's text. @p key is the key, upper-cased. */
OptimBoxSyntax read_optim_box(std::vector<std::string> tokens,
                              const std::string& key);

/** \brief An OPTIM line read without the training configuration. */
struct OptimSyntax {
  std::string key;              ///< upper-cased
  std::string key_as_written;
  OptimSelector selector;
  OptimBoxSyntax box;
};

/** \brief The OPTIM line whose text after the word OPTIM is @p rest, read
 *  as tadah hpo reads its syntax: the key, read_optim_selector() and
 *  read_optim_box(). Throws tadah hpo's text. */
OptimSyntax read_optim_syntax(std::istringstream& rest);

/** The options of an AUTO box: RATIO, FRAC, FLOOR, GROUP_FLOOR. */
const std::array<std::string_view, 4>& optim_auto_options();
/** The shape tags of an OPTIM line: LIN, LOG, ASINH. */
const std::array<std::string_view, 3>& optim_shape_tags();
/** The word of an AUTO box. */
inline constexpr std::string_view kOptimAutoWord = "AUTO";
/** The distance cutoffs of the training configuration, which take no AUTO
 *  box (each index: a joined descriptor has a cutoff for each of its
 *  descriptors). A key is listed by what it is, not by its name: RCUTENV,
 *  the width of the envelope of DM_mBlipEnv in units of the density, takes
 *  an AUTO box. */
inline constexpr std::array<std::string_view, 2> kOptimAutoExcludedKeys = {
    "RCUT2B", "RCUTMB"};

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_HPOTARGET_SYNTAX_H
