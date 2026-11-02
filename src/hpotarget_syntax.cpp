#include <tadah/mlip/hpotarget_syntax.h>

#include <tadah/core/number_format.h>
#include <tadah/core/utils.h>

#include <algorithm>
#include <cctype>
#include <utility>

namespace tadah {
namespace mlip {

const char* const kTargetUnfinishedLineMsg =
    "the line ends with \\, which continues it on the next line, and the "
    "file has no next line";

TargetLineError::TargetLineError(std::size_t l, std::string t,
                                 const std::string& what)
    : std::runtime_error(what), line(l), text(std::move(t)) {}

std::vector<TargetLine> target_logical_lines(std::istream& in) {
  auto strip_comment = [](std::string& s) {
    auto p = s.find(kTargetComment);
    if (p != std::string::npos) s.resize(p);
  };
  auto rtrim = [](std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         [](unsigned char c) { return !std::isspace(c); })
                .base(),
            s.end());
  };
  std::vector<TargetLine> out;
  std::string line;
  std::size_t nline = 0;
  while (std::getline(in, line)) {
    ++nline;
    strip_comment(line);
    rtrim(line);
    if (line.empty()) continue;
    TargetLine l;
    l.first = nline;
    while (!line.empty() && line.back() == kTargetContinuation) {
      line.pop_back();
      std::string cont;
      /* A blank or comment line after the backslash is joined: only the
       * end of the file leaves the line without a next one. */
      if (!std::getline(in, cont)) {
        line += kTargetContinuation;
        l.unfinished = true;
        break;
      }
      ++nline;
      strip_comment(cont);
      rtrim(cont);
      line += cont;
    }
    l.last = nline;
    l.text = line;
    std::istringstream ks(line);
    ks >> l.key;
    if (l.key.empty()) continue;
    out.push_back(std::move(l));
  }
  return out;
}

std::string number_underflow_note(const std::string& tok) {
  double v = 0.0;
  if (!core::parse_decimal_finite(tok, v) || core::parse_decimal_normal(tok, v))
    return "";
  return " (too small: its magnitude is below the smallest normal double, "
         "2.2250738585072014e-308)";
}

// OPTIM and indexed PC lines share the same parenthesized selector grammar.
// @p what names the line in a refusal: "OPTIM <KEY>" or the metric key.
std::string read_index_selector(std::istringstream& iss, const std::string& what) {
    iss >> std::ws;
    if (iss.get() != '(')
        throw std::runtime_error(what + ": the key is followed by its indices "
                                 "in parentheses, such as " + what + " (1)");
    std::string indices;
    if (!std::getline(iss, indices, ')') || iss.eof())
        throw std::runtime_error(what + ": the indices have no closing "
                                 "parenthesis");
    if (indices.find('(') != std::string::npos)
        throw std::runtime_error(what + ": the indices hold another opening "
                                 "parenthesis");
    if (core::trim(indices).empty())
        throw std::runtime_error(what + ": the parentheses hold no index");
    const int next = iss.peek();
    if (next != std::char_traits<char>::eof() && !std::isspace(static_cast<unsigned char>(next)))
        throw std::runtime_error(what + ": a space follows the closing "
                                 "parenthesis of the indices");
    return indices;
}

OptimSelector read_optim_selector(std::istringstream& iss,
                                  const std::string& k) {
  const std::string what = "OPTIM " + k;
  /* The (indices) and the tokens after them. */
  std::vector<std::string> tokens;
  {
    const auto indices = read_index_selector(iss, what);
    std::string token;
    while (iss >> token) {
      if (token.find_first_of("()") != std::string::npos)
        throw std::runtime_error(what + ": the line takes one list of "
                                 "indices in parentheses (unexpected '" +
                                 token + "')");
      tokens.push_back(token);
    }
    tokens.push_back(indices);
  }

  /* At least the (indices) group plus one token (AUTO, or <low> <high>);
   * the exact grammar is checked by read_optim_box. */
  if (tokens.size() < 2) {
    throw std::runtime_error(
      "OPTIM " + k + ": expected `(indices) <low> <high>` or `(indices) AUTO "
      "...` (got " + core::to_string(tokens.size() - (tokens.empty() ? 0 : 1)) +
      " token(s))");
  }

  OptimSelector sel;
  try {
    sel.indices = core::parse_indices_strict(tokens.back());
    if (std::find(sel.indices.begin(), sel.indices.end(), 0) != sel.indices.end())
      throw std::runtime_error("OPTIM indices start at 1");
  } catch (const std::exception& e) {
    throw std::runtime_error("OPTIM " + k + ": bad index list (" +
                             tokens.back() + "): " + e.what());
  }
  sel.index_text = tokens.back();
  tokens.pop_back();
  sel.tokens = std::move(tokens);
  return sel;
}

OptimBoxSyntax read_optim_box(std::vector<std::string> tokens,
                              const std::string& k) {
  /* Strict grammar: OPTIM <KEY> (indices) <low> <high>. Anything else
   * used to be silently ignored (the bounds were popped from the END of
   * the token list), which swallowed typos and stray tokens. */
  /* Two forms after the (indices) group has been extracted:
   *   <low> <high> [LIN | LOG | ASINH <c>]                  explicit box
   *   AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>] [LIN | LOG | ASINH <c>]
   *                                                        seed-relative box
   * AUTO derives the box from the CONFIG seed of each selected index
   * (OptimReader, hpotarget_optim.h). */
  OptimBoxSyntax box;
  auto upper = [](std::string t) {
    for (auto& ch : t) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    return t;
  };
  /* Every number on an OPTIM line: a finite number of the number grammar
   * (core::parse_decimal_normal, tadah hpo's parse_number). */
  auto parse_num = [&](const std::string& tok, const char* what) {
    double v = 0.0;
    if (!core::parse_decimal_normal(tok, v))
      throw std::runtime_error("OPTIM " + k + ": " + what +
                               " must be a finite number (got '" + tok + "')" +
                               number_underflow_note(tok));
    return v;
  };
  /* trailing shape tag (both forms) */
  auto take_tag = [&](std::vector<std::string>& toks) {
    if (toks.size() >= 2 && upper(toks[toks.size() - 2]) == "ASINH") {
      box.shape = OptimShape::Asinh;
      box.c = parse_num(toks.back(), "ASINH <c>");
      if (!(box.c > 0.0))
        throw std::runtime_error("OPTIM " + k + ": ASINH requires a positive "
                                 "linear scale c (got " + toks.back() + ")");
      box.c_tok = toks.back();
      toks.pop_back(); toks.pop_back();
      box.tagged = true;
      return;
    }
    if (!toks.empty()) {
      const std::string tag = upper(toks.back());
      if (tag == "LIN") {
        box.shape = OptimShape::Linear; toks.pop_back(); box.tagged = true;
      } else if (tag == "LOG") {
        box.shape = OptimShape::Log; toks.pop_back(); box.tagged = true;
      }
      else if (tag == "ASINH")
        throw std::runtime_error("OPTIM " + k + ": ASINH needs one argument <c> "
                                 "(the linear scale)");
    }
  };
  const std::string grammar =
    " -- the grammar is OPTIM <KEY> (indices) <low> <high> [LIN | LOG | ASINH <c>]"
    " or OPTIM <KEY> (indices) AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>]"
    " [LIN | LOG | ASINH <c>]";

  box.auto_box = !tokens.empty() && upper(tokens.front()) == "AUTO";
  bool floor_seen = false, group_floor_seen = false;
  bool ratio_seen = false, frac_seen = false;
  if (box.auto_box) {
    tokens.erase(tokens.begin());
    take_tag(tokens);
    /* The name of each option first: a word that is no AUTO option is
     * named as such whatever follows it. */
    for (std::size_t t = 0; t < tokens.size(); t += 2) {
      const std::string opt = upper(tokens[t]);
      if (opt != "RATIO" && opt != "FRAC" && opt != "FLOOR" &&
          opt != "GROUP_FLOOR")
        throw std::runtime_error("OPTIM " + k + ": '" + tokens[t] + "' is not "
                                 "an AUTO option (RATIO, FRAC, FLOOR or "
                                 "GROUP_FLOOR)" + grammar);
      if (t + 1 >= tokens.size())
        throw std::runtime_error("OPTIM " + k + ": AUTO option '" + tokens[t] +
                                 "' needs a value" + grammar);
      /* Each is given once: a second replaced the first's value. */
      bool& seen = opt == "RATIO" ? ratio_seen
                 : opt == "FRAC"  ? frac_seen
                 : opt == "FLOOR" ? floor_seen
                                  : group_floor_seen;
      if (seen)
        throw std::runtime_error("OPTIM " + k + ": AUTO " + opt + " is given "
                                 "twice on this line; each option is given once");
      seen = true;
      if ((opt == "RATIO" || opt == "FRAC") && box.mode_set)
        throw std::runtime_error("OPTIM " + k + ": AUTO takes RATIO or FRAC, "
                                 "not both");
      const double v = parse_num(tokens[t + 1], ("AUTO " + opt).c_str());
      if (opt == "RATIO") {
        if (!(v > 1.0)) throw std::runtime_error("OPTIM " + k + ": AUTO RATIO must be > 1 (got " + tokens[t + 1] + ")");
        box.geometric = true; box.factor = v; box.mode_set = true;
        box.factor_tok = tokens[t + 1];
      } else if (opt == "FRAC") {
        if (!(v >= 0.0))
          throw std::runtime_error("OPTIM " + k + ": AUTO FRAC must be >= 0 (got " +
                                   tokens[t + 1] + "); FRAC 0 with FLOOR or "
                                   "GROUP_FLOOR gives a constant half-width");
        box.geometric = false; box.factor = v; box.mode_set = true;
        box.factor_tok = tokens[t + 1];
      } else {
        if (v < 0.0)
          throw std::runtime_error("OPTIM " + k + ": AUTO " + opt +
                                   " must be >= 0 (got " + tokens[t + 1] + ")");
        if (opt == "FLOOR") {
          box.floor = v;
        } else {
          box.group_floor = v;
          box.group_floor_tok = tokens[t + 1];
        }
        box.floor_set = true;
      }
    }
  } else {
    take_tag(tokens);
    if (tokens.size() != 2) {
      throw std::runtime_error(
        "OPTIM " + k + ": unexpected token(s) '" + core::join(tokens) + "'" +
        grammar);
    }
    box.low  = parse_num(tokens[0], "<low>");
    box.high = parse_num(tokens[1], "<high>");
    box.low_tok = tokens[0];
    box.high_tok = tokens[1];
    /* An empty or reversed box, at the line and with the bounds as
     * written. */
    if (!(box.high > box.low))
      throw std::runtime_error("OPTIM " + k + ": <high> " + box.high_tok +
                               " is not greater than <low> " + box.low_tok);
  }
  return box;
}

OptimSyntax read_optim_syntax(std::istringstream& iss) {
  OptimSyntax out;
  if (!(iss >> out.key_as_written)) {
    throw std::runtime_error("Failed to read key from input.");
  }
  /* A configuration key, and configuration keys are case-insensitive. */
  out.key = core::to_upper(out.key_as_written);
  out.selector = read_optim_selector(iss, out.key);
  out.box = read_optim_box(out.selector.tokens, out.key);
  return out;
}

const std::array<std::string_view, 4>& optim_auto_options() {
  static const std::array<std::string_view, 4> options = {
      "RATIO", "FRAC", "FLOOR", "GROUP_FLOOR"};
  return options;
}

const std::array<std::string_view, 3>& optim_shape_tags() {
  static const std::array<std::string_view, 3> tags = {"LIN", "LOG", "ASINH"};
  return tags;
}

}  // namespace mlip
}  // namespace tadah
