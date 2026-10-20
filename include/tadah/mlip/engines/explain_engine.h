#ifndef TADAH_MLIP_ENGINES_EXPLAIN_ENGINE_H
#define TADAH_MLIP_ENGINES_EXPLAIN_ENGINE_H

#include <tadah/core/blueprint_parser.h>
#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>

#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Command for explaining other commands, options and configuration
 *        keys.
 *
 * OPTION is `<command>`, `<command>.<option>` or
 * `<command>.<subcommand>.<option>` (a subcommand is also written
 * `<command>.<subcommand>`), or `<KEY>`, each name in any letter case.
 * run() prints the long description of the command or subcommand it names
 * (followed, for one this program was built without, by a blank line and
 * EngineFactory::notBuiltMessage with a full stop) and the text of the
 * option (explain_key_text). A last name that is no
 * command, subcommand or option of its command is looked up among the
 * configuration keys (ALLOWED_KEYS), so `tadah explain SETFL` and `tadah
 * explain refit.rhoform` print the text of a key that is no option of a
 * command; a command's own subcommand or option comes first (its text may
 * differ: data.MERGE is the merge subcommand, not the MERGE key). A single
 * name is also asked of the application's explain hook (set_explain_hook):
 * tadah's prints the optimisers of `tadah hpo` (ALGO, LIB:ALGO). A name
 * that is no command and no configuration key prints the hook's text; one
 * that is either prints that text, then, when the hook knows the name too,
 * a blank line, "<NAME> also names <what the hook's names are>:" and the
 * hook's text (`tadah explain RANDOM`: the data key RANDOM, then "RANDOM
 * also names an optimiser of `tadah hpo`:" and TADAH:RANDOM's help). It
 * throws
 * std::runtime_error, printing nothing, when the option names none of
 * these: "tadah explain: no command or configuration key 'bogus'", "tadah
 * explain: no command 'bogus'" (bogus.key: the first name of a dotted
 * option is a command), "tadah explain: data has no subcommand 'nosub'",
 * "tadah explain: refit has no option 'nosuchkey', and it is no
 * configuration key", "tadah explain: data has no subcommand or option
 * 'nosuch', and it is no configuration key" (a command with subcommands),
 * "tadah explain: 'a.b.c.d' is not <command>, <command>.<option> or
 * <command>.<subcommand>.<option>".
 */
class ExplainEngine: public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
};

/**
 * @brief A source of `tadah explain` texts beyond the commands and the
 *        configuration keys.
 *
 * Called with a single name, as the user wrote it, after the text of a
 * command or configuration key of that name if there is one. It writes the
 * name's text to @p out and returns true, or returns false when the name
 * is not one of its own; it may throw std::runtime_error, whose message is
 * the command's error (a name it knows of but cannot explain). The
 * application sets it, so MLIP needs no module it serves: tadah sets the
 * optimiser help of HPO, or with HPO not built one that says so.
 */
using ExplainHook =
    std::function<bool(const std::string &name, std::ostream &out)>;

/**
 * @brief Sets the hook ExplainEngine::run asks (see ExplainHook), replacing
 *        an earlier one; an empty function removes it.
 *
 * @p what says what the hook's names are, for the line that introduces its
 * text after a command's or key's of the same name: "an optimiser of
 * `tadah hpo`" gives "RANDOM also names an optimiser of `tadah hpo`:".
 */
void set_explain_hook(ExplainHook hook, std::string what);

/**
 * @brief The lines of a blueprint long description as `tadah explain`
 *        prints them.
 *
 * The text is RST (the configuration-keys page renders it); the terminal
 * gets it as text, as the CLI page of the documentation shows it
 * (generate_cli.py's literal_text_lines): the indentation all its lines
 * share is removed and the rest kept, so a layout stays nested; the blank
 * lines around it are left out and those inside it are empty; and the RST
 * marker of a literal block, '::' closing a paragraph, is shown as RST
 * shows it: 'text::' as 'text:', 'text ::' as 'text', and a paragraph of
 * '::' alone, with the blank line after it, not at all. Inside the literal
 * block it opens (the lines indented more than that paragraph, and blank
 * lines), and inside a line, '::' is text. Outside a literal block an
 * inline role is shown as the documentation shows it, by its text:
 * ':ref:`Title <label>`' as 'Title', ':ref:`LAMBDA`' as 'LAMBDA',
 * ':math:`x^2`' as 'x^2' (the CLI page renders an option's text in a
 * parsed-literal block, which reads roles). Other RST markup is printed as
 * written.
 */
std::vector<std::string> explain_text_lines(const std::string &text);

/**
 * @brief What `tadah explain <command>.<key>` (or `<KEY>`) prints for
 *        @p key.
 *
 * Its long description, or its description when it has none, as
 * explain_text_lines renders it, one line each; for a retired key
 * (BlueprintKey::isRetired()) then a blank line and the lines "RETIRED:
 * <reason>", "REPLACED BY: <replacement>" and "CONVERSION: <conversion>";
 * then, when it has examples, a blank line, "EXAMPLES: " and each line of
 * each example (an example of several lines is one value per line)
 * indented by two spaces.
 */
std::string explain_key_text(const core::BlueprintKey &key);

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_EXPLAIN_ENGINE_H
