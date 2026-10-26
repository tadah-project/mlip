#ifndef TADAH_MLIP_RETIRED_OPTIONS_H
#define TADAH_MLIP_RETIRED_OPTIONS_H

#include <tadah/core/blueprint_parser.h>

#include <string>

namespace tadah {
namespace mlip {

/** The option @p name (any letter case) of the blueprint task @p task
 *  ("DATA.balance") when the task retires it (the blueprint field
 *  `retired` of the task's key, as tadah data balance retires --rescale);
 *  nullptr when the task does not exist, has no such option or does not
 *  retire it, and for a key that is retired itself (EWEIGHT,
 *  WDBFILE_AUTO, UNCERTAINTY: core::retired_key_text() says why, for every
 *  command) unless the task retires the option with a `retired` of its
 *  own, another reason or conversion than the key's (--error of tadah
 *  predict, whose key ERROR is retired too). A key of the configuration of
 *  that name stays what it is for the other commands (RESCALE of tadah
 *  analysis bfunc). */
const core::BlueprintKey *retired_task_option(const std::string &task,
                                              const std::string &name);

/** "--rescale of tadah data balance is retired: <reason> Replaced by:
 *  <replacement>. <conversion>": the text of the retired option @p option
 *  of the task @p task, named by its command-line option and its short
 *  form ("--force (-F)"; its key in upper case without one). TaskFactory
 *  refuses the option with it. */
std::string retired_task_option_text(const std::string &task,
                                     const core::BlueprintKey &option);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_RETIRED_OPTIONS_H
