#ifndef TADAH_MLIP_KEY_PROVENANCE_H
#define TADAH_MLIP_KEY_PROVENANCE_H

#include <tadah/core/context.h>
#include <tadah/core/iuser_config_parser.h>  // core::SkippedKeyLine

#include <string>
#include <vector>

namespace tadah {
namespace mlip {

/**
 * @brief Where one key of a task's context came from.
 */
struct KeyOrigin {
  enum class Kind {
    NotRecorded,  ///< the key is in the context, but no source is known
    CommandLine,  ///< a command-line option (flag names it)
    ConfigFile,   ///< the file given with -c/--config (or one it includes)
    TaskFile,     ///< the file given with --task (or one it includes)
    Default       ///< the blueprint default, added because nothing set it
  };

  Kind kind = Kind::NotRecorded;
  /** CommandLine: the option as the user can write it, e.g. "-F/--force". */
  std::string flag;
  /** ConfigFile/TaskFile: that file. CommandLine: the configuration file
   *  whose value of the key the option replaced (empty: none). */
  std::string file;
  /** CommandLine: the configuration file's values the option replaced
   *  (empty when the file did not set the key). */
  std::vector<std::string> overridden;

  /**
   * @brief The source in words, for a run journal:
   *   "command line -F/--force; overrides FORCE false in c.tadah",
   *   "command line -F/--force", "configuration file c.tadah",
   *   "task file run.task", "default" or "source not recorded".
   */
  std::string describe(const std::string &key) const;
};

/**
 * @brief Where the keys of a task's context came from: the command line,
 *        the configuration or task file, or the blueprint defaults.
 *
 * TaskFactory fills it while it merges the command-line options, the file
 * and the defaults into the task's context, so that a run can say, for
 * example, that FORCE true came from -F and replaced FORCE false in the
 * configuration file. It answers for any key on demand; a key that none of
 * the three set is KeyOrigin::Kind::NotRecorded.
 */
class KeyProvenance {
public:
  KeyProvenance();

  /**
   * @param command_line The keys given as command-line options (upper case).
   * @param file         The keys read from the configuration or task file.
   * @param file_path    That file, as given ("" when there is none).
   * @param file_kind    ConfigFile or TaskFile.
   * @param defaults     The keys added as blueprint defaults (upper case).
   * @param skipped      The key lines of the file, or of a file it
   *                     includes, that its parse did not read (TaskFactory:
   *                     a potential's lines of the keys of its training
   *                     record but DBFILE, potential_line_not_read).
   */
  KeyProvenance(core::Context command_line, core::Context file,
                std::string file_path, KeyOrigin::Kind file_kind,
                std::vector<std::string> defaults,
                std::vector<core::SkippedKeyLine> skipped = {});

  KeyProvenance(const KeyProvenance &);
  KeyProvenance(KeyProvenance &&);
  KeyProvenance &operator=(const KeyProvenance &);
  KeyProvenance &operator=(KeyProvenance &&);
  ~KeyProvenance();

  /** The origin of @p key (any letter case) in the task's context. */
  KeyOrigin origin(const std::string &key) const;

  /** The configuration or task file, as given ("" when there is none). */
  const std::string &file_path() const { return file_path_; }
  /** ConfigFile (-c/--config) or TaskFile (--task). */
  KeyOrigin::Kind file_kind() const { return file_kind_; }
  /** The key lines of the file, or of a file it includes with CONFIG, that
   *  its parse did not read, in the order read: for a task of TaskFactory,
   *  a potential's lines of the keys of keys_not_read_from_a_potential()
   *  -- those that prepared the training data, UNCERTAINTY and the run
   *  settings VALIDATION, HPOTARGET, VERBOSE, NUMERIC, OUTFILE, AUDIT,
   *  AUDIT_SAMPLE and HEALTH_LOG (potential_line_not_read,
   *  training_record.h).
   *  Their keys are not in the task's context unless another line or the
   *  command line gives them. */
  const std::vector<core::SkippedKeyLine> &skipped_key_lines() const {
    return skipped_;
  }

private:
  core::Context command_line_;
  core::Context file_;
  std::string file_path_;
  KeyOrigin::Kind file_kind_ = KeyOrigin::Kind::ConfigFile;
  std::vector<core::SkippedKeyLine> skipped_;
  /* Sorted and unique. (A std::set member here made GCC 13.3's LTO stop
   * with an internal compiler error while linking test_mlip.) */
  std::vector<std::string> defaults_;
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_KEY_PROVENANCE_H
