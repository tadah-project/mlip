#!/usr/bin/env bash
#
# Usage:
#   ./update_core_symbols.sh <symbols_file> <models_directory>
#
#   <symbols_file>     = file listing one CORE symbol per line (e.g., core_symbols.txt)
#   <models_directory> = path to the MODELS code where references exist
#
# This script searches for each symbol in <symbols_file> as a separate word
# and replaces it with tadah::core::<symbol> throughout <models_directory>.
# It creates .bak backups for each modified file.
#
# After all replacements, it consolidates repeated "tadah::core::" sequences
# into a single instance, preventing lines like "tadah::core::tadah::core::Config".

set -e  # Exit on command errors (optional)

if [ $# -lt 2 ]; then
  echo "Usage: $0 <symbols_file> <models_directory>"
  exit 1
fi

SYMBOLS_FILE="$1"
MODELS_DIR="$2"

if [ ! -f "$SYMBOLS_FILE" ]; then
  echo "Error: symbols file '$SYMBOLS_FILE' does not exist."
  exit 1
fi

if [ ! -d "$MODELS_DIR" ]; then
  echo "Error: directory '$MODELS_DIR' does not exist."
  exit 1
fi

# Read all symbols from <symbols_file> line by line (avoiding 'mapfile' for portability).
SYMBOLS=()
while IFS= read -r symbol; do
  SYMBOLS+=("$symbol")
done < "$SYMBOLS_FILE"

echo "Starting replacement of CORE symbols in $MODELS_DIR..."

for SYMBOL in "${SYMBOLS[@]}"; do
  echo "Processing symbol: $SYMBOL"

  # Search references of this symbol (as a word) in the MODELS directory.
  # -r   = recursive 
  # -I   = ignore binary files
  # -n   = show line numbers
  # -E   = extended regex
  # The pattern \b is not perfectly reliable for C++ word boundaries in all grep versions,
  # so partial matches might still occur. Adjust or refine as needed.
  RESULTS=$(grep -rIn -I "\b$SYMBOL\b" "$MODELS_DIR" || true)

  if [ -z "$RESULTS" ]; then
    echo "  No references found for symbol: $SYMBOL"
    continue
  fi

  echo "  Found references:"
  echo "$RESULTS"

  # Replace each occurrence in each file
  while IFS= read -r LINE; do
    FILE_PATH=$(echo "$LINE" | cut -d':' -f1)
    
    # In-place replacement with .bak backup
    # The sed expression inserts "tadah::core::" before the symbol only if
    # it's not already preceded by "tadah::core::" (see negative lookbehind approach below).
    #
    # However, standard sed does not always support lookbehinds. If your sed doesn't,
    # skip the lookbehind but rely on the final cleanup pass (step below).
    #
    # For example, if your sed supports -E for extended regex, we might try:
    #   sed -E -i.bak 's#([^:])\b'"$SYMBOL"'\b#\1tadah::core::'"$SYMBOL"#g' "$FILE_PATH"
    #
    # In the generic case, do a straightforward replacement:
    
    sed -i.bak "s/\b$SYMBOL\b/tadah::core::$SYMBOL/g" "$FILE_PATH"
  
  done <<< "$RESULTS"

  echo "  Finished processing $SYMBOL."
done

# Final pass to collapse repeated "tadah::core::" sequences:
# e.g., tadah::core::tadah::core:: => tadah::core::
echo "Consolidating repeated 'tadah::core::' references..."
find "$MODELS_DIR" -type f -name "*.*" -exec sed -i.bak -E 's#(tadah::core::){2,}#tadah::core::#g' {} \;

echo "Done! Check the .bak files and run 'git diff' to ensure correctness."

