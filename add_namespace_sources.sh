#!/usr/bin/env bash

# Processes multiple .h files to wrap their contents in the tadah::mlip namespace.
# The namespace opening is positioned right after the last #include directive.
# The namespace closing is placed at the end of the file.
# Usage example:  ./wrap_headers.sh file1.h file2.h ...

for file in "$@"; do
  # Temporary file for transformed output
  tmpfile="$(mktemp)"

  # Use AWK to:
  # 1) Identify the line of the last #include directive.
  # 2) Store all lines in an array.
  # 3) Print everything, inserting the namespace opening right after the last #include.
  # 4) Append the namespace closing brace at the very end.
  awk '
    BEGIN {
      lastIncludeLine = 0
    }
    {
      lines[NR] = $0
      if ($0 ~ /^#include/) {
        lastIncludeLine = NR
      }
    }
    END {
      for (i = 1; i <= NR; i++) {
        if (i == lastIncludeLine) {
          print lines[i]
          print "namespace tadah {"
          print "namespace mlip {"
        } else {
          print lines[i]
        }
      }
      print "} // namespace mlip"
      print "} // namespace tadah"
    }
  ' "$file" > "$tmpfile"

  # Overwrite the original file with transformed content
  mv "$tmpfile" "$file"
done

