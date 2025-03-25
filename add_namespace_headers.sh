#!/usr/bin/env bash

# Processes multiple .h files to wrap their contents in the tadah::mlip namespace.
# The namespace opening is positioned right after the last #include directive.
# The namespace closing is placed just before the last #endif directive.
# Usage example:  ./wrap_headers.sh file1.h file2.h ...

for file in "$@"; do
  # Temporary file to hold modifications
  tmpfile="$(mktemp)"

  # AWK script:
  # 1) Store each line in an array.
  # 2) Track the line number of the last #include and #endif.
  # 3) Print contents, injecting namespace lines at the appropriate positions.
  awk '
    BEGIN {
      lastIncludeLine = 0
      lastEndIfLine   = 0
    }
    {
      lines[NR] = $0
      if ($0 ~ /^#include/) {
        lastIncludeLine = NR
      }
      if ($0 ~ /^#endif/) {
        lastEndIfLine = NR
      }
    }
    END {
      for (i = 1; i <= NR; i++) {
        if (i == lastIncludeLine) {
          # Insert namespace opening after the last #include
          print lines[i]
          print "namespace tadah {"
          print "namespace mlip {"
        } else if (i == lastEndIfLine) {
          # Insert namespace closing before the last #endif
          print "} // namespace mlip"
          print "} // namespace tadah"
          print lines[i]
        } else {
          print lines[i]
        }
      }
    }
  ' "$file" > "$tmpfile"

  # Move updated content back to original file
  mv "$tmpfile" "$file"
done

