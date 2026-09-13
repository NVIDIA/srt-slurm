#!/usr/bin/env bash
# Enforce conventional-commit PR titles so release.yaml can pick the semver bump.
#   check_pr_title.sh "<title>"
set -euo pipefail
title="${1:-}"
types="feat|fix|docs|refactor|perf|test|chore|ci|build|revert"
# Regexes live in variables: bash's [[ =~ ]] parser trips over ")" inside a bracket expression.
re_conventional="^($types)(\([A-Za-z0-9._/-]+\))?!?: .+"
re_github_revert='^Revert ".+"$'
if [[ "$title" =~ $re_conventional ]] || [[ "$title" =~ $re_github_revert ]]; then
  exit 0
fi
cat >&2 <<MSG
PR title must be a conventional commit so the release version bumps correctly:

  <type>(<optional scope>): <summary>      types: ${types//|/, }
  <type>!: <summary>                       breaking change -> MAJOR bump
  feat: ...                                -> MINOR bump
  anything else                            -> PATCH bump

Got: "$title"
MSG
exit 1
