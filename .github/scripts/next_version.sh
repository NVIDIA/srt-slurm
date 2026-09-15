#!/usr/bin/env bash
# Compute the next release tag from the latest tag and a conventional-commit PR title.
#
#   next_version.sh <latest-tag-or-empty> <pr-title> [<pr-body>]
#
# Rules (semver, MAJOR.MINOR.PATCH, tags are vX.Y.Z):
#   - "<type>!: ..." or a body containing "BREAKING CHANGE:"  -> MAJOR bump
#   - "feat: ..." / "feat(scope): ..."                         -> MINOR bump
#   - anything else (fix, docs, refactor, perf, test, chore, ci, build, revert) -> PATCH bump
#   - no previous tag                                          -> v1.0.0
# Prints the new tag on stdout and the bump kind on stderr.
set -euo pipefail

latest="${1:-}"
title="${2:-}"
body="${3:-}"

# Regexes live in variables: bash's [[ =~ ]] parser trips over ")" inside a bracket expression.
re_breaking='^[a-z]+(\([^)]*\))?!:'
re_feat='^feat(\([^)]*\))?:'

kind="patch"
if [[ "$title" =~ $re_breaking ]] || grep -q "BREAKING CHANGE:" <<<"$body"; then
  kind="major"
elif [[ "$title" =~ $re_feat ]]; then
  kind="minor"
fi

if [ -z "$latest" ]; then
  echo "seed" >&2
  echo "v1.0.0"
  exit 0
fi

core=${latest#v}; core=${core%%.post*}; core=${core%%-*}; core=${core%%+*}
IFS='.' read -r major minor patch <<<"$core"
case "$kind" in
  major) next="v$((major + 1)).0.0" ;;
  minor) next="v${major}.$((minor + 1)).0" ;;
  patch) next="v${major}.${minor}.$((patch + 1))" ;;
esac
echo "$kind" >&2
echo "$next"
