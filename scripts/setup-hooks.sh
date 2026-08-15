#!/bin/sh
# Installs this repo's git hooks. .git/hooks isn't versioned by git, so this
# script is the source of truth — re-run it after a fresh clone or if hooks
# go missing (e.g. .git recreated).
set -e

repo_root=$(git rev-parse --show-toplevel)
cp "$repo_root/scripts/hooks/pre-commit" "$repo_root/.git/hooks/pre-commit"
chmod +x "$repo_root/.git/hooks/pre-commit"
echo "installed pre-commit hook"
