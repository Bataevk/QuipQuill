#!/usr/bin/env bash
# Publish the prompt originals (prompts/<name>.md) as variables PROMPT_<NAME> of the GitHub
# environment `production` (ADR 57); the next deploy of `cloud` writes them to the server.
#   ./deploy/push-prompts.sh             all prompts
#   ./deploy/push-prompts.sh gm actor    only these
# Needs the GitHub CLI signed in with admin rights on the repository.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -eq 0 ]]; then
  for file in prompts/*.md; do set -- "$@" "$(basename "$file" .md)"; done
fi
for name in "$@"; do
  file="prompts/$name.md"
  [[ -f "$file" ]] || { echo "no $file" >&2; exit 1; }
  key="PROMPT_$(tr '[:lower:]' '[:upper:]' <<<"$name")"
  gh variable set "$key" --env production < "$file"
  echo "$key ← $file"
done
echo "Now run the cloud workflow (Run workflow) to put them on the server."
