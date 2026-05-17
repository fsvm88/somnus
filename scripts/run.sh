#!/usr/bin/env bash
# Build somnus in release mode then exec the binary directly.
# This avoids the broken-pipe problem that occurs when running under
# `cargo run`: cargo creates an internal pipe to the child's stderr, and
# when cargo exits (after the build) the pipe has no reader, so tracing
# writes silently fail.
#
# Usage: scripts/run.sh [somnus args…]
#   e.g. scripts/run.sh --file clips/foo.mp4 --file-loop --metrics-file /tmp/somnus.jsonl

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[run.sh] building somnus (release)…" >&2
cargo build --release -p somnus 2>&1

echo "[run.sh] starting somnus…" >&2
exec "$REPO_ROOT/target/release/somnus" "$@"
