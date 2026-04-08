#!/usr/bin/env bash
set -u

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_BUILD_TIMEOUT="${DOCKER_BUILD_TIMEOUT:-900}"

log() { printf "%b\n" "$1"; }
pass() { printf "${GREEN}[PASS]${NC} %s\n" "$1"; }
fail() { printf "${RED}[FAIL]${NC} %s\n" "$1"; }
hint() { printf "${YELLOW}[HINT]${NC} %s\n" "$1"; }

stop_at() {
  printf "${RED}Stopped at %s${NC}\n" "$1"
  exit 1
}

run_with_timeout() {
  local timeout_secs="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$timeout_secs" "$@"
  else
    "$@"
  fi
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD} OpenEnv Submission Pre-Validation${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

log "${BOLD}Step 1/3: Checking inference.py${NC} ..."

if [ ! -f "$REPO_DIR/inference.py" ]; then
  fail "Missing inference.py in project root"
  hint "Create CWD/inference.py and follow the mandatory stdout format"
  stop_at "Step 1"
fi

if ! grep -q "\[START\]" "$REPO_DIR/inference.py" || ! grep -q "\[STEP\]" "$REPO_DIR/inference.py" || ! grep -q "\[END\]" "$REPO_DIR/inference.py"; then
  fail "inference.py does not appear to emit required [START]/[STEP]/[END] lines"
  hint "Ensure stdout format exactly matches the hackathon requirement"
  stop_at "Step 1"
fi

pass "inference.py is present with required markers"

log "${BOLD}Step 2/3: Building Docker image${NC} ..."

if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  log "  Default docker build failed, retrying with --network=host"
  BUILD_OK=false
  BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build --network=host "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true
  if [ "$BUILD_OK" = true ]; then
    pass "Docker build succeeded with --network=host"
  else
    fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
    printf "%s\n" "$BUILD_OUTPUT" | tail -20
    stop_at "Step 2"
  fi
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv >/dev/null 2>&1; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
