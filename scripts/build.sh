#!/bin/bash

GREEN="\033[0;32m"
NC="\033[0m"

SKIP_TESTING=false
FILEPATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--skip-tests)
      SKIP_TESTING=true
      shift
      ;;
    -r|--run)
      FILEPATH="$2"
      shift 2
      ;;

    *)
      echo "Unknown flag: $1" >&2
      exit 1
      ;;
  esac
done

cmake -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
cmake --build build

if ! $SKIP_TESTING; then
  echo -e "\n${GREEN}Running tests${NC}\n"
  ctest --test-dir build --output-on-failure
else
  echo -e "\n${GREEN}Skipped Tests${NC}\n"
fi

if ! [ "$FILEPATH" = "" ]; then
  echo -e "${GREEN}Running ${FILEPATH}${NC}\n"
  "$FILEPATH"
fi