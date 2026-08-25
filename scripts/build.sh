#!/bin/bash

SKIP_TESTING=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--skip-tests)
      SKIP_TESTING=true
      shift
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
  echo "Running tests"
  ctest --test-dir build --output-on-failure
else
  echo "Skipped Tests"
fi