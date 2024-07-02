
#/bin/bash
set -eo pipefail

run_tg_tests() {

  echo "LOG_METAL: running run_tg_unit_tests"

  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter="CommandQueueSingleCardFixture.*"

  pytest tests/ttnn/multichip_unit_tests/test_multidevice_TG.py
}

main() {
  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_tg_tests
}

main "$@"
