
#/bin/bash
# set -eo pipefail

run_t3000_falcon7b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon7b_tests"

  env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/falcon7b/tests -m "model_perf_t3000" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon7b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_mixtral_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mixtral_tests"

  env pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_perf.py -m "model_perf_t3000" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama2_70b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama2_70b_tests"

  env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/llama2_70b/tests/test_llama_perf_decode.py -m "model_perf_t3000"

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama2_70b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_falcon40b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_tests"

  env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/falcon40b/tests/test_perf_falcon.py -m "model_perf_t3000" --timeout=600 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon40b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llm_tests() {
  # Run falcon7b tests
  run_t3000_falcon7b_tests

  # Run mixtral tests
  run_t3000_mixtral_tests

  # Run llama2-70b tests
  run_t3000_llama2_70b_tests

  # Run falcon40b tests
  run_t3000_falcon40b_tests

  # Merge all the generated reports
  env python models/perf/merge_perf_results.py
}

run_t3000_cnn_tests() {
  # Merge all the generated reports
  env python models/perf/merge_perf_results.py
}

fail=0
main() {
  # For CI pipeline - source func commands but don't execute tests if not invoked directly
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

  # Parse the arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --pipeline-type)
        pipeline_type=$2
        shift
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
  done

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$pipeline_type" ]]; then
    echo "--pipeline-type cannot be empty" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  if [[ "$pipeline_type" == "llm_model_perf_t3000_device" ]]; then
    run_t3000_llm_tests
  elif [[ "$pipeline_type" == "cnn_model_perf_t3000_device" ]]; then
    run_t3000_cnn_tests
  else
    echo "$pipeline_type is invalid (supported: [cnn_model_perf_t3000_device, cnn_model_perf_t3000_device])" 2>&1
    exit 1
  fi

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
