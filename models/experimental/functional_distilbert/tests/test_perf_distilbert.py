# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
import time

from models.experimental.functional_distilbert.tt import ttnn_optimized_distilbert
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    profiler,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.perf.perf_utils import prep_perf_report
from transformers import DistilBertForQuestionAnswering, AutoTokenizer
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


def get_expected_times_qa():
    return (14.5, 3.5)


@pytest.mark.parametrize("model_name", ["distilbert-base-uncased-distilled-squad"])
@pytest.mark.parametrize(
    "batch_size, seq_len, expected_inference_time, expected_compile_time",
    ([8, 384, 3.5, 14.5],),
)
def test_performance_distilbert_for_qa(
    batch_size,
    model_name,
    seq_len,
    expected_inference_time,
    expected_compile_time,
    device,
):
    hugging_face_reference_model = DistilBertForQuestionAnswering.from_pretrained(model_name)
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = hugging_face_reference_model.config

    disable_persistent_kernel_cache()

    cpu_key = "ref_key"

    context = batch_size * [
        "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
    ]
    question = batch_size * ["What discipline did Winkelmann create?"]
    inputs = tokenizer(
        question,
        context,
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=f"ttnn_{model_name}_optimized",
        initialize_model=lambda: DistilBertForQuestionAnswering.from_pretrained(model_name, torchscript=False).eval(),
        custom_preprocessor=ttnn_optimized_distilbert.custom_preprocessor,
        device=device,
    )
    profiler.end(f"preprocessing_parameter")

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_out = hugging_face_reference_model(**inputs)
        profiler.end(cpu_key)

        durations = []
        for _ in range(2):
            profiler.start(f"preprocessing_input")
            position_ids = None
            input_ids, position_ids, attention_mask = ttnn_optimized_distilbert.preprocess_inputs(
                inputs["input_ids"],
                position_ids,
                inputs["attention_mask"],
                device=device,
            )
            profiler.end(f"preprocessing_input")

            start = time.time()
            tt_output = ttnn_optimized_distilbert.distilbert_for_question_answering(
                config,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                parameters=parameters,
                device=device,
            )
            tt_output = ttnn.from_device(tt_output)
            end = time.time()

            durations.append(end - start)
            enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times_qa()

    prep_perf_report(
        model_name=f"ttnn_{model_name}_optimized",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")

    assert (
        inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_time}"
    logger.info("Exit SD perf test")


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [8, "distilbert-base-uncased-distilled-squad", 30.8],
    ],
)
def test_distilbert_perf_device(batch_size, test, expected_perf, reset_seeds):
    subdir = "ttnn_distilbert"
    margin = 0.03
    num_iterations = 1
    command = f"pytest tests/ttnn/integration_tests/distilbert/test_ttnn_distilbert.py::test_bert_for_question_answering[sequence_size=768-batch_size=8-model_name=distilbert-base-uncased-distilled-squad]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_distilbert{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
