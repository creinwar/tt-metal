1. There are 7 conv's used in klassify model, out of which 3 are failing due to dilation>1 not supported,One conv gived unusual issue(some assertion) and other three failed due to L1 and circular buffer issue.
2. To test the unit_test, run `pytest tests/ttnn/unit_tests/operations/test_conv2d.py::test_klassify_conv`

Note: For conv checking purpose batch_size=1 is used even though 32 is used in the model.
