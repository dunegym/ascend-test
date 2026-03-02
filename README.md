## Test Environment

Ubuntu == 22.04

CPU == Kunpeng-920

NPU == Ascend910B2

CANN == 8.5.0

torch == 2.9.0+cpu

torch-npu == 2.9.0.rc1

mindspore == 2.8.0

## Environment Configuration

CANN && ascend-toolkit && ops: 

https://www.hiascend.com/cann/download

torch && torch-npu: 

https://pypi.org/project/torch-npu/

mindspore: 

https://www.mindspore.cn/install/

## Using Project

Testing the mindspore ops: 

```python
python ms/ops_test.py
```

Testing the torch-npu ops: 

```python
python pth/ops_test.py
```

The scripts are going to test 49 opeerators' availability on Ascend910B; ops which pass the test are marked as [PASS], those fail on NPU may fall back to CPU, then marked as [FALLBACK]

## Expected Result

48 ops in 49 may be marked as [PASS], "cholesky" operator is marked as [FALLBACK]; if your test result is the same or even better, the mindspore/torch-npu environment is actually perfect on your machine
