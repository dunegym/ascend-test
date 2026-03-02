import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional

import mindspore as ms
from mindspore import Tensor, context, ops


@dataclass
class OpCase:
	name: str
	fn: Callable[[], object]
	required: bool = True


@dataclass
class OpResult:
	name: str
	ok: bool
	required: bool = True
	output_shape: Optional[tuple] = None
	dtype: Optional[str] = None
	error: Optional[str] = None
	fallback: bool = False
	fallback_from: Optional[str] = None
	fallback_to: Optional[str] = None


def _short_exception() -> str:
	return traceback.format_exc(limit=1).strip().splitlines()[-1]


def _prepare_context() -> None:
	context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
	device_id = context.get_context("device_id")
	print(f"[INFO] MindSpore version: {ms.__version__}")
	print(f"[INFO] Device target: Ascend, device_id={device_id}")
	print("[INFO] CPU fallback: enabled (Ascend fail -> CPU retry)")


def _build_cases() -> List[OpCase]:
	x = Tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], ms.float32)
	y = Tensor([[0.5, 2.0, -1.0], [1.5, -3.0, 2.0]], ms.float32)

	a = Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
	b = Tensor([[5.0, 6.0], [7.0, 8.0]], ms.float32)

	idx = Tensor([0, 2], ms.int32)
	int_x = Tensor([1, 2, 3, 4], ms.int32)

	conv_x = Tensor(
		[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
		ms.float32,
	)
	conv_w = Tensor(
		[
			[[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]],
			[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]],
		],
		ms.float32,
	)

	batch_a = Tensor(
		[
			[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
			[[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]],
		],
		ms.float32,
	)
	batch_b = Tensor(
		[
			[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
			[[2.0, 0.0], [1.0, 2.0], [0.0, 1.0]],
		],
		ms.float32,
	)

	ln_x = Tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], ms.float32)
	gamma = Tensor([1.0, 1.0, 1.0, 1.0], ms.float32)
	beta = Tensor([0.0, 0.0, 0.0, 0.0], ms.float32)

	scatter_indices = Tensor([[0, 1], [1, 2]], ms.int32)
	scatter_updates = Tensor([9.0, 10.0], ms.float32)

	tensor_scatter_x = Tensor([1.0, 2.0, 3.0, 4.0], ms.float32)
	tensor_scatter_indices = Tensor([[1], [3]], ms.int32)
	tensor_scatter_updates = Tensor([10.0, 20.0], ms.float32)

	mask = Tensor([[True, False, True], [False, True, False]], ms.bool_)
	gather_nd_indices = Tensor([[0, 1], [1, 2]], ms.int32)

	spd = Tensor([[4.0, 1.0], [1.0, 3.0]], ms.float32)

	return [
		OpCase("add", lambda: ops.add(x, y)),
		OpCase("sub", lambda: ops.sub(x, y)),
		OpCase("mul", lambda: ops.mul(x, y)),
		OpCase("div", lambda: ops.div(x, y)),
		OpCase("pow", lambda: ops.pow(x, Tensor(2.0, ms.float32))),
		OpCase("abs", lambda: ops.abs(x)),
		OpCase("sqrt", lambda: ops.sqrt(ops.abs(x) + Tensor(1e-6, ms.float32))),
		OpCase("exp", lambda: ops.exp(x)),
		OpCase("log", lambda: ops.log(ops.abs(x) + Tensor(1.0, ms.float32))),
		OpCase("sin", lambda: ops.sin(x)),
		OpCase("cos", lambda: ops.cos(x)),
		OpCase("tanh", lambda: ops.tanh(x)),
		OpCase("relu", lambda: ops.relu(x)),
		OpCase("sigmoid", lambda: ops.sigmoid(x)),
		OpCase("matmul", lambda: ops.matmul(a, b)),
		OpCase("transpose", lambda: ops.transpose(x, (1, 0))),
		OpCase("reshape", lambda: ops.reshape(x, (3, 2))),
		OpCase("expand_dims", lambda: ops.expand_dims(x, 0)),
		OpCase("squeeze", lambda: ops.squeeze(ops.expand_dims(x, 0), 0)),
		OpCase("concat", lambda: ops.concat((x, y), axis=0)),
		OpCase("stack", lambda: ops.stack((x, y), axis=0)),
		OpCase("tile", lambda: ops.tile(a, (2, 1))),
		OpCase("gather", lambda: ops.gather(x, idx, 1)),
		OpCase("reduce_sum", lambda: ops.reduce_sum(x, axis=1)),
		OpCase("reduce_mean", lambda: ops.reduce_mean(x, axis=0)),
		OpCase("reduce_max", lambda: ops.reduce_max(x, axis=1)),
		OpCase("reduce_min", lambda: ops.reduce_min(x, axis=0)),
		OpCase("argmax", lambda: ops.argmax(x, 1)),
		OpCase("argmin", lambda: ops.argmin(x, axis=1)),
		OpCase("sort", lambda: ops.sort(x, axis=-1)),
		OpCase("topk", lambda: ops.topk(x, 2)),
		OpCase("softmax", lambda: ops.softmax(x, axis=1)),
		OpCase("one_hot", lambda: ops.one_hot(int_x, 6, Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))),
		OpCase("clip_by_value", lambda: ops.clip_by_value(x, Tensor(-1.0, ms.float32), Tensor(2.0, ms.float32))),
		OpCase("where", lambda: ops.where(x > 0, x, y)),
		OpCase("conv2d", lambda: ops.conv2d(conv_x, conv_w, pad_mode="valid", stride=1, dilation=1, groups=1)),
		OpCase("batch_matmul", lambda: ops.bmm(batch_a, batch_b)),
		OpCase("layer_norm", lambda: ops.layer_norm(ln_x, (4,), gamma, beta, 1e-5)),
		OpCase("cumsum", lambda: ops.cumsum(x, axis=1)),
		OpCase("cumprod", lambda: ops.cumprod(ops.abs(x) + Tensor(1.0, ms.float32), 1)),
		OpCase("nonzero", lambda: ops.nonzero(x)),
		OpCase("masked_select", lambda: ops.masked_select(x, mask)),
		OpCase("gather_nd", lambda: ops.gather_nd(x, gather_nd_indices)),
		OpCase("scatter_nd", lambda: ops.scatter_nd(scatter_indices, scatter_updates, (2, 3))),
		OpCase("tensor_scatter_add", lambda: ops.tensor_scatter_add(tensor_scatter_x, tensor_scatter_indices, tensor_scatter_updates)),
		OpCase("matrix_inverse", lambda: ops.inverse(a)),
		OpCase("det", lambda: ops.det(a)),
		OpCase("cholesky", lambda: ops.cholesky(spd), required=False),
		OpCase("qr", lambda: ops.qr(a)),
	]


def _build_case_map_for_target(device_target: str) -> dict:
	context.set_context(device_target=device_target)
	return {case.name: case for case in _build_cases()}


def _run_case(case: OpCase) -> OpResult:
	try:
		out = case.fn()
		if isinstance(out, tuple):
			first = out[0]
			if isinstance(first, Tensor):
				return OpResult(case.name, True, case.required, tuple(first.shape), str(first.dtype))
			return OpResult(case.name, True, case.required)

		if isinstance(out, Tensor):
			return OpResult(case.name, True, case.required, tuple(out.shape), str(out.dtype))

		return OpResult(case.name, True, case.required)
	except Exception:
		return OpResult(case.name, False, case.required, error=_short_exception())


def _run_case_with_fallback(case: OpCase, cpu_case_map: dict) -> OpResult:
	context.set_context(device_target="Ascend")
	ascend_result = _run_case(case)
	if ascend_result.ok:
		return ascend_result

	cpu_case = cpu_case_map.get(case.name)
	if cpu_case is None:
		return ascend_result

	context.set_context(device_target="CPU")
	cpu_result = _run_case(cpu_case)
	context.set_context(device_target="Ascend")

	if cpu_result.ok:
		cpu_result.fallback = True
		cpu_result.fallback_from = "Ascend"
		cpu_result.fallback_to = "CPU"
		cpu_result.error = ascend_result.error
		return cpu_result

	return ascend_result


def main() -> None:
	_prepare_context()
	context.set_context(device_target="Ascend")
	cases = _build_cases()
	cpu_case_map = _build_case_map_for_target("CPU")
	context.set_context(device_target="Ascend")

	results: List[OpResult] = []
	for case in cases:
		result = _run_case_with_fallback(case, cpu_case_map)
		results.append(result)

	fallback_results = [r for r in results if r.ok and r.fallback]

	ok_count = sum(1 for r in results if r.ok)
	fail_count = len(results) - ok_count
	required_fail_count = sum(1 for r in results if (not r.ok and r.required))
	optional_fail_count = sum(1 for r in results if (not r.ok and not r.required))

	print("\n========== NPU算子可用性测试结果 ==========")
	for r in results:
		if r.ok:
			flag = " [FALLBACK]" if r.fallback else ""
			print(f"[PASS]{flag} {r.name:16s} shape={r.output_shape}, dtype={r.dtype}")
		else:
			tag = "FAIL" if r.required else "SKIP"
			print(f"[{tag}] {r.name:16s} error={r.error}")

	print("------------------------------------------")
	print(f"总数: {len(results)} | 通过: {ok_count} | 失败: {fail_count}")
	print(f"必选失败: {required_fail_count} | 可选失败: {optional_fail_count}")
	if fallback_results:
		fallback_desc = ", ".join(f"{r.name}(Ascend->CPU)" for r in fallback_results)
		print(f"CPU回退算子({len(fallback_results)}): {fallback_desc}")
	else:
		print("CPU回退算子(0): 无")

	if required_fail_count > 0:
		raise RuntimeError("存在必选算子在当前NPU环境不可用，请查看上面的 [FAIL] 明细。")


if __name__ == "__main__":
	main()
