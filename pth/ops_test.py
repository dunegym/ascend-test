import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

try:
	import torch_npu  # noqa: F401
except Exception as exc:
	raise RuntimeError("未检测到 torch_npu，请先安装并配置 torch-npu 环境。") from exc


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


def _prepare_context() -> torch.device:
	if not hasattr(torch, "npu"):
		raise RuntimeError("当前 PyTorch 未暴露 torch.npu 接口，请检查 torch-npu 安装与版本匹配。")

	if not torch.npu.is_available():
		raise RuntimeError("torch.npu.is_available() 为 False，请检查 NPU 驱动/CANN/环境变量。")

	device = torch.device("npu:0")
	torch.npu.set_device(device)

	print(f"[INFO] Torch version: {torch.__version__}")
	print(f"[INFO] Torch NPU available: {torch.npu.is_available()}")
	print(f"[INFO] Device target: NPU, device={device}")
	print("[INFO] CPU fallback: enabled (NPU fail -> CPU retry; CPU output also marked)")
	return device


def _build_cases(device: torch.device) -> List[OpCase]:
	x = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=torch.float32, device=device)
	y = torch.tensor([[0.5, 2.0, -1.0], [1.5, -3.0, 2.0]], dtype=torch.float32, device=device)

	a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device)
	b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, device=device)

	idx_1d = torch.tensor([0, 2], dtype=torch.int64, device=device)
	int_x = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)

	conv_x = torch.tensor(
		[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
		dtype=torch.float32,
		device=device,
	)
	conv_w = torch.tensor(
		[
			[[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]],
			[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]],
		],
		dtype=torch.float32,
		device=device,
	)

	batch_a = torch.tensor(
		[
			[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
			[[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]],
		],
		dtype=torch.float32,
		device=device,
	)
	batch_b = torch.tensor(
		[
			[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
			[[2.0, 0.0], [1.0, 2.0], [0.0, 1.0]],
		],
		dtype=torch.float32,
		device=device,
	)

	ln_x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], dtype=torch.float32, device=device)
	mask = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool, device=device)

	gather_src = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=torch.float32, device=device)
	gather_index = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64, device=device)

	scatter_base = torch.zeros((2, 3), dtype=torch.float32, device=device)
	scatter_index = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64, device=device)
	scatter_src = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device)

	tensor_scatter_base = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device)
	tensor_scatter_index = torch.tensor([1, 3], dtype=torch.int64, device=device)
	tensor_scatter_src = torch.tensor([10.0, 20.0], dtype=torch.float32, device=device)

	spd = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float32, device=device)

	return [
		OpCase("add", lambda: torch.add(x, y)),
		OpCase("sub", lambda: torch.sub(x, y)),
		OpCase("mul", lambda: torch.mul(x, y)),
		OpCase("div", lambda: torch.div(x, y)),
		OpCase("pow", lambda: torch.pow(x, 2.0)),
		OpCase("abs", lambda: torch.abs(x)),
		OpCase("sqrt", lambda: torch.sqrt(torch.abs(x) + 1e-6)),
		OpCase("exp", lambda: torch.exp(x)),
		OpCase("log", lambda: torch.log(torch.abs(x) + 1.0)),
		OpCase("sin", lambda: torch.sin(x)),
		OpCase("cos", lambda: torch.cos(x)),
		OpCase("tanh", lambda: torch.tanh(x)),
		OpCase("relu", lambda: torch.relu(x)),
		OpCase("sigmoid", lambda: torch.sigmoid(x)),
		OpCase("matmul", lambda: torch.matmul(a, b)),
		OpCase("transpose", lambda: x.transpose(1, 0)),
		OpCase("reshape", lambda: x.reshape(3, 2)),
		OpCase("expand_dims", lambda: x.unsqueeze(0)),
		OpCase("squeeze", lambda: x.unsqueeze(0).squeeze(0)),
		OpCase("concat", lambda: torch.cat((x, y), dim=0)),
		OpCase("stack", lambda: torch.stack((x, y), dim=0)),
		OpCase("tile", lambda: a.tile((2, 1))),
		OpCase("gather", lambda: torch.gather(gather_src, 1, gather_index)),
		OpCase("reduce_sum", lambda: torch.sum(x, dim=1)),
		OpCase("reduce_mean", lambda: torch.mean(x, dim=0)),
		OpCase("reduce_max", lambda: torch.max(x, dim=1).values),
		OpCase("reduce_min", lambda: torch.min(x, dim=0).values),
		OpCase("argmax", lambda: torch.argmax(x, dim=1)),
		OpCase("argmin", lambda: torch.argmin(x, dim=1)),
		OpCase("sort", lambda: torch.sort(x, dim=-1)),
		OpCase("topk", lambda: torch.topk(x, 2)),
		OpCase("softmax", lambda: torch.softmax(x, dim=1)),
		OpCase("one_hot", lambda: F.one_hot(int_x, num_classes=6).to(dtype=torch.float32)),
		OpCase("clip_by_value", lambda: torch.clamp(x, min=-1.0, max=2.0)),
		OpCase("where", lambda: torch.where(x > 0, x, y)),
		OpCase("conv2d", lambda: F.conv2d(conv_x, conv_w, stride=1, padding=0, dilation=1, groups=1)),
		OpCase("batch_matmul", lambda: torch.bmm(batch_a, batch_b)),
		OpCase("layer_norm", lambda: F.layer_norm(ln_x, (4,))),
		OpCase("cumsum", lambda: torch.cumsum(x, dim=1)),
		OpCase("cumprod", lambda: torch.cumprod(torch.abs(x) + 1.0, dim=1)),
		OpCase("nonzero", lambda: torch.nonzero(x)),
		OpCase("masked_select", lambda: torch.masked_select(x, mask)),
		OpCase("scatter", lambda: scatter_base.scatter(1, scatter_index, scatter_src)),
		OpCase("tensor_scatter_add", lambda: tensor_scatter_base.index_add(0, tensor_scatter_index, tensor_scatter_src)),
		OpCase("matrix_inverse", lambda: torch.linalg.inv(a)),
		OpCase("det", lambda: torch.linalg.det(a)),
		OpCase("cholesky", lambda: torch.linalg.cholesky(spd), required=False),
		OpCase("qr", lambda: torch.linalg.qr(a)),
		OpCase("index_select", lambda: torch.index_select(x, dim=1, index=idx_1d)),
	]


def _run_case(case: OpCase) -> OpResult:
	try:
		out = case.fn()
		if isinstance(out, tuple):
			first = out[0]
			if isinstance(first, torch.Tensor):
				result = OpResult(case.name, True, case.required, tuple(first.shape), str(first.dtype))
				result.fallback_to = first.device.type
				return result
			return OpResult(case.name, True, case.required)

		if isinstance(out, torch.Tensor):
			result = OpResult(case.name, True, case.required, tuple(out.shape), str(out.dtype))
			result.fallback_to = out.device.type
			return result

		return OpResult(case.name, True, case.required)
	except Exception:
		return OpResult(case.name, False, case.required, error=_short_exception())


def _run_case_with_fallback(case: OpCase, cpu_case_map: dict) -> OpResult:
	npu_result = _run_case(case)
	if npu_result.ok:
		if npu_result.fallback_to == "cpu":
			npu_result.fallback = True
			npu_result.fallback_from = "NPU"
			npu_result.fallback_to = "CPU"
		else:
			npu_result.fallback_to = None
		return npu_result

	cpu_case = cpu_case_map.get(case.name)
	if cpu_case is None:
		return npu_result

	cpu_result = _run_case(cpu_case)
	if cpu_result.ok:
		cpu_result.fallback = True
		cpu_result.fallback_from = "NPU"
		cpu_result.fallback_to = "CPU"
		cpu_result.error = npu_result.error
		return cpu_result

	return npu_result


def main() -> None:
	npu_device = _prepare_context()
	cases = _build_cases(npu_device)
	cpu_case_map = {case.name: case for case in _build_cases(torch.device("cpu"))}

	results: List[OpResult] = []
	for case in cases:
		results.append(_run_case_with_fallback(case, cpu_case_map))

	fallback_results = [r for r in results if r.ok and r.fallback]

	ok_count = sum(1 for r in results if r.ok)
	fail_count = len(results) - ok_count
	required_fail_count = sum(1 for r in results if (not r.ok and r.required))
	optional_fail_count = sum(1 for r in results if (not r.ok and not r.required))

	print("\n========== Torch-NPU 算子可用性测试结果 ==========")
	for result in results:
		if result.ok:
			flag = " [FALLBACK]" if result.fallback else ""
			print(f"[PASS]{flag} {result.name:16s} shape={result.output_shape}, dtype={result.dtype}")
		else:
			tag = "FAIL" if result.required else "SKIP"
			print(f"[{tag}] {result.name:16s} error={result.error}")

	print("------------------------------------------")
	print(f"总数: {len(results)} | 通过: {ok_count} | 失败: {fail_count}")
	print(f"必选失败: {required_fail_count} | 可选失败: {optional_fail_count}")
	if fallback_results:
		fallback_desc = ", ".join(f"{r.name}(NPU->CPU)" for r in fallback_results)
		print(f"CPU回退算子({len(fallback_results)}): {fallback_desc}")
	else:
		print("CPU回退算子(0): 无")

	if required_fail_count > 0:
		raise RuntimeError("存在必选算子在当前 Torch-NPU 环境不可用，请查看上面的 [FAIL] 明细。")


if __name__ == "__main__":
	main()
