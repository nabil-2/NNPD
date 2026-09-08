# CPU, CUDA, and multiple GPUs

These are the execution modes implemented by `nnpd/core/runtime.py` and the
bundled trainer. Custom trainers must implement their own distributed contract.
The shipped application test report documents CPU execution; physical CUDA and
multi-node execution were not validated in that report.

## Choose a device

Edit `runtime.device` in `settings.py`:

| Value | Behavior |
|---|---|
| `"auto"` | CUDA when available, otherwise CPU. |
| `"cpu"` | Explicit CPU execution. |
| `"cuda"` | Require CUDA; fail clearly if unavailable. |
| `"cuda:1"` | Select a single indexed device outside a distributed launch. |

Use a compatible CUDA-enabled PyTorch installation for GPU runs. Hardware and
package availability do not follow automatically from selecting a configuration
string. See the official [PyTorch installation selector](https://pytorch.org/get-started/locally/)
for installation commands rather than using a fixed CUDA wheel URL here.

## Independent jobs on two GPUs

Set `runtime.parallel="jobs"` and `runtime.device="cuda"` (or `"auto"`) in the
settings file, then launch from a shell:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run.py run --profile smoke
```

Workers handle disjoint complete jobs, including their analysis. Each model fits
on one GPU. This is not tensor/model parallelism and does not combine GPU memory
for one network. Common compatible products can be shared through the output
store.

## Collective DDP training

The `smoke_ddp` preset selects `runtime.parallel="ddp"` and a single uniform-prior
cohort member:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run.py run --profile smoke_ddp
```

All workers train each model together. Its analysis runs on rank zero.
`training.batch_size` is the **global** batch size, not the batch size per GPU.
The bundled trainer handles unequal or empty final shards without intentionally
dropping or duplicating real examples.

Use `device="cuda"`, not `"cuda:0"`, under a multi-worker launch; local rank selects
the GPU. Use one worker per visible GPU. The CUDA collective backend is NCCL;
CPU DDP uses Gloo. Setting `parallel="ddp"` with only one process does not itself
launch additional processes.

## Precision, reproducibility, and restoration

`training.precision` accepts `"off"`, `"float16"`, or `"bfloat16"`. The bundled
float16 path requires CUDA. bfloat16 also depends on device/operation support.
Deterministic settings do not promise identical weights across all devices,
world sizes, or dependency versions.

Backend identity participates in the model cache key. Changing CPU/CUDA or DDP
world size is not treated as an identical training artifact. To inspect an
already trained model on CPU, use `restore(..., device="cpu")`; do not expect a
new CPU `analyze` launch to match a GPU training recipe automatically.

Interrupted training restarts at stage level: complete model weights are saved,
but mid-epoch optimizer, RNG position, and scaler state are not checkpointed.

## Distributed limits

Runtime settings cannot be swept within one launch. Multi-node launches need
correct `torchrun` rendezvous configuration and a shared filesystem whose locks
and atomic operations work across workers. No site-specific scheduler settings
are supplied. Do not run two independent analyses that write the same result
directory concurrently.

See [Storage](../STORAGE.md) and the unchanged [application test report](../TEST_REPORT.md)
for the precise guarantees and validation boundaries.
