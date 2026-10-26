# Modules 3 & 4 · The Cluster

**ODSC 2026** | Serving LLMs Efficiently at Scale | [Harshul Jain](https://github.com/harshuljain13)

Module 2 ends on a load that will not fit on one card. Modules 3 and 4 go and run
it anyway, on real hardware — so they leave this repo.

Everything is in **[Dodona](https://github.com/harshuljain13/Dodona)**.

| Module | | Start at |
|---|---|---|
| **3** | Bring up the cluster — one card sliced, vLLM prefill and decode split across the slices | [Dodona README](https://github.com/harshuljain13/Dodona/blob/main/README.md) |
| **4** | Put load on it and watch admission control decide what to let in | [`benchmarking/`](https://github.com/harshuljain13/Dodona/blob/main/benchmarking/README.md) |

**Setup first** — [`PREREQUISITES.md`](https://github.com/harshuljain13/Dodona/blob/main/PREREQUISITES.md).
Begin at [`../SETUP.md`](../SETUP.md), which routes you there.

**No GPU?** Dodona commits its runs, so these modules can be presented cold —
[`01-health`](https://github.com/harshuljain13/Dodona/tree/main/benchmarking/01-health),
[`02-baseline`](https://github.com/harshuljain13/Dodona/tree/main/benchmarking/02-baseline),
[`03-capacity`](https://github.com/harshuljain13/Dodona/tree/main/benchmarking/03-capacity),
each with its notebook, reports and Grafana screenshots.

---

Links track Dodona's `main`. To freeze them for a delivery, tag Dodona
`odsc-2026` and swap `blob/main` → `blob/odsc-2026` here.
