# Modules 3 & 4 · The Cluster

**ODSC 2026** | Serving LLMs Efficiently at Scale | [Harshul Jain](https://github.com/harshuljain13)

Module 2 ends on a load one replica cannot keep up with. Not because the model
does not fit — a 32B at int4 is about 19 GB and sits comfortably on an 80 GB
card — but because a single engine runs a bounded number of sequences at once,
and past that requests wait. Modules 3 and 4 run it on real hardware, so they
leave this repo.

Everything is in **[Dodona](https://github.com/harshuljain13/Dodona)**.

| Module | | Start at |
|---|---|---|
| **3** | Bring up the cluster — 2 × A100-80, one vLLM per card, each serving the whole 32B, with a gateway in front | [Dodona README](https://github.com/harshuljain13/Dodona/blob/main/README.md) |
| **4** | Put load on it: routing picks the less loaded replica, then admission control decides what to let in | [`benchmarking/`](https://github.com/harshuljain13/Dodona/blob/main/benchmarking/README.md) |

**Two whole cards, not one sliced one.** Each card runs an independent replica
of the same model — data parallel, TP=1 — and both do prefill and decode. That
is the configuration most teams actually run, and it is what makes module 4
possible: with two replicas there is a routing decision to make, and a request
can be sent to the emptier one before anything has to be refused.

Prefill/decode disaggregation and tensor parallelism are real and worth knowing,
but they are different fleets with different failure modes. They live on their
own Dodona branches (`lambda-pd-disaggregation`, `runpod-tp-cluster`) so that
`main` promises exactly one architecture and delivers it.

**Setup first** — [`PREREQUISITES.md`](https://github.com/harshuljain13/Dodona/blob/main/PREREQUISITES.md):
the box, the keys, the `.env`. Do it before the session.

**No GPU?** Two of the four can be presented cold, because their runs are
committed — notebook, reports and Grafana screenshots:

- [`01-health`](https://github.com/harshuljain13/Dodona/tree/main/benchmarking/01-health) — is the fleet actually serving
- [`02-baseline`](https://github.com/harshuljain13/Dodona/tree/main/benchmarking/02-baseline) — what one replica does unloaded

The other two carry the notebook but **no committed output**, so they need a live
fleet or a run of your own beforehand:

- [`03-capacity`](https://github.com/harshuljain13/Dodona/tree/main/benchmarking/03-capacity) — where throughput stops improving
- [`04-routing-admission`](https://github.com/harshuljain13/Dodona/tree/main/benchmarking/04-routing-admission) — routing between the replicas, then shedding

Worth knowing before the room, not during it: a notebook that has never been run
in anger looks identical to one that has, right up to the first empty cell.

---

Links track Dodona's `main`. To freeze them for a delivery, tag Dodona
`odsc-2026` and swap `blob/main` → `blob/odsc-2026` here.
