# Modules 3 & 4 · Serving It For Real — the Dodona Cluster

**ODSC 2026** | Serving LLMs Efficiently at Scale | [Harshul Jain](https://github.com/harshuljain13)

Modules 1 and 2 are notebooks. These two are not, and the reason is the point of
them: **everything up to here fits in one process on one card, and nothing you
have learned so far tells you what happens when real traffic arrives.**

Module 1 measured the wall. Module 2 lowered it. Modules 3 and 4 put a real
cluster behind it and ask the question neither notebook can:

> when the card is genuinely full, who gets turned away, and did anyone decide?

That needs a scheduler, a control plane, a metrics stack, and load arriving from
somewhere other than the cell above. It does not fit in a notebook, so it lives
in its own repo.

**The cluster is [Dodona](https://github.com/harshuljain13/Dodona).** Everything
below links there. Nothing below restates an instruction from it — if a command
appears in both places, one of them is already wrong.

> **Link policy.** These links track Dodona's `main`, so they always resolve but
> can shift as Dodona develops. To freeze them for a specific delivery, tag
> Dodona (`git tag odsc-2026 && git push origin odsc-2026`) and swap `blob/main`
> for `blob/odsc-2026` throughout this file. That is the only change needed.

---

## Before you start

Setup for these modules is **not** described here. Two files own it, and both are
upstream of this one:

| | Owns |
|---|---|
| [`../SETUP.md`](../SETUP.md) | The pre-session gate — what each half of the workshop needs, and the two RunPod facts that will cost you the session if you get them wrong |
| [Dodona `PREREQUISITES.md`](https://github.com/harshuljain13/Dodona/blob/main/PREREQUISITES.md) | Every actual step: `.env`, keys, the SSH key, the box, and a check that tells you whether you are ready |

Read `../SETUP.md` first. The short version of why: it takes about twenty
minutes, most of it waiting for accounts, it needs a GPU box that **bills by the
hour**, and the HuggingFace token from modules 1–2 gets reused — so the ordering
matters.

**Do not bring the cluster up before the session.** That is the first thing we do
together, it takes fifteen to twenty minutes, and the meter is running.

**You can follow modules 3 and 4 without any of this.** See
[If you are not building the cluster](#if-you-are-not-building-the-cluster) at the
bottom — the measurements are committed to the repo, so the argument survives
without the hardware.

---

## Module 3 · One Card, Three Slices, Two Phases

Module 1 drove prefill and decode by hand in one Python process to time them
apart. A real deployment separates them *physically*, because they want different
things from the hardware — and now you can see why that is worth the trouble.

What runs:

| Piece | What it is | Why it is there |
|---|---|---|
| **k3s + [HAMi](https://github.com/Project-HAMi/HAMi)** | One physical GPU presented as three slices | Module 1's memory arithmetic, enforced by the scheduler instead of hoped for |
| **vLLM `prefill-0`** | Builds the prompt's KV cache | The compute-shaped half of Module 1 §05 |
| **vLLM `decode-0`, `decode-1`** | Generate tokens | The memory-shaped half. Two of them, because decode is what you scale |
| **[Mooncake](https://github.com/kvcache-ai/Mooncake)** | Moves KV between slices | What makes the split possible at all |
| **Prometheus + Grafana** | `vllm:kv_cache_usage_perc` and friends | Module 1 §06's growth curve, live, on real traffic |
| **Open WebUI** | In-cluster at `:30030`, no agent, no tools | Hand-testing the engine without the app in the way |

The model is
[Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ) —
which is Module 2's quantization lever, already pulled, in production.

```bash
make cluster          # k3s, HAMi, vLLM slices, Mooncake
make observability    # Prometheus + Grafana
make kv               # watch the KV signal
```

→ **[Dodona · README](https://github.com/harshuljain13/Dodona/blob/main/README.md)**
for the full architecture diagram and what each service does.

**The moment to watch for.** Module 1 §06b drew a curve and marked where it
crossed the budget. Here that crossing is not a chart — it is a slice filling up
while you watch `make kv`, and a request that has to go somewhere.

---

## Module 4 · Admission Control: Who Gets Turned Away

This is the module that does not exist anywhere else in the workshop, and the
reason the other three lead here.

When the card is full, something has to give. Most deployments decide this **by
accident** — a queue fills, latency rises for everyone equally, and no component
ever made a choice. Dodona decides on purpose and writes down what it saw.

The control plane (`orch-serve`) is a four-stage pipeline:

| Stage | Decision |
|---|---|
| **guard** | malformed → `400`, before anything expensive happens |
| **admit** | `kv_pressure ≥ 0.90` → refuse with `503` and a `kv_free` reason |
| **place** | `least_loaded` plus queue depth |
| **hop** | source ≠ destination → record it |

Refusal is where the engineering is. An overflow gate enforces which failures
may leave the building: **`503`/`529` may, `429` never does.** Every decision
lands in [AdmitPerf](https://github.com/harshuljain13/AdmitPerf)'s log, so a run
produces a verdict rather than a vibe.

### Measuring it

→ **[Dodona · benchmarking/](https://github.com/harshuljain13/Dodona/blob/main/benchmarking/README.md)**

Three self-contained experiments, each with its own notebook, reports, and
screenshots. **Every one is independent** — you can run `02` without `01`, and a
failure in one does not invalidate another:

| | Asks |
|---|---|
| [`01-health`](https://github.com/harshuljain13/Dodona/tree/main/benchmarking/01-health) | is it there and wired? |
| [`02-baseline`](https://github.com/harshuljain13/Dodona/tree/main/benchmarking/02-baseline) | what do the numbers mean? |
| [`03-capacity`](https://github.com/harshuljain13/Dodona/tree/main/benchmarking/03-capacity) | **what binds first, and at what load?** How many concurrent sessions hold the TTFT budget — and *which* resource actually runs out |

`03-capacity` carries the sharpest argument in either repo: **an admission
policy reads one signal, so if that signal never moves, the policy never fires —
and an A/B against it compares two identical configurations.** That is why
establishing which resource binds comes *before* testing any policy, and why
picking a concurrency target up front is a wish rather than a measurement.

The load generators are agent-shaped rather than synthetic — `load.py` sends 4
sequential turns with a ~1k shared prefix and a growing context, which is the
traffic pattern that makes prefix caching matter. `crew_flood.py` runs concurrent
multi-role agents; `agent_flood.py` holds sustained RPS and watches abort
behaviour.

```bash
make load                                               # 8 sessions, seed 1
make load ARGS="--sessions 32 --max-tokens 400 --run high"
```

### Three methodology points worth stealing

These are the parts that transfer to whatever you run at work, and each one is a
mistake someone has already made:

1. **Run the load on the box, not through the tunnel.** An ssh tunnel puts your
   WAN round-trip inside every TTFT *and* serialises concurrency through one TCP
   stream. A bundle taken from a laptop measures your internet. `make load` runs
   on the box and rsyncs the results back for exactly this reason.
2. **Nothing in `benchmarking/` imports the control plane.** A load generator
   that shares code with the thing it measures can agree with it about something
   they are both wrong about. Dodona enforces that boundary with a test rather
   than a convention.
3. **Seed the workload.** `--seed` makes two arms differ only by the thing under
   test. Without it you are comparing noise and calling it a result.

### The question every run has to answer

> was anything actually scarce, and could an admission policy have fired?

`load.py` prints it next to the latency numbers, and the ordering is deliberate.
A p99 on its own cannot distinguish **a policy that worked** from **a policy that
never had anything to decide** — and that distinction is the difference between a
result and a number.

This is also the honest end of the workshop's arc. Module 1 measured a limit,
Module 2 moved it, Module 3 built something real against it — and Module 4 is the
admission that you will still hit it, so the only remaining question is whether
you hit it on purpose.

---

## If you are not building the cluster

**Nothing above requires you to have a GPU to follow it**, and a live cluster at
a conference is a coin flip. Dodona commits its measurements, so the evidence is
readable cold:

| Committed artefact | What it shows |
|---|---|
| [`01-health/reports/01-health.html`](https://github.com/harshuljain13/Dodona/blob/main/benchmarking/01-health/reports/01-health.html) | The cluster wired up and answering |
| [`02-baseline/reports/02-metrics.html`](https://github.com/harshuljain13/Dodona/blob/main/benchmarking/02-baseline/reports/02-metrics.html) | A full baseline run with the metric catalogue |
| [`03-capacity/traces`](https://github.com/harshuljain13/Dodona/tree/main/benchmarking/03-capacity) | Traces from the runs that filled the card |
| `screenshots/` in each folder | Grafana as it looked during the run |

**If you are presenting this:** open these *before* you need them. Walking an
audience through a committed report is a fine outcome; debugging `make cluster`
in front of them is not. Decide which you are doing before the session, not
during it.

---

## Where each thing lives

| | Owned by |
|---|---|
| Why modules 3–4 exist, what they cost, live-vs-recorded | **this file** |
| Setup, keys, accounts, the GPU box | [Dodona `PREREQUISITES.md`](https://github.com/harshuljain13/Dodona/blob/main/PREREQUISITES.md) |
| Architecture, services, `make` targets | [Dodona `README.md`](https://github.com/harshuljain13/Dodona/blob/main/README.md) |
| Load generators, run comparability, SLOs | [Dodona `benchmarking/`](https://github.com/harshuljain13/Dodona/blob/main/benchmarking/README.md) |
| Admission policy and decision logs | [AdmitPerf](https://github.com/harshuljain13/AdmitPerf) |

Setup instructions are deliberately **not** copied into this repo. They would go
stale silently, and stale setup instructions fail in the worst possible way — six
months from now, for someone who then concludes the workshop is broken.
