# bandit-lb

High-performance, asynchronous contextual bandit dynamic request router and load balancer in Python.

## Core Features
* **Adaptive Routing:** Uses LinUCB and Linear Thompson Sampling to select optimal upstream backends based on real-time request context and response telemetry.
* **Non-Blocking Architecture:** Built on FastAPI, Starlette, and `httpx.AsyncClient` connection pooling.
* **Synthetic Latency Simulator:** Configurable upstream mock backends with Normal, Gamma, and Pareto latency distributions, jitter, and failure injection.
