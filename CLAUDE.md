
# Image Spectral Segmentation

## Goal
Build a reliable full-stack C++ image segmentation application that demonstrates numerical computing and backend engineering skills.

Stack: C++17, CMake, Eigen, Spectra, OpenCV, PostgreSQL, React, Docker

## Engineering principles
- Prioritize correctness, simplicity, and end-to-end functionality
- Reuse existing infrastructure; avoid unnecessary dependencies and abstractions
- Keep numerical computation, HTTP handling, and persistence separate
- Prefer established libraries over reimplementing standard algorithms

C++ specifics:
- Use RAII and appropriate synchronization for shared state; avoid std::shared_ptr when weak_ptr or unique_ptr can be used
- take advantage of cache locality (e.g. use std::vector over std::list)
- Ensure modern C++ solutions over C-style code (e.g. type casting and passing, passing read-only parameters as const&)
- Apply Banker's Algorithm on std::mutex instances
- Follow Rule of Three, Five, and Zero precisely for classes


## Backend invariants
- Every result has a unique ID and immutable parameters/metrics
- Saving results is idempotent and uses the exact result ID
- PostgreSQL is the source of truth for saved experiments
- Cache entries must be correctly keyed and safely invalidated
- Prevent stale requests from overwriting current frontend state

## Workflow
- Implement only the requested stage
- Make routine decisions independently; ask only about major trade-offs
- Don't create tests unless prompted
- Keep responses concise: decisions, test results, and blockers
- Minimal commenting; only explain semantics of underlying logic and any clarification on what the declared object stores or what the function does