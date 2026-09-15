# Architecture

TODO (fill in as each phase lands):
- System diagram: model -> cache -> kernels -> runtime -> serving (see project plan doc, Section 48).
- Package layout and how each top-level dir maps to a phase.
- Data flow for a single request: ARRIVED -> QUEUED -> PREFILL -> DECODING -> FINISHED.

Populate this once Phase 4 (serving runtime) exists — no point documenting
an architecture that's still just directory stubs.
