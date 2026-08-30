# Security Policy

Report vulnerabilities privately through GitHub Security Advisories. Do not open public issues containing exploit details, tokens, private paths, or user positions.

The supported boundary is the 0.2 loopback server under `src/gemmafischer`.
The historical Flask/MoE application was removed from `main` and is recoverable
only from `archive/pre-recovery-2026-08-30`; it is unsupported and must not be
restored and exposed to a network.

GemmaFischer is a local application, not a LAN service. Report any path that
accepts a non-loopback Host or foreign Origin, bypasses the capability token on
a mutation, exposes arbitrary filesystem/process controls, or loads unpinned
model code or assets.
