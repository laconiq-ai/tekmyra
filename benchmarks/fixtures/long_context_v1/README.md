# Long-Context Fixture Suite v1

Schema version: `2.1.0`
Generation seed: `20260723`
Generator: `benchmarks/fixtures/long_context_v1/generate_long_context_fixtures.py`

## Contents

This suite contains synthetic, sanitized long-context fixtures for the
LaconIQ NextGen Compression Router evaluation. All content is generated; no real
PII or customer data is included.

### Category counts

- agent_handoffs: 6
- distractor_dense: 6
- incident_reports: 6
- issue_threads: 6
- long_prose: 8
- rag_multi_doc: 8

### Length-bucket counts

- calibration: 12
- eligible: 12
- primary_5k_10k: 8
- primary_10k_plus: 8

### Total fixtures

40

## Held-out split

File: `held_out_split.json`
SHA-256: `55d3f06b1eab6680ed03b36294b209f81bb9bc72d0e66d6e2c82552fb4e37ea4`
Holdout ratio: 20% stratified by category and length bucket.

## Regeneration

```bash
python benchmarks/fixtures/long_context_v1/generate_long_context_fixtures.py
```
