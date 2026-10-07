# Development Notes

## Final benchmark

Input: 1280 x 1280 RGB
Output: 100 x 100 RGB
Window: 5 x 5

Before optimization: 5366.72 s
After one-term-A optimization: 3952.68 s

Improvement:
- 1414.04 s saved
- ~26.35% less runtime
- ~1.36x faster

Correctness:
- Expected pixels: 10000
- Known pixels: 10000
- Unknown pixels: 0

## Rejected optimization

A bounded mask cache was tested but produced only 1.94% hit rate on the 30 x 30 test and increased runtime. It was removed from the final pipeline.

A batched vectorized patch matcher was also tested on one frontier pixel:
- correlate2d: 0.590844 s
- batched vectorized: 0.645236 s
- speedup: 0.92x

The batched version matched the candidate count and best distance but was slower, so it was not integrated.
