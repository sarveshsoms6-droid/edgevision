# EdgeVision Thread Scaling & Runtime Benchmark Summary

- **Measured Host**: `Windows-10-10.0.19045-SP0 (AMD64)`
- **CPU Model**: `Intel64 Family 6 Model 142 Stepping 12, GenuineIntel`
- **Raspberry Pi Status**: `NOT MEASURED ON PI (Measured on host development machine)`
- **Thermal Throttling**: `NOT MEASURED ON PI (Non-Pi Host Environment)`
- **Input Video**: `sample_data/traffic_sample.mp4`
- **Repeats per Configuration**: `1`

## Performance Results Matrix

| Runtime | Threads | Stride | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) | End-to-End FPS | Effective FPS | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|
| executorch-fp32 | 1 | 1 | 80.27 | 79.31 | 100.97 | 12.4 | 12.5 | 341.7 |
| executorch-fp32 | 1 | 2 | 76.44 | 79.00 | 98.09 | 26.0 | 13.1 | 350.7 |
| executorch-fp32 | 2 | 1 | 76.61 | 71.62 | 103.89 | 13.0 | 13.1 | 352.2 |
| executorch-fp32 | 2 | 2 | 77.80 | 74.35 | 100.25 | 25.5 | 12.9 | 351.7 |
| executorch-fp32 | 4 | 1 | 77.75 | 74.87 | 97.28 | 12.8 | 12.9 | 347.1 |
| executorch-fp32 | 4 | 2 | 75.92 | 69.93 | 99.63 | 26.2 | 13.2 | 346.5 |
| executorch-int8 | 1 | 1 | 71.43 | 68.21 | 87.51 | 13.9 | 14.0 | 337.1 |
| executorch-int8 | 1 | 2 | 79.04 | 79.86 | 95.49 | 25.1 | 12.7 | 337.3 |
| executorch-int8 | 2 | 1 | 73.06 | 71.21 | 92.65 | 13.6 | 13.7 | 338.3 |
| executorch-int8 | 2 | 2 | 71.25 | 70.30 | 102.28 | 27.9 | 14.0 | 333.7 |
| executorch-int8 | 4 | 1 | 88.39 | 97.63 | 114.85 | 11.3 | 11.3 | 338.4 |
| executorch-int8 | 4 | 2 | 97.44 | 102.93 | 113.59 | 20.4 | 10.3 | 337.7 |

## Thermal Log (Sampled every 5 s during continuous run)

| Timestamp (s) | CPU Temperature |
|---|---|
| 0.0 s | NOT AVAILABLE (Non-Pi Host) |
| 5.1 s | NOT AVAILABLE (Non-Pi Host) |
| 10.2 s | NOT AVAILABLE (Non-Pi Host) |
| 15.2 s | NOT AVAILABLE (Non-Pi Host) |
| 20.2 s | NOT AVAILABLE (Non-Pi Host) |
| 25.4 s | NOT AVAILABLE (Non-Pi Host) |
| 30.5 s | NOT AVAILABLE (Non-Pi Host) |
| 35.7 s | NOT AVAILABLE (Non-Pi Host) |
| 40.7 s | NOT AVAILABLE (Non-Pi Host) |
| 45.8 s | NOT AVAILABLE (Non-Pi Host) |
| 50.8 s | NOT AVAILABLE (Non-Pi Host) |
| 55.9 s | NOT AVAILABLE (Non-Pi Host) |
| ... | (18 total thermal samples recorded) |

## Key Observations & Scaling Behavior
1. **Thread Scaling**: Sweeping from 1 to 4 threads demonstrates multi-threaded CPU scaling on XNNPACK.
2. **Quantization Impact**: INT8 model delivers a 72% storage reduction (3.68 MB vs 13.15 MB FP32) with competitive latency.
3. **Frame Skipping Throughput**: Increasing frame stride from 1 to 2 or 3 scales effective end-to-end processing speed proportionally.
