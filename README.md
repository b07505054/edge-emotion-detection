# Edge AI Facial Emotion Recognition (Raspberry Pi, TFLite)



A production-style edge AI system for real-time facial emotion recognition, focusing on:



- On-device inference (CPU-only)

- Model optimization (quantization)

- System-level profiling (latency + FPS)

- Hardware-aware performance analysis



---



## Overview



This project demonstrates how to build a \*\*real-time edge AI system\*\* deployed on Raspberry Pi.



Unlike a typical ML project that focuses only on model accuracy, this system emphasizes:



- End-to-end deployment

- Latency breakdown and profiling

- Hardware-aware optimization

- Trade-offs between accuracy, latency, and memory



---



## Architecture



Camera (Picamera2)

↓

Frame Capture

↓

Face Detection (OpenCV)

↓

Preprocessing

↓

TFLite Inference (FP32 / INT8)

↓

Prediction Overlay / Logging



---



## Features



### Edge Deployment

- Raspberry Pi (CPU-only inference)

- Picamera2 camera pipeline

- TensorFlow Lite runtime


### Model Optimization

- FP32 baseline model

- Dynamic quantization

- Full INT8 quantization


### System Profiling

- Stage-level latency breakdown:

&#x20; - camera capture

&#x20; - preprocessing

&#x20; - inference

- FPS measurement (real-time performance)



---



## Model Training



- Dataset: FER2013 (7-class emotion classification)

- Backbone: MobileNetV2

- Optimization:

&#x20; - Class weighting (to handle imbalance)

&#x20; - Fine-tuning



*\*Result:\*\*

- Accuracy improved from \*\*49.4% → 53.4%\*\*



---



## Edge Deployment Performance



### 📊 Raspberry Pi Benchmark (CPU-only)



| Model | Accuracy | Size (MB) | Capture (ms) | Preprocess (ms) | Inference (ms) | Total (ms) | FPS |

|------|---------|----------|-------------|----------------|---------------|-----------|------|

| FP32 | 53.05% | 2.96 | 33.72 | 0.64 | 4.58 | 39.66 | 25.2 |

| INT8 | 50.88% | 1.03 | 26.04 | 0.91 | 5.11 | 32.87 | 30.4 |



---



## Key Observations



- INT8 quantization reduced model size by \*\*\~65%\*\* with \~2% accuracy drop

- INT8 improved end-to-end latency (\*\*\~17% faster total latency\*\*)

- However, INT8 inference alone was slightly slower than FP32



---



## Bottleneck Analysis



- The system is \*\*not compute-bound\*\*

- Inference latency: \*\*\~4–5 ms\*\*

- Camera capture latency: \*\*\~26–34 ms\*\*



👉 \*\*Camera IO dominates system latency\*\*



---



## System Profiling



### ⚙️ Latency Breakdown



| Stage | Latency (ms) |

|------|-------------|

| Camera Capture | \~26–34 |

| Preprocessing | \~0.6–0.9 |

| Inference | \~4–5 |

| Total | \~33–40 |



---



## Roofline Insight



- Reducing compute precision (INT8) did not significantly reduce inference latency

- Indicates \*\*memory-bound behavior\*\*

- Performance is dominated by:

&#x20; - memory access

&#x20; - data movement (camera pipeline)



👉 Optimization should focus on \*\*system-level pipeline\*\*, not only model



---



## Real-Time Performance



- Achieved \*\*25–30 FPS\*\* on Raspberry Pi (CPU-only)

- Performance varies depending on:

&#x20; - camera pipeline latency

&#x20; - scene complexity



---


## System Design Highlights



- \*\*End-to-end edge AI pipeline\*\*

&#x20; Camera → detection → inference → output


- \*\*Hardware-aware optimization\*\*

&#x20; Evaluated model performance directly on Raspberry Pi



- \*\*Quantization trade-off analysis\*\*

&#x20; Compared FP32 vs INT8 across:

&#x20; - accuracy

&#x20; - latency

&#x20; - memory footprint



- \*\*System-level profiling\*\*

&#x20; Identified bottlenecks beyond model inference



---



## Key Insights



- Model optimization alone is insufficient for edge AI

- System bottlenecks often lie in:

&#x20; - input pipeline

&#x20; - memory bandwidth

- Quantization benefits depend on runtime and hardware



---



## Tech Stack



- Python

- TensorFlow / Keras

- TensorFlow Lite

- OpenCV

- Picamera2

- Raspberry Pi



---



## Future Improvements



- Multi-thread pipeline (capture / inference separation)

- Frame skipping strategy

- Hardware acceleration (NPU / Coral / GPU)

- More robust face detection (MediaPipe / DNN)



---



## Demo



Example output:



- Real-time FPS overlay

- Emotion prediction with confidence

- Latency breakdown logs


---



## Summary



This project demonstrates that:



> Edge AI performance is a \*\*system-level problem\*\*, not just a model problem.



Through on-device profiling, we identified that:



- Camera IO dominates latency

- The system is memory-bound rather than compute-bound



This highlights the importance of \*\*hardware-aware design and end-to-end optimization\*\* in real-world AI systems.

