# Face Recognition using FaceNet with INT8 Quantization for FPGA Deployment

## Overview
This project implements a face recognition pipeline using pretrained
FaceNet embeddings and evaluates the impact of INT8 quantization on
classification accuracy. The goal is to design an FPGA-friendly
face recognition system that achieves high accuracy with reduced
computational and memory requirements.

The pipeline separates feature extraction and classification:
- FaceNet is used offline to extract face embeddings.
- Quantized embeddings and linear SVM classification are designed for
  efficient FPGA and edge deployment.

---

## Dataset
- **Dataset:** Labeled Faces in the Wild (LFW)
- **Preprocessing:** Only identities with at least 5 images are used
  to ensure fair train–test evaluation.
- **Embedding Dimension:** 512 (keras-facenet default)

---

## Methodology

### 1. Face Embedding Extraction
- Pretrained FaceNet model (keras-facenet)
- Produces 512-dimensional face embeddings
- This step is GPU-intensive and executed only once

### 2. INT8 Quantization
- Symmetric INT8 quantization is applied to FaceNet embeddings
- Significantly reduces memory footprint
- Suitable for fixed-point hardware implementation

### 3. Classification
- Linear SVM classifier (one-vs-rest)
- Trained separately on FP32 and INT8 embeddings
- Simple multiply–accumulate operations make it FPGA-friendly

### 4. Evaluation Strategy
- 80–20 train–test split
- Accuracy compared between FP32 and INT8 pipelines

---

## Results

| Representation | Test Accuracy |
|---------------|---------------|
| FP32 | 97.66% |
| INT8 | 97.41% |

- **Accuracy drop due to INT8 quantization:** ~0.25%
- Demonstrates robustness of FaceNet embeddings to reduced precision

---

## FPGA-Oriented Design

The FPGA implementation focuses on the post-embedding classification
stage:
- INT8 quantization
- Fixed-point linear SVM inference
- Threshold-based decision logic

FaceNet embedding extraction is performed offline and not mapped to
hardware.

---

## Files in this Repository
- `FaceNet_SVM_QAT.ipynb` – Main project notebook
- `README.md` – Project overview and documentation

---

## How to Run
1. Open the notebook
2. Use cached embeddings (no GPU required)
3. Run SVM training and evaluation cells

> Note: GPU-based embedding extraction is a one-time step and is not
> required for subsequent runs.

---

## Conclusion
This project demonstrates that INT8 quantization of FaceNet embeddings
results in negligible accuracy loss while enabling efficient hardware
deployment. The proposed pipeline is well suited for FPGA-based face
recognition systems.

---

## Future Work
- Implement INT8 SVM inference using HLS or RTL on FPGA
- Measure latency and resource utilization on hardware
- Extend the system for real-time edge deployment

