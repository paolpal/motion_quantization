# Posemi Dataset

A comprehensive data processing pipeline for building multimodal datasets that correlate speech transcriptions with speaker pose sequences.

## Overview

This project provides an end-to-end pipeline for downloading, processing, and preparing video data for pose-speech correlation research. It processes videos from the PATS dataset (or similar sources) to generate structured datasets mapping transcribed text to corresponding speaker pose sequences.

## Architecture

The pipeline consists of five core modules:

### 1. Acquisition
Downloads source video files from online repositories, ensuring data availability for subsequent processing stages.

### 2. Extraction
Extracts essential features from video data:
- **Pose estimation**: Captures speaker body poses using computer vision models
- **Transcription**: Generates text transcriptions from audio tracks

### 3. Quantization
Builds a pose codebook through clustering techniques:
- Currently implements HDBSCAN for unsupervised clustering
- Designed for extensibility with alternative clustering methods
- Generates discrete pose representations (posemes) from continuous pose data

### 4. Dataset
Combines extracted features and the generated codebook to produce the final dataset:
- Outputs JSONL format for efficient streaming and processing
- Correlates transcriptions with corresponding pose sequences
- Provides structured data for downstream machine learning tasks

### 5. Utils
Provides common utilities and helper functions shared across modules.

## Output Format

The final dataset is generated in JSONL (JSON Lines) format, where each line represents a temporal segment containing:
- Transcribed text
- Corresponding pose sequence (posemes)
- Temporal alignment information 