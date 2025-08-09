# LLM-Based Anomaly Detection

This module provides language model-based anomaly detection for text fields using pre-trained transformers.

## Features

- **GPU Memory Management**: Automatic GPU memory management for RTX 3070 (8GB) and other GPUs
- **Safe Batch Sizing**: Dynamic batch size calculation based on available GPU memory
- **Memory Monitoring**: Real-time GPU memory usage tracking
- **Fallback Support**: Automatic fallback to CPU if GPU memory is insufficient
- **Mixed Precision**: FP16 training for memory efficiency on GPU

## GPU Memory Management

The system includes comprehensive GPU memory management for the RTX 3070 (8GB):

### Memory Allocation
- Uses 70% of available GPU memory (5.6GB for RTX 3070)
- Automatic memory fraction setting
- Memory cache clearing after operations

### Safe Batch Sizing
- Calculates safe batch size based on model size and available memory
- DistilBERT: ~260MB model size → safe batch size of 8
- Conservative default: batch size of 4 for most fields
- Automatic batch size reduction if OOM occurs

### Memory Monitoring
- Real-time GPU memory usage tracking
- Memory cleanup after training and inference
- Automatic fallback to CPU if memory < 1GB available

## Usage

```bash
# Basic training with GPU memory management
python anomaly_detectors/llm_based/llm_model_training.py \
    data/esqualo_2022_fall.csv \
    --field material \
    --variation baseline \
    --epochs 3 \
    --batch-size 4

# Force CPU usage
CUDA_VISIBLE_DEVICES="" python anomaly_detectors/llm_based/llm_model_training.py \
    data/esqualo_2022_fall.csv \
    --field material \
    --variation baseline
```

## Configuration

### Model Configurations
- **material**: batch_size=4, max_length=128
- **care_instructions**: batch_size=4, max_length=128  
- **long_description_nl**: batch_size=2, max_length=256
- **product_name_en**: batch_size=4, max_length=128

### GPU Settings
- Memory fraction: 70% of total GPU memory
- Mixed precision: Enabled on GPU
- Gradient accumulation: Automatic based on batch size
- Warmup steps: 100 or batch_size dependent

## Memory Optimization Features

1. **Automatic Memory Management**
   - GPU memory fraction setting
   - Memory cache clearing
   - Memory usage monitoring

2. **Safe Training**
   - Dynamic batch size calculation
   - OOM error handling with automatic retry
   - Gradient accumulation for small batches

3. **Memory Efficient Operations**
   - Mixed precision training (FP16)
   - Memory cleanup after operations
   - Efficient attention mechanisms

## Troubleshooting

### GPU Out of Memory
If you encounter GPU OOM errors:
1. Reduce batch size: `--batch-size 2`
2. Use CPU: Set `CUDA_VISIBLE_DEVICES=""`
3. Check memory usage: Monitor GPU memory in output

### Performance Optimization
- Use batch size 4-8 for RTX 3070
- Enable mixed precision (automatic on GPU)
- Monitor memory usage in training logs