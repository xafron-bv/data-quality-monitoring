# System Architecture Overview

The Data Quality Detection System is built on a modular, extensible architecture that enables multiple detection methods to work together seamlessly. This document provides a comprehensive overview of the system's architecture, design principles, and key components.

## Design Principles

### 1. Modularity
Each detection method is self-contained and implements common interfaces, allowing new methods to be added without modifying existing code.

### 2. Extensibility
The system is designed to be easily extended with new fields, detection methods, and output formats through configuration and plugins.

### 3. Performance
Sequential processing, model caching, and GPU acceleration ensure efficient resource usage even with large datasets.

### 4. Flexibility
Configurable thresholds, weights, and field mappings allow the system to adapt to different domains and use cases.

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Interface"
        UI1[CLI Tools]
        UI2[HTML Viewer]
        UI3[API Endpoints]
    end
    
    subgraph "Entry Points"
        EP1[single_sample_demo]
        EP2[multi_sample_evaluation]
        EP3[ml_curve_generator]
        EP4[analyze_column]
    end
    
    subgraph "Orchestration"
        O1[ComprehensiveFieldDetector]
        O2[ConsolidatedReporter]
        O3[ConfusionMatrixAnalyzer]
    end
    
    subgraph "Detection Engine"
        DE1[Validation Engine]
        DE2[Pattern Detector]
        DE3[ML Detector]
        DE4[LLM Detector]
    end
    
    subgraph "Core Services"
        CS1[Field Mapper]
        CS2[Brand Config]
        CS3[Error Injection]
        CS4[Model Cache]
    end
    
    subgraph "Data Storage"
        DS1[(CSV Data)]
        DS2[(JSON Configs)]
        DS3[(ML Models)]
        DS4[(Detection Results)]
    end
    
    UI1 --> EP1 & EP2 & EP3 & EP4
    EP1 & EP2 --> O1
    O1 --> DE1 & DE2 & DE3 & DE4
    DE1 & DE2 & DE3 & DE4 --> CS1 & CS2 & CS3 & CS4
    CS1 & CS2 & CS3 & CS4 --> DS1 & DS2 & DS3
    O1 --> O2 & O3
    O2 & O3 --> DS4
    UI2 --> DS4
```

## Layer Architecture

The system is organized into distinct layers, each with specific responsibilities:

### 1. Entry Points Layer

This layer provides various ways to interact with the system:

- **Demo Scripts**: Quick demonstration and testing
- **Evaluation Tools**: Performance measurement and comparison
- **Utility Scripts**: Data analysis and preparation

### 2. Orchestration Layer

Coordinates the detection workflow:

- **ComprehensiveFieldDetector**: Manages detection across all fields and methods
- **Evaluator**: Handles performance evaluation and metrics
- **UnifiedInterface**: Provides consistent API for all detection methods

### 3. Detection Methods Layer

Implements the core detection algorithms:

```mermaid
graph LR
    subgraph "Detection Methods"
        V[Validation<br/>100% Confidence]
        P[Pattern-Based<br/>70-80% Confidence]
        M[ML-Based<br/>Configurable]
        L[LLM-Based<br/>Configurable]
    end
    
    V --> VR[Rule Engine]
    P --> PR[Pattern Matcher]
    M --> MR[Similarity Engine]
    L --> LR[Language Model]
```

### 4. Core Services Layer

Provides shared functionality:

- **FieldMapper**: Translates between standard fields and column names
- **BrandConfig**: Manages brand-specific configurations
- **ErrorInjection**: Generates synthetic errors for testing
- **Reporters**: Formats and outputs detection results

### 5. Data Layer

Handles all data storage and retrieval:

- **Input Data**: CSV files with structured data
- **Configuration**: JSON files for settings and rules
- **Models**: Trained ML/LLM models
- **Results**: Detection reports and analyzed data

## Component Interactions

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Orchestrator
    participant Detector
    participant Reporter
    participant Storage
    
    User->>CLI: Run detection command
    CLI->>Orchestrator: Initialize with config
    Orchestrator->>Storage: Load data
    
    loop For each field
        Orchestrator->>Detector: Detect anomalies
        Detector->>Storage: Load model/rules
        Detector-->>Orchestrator: Return results
    end
    
    Orchestrator->>Reporter: Generate report
    Reporter->>Storage: Save results
    Reporter-->>User: Display summary
```

## Detection Flow

The system processes data through a well-defined flow:

```mermaid
flowchart TD
    A[Input Data] --> B{Field Mapping}
    B --> C[Field Selection]
    C --> D{Detection Method}
    
    D --> E[Validation]
    D --> F[Pattern Detection]
    D --> G[ML Detection]
    D --> H[LLM Detection]
    
    E & F & G & H --> I[Result Aggregation]
    I --> J{Combination Strategy}
    
    J --> K[Priority-Based]
    J --> L[Weighted Average]
    
    K & L --> M[Final Results]
    M --> N[Report Generation]
    N --> O[Output Files]
```

## Memory Management

The system implements several strategies for efficient memory usage:

### Sequential Processing
Fields are processed one at a time to minimize memory footprint:

```python
for field in fields:
    results = detect_field(field)
    save_results(results)
    clear_cache()
```

### Model Caching
Models are loaded once and reused:

```mermaid
graph LR
    A[First Request] --> B{Model in Cache?}
    B -->|No| C[Load Model]
    C --> D[Add to Cache]
    D --> E[Use Model]
    B -->|Yes| E
    E --> F[Return Results]
```

### Batch Processing
Data is processed in configurable batches to balance memory and performance.

## Scalability Considerations

### Horizontal Scaling
- Field-level parallelization
- Independent detection methods
- Distributed processing support

### Vertical Scaling
- GPU acceleration for ML/LLM
- Optimized algorithms
- Efficient data structures

## Security Architecture

```mermaid
graph TB
    subgraph "Security Layers"
        S1[Input Validation]
        S2[Access Control]
        S3[Data Sanitization]
        S4[Output Filtering]
    end
    
    I[Input] --> S1
    S1 --> S2
    S2 --> P[Processing]
    P --> S3
    S3 --> S4
    S4 --> O[Output]
```

## Extension Points

The architecture provides several extension points for customization:

1. **New Detection Methods**: Implement `AnomalyDetectorInterface`
2. **Custom Validators**: Implement `ValidatorInterface`
3. **Output Formats**: Implement `ReporterInterface`
4. **Field Types**: Add configuration and rules
5. **Brand Support**: Add brand configuration files

## Performance Optimization

The system includes several performance optimizations:

- **Lazy Loading**: Models loaded only when needed
- **Result Caching**: Avoid redundant computations
- **Parallel Processing**: Multi-threading for independent operations
- **GPU Acceleration**: CUDA support for ML operations

## Next Steps

- Learn about [Core Components](core-components.md) in detail
- Understand [Detection Methods](detection-methods.md) implementation
- Explore [Data Flow](data-flow.md) through the system