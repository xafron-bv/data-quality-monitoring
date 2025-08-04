# Understanding the Application Entrypoints: A Practical Guide

Welcome! This guide will walk you through each entrypoint of the data quality detection system, explaining what they do, how they work, and when to use them. Think of this as a friendly conversation about the tools at your disposal.

## Overview: What Are These Entrypoints?

The system has six main entrypoints, each serving a specific purpose in your data quality journey. You access all of them through the main.py file, which acts as your command center. Let's explore each one.

## Single Demo: Your First Stop for Testing

The single-demo entrypoint is perfect when you want to quickly check data quality on a single CSV file. It's like running a comprehensive health check on your data.

### What It Does

This entrypoint takes your CSV file and runs it through multiple detection methods - validation rules, pattern detection, machine learning models, and optionally, language models. It then generates a beautiful HTML report that you can open in your browser to explore the results interactively.

### How It Works

When you run single-demo, here's what happens behind the scenes:

1. First, it loads your CSV file and maps the columns to standard field types like material, color, size, and category
2. Then, it runs each enabled detection method in parallel for efficiency
3. Validation checks come first - these are fast, rule-based checks for obvious errors
4. Pattern detection looks for statistical anomalies and unusual patterns
5. ML-based detection uses pre-trained models to find subtle anomalies
6. Finally, it compiles everything into an interactive HTML viewer

### Practical Usage

Here's the simplest way to use it:

```bash
python main.py single-demo --data-file your_data.csv
```

This runs all detection methods by default. But maybe you just want quick validation checks:

```bash
python main.py single-demo --data-file your_data.csv --enable-validation
```

Or perhaps you want validation and ML detection, but skip the others:

```bash
python main.py single-demo --data-file your_data.csv --enable-validation --enable-ml
```

The beauty is that you get immediate visual feedback. Open the generated HTML file, and you can filter by field, detection method, or confidence level. It's particularly useful when you're exploring a new dataset or debugging specific data quality issues.

## Multi-Eval: Performance Testing at Scale

While single-demo is great for exploration, multi-eval is your tool for systematic evaluation. It's designed to test how well the detection methods perform on datasets where you already know what the errors are.

### What It Does

Multi-eval processes multiple samples with known errors and generates detailed performance metrics. It tells you things like: How many real errors did we catch? How many false alarms did we raise? What's our precision and recall?

### How It Works

The process is more complex than single-demo:

1. It expects data with injected errors and ground truth labels
2. For each sample, it runs the detection methods
3. It compares detected anomalies against the known errors
4. It calculates confusion matrices, F1 scores, and other metrics
5. It generates both numerical results and visual performance charts

### Practical Usage

Basic usage for evaluation:

```bash
python main.py multi-eval --data-dir ./evaluation_data --output-dir ./results
```

You can focus on specific detection methods:

```bash
python main.py multi-eval --data-dir ./evaluation_data --enable-ml --enable-pattern
```

This is invaluable when you're tuning detection thresholds or comparing different approaches. The generated reports show you exactly where each method succeeds and fails.

## ML-Train: Building Your Detection Models

The ml-train entrypoint is where the magic of machine learning begins. It trains custom anomaly detection models for your specific data patterns.

### What It Does

This entrypoint takes clean training data and builds field-specific models using sentence transformers. These models learn what "normal" looks like for each field type, so they can spot anomalies later.

### How It Works

The training process is sophisticated:

1. It loads your training data and processes each configured field
2. For each field, it uses sentence transformers to create embeddings
3. It employs triplet loss training - learning to distinguish between similar and dissimilar values
4. The models learn representations that cluster similar values together
5. It saves the trained models and their performance metrics

### Practical Usage

Basic training command:

```bash
python main.py ml-train --brand your_brand
```

With custom parameters:

```bash
python main.py ml-train --brand your_brand --epochs 5 --min-samples 100
```

The key insight here is that you need clean, representative training data. The models learn from examples, so garbage in means garbage out. After training, you'll see metrics showing how well each field's model performs.

## LLM-Train: Leveraging Language Models

The llm-train entrypoint is the newest addition, bringing the power of large language models to anomaly detection.

### What It Does

Instead of training from scratch, this entrypoint fine-tunes pre-trained language models to understand your specific data domains. It's particularly powerful for text-heavy fields like descriptions or categories.

### How It Works

The LLM training process:

1. Starts with a pre-trained language model
2. Creates prompts that teach the model about your data patterns
3. Fine-tunes using your training examples
4. Optimizes for both accuracy and efficiency
5. Saves the adapted model for inference

### Practical Usage

Basic LLM training:

```bash
python main.py llm-train --brand your_brand --model-size small
```

With specific configuration:

```bash
python main.py llm-train --brand your_brand --fields "description,category" --training-samples 1000
```

LLM detection is powerful but resource-intensive. Use it for fields where context and semantics matter most.

## Analyze-Column: Deep Dive Analysis

Sometimes you need to understand a specific column in detail. That's where analyze-column comes in.

### What It Does

This entrypoint provides detailed statistical and pattern analysis for a single column. It's like putting that column under a microscope.

### How It Works

The analysis process examines:

1. Basic statistics - unique values, distributions, patterns
2. Common patterns and outliers
3. Data type consistency
4. Missing value patterns
5. Potential data quality issues

### Practical Usage

Analyze a specific column:

```bash
python main.py analyze-column --data-file data.csv --column-name "material"
```

With additional options:

```bash
python main.py analyze-column --data-file data.csv --column-name "size" --show-patterns --export-stats
```

This is particularly useful during initial data exploration or when debugging specific field issues.

## ML-Curves: Finding Optimal Thresholds

The ml-curves entrypoint is your optimization tool. It helps you find the perfect balance between catching errors and avoiding false alarms.

### What It Does

This tool runs your detection methods across multiple threshold values and plots performance curves. It shows you the trade-offs between precision and recall at different sensitivity levels.

### How It Works

The curve generation process:

1. Takes a dataset with known errors
2. Runs detection methods with varying thresholds
3. Calculates precision, recall, and F1 scores at each threshold
4. Generates beautiful plots showing the relationships
5. Identifies optimal threshold values

### Practical Usage

Generate curves for all fields:

```bash
python main.py ml-curves --brand your_brand
```

For specific fields with custom ranges:

```bash
python main.py ml-curves --brand your_brand --fields "material,color" --threshold-range "0.5,0.9"
```

The ROC curves and precision-recall plots help you make informed decisions about threshold settings. You can see exactly how changing thresholds affects your detection performance.

## Putting It All Together

Now that you understand each entrypoint, here's a typical workflow:

1. **Start with analyze-column** to understand your data
2. **Use single-demo** to get a quick quality assessment  
3. **Train models with ml-train** using clean data
4. **Optimize thresholds with ml-curves** 
5. **Evaluate performance with multi-eval**
6. **Deploy with confidence** knowing your detection settings

Each entrypoint serves a specific purpose in the data quality journey. Single-demo is your quick check tool. Multi-eval is your performance validator. ML-train builds your custom models. LLM-train adds semantic understanding. Analyze-column provides deep insights. ML-curves optimizes your settings.

## Tips for Success

Here are some practical tips from experience:

- Always start with validation rules - they're fast and catch obvious errors
- Use pattern detection for statistical anomalies
- Reserve ML detection for subtle patterns that rules miss
- Only enable LLM detection when you need semantic understanding
- Train models on clean, representative data
- Use ml-curves to find the sweet spot between false positives and false negatives
- Run multi-eval regularly to track performance

## Common Scenarios

**Scenario 1: Quick Data Quality Check**
Use single-demo with all methods enabled. Open the HTML report and filter by high-confidence anomalies.

**Scenario 2: Production Deployment Setup**
Train models with ml-train, optimize with ml-curves, validate with multi-eval, then use those settings in production.

**Scenario 3: Investigating Specific Issues**
Use analyze-column to understand the problematic field, then single-demo with targeted detection methods.

**Scenario 4: Continuous Improvement**
Regularly run multi-eval on new data, retrain models when performance drops, adjust thresholds as needed.

Remember, the key to effective anomaly detection is understanding your data and choosing the right tool for each situation. These entrypoints give you everything you need for comprehensive data quality management.