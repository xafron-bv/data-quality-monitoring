# Contributing Guide

Thank you for your interest in contributing to the Data Quality Detection System! This guide will help you get started with development.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, virtualenv)
- GPU (optional, for ML/LLM development)

### Setting Up Your Development Environment

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/data-quality-detection.git
   cd data-quality-detection
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

   This installs hooks that automatically check your code before each commit.

## Code Quality Standards

### Pre-commit Hooks

The project uses pre-commit hooks to maintain code quality. These run automatically before each commit and check for:

- **Import sorting** (isort)
- **Code formatting** (black)
- **Linting** (flake8 with extensions)
- **Type hints** (mypy)
- **Security issues** (bandit)
- **YAML/JSON syntax**
- **Trailing whitespace**
- **File endings**

To run the checks manually:
```bash
pre-commit run --all-files
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines
- Use meaningful variable and function names

### Testing

While the project doesn't currently have a formal test framework, please:

1. **Test your changes manually**:
   ```bash
   # Test with sample data
   python main.py single-demo --data-file data/sample.csv
   ```

2. **Run evaluation mode** to ensure detection accuracy:
   ```bash
   python main.py multi-eval data/sample.csv --field your_field
   ```

3. **Verify no regressions** in existing functionality

## Making Contributions

### Types of Contributions

- **Bug Fixes**: Fix issues in existing code
- **New Features**: Add new detection methods or fields
- **Documentation**: Improve or add documentation
- **Performance**: Optimize existing code
- **Refactoring**: Improve code structure

### Contribution Process

1. **Create an Issue** (optional but recommended)
   - Describe what you plan to work on
   - Get feedback before starting major work

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Follow the existing code structure
   - Update relevant documentation

4. **Run Quality Checks**
   ```bash
   pre-commit run --all-files
   ```

5. **Test Your Changes**
   ```bash
   # Run detection on sample data
   python main.py single-demo --data-file your_test_data.csv
   
   # Run evaluation if applicable
   python main.py multi-eval your_test_data.csv --field affected_field
   ```

6. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add support for new field type"
   ```

   Follow conventional commit format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `perf:` for performance improvements
   - `refactor:` for code refactoring

7. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Guidelines

- **Title**: Clear, descriptive title
- **Description**: Explain what changes you made and why
- **Testing**: Describe how you tested the changes
- **Documentation**: Note any documentation updates
- **Breaking Changes**: Clearly mark if applicable

## Development Guidelines

### Adding New Fields

See the [Adding Fields Guide](adding-fields.md) for detailed instructions.

### Adding Detection Methods

1. Implement the appropriate interface:
   - `ValidatorInterface` for rule-based validation
   - `AnomalyDetectorInterface` for anomaly detection

2. Add configuration support
3. Update documentation
4. Test thoroughly

### Project Structure

Follow the existing structure:
```
project_root/
├── validators/           # Rule-based validators
├── anomaly_detectors/   # Detection method implementations
├── common/              # Shared utilities
├── brand_configs/       # Brand configuration files
└── docs/               # Documentation
```

## Common Development Tasks

### Running with Debug Mode

```bash
python main.py single-demo --data-file data.csv --debug
```

### Analyzing Performance

```bash
python main.py ml-curves data.csv --fields material
```

### Training Models

```bash
python main.py ml-train training_data.csv --fields "new_field"
```

## Getting Help

- Check existing documentation
- Look at similar implementations in the codebase
- Open an issue for questions
- Reach out to maintainers

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

Thank you for contributing!