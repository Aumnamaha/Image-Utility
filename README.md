# Image Utility

A modular image processing utility library for computer vision workflows.

## Features

- Synthetic test image generation (RGB + multi-spectral)
- Configurable resolution bounds and device selection via Pydantic settings
- Integration-ready with OpenCV, scikit-image, and PIL

## Installation

```bash
python -m pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
Image-Utility/
├── src/
│   ├── __init__.py
│   └── config.py          # Pydantic-settings based configuration
├── tests/
│   ├── __init__.py
│   └── conftest.py        # Synthetic image fixtures (RGB + multi-spectral)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## License

MIT
