# Bloodmoon

A Python library for analyzing data from the Wide Field Monitor (WFM) coded mask instrument. Bloodmoon provides tools for:
- Image reconstruction
- Source detection and parameter estimation
- Detector effects modeling (vignetting, PSF)
- Dual-camera source localization

> ⚠️ **Note**: Bloodmoon is under active development. APIs may change between versions.


## Installation

### PyPI (Coming Soon)
```bash
pip install CHANGEME
```

### From Source

Installing from source is necessary when doing development work. The exact process depends on your platform but will generally require:
- Git
- Python 3.11 or later
- pip
- venv or conda (for environment management)

#### Using venv
```bash
# Clone repository
git clone https://github.com/peppedilillo/bloodmoon.git
cd bloodmoon

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

#### Using Conda
```bash
# Clone repository
git clone https://github.com/peppedilillo/bloodmoon.git
cd bloodmoon

# Create and activate conda environment
conda create -n "bloodmoon" python=3.11
conda activate bloodmoon

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import bloodmoon as bm

# Load camera configuration
wfm = bm.codedmask("wfm_mask.fits")

# Load simulation data
sdl = bm.simulation("simdata/")

# Create detector images
detector, bins = bm.count(wfm, sdl.reconstructed["cam1a"])

# Reconstruct sky image
sky = bm.decode(wfm, detector)

# Run iterative source detection
for sources, residuals in bm.iros(wfm, sdl, max_iterations=10):
    # Process detected sources...
    pass
```

For more take a look at our [demo](demo/demo.ipynb).

## Development

### Running Tests

Assuming you installed from sources, and your source living into `bloodmon` directory.

```bash
cd bloodmoon
python -m unittest
```

## Contributing

Contributions are welcome! Before submitting a pull request:

1. Ensure all tests pass: `python -m unittest`
2. Format code with black: `black -l 120 .`
3. Sort imports with isort: `isort --profile=google .`

For bug reports and feature requests, please open an issue.