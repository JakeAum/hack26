# Will Code for Corn 🌽❤️

![hamster.png](hamster.png)

## The Challenge

Accurate, timely corn yield forecasts shape decisions across food security, commodity markets, and agricultural policy yet traditional survey-based estimates remain slow, costly, and coarse. Our team is building a geospatial AI pipeline that turns Earth observation data into faster, more granular, and more trustworthy yield predictions.

## Our Mission

Deliver a transparent, reproducible pipeline that helps farmers, analysts, and policymakers see the season ahead with greater confidence.

# Environment Setup

To install required dependencies, run one of the following:

```bash
pip install -r software/requirements.txt   # Standard installation
# or, for editable install (so code changes are reflected instantly):
pip install -e .
# or, if you have `uv` installed (faster installs):
uv pip sync software/requirements.txt
uv pip install -e . --no-deps
```

### (Optional) AWS EFS Data Mount

If using an AWS compute instance with EFS, you may want to symlink the data directory:

```bash
EFS=/mnt/custom-file-systems/efs/fs-014621b1d53629dd9_fsap-05629cb3c5373e174
ln -s "$EFS/cdls_data" ~/hack26/data
```
