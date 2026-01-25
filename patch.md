# Apply patch to Flashinfer
This repo fix a mask interface bug in Flashinfer v0.5.3, please follow the instruction below to apply the patch

## 1. Initialize submodule to official v0.5.3
```
git submodule update --init --recursive
```

## 2. Apply the patch to the submodule directory
```
patch -p1 -d third_party/flashinfer < patches/flashinfer_v0.5.3.patch
```

## 3. Add editable repo in the uv envrionment
```
uv add --editable ./third_party/flashinfer
```