---
license: cc-by-4.0
task_categories:
  - object-detection
tags:
  - roboflow
  - roboflow-100
  - rf100
  - yolo
  - libreyolo
  - real-world
  - computer-vision
  - bounding-box
pretty_name: "Road Traffic"
size_categories:
  - 1K<n<10K
dataset_info:
  features:
    - name: image
      dtype: image
    - name: label
      dtype: string
  splits:
    - name: train
      num_examples: 494
    - name: validation
      num_examples: 187
    - name: test
      num_examples: 133
---

# Road Traffic

This dataset is part of the **Roboflow 100** benchmark, a diverse collection of 100 object detection datasets spanning 7 imagery domains.

## Dataset Description

- **Source:** [Roboflow 100](https://github.com/roboflow/roboflow-100-benchmark)
- **Category:** Real World
- **License:** CC-BY-4.0
- **Format:** YOLO (LibreYOLO compatible)
- **Mirrored on:** 2026-01-20

## Dataset Statistics

| Split | Images |
|-------|--------|
| Train | 494 |
| Validation | 187 |
| Test | 133 |
| **Total** | **814** |

## Classes (12)

  - bicycles
  - bridges
  - buses
  - chimneys
  - crosswalks
  - fire hydrants
  - motorcycles
  - stairs
  - taxis
  - tractors
  - traffic lights
  - vehicles

## Usage

### With LibreYOLO

```python
from libreyolo import LIBREYOLO

# Load a model
model = LIBREYOLO(model_path="libreyoloXnano.pt")

# Train on this dataset
model.train(data='path/to/data.yaml', epochs=100)
```

### Download from HuggingFace

```python
from huggingface_hub import snapshot_download

# Download the dataset
snapshot_download(
    repo_id="Libre-YOLO/road-traffic",
    repo_type="dataset",
    local_dir="./road-traffic"
)
```

## Directory Structure

```
road-traffic/
├── data.yaml           # Dataset configuration
├── README.md           # This file
├── train/
│   ├── images/         # Training images
│   └── labels/         # Training labels (YOLO format)
├── valid/
│   ├── images/         # Validation images
│   └── labels/         # Validation labels
└── test/
    ├── images/         # Test images (if available)
    └── labels/         # Test labels
```

## Label Format

Labels are in YOLO format (one `.txt` file per image):
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized to [0, 1].

## Citation

If you use this dataset, please cite the Roboflow 100 benchmark:

```bibtex
@misc{rf100_2022,
    Author = {Floriana Ciaglia and Francesco Saverio Zuppichini and Paul Guerrie and Mark McQuade and Jacob Solawetz},
    Title = {Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark},
    Year = {2022},
    Eprint = {arXiv:2211.13523},
}
```

## License

This dataset is released under the **CC-BY-4.0** license. 
Please check the original source for any additional terms.

## Acknowledgments

- Original dataset from [Roboflow Universe](https://universe.roboflow.com/roboflow-100/road-traffic)
- Part of the [Roboflow 100 Benchmark](https://www.rf100.org/)
- Sponsored by Intel
