# DA6401 Assignment 2 - Visual Perception Pipeline

## WandB Report
[Link to be added after training]

## GitHub Repo
[Link to be added]

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
# Step 1: Train classifier
python train.py --task classify --data_dir /path/to/dataset --epochs 20

# Step 2: Train localizer (uses pretrained encoder from classifier)
python train.py --task localize --data_dir /path/to/dataset --epochs 20

# Step 3: Train segmentation (full fine-tuning by default)
python train.py --task segment --data_dir /path/to/dataset --epochs 20 --finetune_strategy full
```

## Inference
```bash
python inference.py --image /path/to/image.jpg --checkpoint_dir checkpoints
```

## Project Structure
```
.
├── checkpoints/
├── data/
│   └── pets_dataset.py
├── losses/
│   ├── __init__.py
│   └── iou_loss.py
├── models/
│   ├── __init__.py
│   ├── classification.py
│   ├── layers.py
│   ├── localization.py
│   ├── multitask.py
│   ├── segmentation.py
│   └── vgg11.py
├── inference.py
├── requirements.txt
├── train.py
└── README.md
```