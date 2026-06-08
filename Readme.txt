Dataset Preparation

The composite 3DB dataset follows the commonly used composite evaluation setting based on CASME II, SMIC and SAMM.

The labels are regrouped into three classes:

Negative: sadness, disgust, contempt, fear and anger;
Positive: happiness;
Surprise: surprise.

Due to dataset license restrictions, this repository does not provide the original video frames. Please obtain the source datasets from their official providers and organize the preprocessed optical-flow data as follows:

data/
└── three_norm_u_v_os/
    ├── subject_001/
    │   ├── u_train/
    │   │   ├── 0/
    │   │   ├── 1/
    │   │   └── 2/
    │   └── u_test/
    │       ├── 0/
    │       ├── 1/
    │       └── 2/
    ├── subject_002/
    └── ...

where class labels are:

0: Negative
1: Positive
2: Surprise

The optical-flow and optical-strain representations should be extracted between the onset and apex frames. Each sample is represented by three channels:

[u, v, os]

where u and v denote the horizontal and vertical optical-flow components, and os denotes optical strain.

Training

To train MSCT-Net under the LOSO protocol, run:

python train.py \
  --data_root ./data/three_norm_u_v_os \
  --metadata_csv ./data/combined_3_class2_for_optical_flow.csv \
  --epochs 200 \
  --batch_size 256 \
  --lr 5e-5 \
  --seed 1

For repeated-run experiments, use different random seeds, for example:

python train.py --seed 1
python train.py --seed 2
python train.py --seed 3
python train.py --seed 4
python train.py --seed 5
Evaluation Metrics

The model is evaluated using two class-balanced metrics:

UF1: Unweighted F1-score;
UAR: Unweighted Average Recall.

These metrics are used because the composite 3DB dataset is class-imbalanced.