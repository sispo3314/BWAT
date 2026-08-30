# Block-Wise Conditional Temporal-Scale Allocation for Wearable Human Activity Recognition
<img width="2752" height="1141" alt="image" src="https://github.com/user-attachments/assets/319e3a7a-74a1-4acb-bd34-db21fa632417" />
This repository implements the methodology proposed in the paper "Block-Wise Conditional Temporal-Scale Allocation for Wearable Human Activity Recognition"

## Paper Overview

**Abstract**: Wearable human activity recognition (HAR) requires temporal context at different scales, yet conventional temporal models typically apply fixed receptive-field schedules, evaluate multiple scales exhaustively, or make a single adaptive decision for an entire input. This work formulates temporal modeling as a block-wise conditional temporal-scale allocation problem, in which each representation block independently selects a temporal scale according to its current representation. We propose a lightweight framework that constructs multiple receptive fields from a nested shared depthwise kernel, performs independent routing at each temporal block, and executes only the selected operator during deterministic inference. A kernel-cost-aware objective further discourages unnecessary use of broad temporal operators while retaining longer context when beneficial. Across four benchmark HAR datasets, the proposed model achieved Macro-F1 scores ranging from 0.9643 to 0.9854 while selectively executing one temporal operator per block. Ablation results showed consistently higher mean Macro-F1 for block-wise routing than for global input-level scale adaptation, while cost-aware optimization and nested kernel sharing improved the recognition--computation trade-off. Routing analysis on UCI-HAR further revealed systematic variation in selected temporal scales across activities and representation blocks. Overall, the results support representation-dependent temporal allocation as an effective alternative to uniformly broad or exhaustive multi-scale computation, providing competitive HAR performance while adaptively reallocating temporal support across network depth.

## Dataset
This repository does not include datasets. Please download them from the official sources below and configure the dataset path accordingly.
- **UCI-HAR** dataset is available at https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **MotionSense** dataset is available at https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
- **MHEALTH** dataset is available at https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- **PAMAP2** dataset is available at https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

## Requirements

PyTorch 2.3.0 (Python 3.10)

​```bash
pip install torch==2.3.0
​```

## Codebase Overview

- [`model.py`](model.py): Implementation of the proposed BWATS architecture, including nested shared-kernel temporal operators, block-wise Gumbel–Softmax routing, deterministic single-operator inference, the input-statistics skip pathway, the cost-aware training objective, and configurations for the independent-kernel and global-routing ablations.


## Citing this Repository
If you use this code in your research, please cite:
```
@article{BWAT,
  title   = {Block-Wise Conditional Temporal-Scale Allocation for Wearable Human Activity Recognition},
  author  = {Jimin Kim and Myung-Kyu Yi},
  journal = {},
  volume  = {},
  number  = {},
  pages   = {},
  year    = {},
  publisher = {}
}
```


## License
This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please contact:
  - Jimin Kim: sispo3314@gmail.com


