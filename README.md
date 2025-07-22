# PIDL-EM: Physics-Informed Deep Learning for Electromagnetic Modeling

A physics-informed deep learning framework for solving electromagnetic problems using neural networks that incorporate physical laws and constraints.

## Overview

This project implements physics-informed neural networks (PINNs) specifically designed for electromagnetic field modeling and simulation. The approach combines the power of deep learning with the fundamental principles of electromagnetism to solve complex EM problems efficiently.

## Features

- Physics-informed neural network implementation
- Electromagnetic field simulation capabilities
- Support for various boundary conditions
- Efficient training algorithms
- Visualization tools for EM fields

## Installation

```bash
# Clone the repository
git clone https://github.com/username/PIDL-EM.git
cd PIDL-EM

# Install dependencies
pip install -r requirements.txt
```

## Usage
Generate synthetic electromagnetic data:
```bash
cd data/msf
python msf.py
```

Train the Fourier Neural Operator (FNO) model for electromagnetic field estimation:
```bash
python fno.py --config configs/msf_fnogru_ez.yml
```
Train the physics-informed neural network (PINN) model for electromagnetic field estimation:
```bash
python pinn.py --config configs/msf_simi2v_ezhxhy.yml
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhang2025time,
  title={Time-Domain 3D Electromagnetic Fields Estimation Based on Physics-Informed Deep Learning Framework},
  author={Zhang, Huifan and Hu, Yun and Zhou, Pingqiang},
  booktitle={2025 Design, Automation \& Test in Europe Conference (DATE)},
  pages={1--7},
  year={2025},
  organization={IEEE}
}
```

## Contact

For questions and support, please contact: [zhanghf@shanghaitech.edu.cn]
