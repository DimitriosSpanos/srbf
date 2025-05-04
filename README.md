# Leveraging Subclass Learning for Improving Uncertainty Estimation in Deep Learning
To reproduce the results reported in our paper (e.g. for MNIST):

Run:
```bash
  python3 train_AE.py --dataset mnist
```
And then:
```bash
  python3 train_SRBF.py --dataset mnist --opt sgd --sgd-lr 5e-2 --num-epochs 30 --gamma 0.999 --cs 64 --subclass 1 --AE-pretrained
```


## Acknowledgments

This work has received partial funding from the Hellenic Foundation for Research & Innovation (H.F.R.I.) scholarship under grant agreement No 20490 (Deep Learning Methodologies for Trustworthy Intelligent Systems) and the research project ”Robotic Safe Adaptation In Unprecedented Situations (RoboSAPIENS)”, which is implemented in the framework of Horizon Europe 2021-2027 research and innovation programme under grant agreement No 101133807. This publication reflects the authors’ views only. The European Commission are not responsible for any use that may be made of the information it contains.

Learn more about [RoboSAPIENS](https://robosapiens-eu.tech/).
