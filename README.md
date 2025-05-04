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

The work presented here is supported by the RoboSAPIENS project funded by the European Commission’s Horizon Europe programme under grant agreement number 101133807. This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains.

Learn more about [RoboSAPIENS](https://robosapiens-eu.tech/).
