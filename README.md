# Probabalistic_Graph_Residual
The full paper describing our project's methodology, strenghts and weaknesses can be found [Report](ProRES_CISS.pdf)

* Replaces the L21 norm in the original AirGNN implementation by computing the distance between probability distributions of all neighboring node features to estimate if they are poisoned.
* We perform and compare our method with L21 norm in noisy and adversarially corrupted data.
* In general our method matches the performance or outperforms in certain cases, but Adaptive residual performs better by tuning L21 cost hyperparameter in noisy and adversarially corrupted data.

* Our model is located in AirGNN

* AirGNN_OG holds the original AirGNN

## Reference

```
@inproceedings{
liu2021graph,
title={Graph Neural Networks with Adaptive Residual},
author={Xiaorui Liu and Jiayuan Ding and Wei Jin and Han Xu and Yao Ma and Zitao Liu and Jiliang Tang},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=hfkER_KJiNw}
}
```
https://github.com/lxiaorui/AirGNN
