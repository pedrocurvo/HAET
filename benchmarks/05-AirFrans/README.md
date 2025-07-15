# HAET for Airfoil Design

The airfoil design task requires the model to estimate the surrounding and surface physical quantities of a 2D airfoil under different Reynolds and angles of attacks.

<p align="center">
<img src=".\fig\task.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Airfoil design task. Left: surrounding pressure; Right: x-direction wind speed.
</p>

## Get Started

This part of code is developed based on the [[AirfRANS]](https://github.com/Extrality/AirfRANS).

1. Install Python 3.8. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

Note: You need to install [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric).

2. Prepare Data.

The experiment data is provided by [[AirfRANS]](https://github.com/Extrality/AirfRANS). You can directly download it with this [link](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip) (9.3GB).

3. Train and evaluate model. We provide the experiment scripts under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash scripts/Training.sh # for Training HAET
bash scripts/Evaluation.sh # for Evaluation
```

Note: You need to change the argument `--my_path` to your dataset path.

4. Test model with different settings. This benchmark supports four types of settings.

| Settings                                     | Argument      |
| -------------------------------------------- | ------------- |
| Use full data                                | `-t full`     |
| Use scarce data                              | `-t scarce`   |
| Test on out-of-distribution Reynolds         | `-t reynolds` |
| Test on out-of-distribution Angle of Attacks | `-t aoa`      |
