In order to generate fairface results on arl model do the following

python train.py
python eval.py
python analyze_results.py

install conda env via environment.yaml in order to run the repo.

download the fairface database from here:
https://github.com/joojs/fairface

Citations 


@inproceedings{karkkainenfairface,
    title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation},
    author={Karkkainen, Kimmo and Joo, Jungseock},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    year={2021},
    pages={1548--1558}
}


@article{lahoti2020fairness,
  title={Fairness without demographics through adversarially reweighted learning},
  author={Lahoti, Preethi and Beutel, Alex and Chen, Jilin and Lee, Kang and Prost, Flavien and Thain, Nithum and Wang, Xuezhi and Chi, Ed},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={728--740},
  year={2020}
}