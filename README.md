# SVDC (CVPR 2025)

This repository contains the source code for our paper:

[SVDC: Consistent Direct Time-of-Flight Video Depth Completion with Frequency Selective Fusion](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_SVDC_Consistent_Direct_Time-of-Flight_Video_Depth_Completion_with_Frequency_Selective_CVPR_2025_paper.pdf)<br/>
Xuan Zhu, Jijun Xiang, Xianqi Wang, Longliang Liu, Yu Wang, Hong Zhang, Fei Guo, Xin Yang<br/>



## Updates
2025/07/04 —— We released our evaluation code and pretrained models. You can now evaluate SVDC on the test datasets.

## Requirements
We provide a conda environment setup file including all of the above dependencies. Create the conda environment by running:
```
conda env create -n SVDC -f environment.yml
```

## Required Data

To evaluate SVDC, you will need to download the required datasets.

* [TartanAir & Dynamicstereo](https://drive.google.com/drive/folders/1WJiZxZtDnrsN1jIvbASCaLzUwbmf_0NT?usp=drive_link)

You need to unrar the file and put them in the datasets folder as follows:

```
|SVDC/
   |--configs/
   |--datasets/
      |--DS_tuihua0/
      |--DS_tuihua1/
      |--DS_tuihua2/
      |--TT_tuihua0/
      |--TT_tuihua1/
      |--test_DS_tuihua_0.txt
      |--test_DS_tuihua_1.txt
    ...
```

## Evaluation

To evaluate on test datasets, run

```Shell
python evaluate.py configs/test_paper_DS_0.txt
python evaluate.py configs/test_paper_DS_1.txt
python evaluate.py configs/test_paper_DS_2.txt
python evaluate.py configs/test_paper_TT_0.txt
python evaluate.py configs/test_paper_TT_1.txt
python calculate_metrics.py
```

## Training

To be done.

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{zhu2025svdc,
  title={Svdc: Consistent direct time-of-flight video depth completion with frequency selective fusion},
  author={Zhu, Xuan and Xiang, Jijun and Wang, Xianqi and Liu, Longliang and Wang, Yu and Zhang, Hong and Guo, Fei and Yang, Xin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={16619--16628},
  year={2025}
}
```

## Contact

Please feel free to contact me (XuanZhu) at xuanzhu@hust.edu.cn.

## Acknowledgements

This project is based on [DVSR](https://github.com/facebookresearch/DVSR) and [DELTAR](https://github.com/zju3dv/deltar), we thank the original authors for their excellent work.
