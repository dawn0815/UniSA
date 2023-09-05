# UniSA: Unified Generative Framework for Sentiment Analysis (ACM MM 2023)
<div align="center">
<strong>Zaijing Li, Ting-En Lin, Yuchuan Wu, Meng Liu, Fengxiao Tang†, Ming Zhao†, Yongbin Li† </strong> 
</div>
<div align="center">
Central South University, Alibaba Group
</div>
<div align="center">
† Corresponding Author
</div>

[[Paper]](https://github.com/dawn0815/SAEval-Benchmark) [[Codes]](https://github.com/dawn0815/UniSA) [[Benchmark]](https://github.com/dawn0815/SAEval-Benchmark)


![image](https://github.com/dawn0815/UniSA/blob/master/f1.png)

## TODO
- [x] Release code and pretrained model.
- [ ] Add the paper link.
- [ ] Code review.
- [ ] Add description of section model weights and running scripts
      
## Introduction
Sentiment analysis is a crucial task that aims to understand people's emotional states and predict emotional categories based on multimodal information. It consists of several subtasks, such as emotion recognition in conversation (ERC), aspect-based sentiment analysis (ABSA), and multimodal sentiment analysis (MSA). However, unifying all subtasks in sentiment analysis presents numerous challenges, including modality alignment, unified input/output forms, and dataset bias. To address these challenges, we propose a Task-Specific Prompt method to jointly model subtasks and introduce a multimodal generative framework called UniSA. Additionally, we organize the benchmark datasets of main subtasks into a new Sentiment Analysis Evaluation benchmark, SAEval. We design novel pre-training tasks and training methods to enable the model to learn generic sentiment knowledge among subtasks to improve the model's multimodal sentiment perception ability. Our experimental results show that UniSA performs comparably to the state-of-the-art on all subtasks and generalizes well to various subtasks in sentiment analysis. 

![image](https://github.com/dawn0815/UniSA/blob/master/p2.png)
Our work represents a step forward in the unified modeling of sentiment analysis subtasks, and we hope that it will inspire future research in this direction. We encourage more researchers to join the study on multi-tasks unified modeling for sentiment analysis, and contribute their wisdom to build sentiment intelligence.

## Installation

1. Clone the repository recursively
    ```
    git clone https://github.com/dawn0815/UniSA.git
    ```

2. Create conda environment
    ```
    cd config
    conda env create -f environment.yaml
    ```
## Datasets
In this paper, we use the proposed [SAEval Benchmark](https://github.com/dawn0815/SAEval-Benchmark) to train and evaluate our UniSA. The SAEval is a benchmark for sentiment analysis to evaluate the model's performance on various subtasks. All datasets were standardized to the same format and divided into training, validation and test sets. For more information about SAEval Benchmark, please refer to the [link](https://github.com/dawn0815/SAEval-Benchmark).

![image](https://github.com/dawn0815/SAEval-Benchmark/blob/master/p1.png)

## Model Weights 
| Model                 | Stage                          | File Size              | Link                                                        |
| :-------------------- | -------------------------------|----------------------- | ----------------------------------------------------------- |
| model_0 |  pretrain_stage_1  |   1.7G  |  [[Google Drive]](https://drive.google.com/file/d/1BCiJV_dg3WmWX1N29KEVHQSn5woGVPTD/view?usp=sharing) |
| model_1 |  pretrain_stage_2  |   1.7G  |  [[Google Drive]](https://drive.google.com/file/d/1hq_ZN0xBlpkgep8HwK_fq2X5IEknuxm3/view?usp=sharing) |
| model_2 |  finetune          |   3.4G  |  [[Google Drive]](https://drive.google.com/file/d/14mwftx3Q7oczECVCWDFktbPaIK8ly53E/view?usp=sharing) |

## Running Scripts

1. pretrain-stage1
    ```
    bash run_pretrain_stage1.sh
    ```
    
2. pretrain-stage2
    ```
    bash run_pretrain_stage2.sh
    ```
        
3. fine-tune
    ```
    bash run_finetune.sh
    ```
    
4. inference
    ```
    bash run_inference.sh
    ```
        
5. few-shot
    ```
    bash run_few_shot.sh
    ```
    
## Evaluation
Here are the experimental results of [UniSA](https://github.com/dawn0815/UniSA) on the SAEval compared to SOTA models of various subtasks. 

![pdf](https://github.com/dawn0815/SAEval-Benchmark/blob/master/p5.png)

We encourage more researchers to join the study on multi-tasks unified modeling for sentiment analysis, and contribute their wisdom to build sentiment intelligence.

## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed: (paper link is coming soon).
```
@inproceedings{li2023unisa,
  title={UniSA: Unified Generative Framework for Sentiment Analysis},
  author={Zaijing Li, Ting-En Lin, Yuchuan Wu, Meng Liu, Fengxiao Tang, Ming Zhao, and Yongbin Li},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  year={2023}
}
```
## Acknowledgement
- Thanks [UniMSE](https://github.com/LeMei/UniMSE) for their contributions to the processing of datasets (e.g., [MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/), [MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/), [IEMOCAP](https://sail.usc.edu/iemocap/) and [MELD](https://github.com/declare-lab/MELD)).
- Thanks [KM-BART](https://github.com/fomalhautb/KM-BART) for their contributions to the multi-modal [BART](https://arxiv.org/abs/1910.13461) framework.
- Thanks Tianshu Yu for his constructive comments.

## License
UniSA is released without any restrictions but restrictions may apply to individual tasks (which are derived from existing datasets) or backbone (e.g., GPT2, T5, and BART). We refer users to the original licenses accompanying each dataset and backbone.



