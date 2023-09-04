# UniSA: Unified Generative Framework for Sentiment Analysis (ACM MM 2023)

#### **Zaijing Li**, **Ting-En Lin**, **Yuchuan Wu**, **Meng Liu**, **Fengxiao Tang**, **Ming Zhao**, **Yongbin Li**  

## Introduction
Sentiment analysis is a crucial task that aims to understand people's emotional states and predict emotional categories based on multimodal information. It consists of several subtasks, such as emotion recognition in conversation (ERC), aspect-based sentiment analysis (ABSA), and multimodal sentiment analysis (MSA). However, unifying all subtasks in sentiment analysis presents numerous challenges, including modality alignment, unified input/output forms, and dataset bias. To address these challenges, we propose a Task-Specific Prompt method to jointly model subtasks and introduce a multimodal generative framework called UniSA. Additionally, we organize the benchmark datasets of main subtasks into a new Sentiment Analysis Evaluation benchmark, SAEval. We design novel pre-training tasks and training methods to enable the model to learn generic sentiment knowledge among subtasks to improve the model's multimodal sentiment perception ability. Our experimental results show that UniSA performs comparably to the state-of-the-art on all subtasks and generalizes well to various subtasks in sentiment analysis. 

Our work represents a step forward in the unified modeling of sentiment analysis subtasks, and we hope that it will inspire future research in this direction. We encourage more researchers to join the study on multi-tasks unified modeling for sentiment analysis, and contribute their wisdom to build sentiment intelligence.

## Installation

1. Clone the repository recursively
    ```
    git clone https://github.com/dawn0815/UniSA.git
    ```

2. Create conda environment
    ```
    conda env create -f environment.yaml
    ```
## Datasets
In this paper, we use the proposed [SAEval Benchmark](https://github.com/dawn0815/SAEval-Benchmark) to train and evaluate our UniSA. The SAEval is a benchmark for sentiment analysis to evaluate the model's performance on various subtasks. All datasets were standardized to the same format and divided into training, validation and test sets. For more information about SAEval Benchmark, please refer to the [link](https://github.com/dawn0815/SAEval-Benchmark).

  
## Pre-train

#### Running Scripts

#### Pretrained Weights

## Fine-tune

#### Running Scripts

#### Finetuned Weights

## Few-shot

#### Datasets for Few-shot

#### Running Scripts

## Citing SAEval
If you use UniSA in your research, please use the following `bib` entry to cite the paper (paper link is coming soon).

## License
UniSA is released without any restrictions but restrictions may apply to individual tasks (which are derived from existing datasets) or backbone (e.g., GPT2, T5, and BART). We refer users to the original licenses accompanying each dataset and backbone.



