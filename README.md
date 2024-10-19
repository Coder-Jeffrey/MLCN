# MLCN 
## Architecture
![image](https://github.com/Coder-Jeffrey/MLCN/assets/76551880/ec3f8fdd-390c-4e88-b2d3-1e322fcc428f)

## Main Environment
```
torch == 1.7.1
python == 3.6.13
transformers == 4.2.2
numpy == 1.19.5
tqdm == 4.6.64
```
## Preprocess
You can download the datasets at: 

AAPD:https://git.uwaterloo.ca/jimmylin/Castor-data/-/tree/master/datasets/AAPD/data

RCV1:https://scikit-learn.org/0.18/datasets/rcv1.html

Our method is to put the text and label tokens in the same space and run **dataset.py** to get the processed data.

## Running
Run **classification.py**

## Result
![image](https://github.com/Coder-Jeffrey/MLCN/assets/76551880/b553a4b4-2190-4277-9f2e-f77b48a74839)

The result demonstrates the following. **MLCN** achieves comparable performance, compared with LDGN, and beats all other models, on both datasets. On AAPD, **MLCN** outperforms LDGN on two metrics; while on RCV1, **MLCN** shows superiority on four indicators, e.g., the improvement on P@3 even reaches 1.15%, which is quite remarkable. The reason for better performance on RCV1 possibly lies in the fact that RCV1 has more informative labels than AAPD, which facilitates our model's learning.
