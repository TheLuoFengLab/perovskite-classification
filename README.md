
# Predicting the formation of fractionally doped perovskite oxides

The **official source code** for our paper, *including the prediction of new compounds using trained model*: 

Zhai, X., Ding, F., Zhao, Z. et al. Predicting the formation of fractionally doped perovskite oxides by a function-confined machine learning method. Commun Mater 3, 42 (2022). https://doi.org/10.1038/s43246-022-00269-9


### Project structure

```
├── Ba_Ti,Ce,Zr,Y,Yb_0.05.xlsx
├── README.md
├── data
│   ├── Fractional_Perovsktie_Oxides_Data.xlsx
│   └── parameters.xlsx
└── main.py
```

- `data` folder: includes all required data files for this project
  - `Fractional_Perovsktie_Oxides_Data.xlsx`: all features of 116 negative samples, 516 positive samples and 47 experimental samples
  - `parameters.xlsx`: all As and Bs parameters for the generation of new compounds 
- `main.py`: the source code for the machine learning model
- `Ba_Ti,Ce,Zr,Y,Yb_0.05.xlsx`: The prediction results for new compounds using the trained model. For example, given step: 0.05, As element: `Ba` and Bs element: Ti,Ce,Zr,Y,Yb, the model predicts all combinations to determine if each is a perovskite oxide. All these results are saved in this Excel file.


### Requirements

- Python3
- numpy==1.21.0
- pandas==1.3.5
- scikit-learn==0.24.2

### Train and Predict

```
git clone https://github.com/TheLuoFengLab/perovskite-classification.git
cd perovskite-classification
python main.py -step 0.05 -As Ba -Bs Ti,Ce,Zr,Y,Yb
```

- It shoud obtain the accuracy of (43/47) for the experimental samples.
- We also provide the functionality to **predict new compounds using trained model**, given the conditions of `step`, `As` and `Bs`.

For example,

`--step: 0.05`, `--As Ba` and `--Bs Ti,Ce,Zr,Y,Yb` will generate all combinations for the predication:

```
Ba 1.0, Ti 0.05 Ce 0.05 Zr 0.05 Y 0.05 Yb 0.8, Negative
Ba 1.0, Ti 0.05 Ce 0.05 Zr 0.05 Y 0.1 Yb 0.75, Negative
...

Ba 1.0, Ti 0.05 Ce 0.05 Zr 0.3 Y 0.05 Yb 0.55, Positive
Ba 1.0, Ti 0.05 Ce 0.05 Zr 0.3 Y 0.1 Yb 0.5,   Positive
...

```

### Cite our paper

If you find this code useful in your research, please consider citing our paper:

Zhai, X., Ding, F., Zhao, Z. et al. Predicting the formation of fractionally doped perovskite oxides by a function-confined machine learning method. Commun Mater 3, 42 (2022). https://doi.org/10.1038/s43246-022-00269-9
