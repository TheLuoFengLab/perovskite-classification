# perovskite-classification

### Project structure

```
├── README.md
├── data
│   ├── exp_samples.xlsx
│   ├── negative_samples.xlsx
│   ├── parameters.xlsx
│   └── positive_samples.xlsx
└── main.py
```

- `data` folder: includes all required data files for this project
  - `negative_samples.xlsx`: all features of 123 negative samples
  - `positive_samples.xlsx`: all features of 639 positive samples
  - `exp_samples.xlsx`: all features of 59 experimental samples
  - `parameters.xlsx`: all As and Bs parameters for the generation of new compounds 
- `main.py`: the source code for the machine learning model

### Requirements

- Python3
- numpy==1.21.0
- pandas==1.3.5
- scikit-learn==0.24.2

### Run

```
git clone https://github.com/TheLuoFengLab/perovskite-classification.git
cd perovskite-classification
python main.py -step 0.05 -As Ba -Bs Ti,Ce,Zr,Y,Yb
```

- It shoud obtain the accuracy of (50/59) for the experimental samples.
- It also can predict new compounds, given step: 0.05, As element: `Ba` and Bs element: Ti,Ce,Zr,Y,Yb
