
# The classification for Fractional Perovskite Oxides

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
- `Ba_Ti,Ce,Zr,Y,Yb_0.05.xlsx`: The prediction results for new compounds, given step: 0.05, As element: `Ba` and Bs element: Ti,Ce,Zr,Y,Yb

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

- It shoud obtain the accuracy of (43/47) for the experimental samples.
- It also can predict new compounds, given step: 0.05, As element: `Ba` and Bs element: Ti,Ce,Zr,Y,Yb
