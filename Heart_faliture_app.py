import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

from scipy import stats
from scipy.stats import norm, skew, boxcox
from collections import Counter

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, plot_confusion_matrix, auc
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

import pickle

#XGBOOST
from xgboost import XGBClassifier

#warning
import warnings
warnings.filterwarnings('ignore')

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore    import *
from PyQt5.QtGui     import *


class MainPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.setWindowIcon(QIcon("ICON.png"))
        self.setWindowTitle("Heart Faliture Predict")
        
    def setupUi(self):
         
         Result = QPushButton(self)
         Result.setText("Predict")
         Result.clicked.connect(self.Prediction)
         
         form   = QFormLayout()
         window = QWidget()
         
         
###############################################################################
         
         self.Radmale   = QRadioButton("Male")
         self.Radfemale = QRadioButton("Female")
         
         self.sexBgroup = QButtonGroup()
         
         self.sexBgroup.addButton(self.Radmale)
         self.sexBgroup.addButton(self.Radfemale)
         
         self.Radmale.toggled.connect(self.toggleRadio)
         self.Radfemale.toggled.connect(self.toggleRadio)
         
         sexLayout = QHBoxLayout()
         sexLayout.addWidget(self.Radmale)
         sexLayout.addStretch()
         sexLayout.addWidget(self.Radfemale)
         
         

###############################################################################
         
         self.Raddiabet  = QRadioButton("Diabet")
         self.Radnotdiabet = QRadioButton("Not Diabet")
         
         self.diabetBgroup = QButtonGroup()         
         
         self.diabetBgroup.addButton(self.Raddiabet)
         self.diabetBgroup.addButton(self.Radnotdiabet)
         
         self.Raddiabet.toggled.connect(self.toggleRadio)
         self.Radnotdiabet.toggled.connect(self.toggleRadio)        
         
         diabetLayout = QHBoxLayout()
         diabetLayout.addWidget(self.Raddiabet)
         diabetLayout.addStretch()
         diabetLayout.addWidget(self.Radnotdiabet)
         
###############################################################################
         
         self.RadHighBloodPressure = QRadioButton("High Blood Pressure")
         self.RadNormalBloodPressure  = QRadioButton("Normal Blood Pressure")
         
         self.HighBloodPressureBgroup = QButtonGroup()
         
         self.HighBloodPressureBgroup.addButton(self.RadHighBloodPressure)
         self.HighBloodPressureBgroup.addButton(self.RadNormalBloodPressure)

         self.RadHighBloodPressure.toggled.connect(self.toggleRadio)
         self.RadNormalBloodPressure.toggled.connect(self.toggleRadio)
         
         HighBloodPressureLayout = QHBoxLayout()
         HighBloodPressureLayout.addWidget(self.RadHighBloodPressure)
         HighBloodPressureLayout.addStretch()
         HighBloodPressureLayout.addWidget(self.RadNormalBloodPressure)
         
###############################################################################

         
         self.RadSmoking  = QRadioButton("Smoking")
         self.RadnotSmoking = QRadioButton("Not Smoking")
         
         self.SmokingBgroup = QButtonGroup()
         
         self.SmokingBgroup.addButton(self.RadSmoking)
         self.SmokingBgroup.addButton(self.RadnotSmoking)

         self.RadSmoking.toggled.connect(self.toggleRadio)
         self.RadnotSmoking.toggled.connect(self.toggleRadio)

         SmokingLayout = QHBoxLayout()
         SmokingLayout.addWidget(self.RadSmoking)
         SmokingLayout.addStretch()
         SmokingLayout.addWidget(self.RadnotSmoking)

###############################################################################

  
         self.RadAnemia  = QRadioButton("Anemia")
         self.RadnotAnemia = QRadioButton("Not Anemia")
         
         self.AnemiaBgroup = QButtonGroup()         
         
         self.AnemiaBgroup.addButton(self.RadAnemia)
         self.AnemiaBgroup.addButton(self.RadnotAnemia)
 
         self.RadAnemia.toggled.connect(self.toggleRadio)
         self.RadnotAnemia.toggled.connect(self.toggleRadio)

         AnemiaLayout = QHBoxLayout()
         AnemiaLayout.addWidget(self.RadAnemia)
         AnemiaLayout.addStretch()
         AnemiaLayout.addWidget(self.RadnotAnemia)
         

###############################################################################       

            
         self.age = QLineEdit()
         self.Time = QLineEdit()
         self.Time.setToolTip('Patient observation period: enter in days') 
         self.CPK = QLineEdit()
         self.CPK.setToolTip('Level of the CPK enzyme in the blood (mcg/L)')
         self.SerumSodium = QLineEdit()
         self.SerumSodium.setToolTip(' Level of serum sodium in the blood (mEq/L)')
         self.SerumCreatinine = QLineEdit()
         self.SerumCreatinine.setToolTip('Level of serum creatinine in the blood (mg/dL)')
         self.Platelets = QLineEdit()
         self.Platelets.setToolTip('Platelets in the blood (kiloplatelets/mL)')
         self.EjectionFraction = QLineEdit()
         self.EjectionFraction.setToolTip('Percentage of blood leaving the heart at each contraction (percentage)')
         
         
         form.addRow(QLabel("Age"),self.age)
         form.addRow(QLabel("Sex"),sexLayout)
         form.addRow(QLabel("Time"),self.Time)
         form.addRow(QLabel("CPK"),self.CPK)
         form.addRow(QLabel("Serum Sodium"),self.SerumSodium)
         form.addRow(QLabel("Serum Creatinine"),self.SerumCreatinine)
         form.addRow(QLabel("Platelets"),self.Platelets)
         form.addRow(QLabel("Ejection Fraction"),self.EjectionFraction)
         form.addRow(QLabel("Anemia"),AnemiaLayout)
         form.addRow(QLabel("Diabet"),diabetLayout)
         form.addRow(QLabel("High Blooad Pressure"),HighBloodPressureLayout)
         form.addRow(QLabel("Smoking"),SmokingLayout)
         form.addRow(Result)
        
    
         self.setLayout(form)
         
         self.show()
         
    def Prediction(self):
         InfoAge  = "zzz zzz zzz\n"
         InfoCPK  = "bla bla bla\n"
         InfoEF   = "tada tada\n"
         InfoSC   = "mrr mrr mrr mrr\n"
         InfoTime = "ring ring ring\n"
         
         def detect_outliers(df,features):
             
             outlier_indices = []
            
             for c in features:
                 # 1st quartile
                 Q1 = np.percentile(df[c],25)
                 # 3st quartile
                 Q3 = np.percentile(df[c],75)
                 # IQR
                 IQR = Q3 - Q1
                 # Outlier Step
                 outlier_step = IQR * 1.5
                 # detect outlier and their indeces
                 outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
                 # store indeces 
                 outlier_indices.extend(outlier_list_col)
                
             outlier_indices = Counter(outlier_indices)
             multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1) 
             
             return multiple_outliers
    
         data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    
         data.loc[detect_outliers(data,["age","creatinine_phosphokinase","ejection_fraction",
                                                                            "platelets","serum_creatinine","serum_sodium","time"])]
    
         data = data.drop(detect_outliers(data,["age","creatinine_phosphokinase","ejection_fraction",
                                           "platelets","serum_creatinine","serum_sodium","time"]),axis = 0).reset_index(drop=True)
         
         skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False) #Çarpıklık kontrolü
         skewness = pd.DataFrame(skewed_feats, columns = ["skewed"])
    
    
         data["creatinine_phosphokinase"], lam = boxcox(data["creatinine_phosphokinase"])
         data["serum_creatinine"], lam_serum_creatine = boxcox(data["serum_creatinine"])         
         data["ejection_fraction"], lam_serum_creatine = boxcox(data["ejection_fraction"])
         data["platelets"], lam_serum_creatine = boxcox(data["platelets"])
    
    
         skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
         skewness_new = pd.DataFrame(skewed_feats, columns = ["skewed"])
    
         X = data.drop("DEATH_EVENT", axis = 1)
         y = data.DEATH_EVENT
    
         sm = SMOTE(random_state=42)
         X_sm, y_sm = sm.fit_resample(X, y)
    
    
         X_train, X_test, Y_train, Y_test = train_test_split(X_sm, y_sm, test_size = 0.2, random_state = 42)
    
    
         model_rnd = RandomForestClassifier()
         model_rnd.fit(X_train, Y_train)
         importance = model_rnd.feature_importances_
    
    
         x_train_random_forest = X_train[["age","creatinine_phosphokinase","ejection_fraction","serum_creatinine","time"]]
         x_test_random_forest = X_test[["age","creatinine_phosphokinase","ejection_fraction","serum_creatinine","time"]]
    
    
         random_forest_model = RandomForestClassifier(max_depth=7, random_state=25)
         random_forest_model.fit(x_train_random_forest, Y_train)
         y_pred_random_forest = random_forest_model.predict(x_test_random_forest)
         cm_random_forest = confusion_matrix(y_pred_random_forest, Y_test)
         acc_random_forest = accuracy_score(Y_test, y_pred_random_forest)
         
         pickle.dump(model_rnd, open('model.pkl','wb'))
         
         model = pickle.load(open('model.pkl','rb'))
         
         DEATH_EVENT=model.predict([[float(self.age.text()),
                                           self.anemia, 
                                     float(self.CPK.text()), 
                                           self.diabet, 
                                     float(self.EjectionFraction.text()), 
                                           self.highbloodpressure,
                                     float(self.Platelets.text()),
                                     float(self.SerumCreatinine.text()),
                                     float(self.SerumSodium.text()),
                                           self.sex,
                                           self.smoking,
                                     float(self.Time.text()) ]])

         if DEATH_EVENT[0]== 0:
             Result =  "no risk of heart failure"
             Detail =  "Happy Healthy Days!"
         else:
             Result =  "high risk of heart failure"
             Detail = InfoCPK + InfoEF + InfoAge + InfoSC + InfoTime
         

         


         msgBox = QMessageBox()
         msgBox.setText(Result)
         msgBox.setWindowTitle("Result")
         msgBox.setDetailedText(Detail)
         restartBtn = msgBox.addButton('Retry', QMessageBox.ActionRole)
         ExitBtn    = msgBox.addButton('Exit', QMessageBox.ActionRole)

        
         ret = msgBox.exec()
        
         if ret == QMessageBox.Close:
                    QCloseEvent()
         elif msgBox.clickedButton() == restartBtn:
             self.age.clear()
             self.Time.clear()
             self.CPK.clear()
             self.SerumSodium.clear()
             self.SerumCreatinine.clear()
             self.Platelets.clear()
             self.EjectionFraction.clear()
         elif msgBox.clickedButton() == ExitBtn:
             self.close()
             
             
            
         
    
    def toggleRadio(self):

        rdButon=self.sender()
 
        if rdButon.isChecked():

            if rdButon.text() == 'Male':
                self.sex = 1
                
            elif rdButon.text() == 'Female':
                self.sex = 0

            if rdButon.text() == 'Diabet':
                self.diabet = 1

            elif rdButon.text() == 'Not Diabet':
                self.diabet = 0

            if rdButon.text() == 'High Blood Pressure':
                self.highbloodpressure = 1

            elif rdButon.text() == 'Normal Blood Pressure':
                self.highbloodpressure = 0

            if rdButon.text() == 'Smoking':
                self.smoking = 1

            elif rdButon.text() == 'Not Smoking':
                self.smoking = 0

            if rdButon.text() == 'Anemia':
                self.anemia = 1

            elif rdButon.text() == 'Not Anemia':
                self.anemia = 0



def main():
    app = QApplication(sys.argv)
    mainPage = MainPage()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()