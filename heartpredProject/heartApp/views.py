from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
#from .forms import *
from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
ListView,DetailView,
CreateView,DeleteView,
UpdateView)
from . import models
from .forms import *
from django.core.files.storage import FileSystemStorage
#from topicApp.Topicfun import Topic
#from ckdApp.funckd import ckd
#from sklearn.tree import export_graphviz #plot tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
##from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
#from sklearn.decomposition import PCA
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
#import eli5 #for purmutation importance
#from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
#from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproduc
class dataUploadView(View):
    form_class = heartForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        #print('inside form')
        if form.is_valid():
            form.save()
            Age= request.POST.get('age')
            data_cp=request.POST.get('cp')
            data_bps=request.POST.get('trestbps')
            data_chol=request.POST.get('chol')
            data_fbs=request.POST.get('fbs')
            data_ecg=request.POST.get('restecg')


            #load and preprocess the dataset
            data= pd.read_csv("heart.csv")
            #heartdiseaseprediction_model
            X=data.drop("target",axis=1)
            Y= data['target']
            X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.25, random_state=0)

            #Train a model
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("Model Accuracy:", accuracy_score(y_test, y_pred))

            #select top 5 feature_selection
            selector= SelectKBest(score_func=chi2,k=5)
            selector.fit(X,Y)
            feature_names = X.columns

            # Get the selected feature names based on the support provided by the selector
            selected_feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
            print(f"Selected {5} features:")
            #reduce the feature set
            X_reduced=X[['age','cp','trestbps','chol','fbs','restecg']]
            X_train,X_test,y_train,y_test=train_test_split(X_reduced, Y, test_size=0.25, random_state=0)

            #pickle model
            model_reduced = KNeighborsClassifier(n_neighbors=5)
            model_reduced.fit(X_train, y_train)
            filename_reduced='reduced_model.sav'
            pickle.dump(model_reduced,open(filename_reduced,'wb'))

            loaded_model_reduced= pickle.load(open(filename_reduced, 'rb'))

            #prepare userinput for Predicition
            def get_user_input():
                return np.array([[int(Age),float(data_cp),float(data_bps),float(data_chol),float(data_fbs),float(data_ecg)]])

            selected_features=get_user_input()
            result=loaded_model_reduced.predict(selected_features)
            #print("Prediction result:",result[0])

            #print (data)
            #dataset1=pd.read_csv("prep.csv",index_col=None)
            dicc={'yes':1,'no':0}
            data = np.array([Age,data_cp,data_bps,data_chol,data_fbs,data_ecg])
            #sc = StandardScaler()
            #data = sc.fit_transform(data.reshape(-1,1))
            out=loaded_model_reduced.predict(data.reshape(1,-1))
# providing an index
            #ser = pd.DataFrame(data, index =['bgr','bu','sc','pcv','wbc'])

            #ss=ser.T.squeeze()
#data_for_prediction = X_test1.iloc[0,:].astype(float)

#data_for_prediction =obj.pca(np.array(data_for_prediction),y_test)
            #obj=ckd()
            ##plt.savefig("static/force_plot.png",dpi=150, bbox_inches='tight')



            return render(request, "succ_msg.html", {'Age':Age,'data_cp':data_cp,'data_bps':data_bps,'data_chol':data_chol,'data_fbs':data_fbs,
                                                    'data_ecg':data_ecg,'out':out})


        else:
            return redirect(self.failure_url)
