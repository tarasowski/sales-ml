import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport

pd.set_option('display.max_columns', 20)

url = './sales-data.csv'

sales_data = pd.read_csv(url)

le = preprocessing.LabelEncoder()

# convert the categorical columns into numeric
sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region'] = le.fit_transform(sales_data['Region'])
sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])

data = sales_data.drop(['Opportunity Result', 'Opportunity Number'], axis=1)
target = sales_data['Opportunity Result']

data_train, data_test, target_train, target_test = train_test_split(
        data, 
        target, 
        test_size= 0.30, 
        random_state=10)


gnb = GaussianNB()
gnb_pred = gnb.fit(data_train, target_train).predict(data_test)

svc_model = LinearSVC(random_state=0)
svc_pred = svc_model.fit(data_train, target_train).predict(data_test)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data_train, target_train)
knn_pred = neigh.predict(data_test)


print('Naive-Bayes accuracy : ', accuracy_score(target_test, gnb_pred, normalize = True))
print('LinearSVC accuracy : ', accuracy_score(target_test, svc_pred, normalize= True))
print('KNN accuracy : ', accuracy_score(target_test, knn_pred, normalize=True))

def visualize(alg):
    visualizer = ClassificationReport(alg, classes=['Won', 'Loss'])
    visualizer.fit(data_train, target_train)
    visualizer.score(data_test, target_test)
    visualizer.poof()

visualize(gnb)
visualize(svc_model)
visualize(neigh)



