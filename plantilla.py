
X, y = dataset1

# Seleccion de hiperparametros

KNN 
n_neighbors: [1, 3, 5, 7]          #4
weights: ["uniform", "distance"]   #2
p: [1, 2]                          #2

GRIDSEARCH
KNN 
X, y
HIPERPARAMETROS
n_neighbors: [1, 3, 5, 7]          #4
weights: ["uniform", "distance"]   #2
p: [1, 2] 

KNN Dataset 1
n_neighbors = 3
weights = unifrom
p = 2

# Entrenamiento
metricsKNN = []
metricsLogis = []
metricsDecision = []
metricsNeural = []

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45, stratify=...)
    knnC = KNNClassifier(n_neighbors = 3, weigths = "uniform", p=2)
    logisticR = logisticRegression()
    decisionT = decisionTreeCl()
    neuralN = neuralNet()

    #Entrenan
    knn.Fit(X_train, Y_test)


    #Sacan las metricas
    knn.Precict(X_test, Y_test)
    Plot de Confusion Matrix
    metricsKNN.append(metrics)

# Analisis de Resultados

#Sacan promedios de cada algoritmo
Accuracy knnC
Recall KNN

#Pueden escoger cualquiera de las 5 corridas para hacer los graáficos aunque sería mas bonito que hagan gráficos del promedio de las 5

#Confusion Matrix