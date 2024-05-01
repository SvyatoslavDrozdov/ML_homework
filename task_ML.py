import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

random_state = 42

df = datasets.load_digits()
X = df.data
y = df.target

# Расчет ---------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)

SVC_model = SVC(random_state=random_state)
DecisionTreeClassifier_model = DecisionTreeClassifier(random_state=random_state)
LogisticRegression_model = LogisticRegression(max_iter=10000, random_state=random_state)

print(
    "\nОбучение на исходных данных -------------------------------------------------------------------------------------\n")
SVC_model.fit(X_train, y_train)
predict_SVC_model = SVC_model.predict(X_test)
print(SVC_model, '\n', classification_report(y_test, predict_SVC_model))

DecisionTreeClassifier_model.fit(X_train, y_train)
predict_DecisionTreeClassifier_model = DecisionTreeClassifier_model.predict(X_test)
print(DecisionTreeClassifier_model, '\n', classification_report(y_test, predict_DecisionTreeClassifier_model))

LogisticRegression_model.fit(X_train, y_train)
predict_LogisticRegression_model = LogisticRegression_model.predict(X_test)
print(LogisticRegression_model, '\n', classification_report(y_test, predict_LogisticRegression_model))

# Снижение размерности -------------------------------------------------------------------------------------------------
PCA_transformer = PCA(n_components=0.90, random_state=random_state)
TSNE_transformer = TSNE(random_state=random_state)

print(
    "Оценка времени препроцессинга -----------------------------------------------------------------------------------\n")
PCA_start = time.time()
PCA_transformer.fit(X)
X_PCA = PCA_transformer.transform(X)
PCA_time = time.time() - PCA_start
print(f"PCA time: {round(PCA_time, 4)} сек.")

TSNE_start = time.time()
X_TSNE = TSNE_transformer.fit_transform(X)
TSNE_time = time.time() - TSNE_start
print(f"TSNE time: {round(TSNE_time, 4)} сек.")

test_size = 0.2

X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = train_test_split(X_PCA, y, test_size=test_size,
                                                                    random_state=random_state)
X_train_TSNE, X_test_TSNE, y_train_TSNE, y_test_TSNE = train_test_split(X_TSNE, y, test_size=test_size,
                                                                        random_state=random_state)

print(f"Количество компонент, необходимое чтобы описать 90% дисперсии в PCA: {PCA_transformer.n_components_}")
# Расчет на сниженной размерности---------------------------------------------------------------------------------------
print(
    "\nОбучение на данных со сниженной размерностью PCA ---------------------------------------------------------------\n")
# PCA
SVC_model.fit(X_train_PCA, y_train_PCA)
predict_SVC_model_PCA = SVC_model.predict(X_test_PCA)
print(SVC_model, '\n', classification_report(y_test_PCA, predict_SVC_model_PCA))

DecisionTreeClassifier_model.fit(X_train_PCA, y_train_PCA)
predict_DecisionTreeClassifier_model_PCA = DecisionTreeClassifier_model.predict(X_test_PCA)
print(DecisionTreeClassifier_model, '\n', classification_report(y_test_PCA, predict_DecisionTreeClassifier_model_PCA))

LogisticRegression_model.fit(X_train_PCA, y_train_PCA)
predict_LogisticRegression_model_PCA = LogisticRegression_model.predict(X_test_PCA)
print(LogisticRegression_model, '\n', classification_report(y_test_PCA, predict_LogisticRegression_model_PCA))

print(
    "\nОбучение на данных со сниженной размерностью TSNE --------------------------------------------------------------\n")
# TSNE
SVC_model.fit(X_train_TSNE, y_train_TSNE)
predict_SVC_model_TSNE = SVC_model.predict(X_test_TSNE)
print(SVC_model, '\n', classification_report(y_test_PCA, predict_SVC_model_TSNE))

DecisionTreeClassifier_model.fit(X_train_TSNE, y_train_TSNE)
predict_DecisionTreeClassifier_TSNE = DecisionTreeClassifier_model.predict(X_test_TSNE)
print(DecisionTreeClassifier_model, '\n', classification_report(y_test_PCA, predict_DecisionTreeClassifier_TSNE))

LogisticRegression_model.fit(X_train_TSNE, y_train_TSNE)
predict_LogisticRegression_model_TSNE = LogisticRegression_model.predict(X_test_TSNE)
print(LogisticRegression_model, '\n', classification_report(y_test_PCA, predict_LogisticRegression_model_TSNE))

print(f"""
Выводы ---------------------------------------------------------------------------------------------------------------\n
1) Как видим, использование алгоритмов PCA и TSNE приводит к увеличению метрики accuracy для DecisionTreeClassifier, однако для SVC и
LogisticRegression_model результаты ухудшаются или не изменяются. Отметим, что для DecisionTreeClassifier TSNE лучше, чем
PCA.\n
2) Количество компонент, необходимое чтобы описать 90% дисперсии в PCA: {PCA_transformer.n_components_}\n
3) Оценка времени препроцессинга:
PCA_time: {round(PCA_time, 4)} сек.
TNSE_time: {round(TSNE_time, 4)} сек.
""")
