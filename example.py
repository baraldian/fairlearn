from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as skm
import pandas as pd
import fairlearn.datasets as fld
import fairlearn.reductions as flr
import fairlearn.metrics as flm
from fairlearn.metrics import MetricFrame
import time
import numpy as np

def report(*, estimator, name, **kwarg):
    y_train_pred = estimator.predict(X_train, **kwarg)
    y_test_pred = estimator.predict(X_test, **kwarg)
    print(
        f"{name}.accuracy.train:  {skm.accuracy_score(y_train, y_train_pred):.3f}\n" +
        f"{name}.accuracy.test:   {skm.accuracy_score(y_test, y_test_pred):.3f}\n" +
        f"{name}.disparity.train: {flm.demographic_parity_difference(y_train, y_train_pred, sensitive_features=A_train):.3f}\n" +
        f"{name}.disparity.test:  {flm.demographic_parity_difference(y_test, y_test_pred, sensitive_features=A_test):.3f}\n")

dataset = fld.fetch_adult(as_frame=True)
X_df = dataset.data.drop(columns = ['workclass','fnlwgt','education','native-country'])
X_df['over_25'] = X_df['age']>=25
X_dummies = pd.get_dummies(X_df)

X = StandardScaler().fit_transform(X_dummies)
A = X_df[['sex']].to_numpy()
#A = X_df[['sex','over_25']].to_numpy()
y = dataset.target.map({'<=50K': 0, '>50K': 1})
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, random_state=19)

rep = 2   # make training set artificially larger

X_train = np.repeat(X_train, rep, axis=0)
y_train = np.repeat(y_train, rep)
A_train = np.repeat(A_train, rep, axis=0)

print(f"n_train = {X_train.shape[0]}\n")

t0 = time.time()
lr = LogisticRegression(C=0.01, random_state=19).fit(X_train, y_train)
elapsed_lr = time.time() - t0
print(f"lr.elapsed: {elapsed_lr:.3f}")
report(estimator=lr, name='lr')

for subsample in [100, 1000, 10000, None]:
    name = "eg"+str(subsample)
    t0 = time.time()
    eg = flr.ExponentiatedGradient(estimator=LogisticRegression(C=0.01, random_state=19),
        constraints=flr.DemographicParity(difference_bound=0.05),
        subsample=subsample, random_state=199)
    eg.fit(X_train, y_train, sensitive_features=A_train)
    elapsed_eg = time.time() - t0
    print(f"{name}.elapsed: {elapsed_eg:.3f}")
    times = pd.DataFrame(eg.oracle_execution_times_)
    times['total'] = times.sum(axis='columns')
    n = times.shape[0]
    print(pd.concat( [times.loc[[0,1,n-1],:],
                      times.sum().to_frame(name='total').T]) )
    report(estimator=eg, name=name, random_state=19)

    if subsample is None:
        continue
    subsample_gs = 2*subsample
    name_gs = name+"gs"+str(subsample_gs)
    t0 = time.time()
    gs = flr.GridSearch(estimator=LogisticRegression(C=0.01, random_state=19),
        constraints=flr.DemographicParity(difference_bound=0.05),
        grid=eg.lambda_vecs_,
        subsample=subsample_gs, random_state=1999)
    gs.fit(X_train, y_train, sensitive_features=A_train)
    elapsed_gs = time.time() - t0
    print(f"{name_gs}.elapsed: {elapsed_gs:.3f}")
    report(estimator=gs, name=name_gs)
