
import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor



def load_data():
    train = pd.read_csv('train_v2.csv')
    properties = pd.read_csv('properties_2017.csv')
    sample = pd.read_csv('sample_submission.csv')
    
    print("Preprocessing...")
    for c, dtype in zip(properties.columns, properties.dtypes):
        if dtype == np.float64:
            properties[c] = properties[c].astype(np.float32)
            
    print("Set train/test data...")
    id_feature = ['heatingorsystemtypeid','propertylandusetypeid', 'storytypeid', 'airconditioningtypeid',
        'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid', 'typeconstructiontypeid']
    for c in properties.columns:
        properties[c]=properties[c].fillna(-1)
        if properties[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(properties[c].values))
            properties[c] = lbl.transform(list(properties[c].values))
        if c in id_feature:
            lbl = LabelEncoder()
            lbl.fit(list(properties[c].values))
            properties[c] = lbl.transform(list(properties[c].values))
            dum_df = pd.get_dummies(properties[c])
            dum_df = dum_df.rename(columns=lambda x:c+str(x))
            properties = pd.concat([properties,dum_df],axis=1)
            properties = properties.drop([c], axis=1)
            #print np.get_dummies(properties[c])
            
    # Add Feature
    #properties['N-LivingAreaError'] = properties['calculatedfinishedsquarefeet'] / properties['finishedsquarefeet12']
    #properties['N-TaxScore'] = properties['taxvaluedollarcnt'] / properties['taxamount']
    #properties['N-location'] = properties['latitude'] / properties['longitude']
    #properties['N-ValueRatio'] = properties['taxvaluedollarcnt'] / properties['taxamount']
    #properties['N-ValueProp'] = properties['structuretaxvaluedollarcnt'] / properties['landtaxvaluedollarcnt']
    #properties['N-LivingAreaProp'] = properties['calculatedfinishedsquarefeet'] / properties['lotsizesquarefeet']
    
    
    # Make train and test dataframe
    train = train.merge(properties, on='parcelid', how='left')
    sample['parcelid'] = sample['ParcelId']
    test = sample.merge(properties, on='parcelid', how='left')

    # drop out ouliers
    train = train[train.logerror > -0.4]
    train = train[train.logerror < 0.419]

    train["transactiondate"] = pd.to_datetime(train["transactiondate"])
    train["Month"] = train["transactiondate"].dt.month
    train["quarter"] = train["transactiondate"].dt.quarter
    
    test["Month"] = 10
    test['quarter'] = 4

    x_train = train.drop(['parcelid', 'logerror','transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    y_train = train["logerror"].values
    
    x_test = test[x_train.columns]
    del test, train    
    print(x_train.shape, y_train.shape, x_test.shape)
    
    return x_train, y_train, x_test

x_train, y_train, x_test = load_data()

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print ("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]                

                S_train[test_idx, i] = y_pred.reshape(-1,)
                S_test_i[:, j] = clf.predict(T)[:].reshape(-1,)
            S_test[:, i] = S_test_i.mean(axis=1)

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res

## MLP
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor

nn = Sequential()
nn.add(Dense(units = 512 , kernel_initializer = 'random_normal', input_dim = 125))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units = 256, kernel_initializer = 'random_normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units=64, kernel_initializer='random_normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units=16, kernel_initializer='random_normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units=4, kernel_initializer='random_normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units=1, kernel_initializer='random_normal'))
nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))


#gbr params
gbr_params = {}
gbr_params['n_estimators'] = 3000
gbr_params['learning_rate'] = 0.05
gbr_params['max_depth'] = 3
gbr_params['max_features'] = 'sqrt'
gbr_params['min_samples_leaf'] = 15
gbr_params['min_samples_split'] = 10
gbr_params['loss'] = 'huber'


# rf params
rf_params = {}
rf_params['n_estimators'] = 30
rf_params['max_depth'] = 8
rf_params['min_samples_split'] = 100
rf_params['min_samples_leaf'] = 30

# xgb params
xgb_params = {}
#xgb_params['n_estimators'] = 50
xgb_params['min_child_weight'] = 12
xgb_params['learning_rate'] = 0.37
xgb_params['max_depth'] = 6
xgb_params['subsample'] = 0.77
xgb_params['reg_lambda'] = 0.8
xgb_params['reg_alpha'] = 0.4
xgb_params['base_score'] = 0
#xgb_params['seed'] = 400
xgb_params['silent'] = 1


# lgb params
lgb_params = {}
#lgb_params['n_estimators'] = 50
lgb_params['max_bin'] = 8
lgb_params['learning_rate'] = 0.37 # shrinkage_rate
lgb_params['metric'] = 'l1'          # or 'mae'
lgb_params['sub_feature'] = 0.35    
lgb_params['bagging_fraction'] = 0.85 # sub_row
lgb_params['bagging_freq'] = 40
lgb_params['num_leaves'] = 512        # num_leaf
lgb_params['min_data'] = 500         # min_data_in_leaf
lgb_params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
lgb_params['verbose'] = 0
lgb_params['feature_fraction_seed'] = 2
lgb_params['bagging_seed'] = 3

# Bagging Regressor
bgg_params = {}
#bgg_params['base_estimator'] = 'None'
bgg_params['n_estimators'] = 3000
#bgg_params['max_samples'] = 1.0
bgg_params['max_features'] = 10
#bgg_params['bootstrap'] = 'True'
#bgg_params['bootstrap_features'] = 'False'
bgg_params['oob_score'] = 'True'
#bgg_params['warm_start'] = 'True'
#bgg_params['n_jobs'] = 1
#bgg_params['random_state'] = 'None'
#bgg_params['verbose'] = 0


# CatBoos params
cb_params = {}
cb_params['iterations'] = 630
cb_params['learning_rate'] = 0.03
cb_params['depth'] = 6
cb_params['l2_leaf_reg'] = 3
cb_params['loss_function'] = 'MAE'
cb_params['eval_metric'] = 'MAE'

cb_model = CatBoostRegressor(**cb_params)

bgg_model = BaggingRegressor(**bgg_params)

gbr_model = GradientBoostingRegressor(**gbr_params)

xgb_model = XGBRegressor(**xgb_params)

lgb_model = LGBMRegressor(**lgb_params)

rf_model = RandomForestRegressor(**rf_params)

et_model = ExtraTreesRegressor()

# SVR model ; SVM is too slow in more then 10000 set
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.05)

# DecsionTree model
dt_model = DecisionTreeRegressor()

# AdaBoost model
ada_model = AdaBoostRegressor()

stack = Ensemble(n_splits=7,
        stacker= HuberRegressor(),
        base_models=(nn, cb_model, gbr_model, rf_model, xgb_model, et_model, ada_model))

y_test = stack.fit_predict(x_train, y_train, x_test)

from datetime import datetime
print("submit...")
pre = y_test
sub = pd.read_csv('sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = pre
submit_file = '{}.csv'.format(datetime.now().strftime('%Y%m%d_%H_%M'))
sub.to_csv(submit_file, index=False,  float_format='%.4f')