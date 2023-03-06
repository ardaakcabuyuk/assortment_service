import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
import urllib.request
from io import BytesIO
from zipfile import ZipFile
import pickle
import random
import sys

from utils import get_R2_test, give_val_ind

## READ PICKLE DUMPS OF X,Y,Z,S ##

#file_X = open(r'pickle/Pickled_static_clustered_same_cabinet_final.p', 'rb')
X_matrix = pickle.load(urllib.request.urlopen('https://drive.google.com/uc?export=download&id=1EXnPKIXuZQI4a_UimXfeWlyutO2c4bt_'))
X_matrix = X_matrix.values
X_matrix = X_matrix[0:15000]

#file_Y = open(r'pickle/Pickled_Y_final.p', 'rb')
resp_Y = urllib.request.urlopen('https://drive.google.com/uc?export=download&id=1XfN59nGId8KmbJ3xsYYiv5K58Yl2yhme')
print(resp_Y)
myzip_Y = ZipFile(BytesIO(resp_Y.read()))
Y_mat = pickle.load(myzip_Y.open('Pickled_Y_final.p'))
Y_matrix = np.zeros((Y_mat.shape[0], Y_mat.shape[2], Y_mat.shape[1]))
for i in range(Y_matrix.shape[0]):
  Y_matrix[i] = Y_mat[i].T

#file_Z = open(r'pickle/Pickled_S_bin_final_son_POS.p', 'rb')
Z_matrix = pickle.load(urllib.request.urlopen('https://drive.google.com/uc?export=download&id=14Du-khiN4_53Or6OUGW0NTPdGxekIS0C'))
Z_matrix = Z_matrix[0:15000]

#file_S = open(r'pickle/Pickled_S_final_son_POS.p', 'rb')
S_matrix = pickle.load(urllib.request.urlopen('https://drive.google.com/uc?export=download&id=1oNiC6ngDgRn8K65g1VdLCqrPL9cjd2Y1'))
S_matrix = S_matrix[0:15000]

## READ PICKLE DUMPS OF X,Y,Z,S ##

## MINMAX NORMALIZE Y ##

scalers = {}
for i in range(Y_matrix.shape[2]):
  scalers[i] = MinMaxScaler()
  Y_matrix[:, :, i] = scalers[i].fit_transform(Y_matrix[:, :, i])

## MINMAX NORMALIZE Y ##

## CREATE TRAINING AND TEST SETS ##

S_vector = np.sum(S_matrix, axis=1)

N = len(X_matrix)
remaining_indices = sorted(list(range(N)),reverse=False)
ts_ind = random.sample(remaining_indices,round(N*0.20))
remaining_indices = list(set(remaining_indices).difference(set(ts_ind)))
val_ind = random.sample(remaining_indices,round(N*0.20))
remaining_indices = list(set(remaining_indices).difference(set(val_ind)))
tr_ind = remaining_indices

X_training = X_matrix[tr_ind,:]
X_validation = X_matrix[val_ind,:]
X_test = X_matrix[ts_ind,:]

Y_training = Y_matrix[tr_ind,:,:]
Y_validation = Y_matrix[val_ind,:,:]
Y_test = Y_matrix[ts_ind,:,:]

Z_training = Z_matrix[tr_ind,:]
Z_validation = Z_matrix[val_ind,:]
Z_test = Z_matrix[ts_ind,:]

S_training = S_vector[tr_ind]
S_validation = S_vector[val_ind]
S_test = S_vector[ts_ind]

## CREATE TRAINING AND TEST SETS ##

## CREATE MODEL ##

# the first branch operates on the first input
inputX = Input(shape=(60,))
x = Dense(100, activation="relu")(inputX)

# the second branch opreates on the second input
inputY = Input(shape=(12, 328, ))
y = LSTM(100, dropout=0.5)(inputY)

inputZ = Input(shape=(328,))

# combine the output of the three branches
combined = Concatenate()([x, y, inputZ])

fc_1 = Dense(100, activation="relu")(combined)
drop_1 = Dropout(rate=0.5)(fc_1)

# Output dense should always be 1.
output = Dense(1, activation="linear")(drop_1)

model = tf.keras.models.load_model(r"model_output.pkl")

S_hat_training = model.predict([X_training, Y_training, Z_training])[:,0]
S_hat_validation = model.predict([X_validation, Y_validation, Z_validation])[:,0]
S_hat_test = model.predict([X_test, Y_test, Z_test])[:,0]

print(get_R2_test(S_training, S_hat_training, S_hat_training))
print(get_R2_test(S_validation, S_hat_validation, S_hat_training))
print(get_R2_test(S_test, S_hat_test, S_hat_training))

output_x = Dense(100, activation="relu")(inputX)
model_x = Model(inputs=inputX, outputs=output_x)
model_x.set_weights(model.layers[2].get_weights())
model_x.compile(loss="mean_squared_error", optimizer="adam")
my_x = model_x.predict(X_validation)

output_y = LSTM(100, dropout=0.5)(inputY)
model_y = Model(inputs=inputY, outputs=output_y)
model_y.set_weights(model.layers[3].get_weights())
model_y.compile(loss="mean_squared_error", optimizer="adam")
my_y = model_y.predict(Y_validation)

np.set_printoptions(threshold=sys.maxsize)
my_z = Z_validation

int_x = Input(shape=(100,))
int_y = Input(shape=(100,))
int_z = Input(shape=(328,))

combined = Concatenate()([int_x, int_y, int_z])
output_int_1 = Dense(100, activation="relu")(combined)
model_int_1 = Model(inputs=[int_x, int_y, int_z], outputs=output_int_1)
model_int_1.set_weights(model.layers[6].get_weights())
model_int_1.compile(loss="mean_squared_error", optimizer="adam")
my_output_int_1 = model_int_1.predict([my_x, my_y, Z_validation])


int_hidden = Input(shape=(100,))
output_int_2 = Dense(1, activation="linear")(int_hidden)
model_int_2 = Model(inputs=int_hidden, outputs=output_int_2)
model_int_2.set_weights(model.layers[8].get_weights())
model_int_2.compile(loss="mean_squared_error", optimizer="adam")
my_output = model_int_2.predict(my_output_int_1)

output = Dense(1, activation="linear")(fc_1)

## CREATE MODEL ##

## READ DATA ##

unique_store_list_original_index = pd.read_csv("csv/MUSTERINOS_FINAL.csv")
unique_store_list_original_index.columns = ["MUSTERINO"]
unique_store_list = unique_store_list_original_index["MUSTERINO"].to_list()

all_products_list=pd.read_csv('csv/products_gsv_sorted.csv')
all_products_list=all_products_list.drop(columns=['Unnamed: 0'])

unique_product_list_1 = pd.read_csv('intervals/interval1.csv')
unique_product_list_1=unique_product_list_1.drop(columns=['Unnamed: 0'])
unique_product_list_2 = pd.read_csv('intervals/interval2.csv')
unique_product_list_2=unique_product_list_2.drop(columns=['Unnamed: 0'])
unique_product_list_3 = pd.read_csv('intervals/interval3.csv')
unique_product_list_3=unique_product_list_3.drop(columns=['Unnamed: 0'])
unique_product_list_4 = pd.read_csv('intervals/interval4.csv')
unique_product_list_4=unique_product_list_4.drop(columns=['Unnamed: 0'])
unique_product_list_5 = pd.read_csv('intervals/interval5.csv')
unique_product_list_5=unique_product_list_5.drop(columns=['Unnamed: 0'])


unique_product_code_list = all_products_list["PRODUCTCODE"].to_list()
unique_product_name_list = all_products_list["SKUName"].to_list()
unique_product_gsv_ltr_list = all_products_list["GSV/LTR"].to_list()

month_list = np.array(["2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", 
                       "2021-08-01", "2021-09-01", "2021-10-01", "2021-11-01", 
                       "2021-12-01","2022-01-01", "2022-02-01", "2022-03-01", 
                       "2022-04-01", "2022-05-01", "2022-06-01"])

pos_month_index=[]
pos_obs_month_val=[]

for i in range(len(val_ind)):
    pos_month_index.append((val_ind[i]//15,val_ind[i]%15))
    
for i in range(len(pos_month_index)):
    pos_obs_month_val.append((val_ind[i],unique_store_list[pos_month_index[i][0]],month_list[pos_month_index[i][1]]))

## READ DATA ##

## SET PARAMETERS and WEIGHTS ##

# Parameters
nr_hidden_layers = 1
nr_nodes_per_hidden_layer = [100] #output_int_1 node sayısı

#total GSV output aggregated
nr_output_nodes = 1 

# Corresponding Weights 1
w1 = model.layers[6].get_weights()
weights_1 = w1[0]
weights_1_bias = w1[1]

weights_1_x= weights_1 [0:100,:]
weights_1_y= weights_1 [100:200,:]
weights_1_z= weights_1 [200:,:]

w2=model.layers[8].get_weights()
weights_2 = w2[0]
weights_2_bias = w2[1]

## SET PARAMETERS and WEIGHTS ##

def get_suggested_assortments(musterino: int, observation_month: str):

    #obs_ind=i
    #eğer input alan fonskiyon çalışsaydı for looptan çıkartıp 270. satır yerine aşağıdaki hidelı linelar ile ilerleyecektik
    obs_ind = give_val_ind(pos_obs_month_val, val_ind, musterino, observation_month)
    #print(obs_ind)
    #print(pos_obs_month_val[obs_ind][0])
    print("The MUSTERINO:",pos_obs_month_val[obs_ind][1]," and the Observation Month is:",pos_obs_month_val[obs_ind][2],".")
     
    feature_x = my_x[obs_ind]   
    feature_y = my_y[obs_ind]    
    feature_z = my_z[obs_ind]
    
    M=100000
    epsilon = 10**(-6)
    
    # Create a new model
    m = gp.Model("Algida Assortment Optimization")
    
    # Create variables
    x = []
    for k in range(100):
        x.append(m.addVar(vtype='C', ub=feature_x[k], lb=feature_x[k], name="x[{}]".format(k))) 
    #    x.append(m.addVar(vtype='B', name="x[{}]".format(i))) 
    m.update()
    
    y = []
    for k in range(100):
        y.append(m.addVar(vtype='C', ub=feature_y[k], lb=feature_y[k], name="y[{}]".format(k))) 
        
    m.update()
     
    z = []
    for k in range(328):
        z.append(m.addVar(vtype='B', name="z[{}]".format(k))) 
    m.update()
    
    h = []
    for l in range(nr_hidden_layers):   
        h.append([]) 
        for k in range(nr_nodes_per_hidden_layer[l]) :
                h[l].append(m.addVar(vtype='C', ub=M, name="h[{}][{}]".format(l,k))) 
    m.update()
       
    u = []
    for k in range(100):
        u.append(m.addVar(vtype='B', name="u[{}]".format(k))) 
    m.update()
    
    s = []
    for k in range(nr_output_nodes):
        s.append(m.addVar(vtype='C', lb=-M, ub=M, name="s[{}]".format(k)))
    m.update()
    
    delta1= [] # for the difference between current and optimal assortment
    for k in range(328):
        delta1.append(m.addVar(vtype='C', name='delta[{}]'.format(k)))
    m.update
    
    delta2=[] # for the absolute difference between current and optimal assortment
    for k in range(328):
        delta2.append(m.addVar(vtype='C', name='abs_delta[{}]'.format(k)))
    m.update
    
    abs_delta=[] # for the absolute difference between current and optimal assortment
    for k in range(328):
        abs_delta.append(m.addVar(vtype='C', name='abs_delta[{}]'.format(k)))
    m.update
    
    # Set objective function (maksimum gsv)
    obj = gp.quicksum(s[k] for k in range(nr_output_nodes))
    m.setObjective(obj, GRB.MAXIMIZE)
      
    l=0 
    for j in range(nr_nodes_per_hidden_layer[l]):
        m.addConstr(h[l][j]>=weights_1_bias[j]+sum(weights_1_x[k,j] * x[k] for k in range(100))+ sum(weights_1_y[k,j] * y[k] for k in range(100))+ sum(weights_1_z[k,j] * z[k] for k in range(328)))
    
    for j in range(nr_nodes_per_hidden_layer[l]):
        m.addConstr(h[l][j]>=0)
        
    for j in range(nr_nodes_per_hidden_layer[l]):
        m.addConstr(h[l][j]<=weights_1_bias[j]+sum(weights_1_x[k,j] * x[k] for k in range(100))+ sum(weights_1_y[k,j] * y[k] for k in range(100))+ sum(weights_1_z[k,j] * z[k] for k in range(328)) + M*(1-u[j])) 
        
    for j in range(nr_nodes_per_hidden_layer[l]):
        m.addConstr(h[l][j]<=M*u[j])
        
    for k in range(nr_output_nodes):
        m.addConstr(s[k]==weights_2_bias[k]+sum(weights_2[j,k] * h[l][j] for j in range(nr_nodes_per_hidden_layer[l])) )
     
    # segmentasyon sayesinde ve farklı assortmentları sağlayan constraintler
    if X_matrix[val_ind[obs_ind]][48] == 1: #high alışverişçi pos için
        m.addConstr(sum(z[k] for k in range(179)) <= 8)
        m.addConstr(sum(z[k] for k in range(179,260)) <= 4)
        m.addConstr(sum(z[k] for k in range(260,315)) <= 4)
        m.addConstr(sum(z[k] for k in range(315,326)) <= 2)
        m.addConstr(sum(z[k] for k in range(326,328)) <= 2)
    elif X_matrix[val_ind[obs_ind]][49] == 1: #low alışverişçi pos için
        m.addConstr(sum(z[k] for k in range(179)) <= 11)
        m.addConstr(sum(z[k] for k in range(179,260)) <= 5)
        m.addConstr(sum(z[k] for k in range(260,315)) <= 2)
        m.addConstr(sum(z[k] for k in range(315,326)) <= 1)
        m.addConstr(sum(z[k] for k in range(326,328)) <= 1)
    elif X_matrix[val_ind[obs_ind]][50] == 1: #lower-mid alışverişçi pos için
        m.addConstr(sum(z[k] for k in range(179)) <= 10)
        m.addConstr(sum(z[k] for k in range(179,260)) <= 5)
        m.addConstr(sum(z[k] for k in range(260,315)) <= 3)
        m.addConstr(sum(z[k] for k in range(315,326)) <= 1)
        m.addConstr(sum(z[k] for k in range(326,328)) <= 1)
    elif X_matrix[val_ind[obs_ind]][51] == 1: #upper-mid alışverişçi pos için
        m.addConstr(sum(z[k] for k in range(179)) <= 9)
        m.addConstr(sum(z[k] for k in range(179,260)) <= 5)
        m.addConstr(sum(z[k] for k in range(260,315)) <= 3)
        m.addConstr(sum(z[k] for k in range(315,326)) <= 2)
        m.addConstr(sum(z[k] for k in range(326,328)) <= 1)
       
    
    # abs difference olayı için gerekli constraintler (şimdilik bunu göz ardı edebilirsin)
    
    # for k in range(328):
    #     m.addConstr(delta1[k]== (z[k] - feature_z[k]))
    #     m.addConstr(delta2[k]== -(z[k] - feature_z[k]))
    #     m.addConstr(abs_delta[k]== gp.abs_(delta1[k]))
        
    # m.addConstr(sum(abs_delta[k]for k in range(328)) <= 5)     
       
    # Solve it!
    m.optimize()
    m.getVars()
    
    assortment_list=np.zeros(328)
    placed_products=[]
    for k in range(328):
        assortment_list[k]=z[k].x
        if assortment_list[k] == 1:
            placed_products.append((unique_product_code_list[k],unique_product_name_list[k]))
    print("The products in the optimal assortment:")
    for k in placed_products:
        print(k)    
    print(f"Optimal objective value: {m.objVal}")
    return placed_products