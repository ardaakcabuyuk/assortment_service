import numpy as np

def get_R2_test(my_y, my_yhat, y_train):
  mean_pt = np.mean(y_train)
  
  diff = my_y - my_yhat
  SSR = sum(diff**2)

  SST = sum((my_y - mean_pt)**2)

  results = 1- (SSR/SST)
  return results, SSR, SST

# musterıno ve obs_month input alıp optimizationa yansıtacak olan fonksiyon denemesi    
def give_val_ind(arr, val_ind, a, b):
  for i in range(len(arr)):
    if a == arr[i][1] and b == arr[i][2]:
      return val_ind.index(arr[i][0])
  return -1