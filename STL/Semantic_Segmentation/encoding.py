import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

id_to_trainId = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 
    14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 
    20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 
    27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 
    33: 18, -1: -1
}



seg_map = Image.open('/Users/jaybhatt/BTP/CityScapes/train/semantic/2.png')
seg_map = np.array(seg_map)

def encode(seg_map, id_to_trainId):
    train_map = np.copy(seg_map)
    
    for id, trainId in id_to_trainId.items():
        train_map[train_map == id] = trainId
        
    return train_map

train_map = encode(seg_map, id_to_trainId)
print(np.unique(train_map))
plt.imshow(train_map)
plt.show()