import os
import warnings

# Silence TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Disable XLA duplication logs
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import os
import random
import cv2
import imutils
import random
import matplotlib.pyplot as plt
import keras
import keras.layers as L
import keras.models as M
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
from keras.utils import Sequence
from jiwer import wer, cer
from tensorflow.keras import backend as K
img_size = 32
BATCH_SIZE = 32


import zipfile
local_zip = "C:\\Users\\Yen's PC\\Downloads\\Handwriting Recognition.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(r"C:\Users\Yen's PC\Downloads\Handwriting Recognition")
zip_ref.close()

train = pd.read_csv(r"C:\Users\Yen's PC\Downloads\Handwriting Recognition\written_name_train_v2.csv")
val = pd.read_csv(r"C:\Users\Yen's PC\Downloads\Handwriting Recognition\written_name_validation_v2.csv")

train.dropna(inplace=True)
train['Length']=train['IDENTITY'].apply(lambda x : len(str(x)))
train=train[train['Length']<=16]
train['IDENTITY']=train['IDENTITY'].str.upper()
train[train['Length']==max(train['Length'])]
train=train.sample(frac=0.6,random_state=42)
val.dropna(inplace=True)
val['Length'] = val['IDENTITY'].apply(lambda x: len(str(x)))
val = val[val['Length'] <= 16]
val['IDENTITY'] = val['IDENTITY'].str.upper()
val = val.sample(frac=0.1)

characters = set()
train['IDENTITY'] = train['IDENTITY'].apply(lambda x: str(x))
for name in train['IDENTITY'].values:
    for char in name:
        if char not in characters:
            characters.add(char)
characters = sorted(list(characters))

char_to_label = {char: idx for idx, char in enumerate(characters)}
label_to_char = {idx: char for idx, char in enumerate(characters)}

path_train = r"C:\Users\Yen's PC\Downloads\Handwriting Recognition\train_v2\train"
path_val = r"C:\Users\Yen's PC\Downloads\Handwriting Recognition\validation_v2\validation"

# Data Generator
class DataGenerator(Sequence):
    def __init__(self,dataframe,path,char_map,batch_size=128,img_size=(256,64),
                 downsample_factor=4,max_length=16,shuffle=True):
        self.dataframe=dataframe
        self.path=path
        self.char_map=char_map
        self.batch_size=batch_size
        self.width=img_size[0]
        self.height=img_size[1]
        self.downsample_factor=downsample_factor
        self.max_length=max_length
        self.shuffle=shuffle
        self.indices = np.arange(len(dataframe))
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.dataframe)//self.batch_size
    def __getitem__(self,idx):
        curr_batch_idx=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_images=np.ones((self.batch_size,self.width,self.height,1),dtype=np.float32)
        batch_labels=np.ones((self.batch_size,self.max_length),dtype=np.float32)
        input_length = np.ones((self.batch_size,1), dtype=np.int32) * ((self.width // self.downsample_factor) - 2)
        label_length=np.zeros((self.batch_size,1),dtype=np.int64)
        
        for i,idx in enumerate(curr_batch_idx):
            img_path=self.dataframe['FILENAME'].values[idx]
            img=cv2.imread(self.path+'/'+img_path)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img=cv2.resize(img,(self.width,self.height))
            img=(img/255).astype(np.float32)
            img=img.T
            img=np.expand_dims(img,axis=-1)
            text=self.dataframe['IDENTITY'].values[idx]
            text=str(text)
            label=[]
            for j in text:
                if j in self.char_map:
                    label.append(self.char_map[j])
                else:
                    label.append(100)

            if len(label) > self.max_length:
                label = label[:self.max_length]

            label_length[i] = len(label)

            label.extend([100]*(self.max_length-len(label)))

            batch_labels[i] = label
            batch_images[i] = img
        batch_inputs= {
                'input_data':batch_images,
                'input_label':batch_labels,
                'input_length':input_length,
                'label_length':label_length
        }
        return batch_inputs,np.zeros((self.batch_size),dtype=np.float32)
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

train_generator=DataGenerator(train,path_train,char_to_label,batch_size=BATCH_SIZE,img_size=(128,32))
val_generator=DataGenerator(val,path_val,char_to_label,batch_size=BATCH_SIZE,img_size=(128,32))

class CTCLayer(L.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        
        # On test time, just return the computed loss
        return loss
    
def make_model():
    inp = L.Input(shape=(128, 32, 1), dtype=np.float32, name='input_data')
    labels = L.Input(shape=[16], dtype=np.float32, name='input_label')
    input_length = L.Input(shape=[1], dtype=np.int64, name='input_length')
    label_length = L.Input(shape=[1], dtype=np.int64, name='label_length')
    
    # CNN
    x = L.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = L.MaxPooling2D(pool_size=(2, 2))(x)
    x = L.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = L.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Reshape: (batch, 32, 8, 64) → (batch, 32, 512)
    x = L.Reshape((32, 512))(x)
    
    # RNN
    x = L.Dense(256, activation='relu')(x)
    x = L.Dropout(0.2)(x)
    x = L.Bidirectional(L.LSTM(128, return_sequences=True))(x)
    x = L.Dropout(0.2)(x)
    x = L.Bidirectional(L.LSTM(64, return_sequences=True))(x)
    
    # Output
    x = L.Dense(len(characters) + 1, activation='softmax')(x)
    output = CTCLayer(name='outputs')(labels, x, input_length, label_length)
    
    model = M.Model([inp, labels, input_length, label_length], output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss={'outputs': lambda y_true, y_pred: y_pred})
    return model
model=make_model()
model.summary()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
print(os.listdir('./'))
if 'prediction_model_ocr.h5' not in os.listdir('./'):
    history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=val_generator,
                    epochs=10,
                    callbacks=[es])  # ✅ Thêm early stopping

prediction_model = tf.keras.models.Model(model.input[0],
                                    model.get_layer(name='dense_1').output)
prediction_model.summary()
print(prediction_model.input)
if 'prediction_model_ocr.h5' not in os.listdir('./'):
    prediction_model.save('prediction_model_ocr.h5')
    prediction_model = tf.keras.models.load_model('prediction_model_ocr.h5', compile=False)

label_to_char[100] = ''
def decode_batch_predictions(pred):
    pred = pred[:, :-2]
    input_len = np.ones(pred.shape[0])*pred.shape[1]
    
    # Use greedy search. For complex tasks, you can use beam search
    results = tf.keras.backend.ctc_decode(pred, 
                                        input_length=input_len,
                                        greedy=True)[0][0]
    
    # Iterate over the results and get back the text
    output_text = []
    for res in results.numpy():
        outstr = ''
        for c in res:
            if c < len(characters) and c >=0:
                outstr += label_to_char[c]
        output_text.append(outstr)
    
    # return final text results
    return output_text

for p, (inp_value, _) in enumerate(val_generator):
    bs = inp_value['input_data'].shape[0]
    X_data = inp_value['input_data']
    labels = inp_value['input_label']
    plt.imshow(X_data[0].squeeze().T, cmap='gray')
    preds = prediction_model.predict(X_data)
    pred_texts = decode_batch_predictions(preds)
    
    
    orig_texts = []
    for label in labels:
        text = ''.join([label_to_char[int(x)] for x in label])
        orig_texts.append(text)
    with open('predictions.txt', 'w') as f:    
        for i in range(bs):
            print(f'Ground truth: {orig_texts[i]} \t Predicted: {pred_texts[i]}')
            f.write(f'Ground truth: {orig_texts[i]} \t Predicted: {pred_texts[i]}\n')
        break
    
test_path = r"C:\Users\Yen's PC\Downloads\Handwriting Recognition\test_v2\test"

# lấy danh sách ảnh
test_images = os.listdir(test_path)

# chọn random 1 ảnh
random_image = random.choice(test_images)

# tạo batch
batch_images = np.ones((1,128,32,1))
img_path = os.path.join(test_path, random_image)
img = cv2.imread(img_path)
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img=cv2.resize(img,(128,32))
img=(img/255).astype(np.float32)
img=img.T
img=np.expand_dims(img,axis=-1)
batch_images[0]=img
x=prediction_model.predict(batch_images)
pred_texts = decode_batch_predictions(x)
pred_texts = pred_texts[0]
im=cv2.imread(img_path)
plt.imshow(im)
print('Predicted Text:',pred_texts)
def calculate_wer(pred_texts, orig_texts):
    total_wer = 0
    for pred, orig in zip(pred_texts, orig_texts):
        total_wer += wer(orig, pred)
    return total_wer / len(pred_texts)

def calculate_cer(pred_texts, orig_texts):
    total_cer = 0
    for pred, orig in zip(pred_texts, orig_texts):
        total_cer += cer(orig, pred)
    return total_cer / len(pred_texts)

def evaluate_model(prediction_model, val_generator):
    all_pred_texts = []
    all_orig_texts = []
    
    for inp_value, _ in val_generator:
        X_data = inp_value['input_data']
        labels = inp_value['input_label']
        
        preds = prediction_model.predict(X_data)
        pred_texts = decode_batch_predictions(preds)
        
        orig_texts = []
        for label in labels:
            text = ''.join([label_to_char[int(x)] for x in label])
            orig_texts.append(text)
        
        all_pred_texts.extend(pred_texts)
        all_orig_texts.extend(orig_texts)
    
    wer_score = calculate_wer(all_pred_texts, all_orig_texts)
    cer_score = calculate_cer(all_pred_texts, all_orig_texts)
    
    return wer_score, cer_score
wer_score, cer_score = evaluate_model(prediction_model, val_generator)
print(f'Word Error Rate (WER): {wer_score:.4f}')
print(f'Character Error Rate (CER): {cer_score:.4f}')
