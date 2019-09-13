from tensorflow.python.keras.callbacks import ModelCheckpoint,TensorBoard
from model import *
from data_set import *
import os

import tensorflow as tf
from tensorflow.python.keras.utils import multi_gpu_model
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

from divideImage import init_box

data_gen_args = dict(rotation_range=2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

run_mode = "train_GPUs"
save_mode = "single"
# Choose Devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

model = SDFCN()
if run_mode == "train":
    myGene = trainGenerator(15,'temp_data/pitches/train','image','label',data_gen_args,save_to_dir = None)
    model_checkpoint = ModelCheckpoint('SDFCN_membrane_0912.hdf5', monitor='loss',verbose=1, save_best_only=False)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(myGene,steps_per_epoch=100, epochs=1e4,
                        callbacks=[model_checkpoint,TensorBoard(log_dir="./log_0912")],
                        shuffle=True,workers=2, use_multiprocessing=True)

elif run_mode == "train_GPUs":
    myGene = trainGenerator(15, 'temp_data/pitches/train', 'image', 'label', data_gen_args, save_to_dir=None)
    parallel_model = multi_gpu_model(model, gpus=3)
    checkpoint = ParallelModelCheckpoint(model, filepath='SDFCN_membrane_0913.hdf5', monitor='loss', verbose=1,
                                         save_best_only=False)  # 解决多GPU运行下保存模型报错的问题
    parallel_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    parallel_model.fit_generator(myGene,steps_per_epoch=100,epochs=10,
                                 callbacks=[TensorBoard(log_dir="./log_0913_multgpu")],
                                 shuffle=True) # workers=2, use_multiprocessing=True
elif run_mode == "test":
    model.load_weights("SDFCN_membrane_0912.hdf5")

# pitch数
num = 1000
# 融合大图时，每张总共的小pitch
each_image_size = 1
# temp_data/pitches/test/image
testGene = testGenerator("temp_data/temp_test",num * each_image_size)
results = model.predict_generator(testGene, num * each_image_size, verbose=1)

if save_mode == "single":
    saveResult("temp_data/temp_test",results)
elif save_mode == "full":
    saveBigResult("temp_data/full", results, init_box, each_image_size, num)