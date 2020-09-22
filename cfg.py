import os

input_size = 640
initial_epoch = 0
epoch_num = 1200
val_epoch = 50
lr = 0.0001
decay = 5e-4

load_weights = False

batch_size = 2

data_dir = 'data/'

train_image_dir = data_dir+'train/img'
train_label_dir = data_dir+'train/txt'
val_image_dir = data_dir+'val/img'
val_label_dir = data_dir+'val/txt'


# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.3
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.6
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1) #[5,4,3,2]
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]

if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('saved_model'):
    os.mkdir('saved_model')

saved_model_weights_file_path = ''

pixel_threshold = 0.1
side_vertex_pixel_threshold = 0.5
trunc_threshold = 0.2
predict_cut_text_line = False
predict_write2txt = True
