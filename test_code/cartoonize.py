import os
import cv2
import numpy as np
import tensorflow as tf 
import network
import guided_filter
from tqdm import tqdm

input_photo = None
final_out = None
sess = None

def init():
    load_model('saved_models')

def load_model(model_path): 
    global input_photo, final_out, sess

    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image
    

def cartoonize(image):
    try:
        image = resize_crop(image)
        batch_image = image.astype(np.float32)/127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        output = sess.run(final_out, feed_dict={input_photo: batch_image})
        output = (np.squeeze(output)+1)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
    except:
        print('cartoonize {} failed'.format(load_path))

def call_cartoonize(ip_file, op_file):
    image = cv2.imread(ip_file)
    cv2.imwrite(op_file, cartoonize(image))

if __name__ == '__main__':
    model_path = 'saved_models'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    init()

    ip_file = os.path.join(load_folder, 'actress2.jpg')
    op_file = os.path.join(save_folder, 'actress2.jpg')
    call_cartoonize(ip_file, op_file)
    
    ip_file = os.path.join(load_folder, 'party5.jpg')
    op_file = os.path.join(save_folder, 'party5.jpg')
    call_cartoonize(ip_file, op_file)

    ip_file = os.path.join(load_folder, 'party7.jpg')
    op_file = os.path.join(save_folder, 'party7.jpg')
    call_cartoonize(ip_file, op_file)
    

    