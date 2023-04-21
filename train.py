import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Add
from dataset import train_dataset_sim, test_dataset_sim
from loss import G_loss
from args import parse_args

from processing.G_select import select_G
from optics.C_select import select_C
import setting.imaging_model as conv
from setting.optics_setting import *

import os
import time
import numpy as np
import scipy.io as sio
import optics.C.MA as optics

from tensorflow.python.training.tracking.tracking import AutoTrackable

import os


## Logging for TensorBoard
def log(name, BL_img, GT_img, G, vgg_model, summary_writer,step, params, args):

    sensor_img = optics.sensor_noise(BL_img, params)
    
    _,G_img0 = params['deconv_fn'](sensor_img, G, training=False)

    # Losses
    gt_img0 = tf.image.resize_with_crop_or_pad(GT_img, params['out_width'], params['out_width'])

    tile_times = tf.constant([1,1,1,3],tf.int32)
    G_img = tf.tile(G_img0,tile_times)
    gt_img = tf.tile(gt_img0,tile_times)

    gt_img = tf.clip_by_value(gt_img, 0.0, 1.0)
    G_img = tf.clip_by_value(G_img, 0.0, 1.0)
    
    G_Content_loss_val, G_loss_components, G_metrics = G_loss(G_img, gt_img, vgg_model, args)   

    # Save records to TensorBoard
    with summary_writer.as_default():
        # Images
        tf.summary.image(name = name+ 'Input/Input' , data=gt_img, step=step)

        num_patches = params['batchSize']
        for i in range(num_patches):
            tf.summary.image(name = name+'Output/Output_'+str(i), data=G_img[i:i+1,:,:,:], step=step)
            tf.summary.image(name = name+'Blur/Blur_'+str(i), data=sensor_img[i:i+1,:,:,:], step=step)
        
        # Metrics
        tf.summary.scalar(name = name+'metrics/G_PSNR', data = G_metrics['PSNR'], step=step)
        tf.summary.scalar(name = name+'metrics/G_SSIM', data = G_metrics['SSIM'], step=step)
        

        # Content losses
        tf.summary.scalar(name = name+'loss/G_Content_loss', data = G_Content_loss_val, step=step)
        tf.summary.scalar(name = name+'loss/G_Norm_loss'   , data = G_loss_components['Norm'], step=step)
        tf.summary.scalar(name = name+'loss/G_P_loss'      , data = G_loss_components['P'], step=step)
        tf.summary.scalar(name = name+'loss/G_Spatial_loss', data = G_loss_components['Spatial'], step=step)


def train_step(mode, BL_img, GT_img, G, G_optimizer, vgg_model, params, args):
    with tf.GradientTape() as G_tape:
        
        sensor_img = optics.sensor_noise(BL_img, params)
        
        _,G_img0 = params['deconv_fn'](sensor_img, G, training= True)

        # Losses
        gt_img0 = tf.image.resize_with_crop_or_pad(GT_img, params['out_width'], params['out_width'])

        tile_times = tf.constant([1,1,1,3],tf.int32)
        G_img = tf.tile(G_img0,tile_times)
        gt_img = tf.tile(gt_img0,tile_times)

        
        G_loss_val, G_loss_components, G_metrics = G_loss(G_img, gt_img, vgg_model, args)  

    # Apply gradients
        
    if mode == 'G':
        G_vars = G.trainable_variables
        G_gradients = G_tape.gradient(G_loss_val, G_vars)
        G_optimizer.apply_gradients(zip(G_gradients, G_vars))
    else:
        assert False, "Non-existant training mode"    

## Training loop
def train(args):
    ## Metasurface
    params = initialize_params(args)

    params['deconv_fn'] = conv.deconvolution_tf(params, args)
 
    ## Network architectures
    G = select_G(params, args)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(args.G_lr, 80000, end_learning_rate=1e-10, power=1.0)
    G_optimizer = tf.keras.optimizers.Adam(args.G_lr, beta_1=args.G_beta1)

    ## Construct vgg for perceptual loss
    if not args.P_loss_weight == 0:
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg_layers = [vgg.get_layer(name).output for name in args.vgg_layers.split(',')]
        vgg_model = tf.keras.Model(inputs=vgg.input, outputs=vgg_layers)
        vgg_model.trainable = False
    else:
        vgg_model = None

    ## Saving the model 
    checkpoint = tf.train.Checkpoint(G=G)

    max_to_keep = args.max_to_keep
    if args.max_to_keep == 0:
        max_to_keep = None
    manager = tf.train.CheckpointManager(checkpoint, directory=args.save_dir, max_to_keep=max_to_keep)
    ## Loading pre-trained model if exists
    if not args.ckpt_dir == None:
        status = checkpoint.restore(tf.train.latest_checkpoint(args.ckpt_dir, latest_filename=None))
        status.expect_partial() # Silence warnings
        #status.assert_existing_objects_matched() # Only partial load for networks (we don't load the optimizers)
        #status.assert_consumed()

    ## Create summary writer for TensorBoard
    summary_writer = tf.summary.create_file_writer(args.save_dir)
    ## Dataset
    train_ds = iter(train_dataset_sim(params['load_width'], args))
    test_ds  = list(test_dataset_sim(params['load_width'], args).take(1))

    ## Do training
    for step in range(args.steps):
        start = time.time()
        if step % args.save_freq == 0:
            print('Saving', flush=True)
            manager.save()
            
        for _ in range(args.G_iters):
            print('training_Net', flush=True)
            img = next(train_ds)
            BL_img = img[:,:,:,0:1]
            GT_img = img[:,:,:,2:]

            train_step('G', BL_img, GT_img, G, G_optimizer, vgg_model, params, args)

            if step % args.log_freq == 0:
                print('Logging', flush=True)
                log('train_',BL_img, GT_img,  G, vgg_model, summary_writer,step, params, args)
                img = test_ds[0]
                BL_img = img[:,:,:,0:1]
                GT_img = img[:,:,:,2:]
                log('test_',BL_img, GT_img,   G, vgg_model, summary_writer,step, params, args) 
         
        print("Step time: {}\n".format(time.time() - start), flush=True)
        

## Entry point
def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_flag
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
       try:
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu,True)
       except RuntimeError as e:
         print(e)
    train(args)

if __name__ == '__main__':
    main()
