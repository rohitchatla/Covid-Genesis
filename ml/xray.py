# to ignore the warnings 
import warnings
warnings.filterwarnings("ignore")


# use this link to get the data from drop box

# !wget https://www.dropbox.com/s/e1r2laj50nh4tez/COVID-19_Radiography_Dataset.zip?dl=0



# add the path where you have saved the data set i.e the zip file

# !unzip "/content/COVID-19_Radiography_Dataset.zip?dl=0"

import cv2
import numpy as np
from keras.preprocessing import image


import pandas as pd
# import numpy as np
import os
import shutil
import glob
import matplotlib.pyplot as plt 

from keras.models import load_model

import cv2
import numpy as np
from keras.preprocessing import image


import tensorflow as tf

import matplotlib.cm as cm

from IPython.display import Image, display

import keras

from keras.applications.vgg16 import VGG16
from keras.layers import Flatten , Dense, Dropout , MaxPool2D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import base64


class Xray():
    def __init__(self):
      pass
    def routi(self):


      # Going Through Meta Data

      # covid_imgs = pd.read_excel("./COVID-19_Radiography_Dataset/COVID.metadata.xlsx",engine='openpyxl') #df1 = pd.read_excel(os.path.join(APP_PATH, "Data", "aug_latest.xlsm"), engine='openpyxl',)

      # covid_imgs.head(2)

      # opacity_images = pd.read_excel("./COVID-19_Radiography_Dataset/Lung_Opacity.metadata.xlsx", engine='openpyxl')
      # opacity_images.head(2)

      # normal_images = pd.read_excel("./COVID-19_Radiography_Dataset/Normal.metadata.xlsx", engine='openpyxl')
      # normal_images.head(2)

      # pneumonia_images = pd.read_excel("./COVID-19_Radiography_Dataset/Viral Pneumonia.metadata.xlsx", engine='openpyxl')
      # pneumonia_images.head(2)

      # # Working with images

      # ROOT_DIR = "./COVID-19_Radiography_Dataset/"#"/content/COVID-19_Radiography_Dataset/"
      # imgs = ['COVID','Lung_Opacity','Normal','Viral Pneumonia']

      # NEW_DIR = "./all_images/"

      # # Copy all my images to a new folder i.e all_images

      # if not os.path.exists(NEW_DIR):
      #   os.mkdir(NEW_DIR)

      #   for i in imgs:
      #     org_dir = os.path.join(ROOT_DIR, i+"/")
          
      #     for imgfile in glob.iglob(os.path.join(org_dir, "*.png")):
      #       shutil.copy(imgfile, NEW_DIR)
            
      # else:
      #   print("Already Exist")

      # counter = {'COVID':0,'Lung_Opacity':0,'Normal':0,'Viral Pneumonia':0}

      # for image in imgs:
      #   for count in glob.iglob(NEW_DIR+image+"*"):
      #     counter[image] += 1

      # # the number of images i have in each class
      # print(counter)

      # #visualizing the number of images 

      # plt.figure(figsize=(10,5))
      # plt.bar(x = counter.keys(), height= counter.values())
      # plt.show()


      # if not os.path.exists(NEW_DIR+"train_test_split/"):

      #   os.makedirs(NEW_DIR+"train_test_split/")

      #   os.makedirs(NEW_DIR+"train_test_split/train/Normal")
      #   os.makedirs(NEW_DIR+"train_test_split/train/Covid")

      #   os.makedirs(NEW_DIR+"train_test_split/test/Normal")
      #   os.makedirs(NEW_DIR+"train_test_split/test/Covid")

      #   os.makedirs(NEW_DIR+"train_test_split/validation/Normal")
      #   os.makedirs(NEW_DIR+"train_test_split/validation/Covid")


      #   # Train Data
      #   for i in np.random.choice(replace= False , size= 3000 , a = glob.glob(NEW_DIR+imgs[0]+"*") ):
      #     shutil.copy(i , NEW_DIR+"train_test_split/train/Covid" )
      #     os.remove(i)

      #   for i in np.random.choice(replace= False , size= 3900 , a = glob.glob(NEW_DIR+imgs[2]+"*") ):
      #     shutil.copy(i , NEW_DIR+"train_test_split/train/Normal" )
      #     os.remove(i)

      #   for i in np.random.choice(replace= False , size= 900 , a = glob.glob(NEW_DIR+imgs[3]+"*") ):
      #     shutil.copy(i , NEW_DIR+"train_test_split/train/Covid" )
      #     os.remove(i)

      #   # Validation Data
      #   for i in np.random.choice(replace= False , size= 308 , a = glob.glob(NEW_DIR+imgs[0]+"*") ):
      #     shutil.copy(i , NEW_DIR+"train_test_split/validation/Covid" )
      #     os.remove(i)

      #   for i in np.random.choice(replace= False , size= 500 , a = glob.glob(NEW_DIR+imgs[2]+"*") ):
      #     shutil.copy(i , NEW_DIR+"train_test_split/validation/Normal" )
      #     os.remove(i)

      #   for i in np.random.choice(replace= False , size= 200 , a = glob.glob(NEW_DIR+imgs[3]+"*") ):
      #     shutil.copy(i , NEW_DIR+"train_test_split/validation/Covid" )
      #     os.remove(i)

      #   # Test Data
      #   for i in np.random.choice(replace= False , size= 300 , a = glob.glob(NEW_DIR+imgs[0]+"*") ):
      #     shutil.copy(i , NEW_DIR+"train_test_split/test/Covid" )
      #     os.remove(i)

      #   for i in np.random.choice(replace= False , size= 300 , a = glob.glob(NEW_DIR+imgs[2]+"*") ):
      #     shutil.copy(i , NEW_DIR+"train_test_split/test/Normal" )
      #     os.remove(i)

      #   for i in np.random.choice(replace= False , size= 200 , a = glob.glob(NEW_DIR+imgs[3]+"*") ):
      #     shutil.copy(i , NEW_DIR+"train_test_split/test/Covid" )
      #     os.remove(i)


      # train_path  = "/content/all_images/train_test_split/train"
      # valid_path  = "/content/all_images/train_test_split/validation"
      # test_path   = "/content/all_images/train_test_split/test"
      train_path  = "./all_images/train_test_split/train"
      valid_path  = "./all_images/train_test_split/validation"
      test_path   = "./all_images/train_test_split/test"


      from keras.preprocessing.image import ImageDataGenerator
      from keras.applications.resnet50 import preprocess_input, ResNet50
      #from keras.applications import vgg16
      from keras.models import Model
      from keras.layers import Dense, MaxPool2D, Conv2D
      import keras

      train_data_gen = ImageDataGenerator(preprocessing_function= preprocess_input, #vgg16.preprocess_input
                                          zoom_range= 0.2, 
                                          horizontal_flip= True, 
                                          shear_range= 0.2,
                                          #, rescale= 1./255
                                          )

      train = train_data_gen.flow_from_directory(directory= train_path, 
                                                target_size=(224,224))

      validation_data_gen = ImageDataGenerator(preprocessing_function= preprocess_input  )##vgg16.preprocess_input, rescale= 1./255

      valid = validation_data_gen.flow_from_directory(directory= valid_path, 
                                                      target_size=(224,224))

      test_data_gen = ImageDataGenerator(preprocessing_function= preprocess_input )##vgg16.preprocess_input, rescale= 1./255

      test = train_data_gen.flow_from_directory(directory= test_path , 
                                                target_size=(224,224), 
                                                shuffle= False)

      # Covid +ve X-Ray is represented by 0 and Normal is represented by 1

      class_type = {0:'Covid',  1 : 'Normal'}

      # to visualize the images in the traing data denerator 

      t_img , label = train.next()

      # function when called will prot the images 

      def plotImages(img_arr, label):
        """
        input  :- images array 
        output :- plots the images 
        """

        for im, l in zip(img_arr,label) :
          plt.figure(figsize= (5,5))
          plt.imshow(im, cmap = 'gray')
          plt.title(im.shape)
          plt.axis = False
          plt.show()

      # function call to plot the images 

      # plotImages(t_img, label)

      ##we will be using our model Resnet 50

      # The mode that we are using here is ResNet50

      from keras.applications.resnet50 import ResNet50
      from keras.layers import Flatten , Dense, Dropout , MaxPool2D

      from keras.applications.vgg16 import VGG16
      from keras.layers import Flatten , Dense, Dropout , MaxPool2D

      #res = ResNet50( input_shape=(224,224,3), include_top= False) # include_top will consider the new weights
      vgg = VGG16( input_shape=(224,224,3), include_top= False) # include_top will consider the new weights

      # for layer in res.layers:           # Dont Train the parameters again 
      #   layer.trainable = False
      for layer in vgg.layers:           # Dont Train the parameters again 
        layer.trainable = False

      # x = Flatten()(res.output)
      # x = Dense(units=2 , activation='sigmoid', name = 'predictions' )(x)



      # creating our model.
      #model = Model(res.input, x)
      #model = keras.models.load_model('./initial_model.h5')

      # '''
      # Saving a Keras model:


      # model = ...  # Get model (Sequential, Functional Model, or Model subclass)
      # model.save('path/to/location')
      # Loading the model back:

      # x = Flatten()(vgg.output)
      # x = Dense(units=2 , activation='sigmoid', name = 'predictions' )(x)

      # model = Model(vgg.input, x)

      model = keras.models.load_model('./initial_model_vgg.h5')


      from tensorflow import keras
      # model = keras.models.load_model('path/to/location')
      # '''


      #model.summary()

      #model.compile( optimizer= 'adam' , loss = 'categorical_crossentropy', metrics=['accuracy'])

      #model.save("initial_model_vgg.h5")


      # implementing early stopping and model check point 

      from keras.callbacks import EarlyStopping
      from keras.callbacks import ModelCheckpoint

      #early_stop, checkpoint
      es = EarlyStopping(monitor= "val_accuracy" , min_delta= 0.01, patience= 3, verbose=1)
      mc = ModelCheckpoint(filepath="bestmodel_vgg.h5", monitor="val_accuracy", verbose=1, save_best_only= True)

      #hist = model.fit_generator(train, steps_per_epoch= 10, epochs= 30, validation_data= valid , validation_steps= 16, callbacks=[es,mc])
      # hist.save("hist_model.h5")
      # filename='log.csv'
      # history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
      # np.save('history1.npy',hist.history) 
      # hist=np.load('history1.npy',allow_pickle='TRUE').item()#history1
      # hist=np.load('history1.npy',allow_pickle='TRUE')#history1

      '''
      # model_json = model.to_json()
      # with open("hist_model2.json", "w") as json_file:#model.json
      #     json_file.write(model_json)


      # load json and create model
      # json_file = open('model.json', 'r')
      # loaded_model_json = json_file.read()
      # json_file.close()
      # loaded_model = model_from_json(loaded_model_json)
      # # load weights into new model
      # loaded_model.load_weights("model_best_weights.h5")
      # print("Loaded model from disk")

      # # evaluate loaded model on test data
      # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
      # score = loaded_model.evaluate(X, Y, verbose=0)
      '''

      ## load only the best model 
      from keras.models import load_model
      model = load_model("bestmodel_vgg.h5")

      # h = hist.history
      # h.keys()

      # plt.plot(h['accuracy'])
      # plt.plot(h['val_accuracy'] , c = "red")
      # plt.title("acc vs v-acc")
      # plt.show()

      # checking out the accurscy of our model 

      acc = model.evaluate_generator(generator= test)[1] 

      print(f"The accuracy of your model is = {acc} %")

      from keras.preprocessing import image

      def get_img_array(img_path):
        """
        Input : Takes in image path as input 
        Output : Gives out Pre-Processed image
        """
        path = img_path
        img = image.load_img(path, target_size=(224,224,3))
        img = image.img_to_array(img)
        img = np.expand_dims(img , axis= 0 )
        
        return img
      

      def imageutil():
        im_bytes = base64.b64decode(b64)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)# im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        # img=np.array(im_bytes)
        img = cv2.resize(img, (224,224))
        img = img[...,::-1] # Added
        img = image.img_to_array(img)/255
        img = np.expand_dims(img , axis= 0 )
        
        return img

      # path for that new image. ( you can take it either from google or any other scource)

      # path = "/content/all_images/COVID-2228.png"       # you can add any image path
      # path = "./content/all_images/COVID-1905.png"       # you can add any image path (--> . before/content wrong won't work in collab, but works in local)
      path = "./all_images/COVID-1091.png"       # you can add any image path

      #predictions: path:- provide any image from google or provide image from all image folder
      img = get_img_array(path)
      #print(img.shape)

      res = class_type[np.argmax(model.predict(img))]
      #model.predict_classes(img)
      print(f"The given X-Ray image is of type = {res}")
      print()
      print(f"The chances of image being Covid is : {model.predict(img)[0][0]*100} percent")
      print()
      print(f"The chances of image being Normal is : {model.predict(img)[0][1]*100} percent")


      # to display the image  
      #plt.imshow(img[0]/255, cmap = "gray")
      plt.title("input image")
      plt.show()

      # Grad CAM Visualization


      import tensorflow as tf



      # this function is udes to generate the heat map of aan image
      def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
          # First, we create a model that maps the input image to the activations
          # of the last conv layer as well as the output predictions
          grad_model = tf.keras.models.Model(
              [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
          )

          # Then, we compute the gradient of the top predicted class for our input image
          # with respect to the activations of the last conv layer
          with tf.GradientTape() as tape:
              last_conv_layer_output, preds = grad_model(img_array)
              if pred_index is None:
                  pred_index = tf.argmax(preds[0])
              class_channel = preds[:, pred_index]

          # This is the gradient of the output neuron (top predicted or chosen)
          # with regard to the output feature map of the last conv layer
          grads = tape.gradient(class_channel, last_conv_layer_output)

          # This is a vector where each entry is the mean intensity of the gradient
          # over a specific feature map channel
          pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

          # We multiply each channel in the feature map array
          # by "how important this channel is" with regard to the top predicted class
          # then sum all the channels to obtain the heatmap class activation
          last_conv_layer_output = last_conv_layer_output[0]
          heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
          heatmap = tf.squeeze(heatmap)

          # For visualization purpose, we will also normalize the heatmap between 0 & 1
          heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
          return heatmap.numpy()

      # Now we will mask the heat map on the image



      import matplotlib.cm as cm

      from IPython.display import Image, display

      def save_and_display_gradcam(img_path , heatmap, cam_path="cam.jpg", alpha=0.4):
          """
          img input shoud not be expanded 
          """

          # Load the original image
          img = keras.preprocessing.image.load_img(img_path)
          img = keras.preprocessing.image.img_to_array(img)

          
          # Rescale heatmap to a range 0-255
          heatmap = np.uint8(255 * heatmap)

          # Use jet colormap to colorize heatmap
          jet = cm.get_cmap("jet")

          # Use RGB values of the colormap
          jet_colors = jet(np.arange(256))[:, :3]
          jet_heatmap = jet_colors[heatmap]

          # Create an image with RGB colorized heatmap
          jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
          jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
          jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

          # Superimpose the heatmap on original image
          superimposed_img = jet_heatmap * alpha + img
          superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

          # Save the superimposed image
          superimposed_img.save(cam_path)

          # Display Grad CAM
          # display(Image(cam_path))

      # function that is used to predict the image type and the ares that are affected by covid


      def image_prediction_and_visualization(path,last_conv_layer_name = "block5_conv3", model = model):
        #last_conv_layer_name = "conv5_block3_3_conv"-->resnet
        #last_conv_layer_name = "block5_conv3"-->vgg
        """
        input:  is the image path, name of last convolution layer , model name
        output : returs the predictions and the area that is effected
        """
        
        img_array = get_img_array(path)

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        plt.title("the heat map of the image is ")
        #plt.imshow(heatmap)
        plt.show()
        print()
        img = get_img_array(path)

        res = class_type[np.argmax(model.predict(img))]
        print(f"The given X-Ray image is of type = {res}")
        print()
        print(f"The chances of image being Covid is : {model.predict(img)[0][0]*100} %")
        print(f"The chances of image being Normal is : {model.predict(img)[0][1]*100} %")

        print()
        print("image with heatmap representing region on interest")

        # function call
        save_and_display_gradcam(path, heatmap)

        print()
        print("the original input image")
        print()

        a = plt.imread(path)
        plt.imshow(a, cmap = "gray")
        plt.title("Original image")
        plt.show()

    

    def get_img_array(self, img_path):
      """
      Input : Takes in image path as input 
      Output : Gives out Pre-Processed image
      """
      path = img_path
      img = image.load_img(path, target_size=(224,224,3))
      img = image.img_to_array(img)/255
      img = np.expand_dims(img , axis= 0 )
      
      return img

    def imageutil(self, b64):
      im_bytes = base64.b64decode(b64)
      im_arr = np.frombuffer(im_bytes, dtype=np.uint8)# im_arr is one-dim Numpy array
      img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
      # img=np.array(im_bytes)
      img = cv2.resize(img, (224,224))
      img = img[...,::-1] # Added
      # img = image.img_to_array(img)/255
      img = np.expand_dims(img , axis= 0 )
      
      return img


    def predict(self):

      
      model=load_model("bestmodel_vgg_working_collab.h5")
      # self.model=model
      # model.summary()
      class_type = {0:'Covid',  1 : 'Normal'}
      # path for that new image. ( you can take it either from google or any other scource)

      # path = "/content/all_images/COVID-1905.png"       # you can add any image path
      # path = "./content/all_images/COVID-1905.png"       # you can add any image path (--> . before/content wrong won't work in collab, but works in local)
      # path = "/content/drive/MyDrive/all_images/COVID-1334.png"    
      path = "./all_images/COVID-1334.png"    
      #predictions: path:- provide any image from google or provide image from all image folder
      img = self.get_img_array(path)

      res = class_type[np.argmax(model.predict(img))]
      print(f"The given X-Ray image is of type = {res}")
      print()
      print(f"The chances of image being Covid is : {model.predict(img)[0][0]*100} percent")
      print()
      print(f"The chances of image being Normal is : {model.predict(img)[0][1]*100} percent")

      # to display the image  
      plt.imshow(img[0], cmap = "gray")
      plt.title("input image")
      plt.show()
      return res

    # this function is udes to generate the heat map of aan image


    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()


    # put the heatmap to our image to understand the area of interest

    def save_and_display_gradcam(self, img_path , heatmap, cam_path="cam.jpg", alpha=0.4):
        """
        img input shoud not be expanded 
        """

        # Load the original image
        img = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img)

        #print(img)
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)

        # Display Grad CAM
        # display(Image(cam_path))
        a = plt.imread(cam_path)
        plt.imshow(a, cmap = "gray")
        plt.title("Original image")
        plt.show()



    # function that is used to predict the image type and the ares that are affected by covid


    def image_prediction_and_visualization(self, path, b64url, last_conv_layer_name = "block5_conv3"):#, model = model # model=self.model
      """
      input:  is the image path, name of last convolution layer , model name
      output : returs the predictions and the area that is effected
      """
      
      model=load_model("bestmodel_vgg_working_collab.h5")
      # self.model=model
      # model.summary()
      class_type = {0:'Covid',  1 : 'Normal'}


      
      # img_array = self.get_img_array(path)
      img_array=self.imageutil(b64url)

      heatmap = self.make_gradcam_heatmap(img_array, model, last_conv_layer_name)

      # img = self.get_img_array(path)
      img=self.imageutil(b64url)

      res = class_type[np.argmax(model.predict(img))]
      print(f"The given X-Ray image is of type = {res}")
      print()
      print(f"The chances of image being Covid is : {model.predict(img)[0][0]*100} %")
      print(f"The chances of image being Normal is : {model.predict(img)[0][1]*100} %")

      print()
      print("image with heatmap representing the covid spot")

      # function call
      # self.save_and_display_gradcam(path, heatmap)

      print()
      print("the original input image")
      print()

      a = plt.imread(path)
      plt.imshow(a, cmap = "gray")
      plt.title("Original image")
      plt.show()
      return res

  





def check(b64url): # main()
  ## predictions
  # provide the path of any image from google or any other scource 
  # the path is already defigned above , but you can also provide the path here to avoid scrolling up 

  # for covid image :  path:- provide any image from google or provide image from all image folder
  # path = "/content/all_images/COVID-2228.png"
  path = "./all_images/COVID-1091.png"

  xy=Xray()
  # pred_val="ss"
  #xy.routi()
  #xy.routi().image_prediction_and_visualization(path)
  #pred_val=xy.predict()

  pred_val=xy.image_prediction_and_visualization(path, b64url)


  # for normal image :  path:- provide any image from google or provide image from all image folder
  # path = "/content/all_images/train_test_split/validation/Normal/Normal-10022.png"
  # path = "./all_images/train_test_split/validation/Normal/Normal-10191.png"
  # path = "./all_images/train_test_split/validation/Normal/Normal-341.png"
  
  #path = "./all_images/train_test_split/test/Normal/Normal-10186.png"
  #xy.image_prediction_and_visualization(path)

  # for a healthey chest x-Ray heap map will be white thus the x-ray will look blue



  # xy=Xray()

  # pred_val=xy.predict()#parameter/args(preferably):: path
  return pred_val
  
  
  ##