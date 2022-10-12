import streamlit as st 
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import from_pretrained_keras

st.header("Segmentación de dientes con rayos X")

st.markdown('''

Hola estudiantes de Platzi 🚀. Este modelo usan UNet para segmentar imágenes
de dientos en rayos X. Se utila un modelo de Keras importado con la función
`huggingface_hub.from_pretrained_keras`. Recuerda que el Hub de Hugging Face está integrado
con muchas librerías como Keras, scikit-learn, fastai y otras.

El modelo fue creado por [SerdarHelli](https://huggingface.co/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net).

''')

model_id = "SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net"
model=from_pretrained_keras(model_id)

## Si una imagen tiene más de un canal entonces se convierte a escala de grises (1 canal) 
def convertir_one_channel(img):
    if len(img.shape)>2:
        img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    else:
        return img
    
def convertir_rgb(img):
    if len(img.shape)==2:
        img= cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)  
        return img
    else:
        return img

    
image_file = st.file_uploader("Sube aquí tu imagen.", type=["png","jpg","jpeg"])

    
if image_file is not None:

      img= Image.open(image_file)
      
      st.image(img,width=850)
      
      img=np.asarray(img)
  
      img_cv=convertir_one_channel(img)
      img_cv=cv2.resize(img_cv,(512,512), interpolation=cv2.INTER_LANCZOS4)
      img_cv=np.float32(img_cv/255)
      
      img_cv=np.reshape(img_cv,(1,512,512,1))
      prediction=model.predict(img_cv)
      predicted=prediction[0]
      predicted = cv2.resize(predicted, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
      mask=np.uint8(predicted*255)# 
      _, mask = cv2.threshold(mask, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      kernel =( np.ones((5,5), dtype=np.float32))
      mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=1 )  
      mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=1 )
      cnts,hieararch=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      output = cv2.drawContours(convertir_one_channel(img), cnts, -1, (255, 0, 0) , 3)


      if output is not None :      
          st.subheader("Segmentación:")  
          st.write(output.shape)
          st.image(output,width=850)