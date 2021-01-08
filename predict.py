#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
import tensorflow as tf
from PIL import Image

from frcnn import FRCNN

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

frcnn = FRCNN()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
        #-------------------------------------#
        #   转换成RGB图片，可以用于灰度图预测。
        #-------------------------------------#
        image = image.convert("RGB")
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()
