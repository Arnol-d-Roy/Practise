import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.compiler.tensorrt import trt_convert as trt


mobilenet_v2 = tf.keras.applications.MobileNetV2(weights='imagenet')
mobilenet_v2.save('mobilenet_v2')


img = tf.keras.preprocessing.image.load_img('elephant.jpg', target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

sample_image = plt.imread('elephant.jpg')
plt.imshow(sample_image)
plt.show()

preds = mobilenet_v2.predict(x)
print('Predicted:', tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0])


def time_my_model(model, data):
    times = []
    for i in range(20):
        start_time = time.time()
        one_prediction = model.predict(data)
        delta = (time.time() - start_time)
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))

time_my_model(mobilenet_v2, x)
time_my_model(mobilenet_v2, x)
time_my_model(mobilenet_v2, x)
time_my_model(mobilenet_v2, x)
time_my_model(mobilenet_v2, x)

# Convert the SavedModel using TF-TRT

# Prepare the configuration params for the conversion
params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode='FP16',
    is_dynamic_op=True)

# Perform conversion
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='mobilenet_v2',
    conversion_params=params)
converter.convert()


saved_model_dir_trt = 'mobilenet_v2.trt'
converter.save(saved_model_dir_trt)

root = tf.saved_model.load(saved_model_dir_trt)
concrete_func = root.signatures['serving_default']

concrete_func.structured_outputs

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

labeling = concrete_func(tf.constant(x.astype('float32')))
activations = tf.nn.softmax(labeling['predictions'])
imagenet_labels[np.argsort(activations)[0,::-1][:5]+1]

def time_trt_model():
    image_input = tf.constant(x.astype('float32'))
    times = []
    for i in range(20):
        start_time = time.time()
        one_prediction = concrete_func(input_1=image_input)
        delta = (time.time() - start_time)
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))

time_trt_model()
time_trt_model()
time_trt_model()
time_trt_model()
time_trt_model()
