import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as kp_image
import PIL.Image

# Helper to load and preprocess image
def load_and_process_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = vgg19.preprocess_input(img)
    return tf.expand_dims(img, axis=0)

# Helper to deprocess image (convert back to displayable format)
def deprocess_img(processed_img):
    x = processed_img.copy()
    x = x.reshape((224, 224, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Content and style layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

def get_model():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    model = Model(inputs=vgg.input, outputs=outputs)
    return model

# Gram matrix (for style)
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

# Compute style and content features
def get_feature_representations(model, content_path, style_path):
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    style_outputs = model(style_image)
    content_outputs = model(content_image)

    style_features = [gram_matrix(style_layer) for style_layer in style_outputs[:len(style_layers)]]
    content_features = content_outputs[len(style_layers):]

    return style_features, content_features

# Compute loss
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)

    style_output_features = model_outputs[:len(style_layers)]
    content_output_features = model_outputs[len(style_layers):]

    style_score = tf.add_n([tf.reduce_mean((gram_matrix(style_output) - gram_style)**2)
                            for style_output, gram_style in zip(style_output_features, gram_style_features)])
    content_score = tf.add_n([tf.reduce_mean((content_output - content_target)**2)
                              for content_output, content_target in zip(content_output_features, content_features)])

    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score
    return loss

# Apply style transfer
@tf.function()
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss
    return tape.gradient(total_loss, cfg['init_image']), total_loss

def run_style_transfer(content_path, style_path, epochs=250, style_weight=1e-2, content_weight=1e4):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    style_features, content_features = get_feature_representations(model, content_path, style_path)
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    opt = tf.optimizers.Adam(learning_rate=5)

    cfg = {
        'model': model,
        'loss_weights': (style_weight, content_weight),
        'init_image': init_image,
        'gram_style_features': style_features,
        'content_features': content_features
    }

    best_loss = float('inf')
    best_img = None

    for i in range(epochs):
        grads, loss = compute_grads(cfg)
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, -128.0, 128.0)
        init_image.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = init_image.numpy()

        if i % 50 == 0:
            print(f"Iteration {i}: loss={loss:.2f}")

    return deprocess_img(best_img)

# Paths to content and style images
content_path = 'path_to_your_content_image.jpg'
style_path = 'path_to_your_style_image.jpg'

stylized_image = run_style_transfer(content_path, style_path)
plt.imshow(stylized_image)
plt.title("Stylized Image")
plt.axis('off')
plt.show()
