import vgg16
import tensorflow as tf
import numpy as np
import argparse
from pprint import pprint
import helper


# check for VGG16 model
vgg16.maybe_download()


def mean_square_error(tensor_a, tensor_b):
    return tf.reduce_mean(tf.square(tensor_a - tensor_b))


def get_gram_matrix(tensor):
    shape = tensor.get_shape()
    depth = shape[3]
    matrix = tf.reshape(tensor, [-1, depth])
    gram_matrix = tf.matmul(tf.transpose(matrix), matrix)
    return gram_matrix


def content_loss(session, model, content_img, layer_ids):
    feed_dict = model.create_feed_dict(content_img)
    layers = model.get_layer_tensors(layer_ids)
    layer_output = session.run(layers, feed_dict=feed_dict)

    with model.graph.as_default():
        loss_list = []

        for output, layer in zip(layer_output, layers):
            loss = mean_square_error(layer, output)
            loss_list.append(loss)

        total_loss = tf.reduce_mean(loss_list)

    return total_loss


def style_loss(session, model, style_img, layer_ids):
    feed_dict = model.create_feed_dict(style_img)
    layers = model.get_layer_tensors(layer_ids)

    with model.graph.as_default():
        loss_list = []
        gram_matrix_values = [get_gram_matrix(layer) for layer in layers]
        layers_output = session.run(gram_matrix_values, feed_dict=feed_dict)

        for output, layer in zip(layers_output, layers):
            loss = mean_square_error(layer, output)
            loss_list.append(loss)

        total_loss = tf.reduce_mean(loss_list)

    return total_loss


def denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:, 1:, :, :] - model.input[:, :-1, :, :])) + \
           tf.reduce_sum(tf.abs(model.input[:, :, 1:, :] - model.input[:, :, :-1, :]))

    return loss


def neural_style_transfer(content_img, style_img, content_layer_ids, style_layer_ids, content_weight=1.5,
                          style_weight=10.0, denoise_weight=0.3, iters=120, step_size=10.0):
    model = vgg16.VGG16()
    session = tf.InteractiveSession(graph=model.graph)

    print("Content layers:")
    print(model.get_layer_names(content_layer_ids))
    print()

    print("Style layers:")
    print(model.get_layer_names(style_layer_ids))
    print()

    c_loss = content_loss(session, model, content_img, content_layer_ids)
    s_loss = style_loss(session, model, style_img, style_layer_ids)
    d_loss = denoise_loss(model)

    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    session.run([adj_content.initializer, adj_style.initializer, adj_denoise.initializer])

    update_adj_content = adj_content.assign(1.0 / (c_loss + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (s_loss + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (d_loss + 1e-10))

    loss_combined = content_weight * adj_content * c_loss + \
                    style_weight * adj_style * s_loss + \
                    denoise_weight * adj_denoise * d_loss

    gradient = tf.gradients(loss_combined, model.input)

    run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

    output_image = np.random.rand(*content_img.shape) + 128

    for i in range(iters):
        feed_dict = model.create_feed_dict(image=output_image)

        grad, adj_content_val, adj_style_val, adj_denoise_val \
            = session.run(run_list, feed_dict=feed_dict)

        grad = np.squeeze(grad)

        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        output_image -= grad * step_size_scaled

        output_image = np.clip(output_image, 0.0, 255.0)

        print(". ", end="")

        if (i % 10 == 0) or (i == iters - 1):
            print()
            print("Iteration:", i)

            msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

    session.close()

    # Return the mixed-image.
    return output_image


parser = argparse.ArgumentParser(description='A neural algorithm of artistic style')
parser.add_argument('--layer_ids', help='Get the ids for the layers of VGG16 model',
                    action='store_true')
parser.add_argument('--ci', help='Path for the content image.')
parser.add_argument('--si', help='Path for the style image.')
args = parser.parse_args()


if __name__ == 'main':
    if args.layer_ids:
        model = vgg16.VGG16()
        pprint(model.get_layer_names(list(range(13))))
    else:
        content_image = helper.load_image(args.ci)
        style_image = helper.load_image(args.si)
        content_layer_id = 4
        style_layer_id = list(range(13))
        content_weight = 1.5
        style_weight = 10.0
        denoise_weight = 0.3
        iters = 120
        step_size = 10.0
        output_image = neural_style_transfer(content_img=content_image, style_img=style_image, content_layer_ids=content_layer_id,
                              style_layer_ids=style_layer_id, content_weight=content_weight, style_weight=style_weight,
                              denoise_weight=denoise_weight, iters=iters, step_size=step_size)

        helper.save_image(output_image, 'output.png')