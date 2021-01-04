# coding=utf-8
#from __future__ import absolute_import, division, print_function, unicode_literals
import os
import glob
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow.compat.v1 as tf
import sys

tf.disable_v2_behavior()

root_dir = 'colab_data/'
img_width = 100
img_height = 397
test_width = 100
test_height = 397
channels = 1
num_way = 2 # number of classes
num_shot = 20#20 # number of examples per class for support set
num_query = 20 #20
num_examples = 10737 #per class
num_epochs =  15 #
num_episodes = 300

n_classes = 2

n_test_episodes = 1000#1000
num_test_empathy0 = 124
num_test_empathy1 = 595
n_test_way = 2
n_test_shot = 30
n_test_query = 30

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def load_dataset(split_file='train.txt'):

    ### Load Dataset
    dataset = np.zeros([n_classes, num_examples, img_height, img_width],
        dtype=np.float32)

    dataset[0] = np.load("../processed_final_texts/imdb_neg_lem.npy")[:10737]
    dataset[1] = np.load("../processed_final_texts/imdb_pos_lem.npy")


    print(dataset.shape)
    return dataset, n_classes

def load_test(split_file):

    ### Initialize dataset with a shape as number of classes, examples, height, and width
    dataset0 = np.zeros([1, num_test_empathy0, test_height, test_width], dtype=np.float32)
    dataset1 = np.zeros([1, num_test_empathy1, test_height, test_width], dtype=np.float32)
    print(dataset0.shape,dataset1.shape)

    dataset0[0] = np.load("../processed_final_texts/neg_lemma.npy")
    dataset1[0] = np.load("../processed_final_texts/pos_lemma.npy")

    return dataset0, dataset1, n_classes

def convolution_block1(inputs,batch_size,name='conv'):
    with tf.variable_scope(name):

        conv1 = tf.layers.conv2d(inputs,filters = 1024, kernel_size=(3,100), strides = 1)
        conv2 = tf.layers.conv2d(inputs,filters = 1024, kernel_size=(4,100), strides = 1)
        conv3 = tf.layers.conv2d(inputs, filters = 1024, kernel_size=(5,100), strides = 1)

        conv1 = tf.layers.max_pooling2d(conv1, (395,1), strides = 1)
        conv2 = tf.layers.max_pooling2d(conv2, (394,1), strides = 1)
        conv3 = tf.layers.max_pooling2d(conv3, (393,1), strides = 1)
  
        conv = tf.keras.layers.Concatenate(axis = 1)([conv1,conv2,conv3])

        conv = tf.nn.relu(conv)
        return conv

#encoder is different from protoTensor_original_implementation
def encoder(support_set, n_shot, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('encoder', reuse = reuse): #possible reuse = Truetf.AUTO_REUSE
        net = convolution_block1(support_set,batch_size = n_shot ,name='conv_1')
        net = tf.layers.flatten(net)
        return net


def euclidean_distance(a, b):
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)

x_train, x_classes = load_dataset()
print("nacital som train")
x_test0, x_test1, x_test_classes = load_test(split_file='test.txt')
print("nacital som test")

support_set = tf.placeholder(tf.float32, [num_way, num_shot, img_height,
    img_width, channels],name = "support_set")
query_set = tf.placeholder(tf.float32, [num_way,num_query, img_height,
    img_width, channels], name = "query_set")
test_support_set = tf.placeholder(tf.float32, [n_test_way, n_test_shot, test_height,
    test_width, channels], name = "test_support_set")
test_query_set = tf.placeholder(tf.float32, [n_test_way, n_test_query, test_height,
    test_width, channels], name = "test_query_set")
test_predicted = tf.placeholder(tf.float32, [n_test_way, n_test_query, test_height,
    test_width, channels], name = "test_predicted")
support_set_shape = tf.shape(support_set)
query_set_shape = tf.shape(query_set)
num_classes, num_support_points = support_set_shape[0],support_set_shape[1]
num_query_points = query_set_shape[1]
y = tf.placeholder(tf.int64, [None, None], name = "y")
y_one_hot = tf.one_hot(y, depth=num_classes)
test_y = tf.placeholder(tf.int64, [None, None], name = "test_y")
test_y_one_hot = tf.one_hot(test_y, depth=num_classes)
support_set_embeddings = encoder(tf.reshape(support_set,
#h_dim and z_dim is not at the end !!!
    [num_classes * num_support_points, img_height, img_width, channels]), n_shot = num_shot)
embedding_dimension = tf.shape(support_set_embeddings)[-1]
#embedding_dimension = tf.shape(support_set_embeddings)
class_prototype = tf.reduce_mean(tf.reshape(support_set_embeddings, [num_classes, num_support_points, embedding_dimension]), axis=1)
query_set_embeddings = encoder(tf.reshape(query_set, [num_classes * num_query_points, img_height, img_width, channels]),
 n_shot = num_shot, reuse=True)
distance = euclidean_distance(query_set_embeddings, class_prototype)
predicted_probability = tf.reshape(tf.nn.log_softmax(-distance), [num_classes, num_query_points, -1])
loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, predicted_probability), axis=-1), [-1]))
accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(predicted_probability, axis=-1), y)))

##### Getting predicted values for each iteration
predicted = tf.argmax(predicted_probability, axis= -1)


#### test phase placeholders
#num_support_points can stay as it is if support is equally large in train and test
test_support_set_embeddings = encoder(tf.reshape(test_support_set, [num_classes * n_test_shot, test_height, test_width, channels]),
    n_shot = n_test_shot)
test_embedding_dimension = tf.shape(test_support_set_embeddings)[-1]
test_class_prototype = tf.reduce_mean(tf.reshape(test_support_set_embeddings, [num_classes, n_test_shot, test_embedding_dimension]), axis=1)
#num_query_points can stay as it is if query size is the same in train and test
test_query_set_embeddings = encoder(tf.reshape(test_query_set, [num_classes * n_test_query, img_height, img_width, channels]),
n_shot = n_test_shot, reuse=True)
test_distance = euclidean_distance(test_query_set_embeddings, test_class_prototype)
test_predicted_probability = tf.reshape(tf.nn.log_softmax(-test_distance), [num_classes, n_test_query, -1])
test_predicted = tf.argmax(test_predicted_probability, axis= -1)
test_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(test_y_one_hot, test_predicted_probability), axis=-1), [-1]))
test_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(test_predicted_probability, axis=-1), test_y)))
ones= tf.ones([n_test_query, 1], dtype=tf.int64)
get_sums = tf.linalg.matmul(test_predicted,ones)


best_loss = 1000
best_accuracy = 0
train_op = tf.train.AdamOptimizer().minimize(loss)

best_saver = tf.train.Saver(max_to_keep=1)

sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)

#def model_summary():
#    model_vars = tf.trainable_variables()
#    .model_analyzer.analyze_vars(model_vars, print_info=True)
#model_summary()

for epoch in range(num_epochs):
    for episode in range(num_episodes):
        #episodic_classes = np.random.permutation(x_classes)[:num_way]
        episodic_classes = [0,1]
        #print(episodic_classes)
        support = np.zeros([num_way, num_shot, img_height, img_width], dtype=np.float32)
        query = np.zeros([num_way, num_query, img_height, img_width], dtype=np.float32)
        #print(episodic_classes)

        for index, class_ in enumerate(episodic_classes):
            #print(index, class_)
            selected = np.random.permutation(num_examples)[:num_shot + num_query]
            support[index] = x_train[class_, selected[:num_shot]]
            #print(x_train.shape)
            query[index] = x_train[class_, selected[num_shot:]]

        support = np.expand_dims(support, axis=-1)
        query = np.expand_dims(query, axis=-1)
        labels = np.tile(np.arange(num_way)[:, np.newaxis], (1, num_query)).astype(np.uint8)
        #print(labels)
        _, loss_, accuracy_, predicted_values = sess.run([train_op, loss, accuracy,predicted], feed_dict={support_set: support, query_set: query, y:labels})

        #print(predicted_values)
        if (episode+1) % 10 == 0:
            print('Epoch {} : Episode {} : Loss: {}, Accuracy: {}'.format(epoch+1, episode+1, loss_, accuracy_))

             #print(class_prototype_)
        if loss_ < best_loss:
            best_loss = loss_
            best_accuracy = accuracy_
                #best_class_prototype = class_prototype_
                #no_improvement = 0
                #experimental follows
                #best_save_path = os.path.join(,'checkpoint', 'best_weights', 'after-epoch')
            best_save_path = best_saver.save(sess, "best_weights.ckpt")
            print("Model saved in path: %s" % best_save_path)
            print('Episode: {} Best loss: {} Best Accuracy: {}'.format(episode + 1, best_loss, best_accuracy))
            #no_improvement += 1
            #if no_improvement == early_stopping:
            #    break
        #print(tf.trainable_variables())
        #results.append(test_probability_)


print('Testing...')

tf.reset_default_graph()
best_saver.restore(sess, "best_weights.ckpt")
results = pd.DataFrame(np.zeros([n_test_episodes,4]))
results.columns = ["TP","TN","FP","FN"]
avg_acc = 0.
for epi in range(n_test_episodes):

    #in case of imdb, it epi_classes = 2
    epi_classes = [0,1]
    test_support = np.zeros([n_test_way, n_test_shot, test_height, test_width], dtype=np.float32)
    test_query = np.zeros([n_test_way, n_test_query, test_height, test_width], dtype=np.float32)
    #support = np.zeros([n_test_way, n_test_shot, test_height, test_width], dtype=np.float32)
    #query = np.zeros([n_test_way, n_test_query, test_height, test_width], dtype=np.float32)

    selected0 = np.random.permutation(num_test_empathy0 )[:n_test_shot + n_test_query]
    #print(selected0)
    selected1 = np.random.permutation(num_test_empathy1 )[:n_test_shot + n_test_query]
    #print(selected1)
    test_support[0] = x_test0[0, selected0[:n_test_shot]]
    test_query[0] = x_test0[0, selected0[n_test_shot:]]
    test_support[1] = x_test1[0, selected1[:n_test_shot]]
    test_query[1] = x_test1[0, selected1[n_test_shot:]]
    test_support = np.expand_dims(test_support, axis=-1)
    test_query = np.expand_dims(test_query, axis=-1)
    test_labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)

    #support[0] = x_test0[0, selected0[:n_test_shot]]
    #query[0] = x_test0[0, selected0[n_test_shot:]]

    #support[1] = x_test1[0, selected1[:n_test_shot]]
    #query[1] = x_test1[0, selected1[n_test_shot:]]
    #support = np.expand_dims(support, axis=-1)
    #query = np.expand_dims(query, axis=-1)
    #labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)



    ls, ac, predicted_values, y_values = sess.run([test_loss, test_accuracy, test_predicted, test_y], feed_dict={test_support_set: test_support, test_query_set: test_query, test_y:test_labels})
    sums = sess.run([get_sums], feed_dict = {test_predicted : predicted_values})
    #print(predicted_values)

    #sum of 1 and 2. row
    #ones= tf.ones([n_test_query, 1], dtype=tf.int64)
    #with tf.Session() as sess:
    #    sums = tf.linalg.matmul(predicted_values,ones).eval()

    print(ac)
    avg_acc += ac
    results.iloc[epi,0] = sums[0][1][0]
    results.iloc[epi,1] = n_test_query - sums[0][0][0]
    results.iloc[epi,2] = sums[0][0][0]
    results.iloc[epi,3] = n_test_query - sums[0][1][0]
    if (epi+1) % 50 == 0:
        print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_test_episodes, ls, ac))
avg_acc /= n_test_episodes
print('Average Test Accuracy: {:.5f}'.format(avg_acc))
print('Best training accuracy was',best_accuracy)
results.to_csv("../results/protonets/v4_4.csv",index = False)
