import tensorflow as tf
import numpy as np
import os

import pickle
import time

from sklearn.model_selection import train_test_split

from utils import save_param
from utils import embed_dim, combiner
from utils import uid_max, gender_max, age_max
from utils import movie_id_max, movie_categories_max, movie_title_max
from utils import window_sizes, filter_num
from utils import num_epochs, batch_size, dropout_keep, learning_rate
from utils import show_every_n_batches, save_dir

#get data
(title_count, title_set, genres2int, features,
targets_values, ratings, users, movies, data,
movies_orig, users_orig) = pickle.load(open('preprocess.p', mode='rb'))
sentences_size = title_count

#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]:i for i, val in enumerate(movies.values)}


def get_inputs():
    with tf.name_scope('input'):
        uid = tf.placeholder(tf.int32, [None, 1], name='uid')
        user_gender = tf.placeholder(tf.int32, [None, 1], name='user_gender')
        user_age = tf.placeholder(tf.int32, [None, 1], name='user_age')
        user_job = tf.placeholder(tf.int32, [None, 1], name='user_job')

        movie_id = tf.placeholder(tf.int32, [None, 1], name='movie_id')
        movie_categories = tf.placeholder(tf.int32, [None, 18], name='movie_categories')
        movie_title = tf.placeholder(tf.int32, [None, 15], name='movie_title')
        targets = tf.placeholder(tf.int32, [None, 1], name='targets')
        lr = tf.constant(learning_rate, name='learning_rate')
        dropout_keep_prob = tf.placeholder(tf.float32, name='droupout_keep_prob')

        return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_title, targets, lr, dropout_keep_prob



def get_user_embedding(uid, user_gender, user_age, user_job):
    with tf.variable_scope('user_embedding'):
        user_embed_matrix = tf.get_variable(
                    'user_embed_matrix',
                    initializer=tf.random_uniform_initializer(maxval=1, minval=-1),
                    shape=[uid_max, embed_dim])
        user_embedding =  tf.nn.embedding_lookup(
                    user_embed_matrix,
                    uid,
                    name='user_embedding'
        )
        gender_embed_matrix = tf.get_variable(
                    'gender_embed_matrix',
                    initializer=tf.random_uniform_initializer(maxval=1, minval=-1),
                    shape=[gender_max, embed_dim]
        )
        gender_embedding = tf.nn.embedding_lookup(
                    gender_embed_matrix,
                    user_gender,
                    name='gender_embedding'
        )
        age_embed_matrix = tf.get_variable(
                    'age_embed_matrix',
                    initializer=tf.random_uniform_initializer(maxval=1, minval=-1),
                    shape=[age_max, embed_dim]
        )
        age_embedding = tf.nn.embedding_lookup(
                    age_embed_matrix,
                    user_age,
                    name='age_embedding'
        )
        job_embed_matrix = tf.get_variable(
                    'job_embed_matrix',
                    initializer=tf.random_uniform_initializer(maxval=1, minval=-1),
                    shape=[age_max, embed_dim]
        )
        job_embedding = tf.nn.embedding_lookup(
                    job_embed_matrix,
                    user_age,
                    name='job_embedding'
        )
    return user_embedding, gender_embedding, age_embedding, job_embedding



def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    with tf.variable_scope("user_fc"):
        uid_fc_layer = tf.layers.dense(
                uid_embed_layer,
                embed_dim,
                name = "uid_fc_layer",
                activation=tf.nn.relu
        )
        gender_fc_layer = tf.layers.dense(
                gender_embed_layer,
                embed_dim,
                name = "gender_fc_layer",
                activation=tf.nn.relu
        )
        age_fc_layer = tf.layers.dense(
                age_embed_layer,
                embed_dim,
                name="age_fc_layer",
                activation=tf.nn.relu
        )
        job_fc_layer = tf.layers.dense(
                job_embed_layer,
                embed_dim,
                name="job_fc_layer",
                activation=tf.nn.relu
        )

        user_combine_layer = tf.concat(
                [uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer],
                2
        )  #(?, 1, 128)
        user_combine_layer = tf.contrib.layers.fully_connected(
                user_combine_layer,
                200,
                tf.tanh
        )  #(?, 1, 200)

        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
    return user_combine_layer, user_combine_layer_flat



def get_movie_id_embed_layer(movie_id):
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(
                tf.random_uniform([movie_id_max, embed_dim], -1, 1),
                name = "movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(
                movie_id_embed_matrix, movie_id,
                name = "movie_id_embed_layer")
    return movie_id_embed_layer



def get_movie_categories_layers(movie_categories):
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix = tf.get_variable(
                "movie_categories_embed_matrix",
                initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                shape=[movie_categories_max, embed_dim]
        )
        movie_categories_embed_layer = tf.nn.embedding_lookup(
                movie_categories_embed_matrix,
                movie_categories,
                name = "movie_categories_embed_layer"
        )
        if combiner == "sum":
            movie_categories_embed_layer = tf.reduce_sum(
                movie_categories_embed_layer,
                axis=1,
                keepdims=True
            )
    #     elif combiner == "mean":

    return movie_categories_embed_layer



def get_movie_cnn_layer(movie_titles, dropout_keep_prob):
    #从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix = tf.Variable(
            tf.random_uniform([movie_title_max, embed_dim], -1, 1),
            name = "movie_title_embed_matrix"
        )
        movie_title_embed_layer = tf.nn.embedding_lookup(
            movie_title_embed_matrix,
            movie_titles,
            name = "movie_title_embed_layer"
        )
        movie_title_embed_layer_expand = tf.expand_dims(
            movie_title_embed_layer,
            -1
        )

    #对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
            filter_weights = tf.Variable(
                tf.truncated_normal(
                    [window_size, embed_dim, 1, filter_num],
                    stddev=0.1),
                name = "filter_weights")
            filter_bias = tf.Variable(
                tf.constant(0.1, shape=[filter_num]),
                name="filter_bias")

            conv_layer = tf.nn.conv2d(
                movie_title_embed_layer_expand,
                filter=filter_weights,
                strides=[1,1,1,1],
                padding="VALID",
                name="conv_layer")
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer,filter_bias),
                                    name ="relu_layer")

            maxpool_layer = tf.nn.max_pool(
                relu_layer,
                ksize=[1,sentences_size - window_size + 1 ,1,1],
                strides=[1,1,1,1],
                padding="VALID",
                name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    #Dropout层
    with tf.name_scope("pool_dropout"):
        pool_layer = tf.concat(pool_layer_lst, 3, name ="pool_layer")
        max_num = len(window_sizes) * filter_num
        pool_layer_flat = tf.reshape(
            pool_layer ,
            [-1, 1, max_num],
            name = "pool_layer_flat")

        dropout_layer = tf.nn.dropout(
            pool_layer_flat, dropout_keep_prob, name = "dropout_layer")
    return pool_layer_flat, dropout_layer



def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    with tf.name_scope("movie_fc"):
        #第一层全连接
        movie_id_fc_layer = tf.layers.dense(
            movie_id_embed_layer,
            embed_dim,
            name = "movie_id_fc_layer",
            activation=tf.nn.relu
        )
        movie_categories_fc_layer = tf.layers.dense(
            movie_categories_embed_layer,
            embed_dim,
            name = "movie_categories_fc_layer",
            activation=tf.nn.relu
        )

        #第二层全连接
        movie_combine_layer = tf.concat(
            [movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  #(?, 1, 96)
        movie_combine_layer = tf.contrib.layers.fully_connected(
            movie_combine_layer, 200, tf.tanh)  #(?, 1, 200)

        movie_combine_layer_flat = tf.reshape(
            movie_combine_layer,
            [-1, 200]
        )
    return movie_combine_layer, movie_combine_layer_flat



def build():
    tf.reset_default_graph()
    train_graph = tf.Graph()
    with train_graph.as_default():
        #获取输入占位符
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob = get_inputs()
        inputs = [uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob]
        #获取User的4个嵌入向量
        uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender, user_age, user_job)
        #得到用户特征
        user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer)
        #获取电影ID的嵌入向量
        movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
        #获取电影类型的嵌入向量
        movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
        #获取电影名的特征向量
        pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles, dropout_keep_prob)
        #得到电影特征
        movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer,
                                                                                movie_categories_embed_layer,
                                                                                dropout_layer)
        #计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
        with tf.name_scope("inference"):
        #将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
#         inference_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)  #(?, 200)
#         inference = tf.layers.dense(inference_layer, 1,
#                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
#                                     kernel_regularizer=tf.nn.l2_loss, name="inference")
        #简单的将用户特征和电影特征做矩阵乘法得到一个预测评分
#        inference = tf.matmul(user_combine_layer_flat, tf.transpose(movie_combine_layer_flat))
            inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
            inference = tf.expand_dims(inference, axis=1)

        with tf.name_scope("loss"):
            # MSE损失，将计算值回归到评分
            cost = tf.losses.mean_squared_error(targets, inference )
            loss = tf.reduce_mean(cost)
        # 优化损失
#       train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(lr)
        gradients = optimizer.compute_gradients(loss)  #cost
        train_op = optimizer.apply_gradients(gradients, global_step=global_step)

    return train_graph, gradients, loss, train_op, global_step, inputs


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]



def train(train_graph, gradients, loss,train_op, global_step, inputs):
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob = inputs
    losses = {'train':[], 'test':[]}
    with tf.Session(graph=train_graph) as sess:
        #搜集数据给tensorBoard用
        # Keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in gradients:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        #Output directory for models and summaries
        #timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "graphs"))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Inference summaries
        inference_summary_op = tf.summary.merge([loss_summary])
        inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
        inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch_i in range(num_epochs):
            start_time = time.time()
            #将数据集分成训练集和测试集，随机种子不固定
            train_X,test_X, train_y, test_y = train_test_split(features,
                                                           targets_values,
                                                           test_size = 0.2,
                                                           random_state = 0)
            train_batches = get_batches(train_X, train_y, batch_size)
            test_batches = get_batches(test_X, test_y, batch_size)

            #训练的迭代，保存训练损失
            for batch_i in range(len(train_X) // batch_size):
                x, y = next(train_batches)

                categories = np.zeros([batch_size, 18])
                for i in range(batch_size):
                    categories[i] = x.take(6,1)[i]

                titles = np.zeros([batch_size, sentences_size])
                for i in range(batch_size):
                    titles[i] = x.take(5,1)[i]

                feed = {
                    uid: np.reshape(x.take(0,1), [batch_size, 1]),
                    user_gender: np.reshape(x.take(2,1), [batch_size, 1]),
                    user_age: np.reshape(x.take(3,1), [batch_size, 1]),
                    user_job: np.reshape(x.take(4,1), [batch_size, 1]),
                    movie_id: np.reshape(x.take(1,1), [batch_size, 1]),
                    movie_categories: categories,  #x.take(6,1)
                    movie_titles: titles,  #x.take(5,1)
                    targets: np.reshape(y, [batch_size, 1]),
                    dropout_keep_prob: dropout_keep, #dropout_keep
                }
                step, train_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed_dict=feed)  #cost
                losses['train'].append(train_loss)
                train_summary_writer.add_summary(summaries, step)  #

                # Show every <show_every_n_batches> batches
                if (epoch_i * (len(train_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                    #time_str = datetime.datetime.now().isoformat()
                    print('Epoch {:>3}   Batch {:>4}/{}  train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        (len(train_X) // batch_size),
                        train_loss)
                    )

            #使用测试数据的迭代
            for batch_i  in range(len(test_X) // batch_size):
                x, y = next(test_batches)
                categories = np.zeros([batch_size, 18])
                for i in range(batch_size):
                    categories[i] = x.take(6,1)[i]

                titles = np.zeros([batch_size, sentences_size])
                for i in range(batch_size):
                    titles[i] = x.take(5,1)[i]

                feed = {
                    uid: np.reshape(x.take(0,1), [batch_size, 1]),
                    user_gender: np.reshape(x.take(2,1), [batch_size, 1]),
                    user_age: np.reshape(x.take(3,1), [batch_size, 1]),
                    user_job: np.reshape(x.take(4,1), [batch_size, 1]),
                    movie_id: np.reshape(x.take(1,1), [batch_size, 1]),
                    movie_categories: categories,  #x.take(6,1)
                    movie_titles: titles,  #x.take(5,1)
                    targets: np.reshape(y, [batch_size, 1]),
                    dropout_keep_prob: 1,
                 }

                step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op], feed_dict=feed)  #cost

                #保存测试损失
                losses['test'].append(test_loss)
                inference_summary_writer.add_summary(summaries, step)  #

                #:6time_str = datetime.datetime.now().isoformat()
                if (epoch_i * (len(test_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3}   Batch {:>4}/{}   test_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        (len(test_X) // batch_size),
                        test_loss)
                    )

            print('Took time: {}', time.time()-start_time)

        # Save Model
        saver.save(sess, save_dir)  #, global_step=epoch_i
        print('Model Trained and Saved')



if __name__ == "__main__":
    train_graph, gradients, loss, train_op, global_step, inputs = build()
    train(train_graph, gradients, loss, train_op, global_step, inputs)
    save_param((save_dir))
