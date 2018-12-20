import tensorflow as tf
import numpy as np

from utils import load_params
from model import movies, users, movies_orig, users_orig
from model import sentences_size
from model import movieid2idx


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


load_dir = load_params()


def get_tensors(loaded_graph):
    uid = loaded_graph.get_tensor_by_name('input/uid:0')
    user_gender = loaded_graph.get_tensor_by_name('input/user_gender:0')
    user_age = loaded_graph.get_tensor_by_name('input/user_age:0')
    user_job = loaded_graph.get_tensor_by_name('input/user_job:0')
    movie_id = loaded_graph.get_tensor_by_name('input/movie_id:0')

    movie_categories = loaded_graph.get_tensor_by_name('input/movie_categories:0')
    movie_titles = loaded_graph.get_tensor_by_name('input/movie_title:0')
    targets = loaded_graph.get_tensor_by_name('input/targets:0')
    dropout_keep_prob = loaded_graph.get_tensor_by_name('input/droupout_keep_prob:0')
    lr = loaded_graph.get_tensor_by_name('input/learning_rate:0')

    inference = loaded_graph.get_tensor_by_name('inference/ExpandDims:0')
    movie_combine_layer_flat = loaded_graph.get_tensor_by_name('movie_fc/Reshape:0')
    user_combine_layer_flat = loaded_graph.get_tensor_by_name('user_fc/Reshape:0')

    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat


def rating(user_id, movie_id_):
    """
    Rating a movie for a user
    """
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Reload the graph and restore the params
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
        (uid, user_gender, user_age, user_job, movie_id,
        movie_categories, movie_titles,
         targets, lr, dropout_keep_prob,
         inference,_, __) = get_tensors(loaded_graph)

        categories = np.zeros([1, 18])
        categories[0] = movies.values[movieid2idx[movie_id_]][2]

        titles = np.zeros([1, sentences_size])
        titles[0] = movies.values[movieid2idx[movie_id_]][1]

        feed = {
                    uid: np.reshape(users.values[user_id-1][0], [1, 1]),
                    user_gender: np.reshape(users.values[user_id-1][1], [1, 1]),
                    user_age: np.reshape(users.values[user_id-1][2], [1, 1]),
                    user_job: np.reshape(users.values[user_id-1][3], [1, 1]),
                    movie_id: np.reshape(movies.values[movieid2idx[movie_id_]][0], [1, 1]),
                    movie_categories: categories,  #x.take(6,1)
                    movie_titles: titles,  #x.take(5,1)
                    dropout_keep_prob: 1
        }

        # Get Prediction
        inference_val = sess.run([inference], feed)

        print('For user: {} to rate movie {}:\n{:.2f}'.format(user_id, movie_id_, inference_val[0].take(0)))
        return (inference_val)


def generate_user_matrix():
    """
    Build the user matrix
    """
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        results = get_tensors(graph)
        uid = results[0]
        user_gender = results[1]
        user_age = results[2]
        user_job = results[3]
        user_combine_layer_flat = results[-1]
        dropout_keep_prob = results[-4]

        for item in users.values:
            feed = {
                uid: np.reshape(item.take(0), [1, 1]),
                user_gender: np.reshape(item.take(1), [1, 1]),
                user_age: np.reshape(item.take(2), [1, 1]),
                user_job: np.reshape(item.take(3), [1, 1]),
                dropout_keep_prob: 1
            }
            user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)
            yield user_combine_layer_flat_val


user_matrics = np.array(list(generate_user_matrix())).reshape(-1, 200)

def generate_movie_matrix():
    """
    Build the movies matrix
    """
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        results = get_tensors(graph)
        movie_id  = results[4]
        movie_categories = results[5]
        movie_titles = results[6]
        dropout_keep_prob = results[-4]
        movie_combine_layer_flat = results[-2]

        for item in movies.values:
            categories = np.zeros([1, 18])
            categories[0] = item.take(2)

            titles = np.zeros([1, sentences_size])
            titles[0] = titles.take(1)

            feed = {
                movie_id: np.reshape(item.take(0), [1, 1]),
                movie_categories: categories,
                movie_titles: titles,
                dropout_keep_prob: 1
            }
            movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat], feed)
            yield movie_combine_layer_flat_val

movie_matrics = np.array(list(generate_movie_matrix())).reshape(-1, 200)

def recommend_same_type(movie_id_, top=20):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        # Load the model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        norm_movie_matrics = tf.sqrt(
            tf.reduce_sum(tf.square(movie_matrics),
            1,
            keepdims=True
        ))
        normalized_movie_matrics = movie_matrics / norm_movie_matrics

        # Recommend the same type movies
        probs_embedding = (movie_matrics[movieid2idx[movie_id_]]).reshape([1, 200])
        probs_similarity = tf.matmul(
            probs_embedding,
            tf.transpose(normalized_movie_matrics)
        )
        sim = probs_similarity.eval()

        print("The movie you like:\n{}".format(movies_orig[movieid2idx[movie_id_]]))
        print('Perhaps you would love the same type:')
        print("==================================================")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 6:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for i in (results):
            print(movies_orig[i])
        print('==================================================')


def recommend_according_user(user_id_, top=10):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        probs_embedding = (user_matrics[user_id_-1]).reshape([1, 200])
        probs_similarity = tf.matmul(
            probs_embedding,
            tf.transpose(movie_matrics)
        )
        sim = probs_similarity.eval()

        results = set()
        print('Welcome user: {}'.format(user_id_))
        print('According your personal love, we recommend the following movies:')
        print('===============================================================')
        p = np.squeeze(sim)
        p[np.argsort(p)][:-top] = 0
        p = p / np.sum(p)
        while len(results) != 6:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for i in results:
            print(movies_orig[i])
        print('===============================================================')


def recommend_others_choice_according_movie(movie_id_, top=20):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        probs_movie_embedding = (movie_matrics[movieid2idx[movie_id_]]).reshape([1, 200])
        probs_user_like_sim = tf.matmul(
            probs_movie_embedding,
            tf.transpose(user_matrics)
        )
        other_user_id = np.argsort(probs_user_like_sim.eval())[0][-top:]

        probs_user_embedding = (user_matrics[other_user_id-1]).reshape([-1, 200])
        probs_sim = tf.matmul(
            probs_user_embedding,
            tf.transpose(movie_matrics)
        )
        sim = probs_sim.eval()

        print('The movie you love is: {}'.format(movies_orig[movieid2idx[movie_id_]]))
        print('There are another users love it: {}'.format(users_orig[other_user_id-1]))
        print('And he or she also like these movies:')
        print('=================================================================')
        results = set()
        p = np.argmax(sim, 1)
        while len(results) != 6:
            c = p[np.random.randint(top)]
            results.add(c)
        for i in results:
            print(movies_orig[i])
        print('=================================================================')



if __name__ == '__main__':
    print('Rating test...')
    rating(1024, 1024)
    print('\n\n')
    recommend_same_type(1401, 20)
    print('\n\n')
    recommend_according_user(1105, 10)
    print('\n\n')
    recommend_others_choice_according_movie(1401, 20)










