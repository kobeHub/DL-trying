import pickle




def save_param(params):
	"""
	Save parameters to file
	"""
	pickle.dump(params, open('params.p', 'wb'))

def load_params():
	return pickle.load(open('params.p', mode='rb'))

# Define super arguments
#嵌入矩阵的维度
embed_dim = 32
uid_max = 6040
gender_max = 2
age_max = 7
job_max = 21

#电影ID个数
movie_id_max = 3952
#电影类型个数
movie_categories_max = 19
#电影名单词个数
movie_title_max = 5216

#对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"

#电影名长度
sentences_size = 15 # = 15
#文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
#文本卷积核数量
filter_num = 8

num_epochs = 5
batch_size = 256
dropout_keep = 0.5
learning_rate = 0.0001
show_every_n_batches = 20
save_dir = './'

