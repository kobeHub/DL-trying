import pandas as pd
import pickle
import re



def get_data():
    """
    Load dataset from File and convert data.
       Users:
           UserID: specific a user range: 1-6040
           Gender: 'F'->0  'M'->1
           Age:    divide into 7 part: 0-6
                0: under 18
                1: 18-24
                2: 25-34
                3: 35-44
                4: 45-49
                5: 50-55
                6: 56+
           OccupationID:  specific the job of one user
                0: "other" or not specified
                1: "academic/educator"
                2: "artist"
                3: "clerical/admin"
                4: "college/grad student"
                5: "customer service"
                6: "doctor/health care"
                7: "executive/managerial"
                8: "farmer"
                9: "homemaker"
                10: "K-12 student"
                11: "lawyer"
                12: "programmer"
                13: "retired"
                14: "sales/marketing"
                15: "scientist"
                16: "self-employed"
                17: "technician/engineer"
                18: "tradesman/craftsman"
                19: "unemployed"
                20: "writer"
            Zip-code: 邮编
        Movies:
            MoiveID:  specific a movies range: 1-3952
            Genres: convert into dict
    """
    # Users
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine = 'python')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    #改变User数据中性别和年龄
    gender_map = {'F':0, 'M':1}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map = {val:ii for ii,val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    #读取Movie数据集
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
    movies_orig = movies.values
    #将Title中的年份去掉
    pattern = re.compile(r'^(.*)\((\d+)\)$')

    title_map = {val:pattern.match(val).group(1) for ii,val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)

    #电影类型转数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)

    genres_set.add('<PAD>')
    genres2int = {val:ii for ii, val in enumerate(genres_set)}

    #将电影类型转成等长数字列表，长度是18
    genres_map = {val:[genres2int[row] for row in val.split('|')] for ii,val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt,genres2int['<PAD>'])

    movies['Genres'] = movies['Genres'].map(genres_map)

    #电影Title转数字字典
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)

    title_set.add('<PAD>')
    title2int = {val:ii for ii, val in enumerate(title_set)}

    #将电影Title转成等长数字列表，长度是15
    title_count = 15
    title_map = {val:[title2int[row] for row in val.split()] for ii,val in enumerate(set(movies['Title']))}

    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt,title2int['<PAD>'])

    movies['Title'] = movies['Title'].map(title_map)

    #读取评分数据集
    ratings_title = ['UserID','MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')

    #合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)

    #将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

    features = features_pd.values
    targets_values = targets_pd.values



    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig


# Load data to global
# title_count：Title字段的长度（15）
#title_set：Title文本的集合
# genres2int：电影类型转数字的字典
#features：是输入X
#targets_values：是学习目标y
#ratings：评分数据集的Pandas对象
#users：用户数据集的Pandas对象
#movies：电影数据的Pandas对象
#data：三个数据集组合在一起的Pandas对象
#movies_orig：没有做数据处理的原始电影数据
#users_orig：没有做数据处理的原始用户数据
title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = get_data()
#pickle.dump((title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig), open('preprocess.p', 'wb'))

print(users)
print(movies)
print(ratings)

