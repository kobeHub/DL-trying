# Movie Recommendation System

![GCC](https://img.shields.io/badge/Build-pass-brightgreen.svg)  ![gdb](https://img.shields.io/badge/tensorflow-1.12-brightgreen.svg)  ![Hex.pm](https://img.shields.io/hexpm/l/plug.svg?style=flat-square)  

## Introduction

**The program is Information Retrieval course design in software college, ShanDong University.** 

**The program is a Movie Recommendation System based on MovieLens dataset.**


> The basic idea is to use collaborative filtering to calculate the possible scores of each user for the movie. After vectorizing the user information and movie information, construct their respective feature matrices, and score and recommend based on this.


## Requirement

+ python 3.6
+ tensorflow 1.12
+ numpy 

 ## Run and demo

```shell
python3 predict.py
```

+ **Rating a movie for a user and recommend same type movies:**

  ![r1](https://github.com/kobeHub/DL-trying/raw/master/RecommendSys/MovieLens_ml-1m/pic/r1.png)

+ **Recommend movies according to user preference:**

  ![r2](https://raw.githubusercontent.com/kobeHub/DL-trying/master/RecommendSys/MovieLens_ml-1m/pic/r2.png)

+ **Recommend movies according to users who love the same movie:**

  ![r3](https://raw.githubusercontent.com/kobeHub/DL-trying/master/RecommendSys/MovieLens_ml-1m/pic/r3.png)



