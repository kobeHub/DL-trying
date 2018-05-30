# tensorflow 中常用的operations

## 特性分析

+ tensoflow使用图来建立计算任务，
+ 在会话(session)的上下文(context)执行图中的操作，包括变量的实例化，ops的依次执行
+ 使用tensor表示数据
+ 使用variable来维护状态
+ 使用feed和fetch可以为任意的操作(arbitrary operation)赋值或者从中获取数据

## 综述

Tensorflow 是一个编程系统，使用图来表示计算任务，图中的节点被表示为ops,一个op获得０到多个tensor执行计算，输出一定数量的张量．一个tersorflow 描述了图的构建过程但是在执行时必须借助会话执行(或者是借助eager execution)　

由于其延迟加载的特性，需要进行张量的判断时就需要特定的操作来满足

## tf.cond 用于张量的条件判断

![ if...else..error](http://media.innohub.top/180523-error.png)

如图所示