https://developers.google.cn/machine-learning/crash-course/

看的这个教程

# 简介与概念

梯度什么的不提了

## Pandas：

* 用于数据分析和建模的库，用于TensorFlow编码。可以在https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb?hl=zh-cn#scrollTo=s3ND3bgOkB5k&uniqifier=1学习
* 数据结构：
  * DataFrame：关系型数据表格，包括行和列（列有命名）
    * 读取文件示例代码
    ```
    california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu- datasets/california_housing_train.csv", sep=",")
    california_housing_dataframe.describe()
    ```
    * 可以从Series对象构建DataFrame（见下面）
    * 可以直接当作一个dict用，如：
    ```
    cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
    cities['Population density'] = cities['Population'] / cities['Area square miles']
    ​````
    ```
  * Series对象：直接把数组传进构造函数中就可以了，之后可以用`pd.DataFrame({ 'City name': city_names, 'Population': population })`来构造DataFrame
    * 访问可以使用dict的方式（对于DataFrame）和list的方式（对于Series）
    * 可以对Series进行算术操作，相当于对每一个都进行这样的操作。但：注意：布尔值 Series 是使用“按位”而非传统布尔值“运算符”组合的。例如，执行逻辑与时，应使用 &，而不是 and。
    * Series可以用于NumPy的参数
    * 可以使用匿名函数`population.apply(lambda val: val > 1000000)`
    * 每个元素有自己的index，如果修改顺序可以修改index排序的顺序，如`cities.reindex([2, 0, 1])`就可以使得元素按照第二个、第零个、第一个排列；`cities.reindex(np.random.permutation(cities.index))`可以直接随机排列

## TensorFlow

加载数据集：
``` python
california_housing_dataframe = pd.read_csv("https://dl.google.com/mlcc/mledu-datasets/california_housing_train.csv", sep=",")
# 随机化处理与单位转换（换为以千为单位）
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
```

使用total_rooms作为输入，预测median_house_value（标签）

使用特征列表示特征的数据类型，仅储存对特征数据的描述而非数据本身。

``` python
# 定义输入特征为total_rooms
my_feature = california_housing_dataframe[["total_rooms"]]
# california_housing_dataframe[['total_rooms']]可以表示仅有这一列被保留下来，得到的是一个DataFrame，而如果只有一层中括号则剩下的是一个Series
feature_columns = [tf.feature_column.numeric_column("total_rooms")]
# 定义目标
targets = california_housing_dataframe["median_house_value"]

# 定义线性回归模型，使用小批量随机梯度下降训练模型
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# 梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# 用特征列和优化器（梯度下降）定义线性回归模型，设置学习率
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)
```
定义输入函数：
``` python
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """从一个特征训练线性回归模型
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
```

训练模型：（linear_regressor里面放着整个模型，所以直接用它进行train）
```
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)
```

评估模型：使用训练误差来进行
```
# 建立一个预测的输入函数
# 因为我们只对每个样本进行一个预测，所以我们不需要打乱顺序
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# 调用预测函数
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# 转换为NumPy类型以计算误差
predictions = np.array([item['predictions'][0] for item in predictions])

# 输出均方误差与均方根误差
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
```





看到了https://developers.google.cn/machine-learning/crash-course/feature-crosses/video-lecture