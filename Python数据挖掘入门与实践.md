# Python数据挖掘入门与实践

## 第一章 开始数据挖掘之旅

### 1.1 数据挖掘简介

- 数据集主要包括以下两个部分：

  表示真实世界中物体的样本。

  描述数据集中样本的特征。

- 特征抽取是数据挖掘过程的一个重要环节。

### 1.2 使用Python和IPython Notebook

- `pip3 freeze`命令测试pip能否正常运行。
- 只为当前用户安装ipython 可用命令 `pip install --user ipython[all]`
- 使用命令 `jupyter notebook`创建IPython Notebook实例，并打开web浏览器连接到实例。ctrl+C关闭。
- pip安装scikit-learn

### 1.3 亲和性分析示例

- 亲和性分析：根据样本个体之间的相似度，确定它们关系的亲疏。应用场景如下

  向网站用户提供多样化服务或投放定向广告

  为用户推荐电影或商品，而卖给他们一些与之相关的玩意

  根据基因寻找有亲缘关系的人

- 商品推荐。根据数据挖掘，我们希望得到以下规则：如果一个人买了商品X，那么他很有可能购买商品Y。

- 本书源码来自PacktPublishing

- 规则的优劣常见的衡量方法是支持度和置信度。

- 支持度指数据集中规则应验的次数，有时需要对支持度进行规范化（除以总数量）。

- 置信度衡量的是规则准确率如何，即符合前提条件的所有规则里，跟当前规则结论一致的比例有多大。

- 分别为规则应验和规则无效这两种情况创建字典。字典的键是由条件和结论组成的元组，如（3，4）.

- `from collections import defaultdict`使用defaultdict使得查找的键不存在时可以返回默认值

- `from operator import itemgetter`可以获取字典各元素的值，`itemgetter(1)`

### 1.4 什么是分类

- 分类应用的目标是，根据已知类别的数据集，经过训练得到一个分类模型，再用模型对类别未知的数据进行分类。
- 过拟合：模型在训练集上表现很好，但对于没有见过的数据表现很差。不要使用训练数据测试算法。
- `from sklearn.model_selection import train_test_split` 可将数据集划分为训练集和测试集。

## 第二章 用scikit-learn估计器分类

### 2.1 scikit-learn 估计器

- 估计器：用于分类、聚类和回归分析
- 转换器：用于数据预处理和数据转换
- 流水线：组合数据挖掘流程，便于再次使用。

- sklearn估计器包括fit()和predict()两个函数，接收和输出格式均为numpy数组或类似格式。
- 近邻算法：查找训练集，找到和新个体最相似的哪些个体，看这些个体属于哪个类别，就把新个体分到哪个类别。要计算每两个个体之间的距离，计算量大。在特征取离散值的数据集上表现很差。
- 欧式距离：即真实距离，是两个特征向量长度平方和的平方根。直观，但当某些特征取值比其他特征大得多，或很多特征值为0即稀疏矩阵时，结果不准确，此时可使用曼哈顿距离和余弦距离。
- 曼哈顿距离为两个特征在标准坐标系中的绝对轴距之和，受异常值的影响比欧氏距离小。但当某些特征取值比其他特征大得多，这些特征会掩盖其他特征间的近邻关系。
- 余弦距离更适合解决异常值和数据稀疏问题，指的是特征向量夹角的余弦值。适用于特征向量很多的情况，丢弃了向量长度所包含的在某些场景下有用的信息。
- 交叉检验算法描述如下：
  - 将整个大数据集分为几个部分
  - 对于每个部分执行以下操作：
    - 将其中一部分作为当前测试集
    - 用剩余部分训练算法
    - 在当前测试集上测试算法
  - 记录每次得分及平均得分
  - 在上述过程中，每条数据只能在测试集中出现一次，以减少运气成分。
- from sklearn.model_selection import cross_val_score 默认使用Stratified K Fold方法切分数据集，保证切分后的数据集中类别分别大致相同。用此函数进行交叉检验。
- `%matplotlib inline` 来告知要在notebook里作图

### 2.2 流水线在预处理中的应用

- 规范化：特征值的大小和该特征的分类效果没有任何关系，所以要对不同的特征进行规范化，使得它们的特征落在相同的值域或几个确定的类别。
- 选取最具区分度的特征、创建新特征都属于预处理的范畴。sklearn中的预处理工具叫做转换器。
- `from sklearn.preprocessing import MinMaxScaler`进行基于特征的规范化，把每个特征值域规范到0和1之间。
  - 为使每条数据特征值和为1，使用Normalizer
  - 为使各特征的均值为0，使用StandardScaler
  - 为将数值型特征二值化，使用Binarizer，大于阈值为1，反之为0

### 2.3 流水线

- 随着实验的增加，操作复杂程度也在提高，可能导致错误操作或操作顺序不当的问题。流水线就是用来解决这个问题的。流水线将这些步骤保存到工作流中，以便之后的数据读取以及预处理等操作。
- `from sklearn.pipeline import Pipeline` 流水线的输入为一连串的数据挖掘步骤，接着是转换器，最后是估计器，每一步的结果作为下一步的输出。
- 每一步都用元组('名称','步骤')来表示。如`scaling_pipline = Pipeline([('scale', MinMaxScaler()), ('predict', KNeighborsClassifier())])`

## 第三章 用决策树预测获胜球队

### 3.1 加载数据集

- 决策树的一大优点是人和及其都能看懂
- pandas.readcsv函数提供了修复数据的参数，`pd.read_csv(data_filename, parse_dates=["Dates"], skiprows=[0,])`
- 用`dataset.columns=[]`修改头部。
- 用`dataset[].values`提取数组。
- 用`for index, row in dataset.iterrows()`遍历每一行。用for index, row in dataset.sort("Date").iterrows()按某一列顺序遍历。

### 3.2 决策树

- 决策树是一种积极算法，需要进行训练，而近邻算法是惰性算法，分类时才开始干活。
- 决策树在从根节点起每层选取该层的最佳特征用于决策，到达下一个节点，选择下一个最佳特征，以此类推。当无法从增加树的层级中获得更多信息时，启动退出机制。
- sklearn实现了分类回归树算法（CART）并将其作为生成决策树的默认算法，支持连续型和类别型特征。
- 退出准则可以防止过拟合。除了退出准则外，也可以先建立完整的树，再进行剪枝，去掉对整个过程没有提供太多信息的节点。
- 使用
  - min_samples_split：指定创建一个新节点至少需要的个体数量
  - min_samples_leaf：指定为保留节点每个节点至少应该保留的个体数量。
- 创建决策的标准，有：
  - 基尼不纯度：用于衡量决策节点错误预测新个体类别的比例。
  - 信息增益：用信息论中的熵表示决策节点提供多少新信息。
- `from sklearn.tree import DecisionTreeClassifier`创建决策树。

### 3.3 NBA比赛结果预测

- `from sklearn.preprocessing import LabelEncoder`转换器将字符串类型的球队名转化为整型，以满足sklearn决策树的需求。

  encoding = LabelEncoder()

  encoding.fit(dataset[""].values)

  home_teams = encoding.transform(dataset[""].values)

- np.vstack，np.hstack将向量组合起来，形成一个矩阵。

- 使用LabelEncoder转换得到的整型仍被认为是连续型特征，即1与2比1与3更相似。

- 使用OneHotEncoder可以将整数转化为二进制数字，特征有多少种类型就有多少位二进制数字，第几个类型第几个二进制位为1，其余为0。如001，010，100.

### 3.4 随机森林

- 决策树可能出现过拟合的情况，解决方法之一是调整决策树算法，限制它所学到的规则的数量。使用这种方法会导致决策树泛化能力强，但整体表现稍弱。

- 随机森林通过创建多棵决策树，用它们分别进行预测，再根据少数服从多少的原则选择最终预测结果。
  - 装袋：每次随机从数据集中选取一部分数据作为训练集。
  - 选取部分决策特征作为决策依据：前几个决策节点的特征非常突出，随机选取的训练集仍具有较大相似性。

- 方差是由训练集的变化引起的。决策树这种方差大的算法极易受到训练集变化的影响，从而产生过拟合问题。随机森林对大量决策树的预测结果取均值，能有效降低方差。

- 偏误是由算法的假设引起的，与数据集没有关系。

- 决策树集成做出了以下假设：预测过程具有因分类器而异的随机性，使用多个模型得到的预测结果的均值，能够消除随机误差的影响。

- `from sklearn.ensemble import RandomForestClassifier`提供了`DecisionTreeClassifier`的的参数，如决策标准（基尼不纯度/信息增益）、max_features、min_samples_split等。也引入了新参数：

  - n_esimators：指定决策树数量。
  - oob_score：设为真，则 测试时不会使用训练时用过的数据。
  - n_jobs：并行计算使用的内核数量。

- 使用`GridSearchCV()`搜索最佳参数：

  `parameter_space = {`
                     `"max_features": [2, 10, 'auto'],`
                     `"n_estimators": [100,],`
                     `"criterion": ["gini", "entropy"],`
                     `"min_samples_leaf": [2, 4, 6],`
                     `}`
  `clf = RandomForestClassifier(random_state=14)`
  `grid = GridSearchCV(clf, parameter_space)`

  可使用`grid.best_estimator_`查看使用了哪些参数。

- 使用`dataset[""] = feature_creator(dataset)`创建新特征。

## 第四章 用亲和性分析推荐电影

### 4.1 亲和性分析

- 亲和性分析用来找出两个对象共同出现的情况。应用场景如：
  - 欺诈检测
  - 顾客区分
  - 软件优化
  - 产品推荐
- Apriori算法是经典的亲和性分析算法。从数据集中频繁出现的商品中选取共同出现的商品组成频繁项集，避免复杂度呈指数增长的问题。
  - 最小支持度：要生存A,B的频繁项集（A,B），要求最小支持度为30，则A和B都必须在数据集中出现30次。更大的频繁项集如（A,B,C）的子集（A,B）也要是满足最小支持度的频繁项集。
  - 生成频繁项集后，再考虑其他不够频繁的项集。
- Apriori算法过程为：
  - 设定最小支持度，找出频繁项集。
  - 根据置信度选取关联规则。

### 4.2 电影推荐问题

- `all_ratings = pd.read_csv(ratings_filename, delimiter='\t', header=None, names=["UserID", "MovieID", "Rating", "Datetime"])`将识别制表符作为分隔符，没有表头，添加表头。
- `all_ratings["Datetime"] = pd.to_datetime(all_ratings["Datetime"],unit='s')`解析时间戳数据，设定单位为秒。
- 稀疏矩阵格式：即对不存在的数据不存储，而不是存放大量的0。

### 4.3 Apriori算法的实现

- `ratings = all_ratings[all_ratings["UserID"].isin(range(200))]` 选取一部分数据作训练集，减少搜索空间。

- `favorable_ratings = ratings[ratings["Favorable"]]` 新建数据集，只保留某一行。

- `favorable_reviews_by_users = dict((k,frozenset(v.values)) for k, v in favorable_ratings.groupby("UserID")["MovieID"])`用.groupby进行分组，frozenset是固定不变的集合，速度快于列表。

- Apriori算法过程如下：

  - 把各项放到只包含子集的项集中，生成最初的频繁项集。只使用达到最小支持度的项。
  - 查找现有频繁项集的超集，发现新的备选项集。
  - 测试新生成备选项集的频繁程度，如果不够频繁则舍弃。如果没有新的频繁项集，则跳到最后一步。
  - 存储新发现的频繁项集，跳到第二步。
  - 返回所有频繁项集。

- 第一步：

  ```python
   frequent_itemsets[1] = dict((frozenset((movie_id,)),row["Favorable"]) 
  for movie_id, row in num_favorable_by_movie.iterrows() 
  if row["Favorable"] > min_support) 
  ```

- 第二三步：

  ```python
  def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
      counts = defaultdict(int)
      for user, reviews in favorable_reviews_by_users.items():
          for itemset in k_1_itemsets:
              if itemset.issubset(reviews):
                  for other_reviewed_movie in reviews - itemset:
                      current_superset = itemset | frozenset((other_reviewed_movie,))
                      counts[current_superset] += 1
      return dict([(itemset, frequency)
                 for itemset, frequency in counts.item()
                 if frequency >= min_support])
  ```

- 第四五步

  ```python
  for k in range(2, 20):
      cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k-1], min_support)
      frequent_itemsets[k] = cur_frequent_itemsets
      if len(cur_frequent_itemsets) == 0:
          print("Did not find any frequent itemsets of length {}".format(k))
          sys.stdout.flush()
          break
      else:
          print("I found {} frequent itemsets of length {}".format(len(cur_frequent_itemsets), k))
          sys.stdout.flush()
  del frequent_itemsets[1]
  ```

## 第五章 用转换器抽取特征

### 5.1 特征抽取

- 特征抽取是数据挖掘任务最重要的环境，对最终结果的影响高于数据挖掘算法本身。
- 特征选择降低真实世界的复杂度，模型比现实更容易操纵。
- 简化要以数据挖掘的目标为核心。
- 简化会忽略很多细节，甚至会抛弃很多对数据挖掘算法能力起到帮助作用的信息。
- 不是所有特征必须是数值型或类别型值，直接作用于文本、图像和其他数据结构的算法已经研究出来了。
- `adult.dropna(how="all", inplace=True)`删除包含无效数据的行，inplace为真表明在当前数据框中修改，而不是新建一个数据框。
- `adult["Hours-per-week"].describe()`提供了常见统计量的计算。
- `adult["Work-Class"].unique()`得到特征的所有不同情况。

### 5.2 特征选择

- 特征选择的原因如下：

  - 降低复杂度
  - 降低噪音
  - 增加模型可读性

- `X = np.arange(30).reshape((10, 3))`创建0到29，30个数字，分为3列10行。

- 删除方差达不到最低标准的特征。

  ```python
  from sklearn.feature_selection import VarianceThreshold
  vt = VarianceThreshold()
  xt = vt.fit_transform(X)
  print(vt.variances_)
  ```

- 随着特征的增加，寻找最佳特征组合的时间复杂度是呈指数级增长的，变通的方法是寻找表现好的单个特征，一般是测量变量与目标类别之间的某种相关性。

- SelectKBest返回k个最佳特征，SelectPercentile返回最佳的前r%个特征。计算单个特征与某一类别之间相关性的计算方法有卡方检验，互信息和信息熵。

  ```python
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import chi2
  transformer = SelectKBest(score_func=chi2, k=3)
  xt_chi2 = transformer.fit_transform(X, y)
  print(transformer.scores_)
  ```

- 也可使用皮尔逊相关系数计算相关性。皮尔逊相关系数为-1到1的值，绝对值越大，相关性越大。`from scipy.stats import pearsonr`scipy实现的皮尔逊相关系数接收（X，y），返回每个特征的皮尔逊相关系数和p值，X为该特征列。

  ```python
  def multivariate_pearsonr(X, y):
      scores, pvalues = [], []
      for column in range(X.shape[1]):
          cur_score, cur_p = pearsonr(X[:, column], y)
          scores.append(abs(cur_score))
          pvalues.append(cur_p)
      return (np.array(scores), np.array(pvalues))
  transformer = SelectKBest(score_func=multivariate_pearsonr, k=3)
  xt_pearson = transformer.fit_transform(X, y)
  print(transformer.scores_)
  ```

- 哪些特征是好的没有标准答案，取决于度量标准。

### 5.3 创建特征

- 数据集中原始特征可能会出现特征间相关性很强，特征冗余等情况，增加算法除了难度，因此要创建新特征。

- 使用converters修复数据。

  ```python
  def convert_number(x):
      try:
          return float(x)
      except ValueError:
          return np.nan
  from collections import defaultdict
  converters = defaultdict(convert_number)
  converters[1558] = lambda x: 1 if x.strip() == 'ad.' else 0
  ads = pd.read_csv(data_filename, header=None, converters=converters)
  ```

- 主成分分析算法的目的是找到能用较少信息描述数据集的特征组合。意在发现彼此之间没有相关性、能够描述数据集、特征方差与整体方差相近的特征，即主成分。

  ```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=5)
  Xd = pca.fit_transform(X)
  np.set_printoptions(precision=3, suppress=True)
  pca.explained_variance_ratio_
  ```

  

- 用PCA算法不好的地方在于，主成分往往是很多特征的复杂组合，理解起来很困难。

- PCA的一个优点是可以将抽象的数据集绘制成图像，如将前两个特征做成图形。

  ```python
  %matplotlib inline
  from matplotlib import pyplot as plt
  classes = set(y)
  colors = ['red', 'green']
  for cur_class, color in zip(classes, colors):
      mask = (y == cur_class).values
      plt.scatter(Xd[mask,0], Xd[mask,1], marker='o', color=color, label=int(cur_class))
  plt.legend()
  plt.show()
  ```

### 5.4 创建自己的转换器

- 导入TransformerMinxin类，重写其中的fit、transform函数。使用as_float_array判断输入类型是否为float。

  ```python
  from sklearn.base import TransformerMixin
  from sklearn.utils import as_float_array
  class MeanDiscrete(TransformerMixin):
      def fit(self, X):
          X = as_float_array(X)
          self.mean = X.mean(axis=0)
          return self
      def transform(self, X):
          X = as_float_array(X)
          assert X.shape[1] == self.mean.shape[0]
          return X > self.mean
  mean_discrete = MeanDiscrete()
  X_mean = mean_discrete.fit_transform(X)
  ```

## 第六章 使用朴素贝叶斯进行社会媒体挖掘

### 6.1 消歧

- 朴素贝叶斯：朴素是因为假设了各特征之间是相互独立的。
- 文本挖掘的一个难点在于歧义，如bank一词指的是河岸还是银行，消除歧义被称为消歧。
- jupyter中用`%%javascript`表示该代码段为JavaScript语言。
- 只有在相同的测试集上，在相同的条件下进行测试，才能比较算法的优劣。

### 6.2 文本转换器

- 一种简单但高效的文本测量方法是统计数据集中每个单词出现的次数。
- `from collections import Counter`能计算列表中各个元素出现的次数，用`c.most_common(5)`输出出现次数最多的5个词。
- 词袋模型分为三种：
  - 用词语实际出现的次数作为词频。缺点是当文档长度差异明显时，词频差距会很大。
  - 使用归一化后的词频，每篇文档中词频和为1，规避了文档长度对词频的影响。
  - 用二值特征表示，出现为1，不出现为0。
  - 词频-逆文档频率法（tf-idf）：用词频代替词的出现次数，词频除以包含该词的文档数。
- N元语法是指由几个连续的词组成的子序列。会导致特征矩阵变得更稀疏。另一种是字符N元语法，用于发现拼写错误。

### 6.3 朴素贝叶斯

- 在贝叶斯统计学中，使用数据来描述模型，而不是用模型描述数据。频率论者则使用数据证实假设的模型。

- 贝叶斯定理公式如下：
  $$
  P(A|B)=\frac{P(B|A)P(A)}{P(B)}
  $$
  比较后验概率大小时，只需计算分子并比较大小。

### 6.4 应用

- NLTK的word_tokenize函数将原始文档转换为由单词和其是否出现的字典。NLTK与转换器接口不一致，因此要创建包含fit和transform的转换器。

  ```python
  class NLTKBOW(TransformerMixin):
      def fit(self, X, y=None):
          return self
      def transform(self, X):
          return [{word: True for word in work_tokenize(document) for document in X}]
  ```

- `from sklearn.feature_extraction import DictVectorizer`接收元素为字典的列表，将其转换为矩阵。

- `from sklearn.naive_bayes import BernoulliNB`引入二值分类的朴素贝叶斯分类器。

- 组合部件，创建流水线。

  ```python
  from sklearn.pipeline import Pipeline
  pipeline = Pipeline([('bag-of-words', NLTKBOW()),
                      ('vectorizer', DictVectorizer()),
                      ('naive-bayes', BernoulliNB())
                      ])
  ```

- 正确率对于不均匀的数据集来说，并不能反映算法的优劣。更常用的指标为F1值。

- F1值是以每个类别为基础进行定义的。包括两大概念：

  - 准确率：预测结果属于某一类的个体，实际属于该类的比例。
  - 召回率：被正确预测为某类的个体数量与数据集中该类个体总数的比例。

- 在案例中就是：

  - 正确率：在所有被预测为相关的消息中真正相关的占比多少？
  - 召回率：数据集所有相关的消息中，由多少被正确识别为相关？

- F1值是正确率和召回率的调和平均数。
  $$
  F1=2·\frac{precision·recall}{precision+recall}
  $$

- `scores=cross_val_score(pipeline, tweets, labels, scoring='f1')`交叉验证法计算F1得分。
- `nb = model.named_steps('naive-bayes')`访问流水线的每个步骤。
- 当概率较小时，可以使用对数概率，防止下溢。
- `np.argsort()`进行降序排列。
- DictVectorizer保存了特征的名称，可搜索其feature_names_属性查找。

## 第七章 用图挖掘找到感兴趣的人

### 7.1 加载数据集

- 初始化twitter连接实例。

  ```python
  import twitter
  consumer_key = "52Nu7ubm2szT1JyJEOB7V2lGM"
  consumer_secret = "mqA94defqjioyWeMxdJsSduthxdMMGd2vfOUKvOFpm0n7JTqfY"
  access_token = "16065520-USf3DBbQAh6ZA8CnSAi6NAUlkorXdppRXpC4cQCKk"
  access_token_secret = "DowMQeXqh5ZsGvZGrmUmkI0iCmI34ShFzKF3iOdiilpX5"
  authorization = twitter.OAuth(access_token, access_token_secret, consumer_key, consumer_secret)
  t = twitter.Twitter(auth=authorization, retry=True)
  ```

- `search_results = t.search.tweets(q="python", count=100)['statuses']`搜索包含关键词的推文。

- 导入joblib库，保存模型。

  ```python
  from sklearn.externals import joblib
  model_filename = os.path.join(os.path.expanduser("~"), "Models", "twitter", "python_context.pkl")
  joblib.dump(model, output_filename)
  ```

- 定制的类无法直接用joblib加载，因此要重建NLTKBOW。

  ```python
  from sklearn.base import TransformerMixin
  from nltk import word_tokenize
  
  class NLTKBOW(TransformerMixin):
      def fit(self, X, y=None):
          return self
  
      def transform(self, X):
          return [{word: True for word in word_tokenize(document)}
                   for document in X]
  ```

- 调用joblib的load函数加载模型。

  ```python
  context_classifier = joblib.load(model_filename)
  ```

- `results = t.friends.ids(user_id=user_id, cursor=cursor, count=5000)`从推特获取用户关注的好友编号列表，一页5000，使用游标表示第几页，初始设为-1，不为0表示有下一页。返回一个字典，包含'ids'和下一个游标'next_cursor'。

- `friends.extend([friends for friends in results['ids']])`列表扩展，接收参数为一个列表。

- `sys.stdout.flush()`输出缓存到屏幕，避免过长等待。

- 完整的获取好友函数代码：

  ```python
  def get_friends(t, user_id):
      friends = []
      cursor = -1  # Start with the first page
      while cursor != 0:  # If zero, that is the end:
          try:
              results = t.friends.ids(user_id=user_id, cursor=cursor, count=5000)
              friends.extend([friends for friends in results['ids']])
              cursor = results['next_cursor']
              if len(friends) >= 10000:
                  break
              if cursor != 0:
                  print("Collected {} friends so far, but there are more".format(len(friends)))
                  sys.stdout.flush
          except TypeError as e:
              if results is None:
                  print("You probably reached your API limit, waiting for 5 minutes")
                  sys.stdout.flush()
                  time.sleep(5*60) # 5 minute wait
              else:
                  raise e
          except twitter.TwitterHTTPError as e:
              break
          finally:
              time.sleep(60)  # Wait 1 minute before continuing
      return friends
  ```

- 为加快网络构建，从现有用户-好友列表字典中计算每个好友的出现次数，降序排列，依次查看该好友是否已被查找，找到排名最高的未被查找的好友，进行搜索，更新字典，以此类推。

- 使用json保存/加载好友字典：

  ```python
  import json
  friends_filename = os.path.join(data_folder, "python_friends.json")
  with open(friends_filename, 'w') as outf:
      json.dump(friends, outf)
  with open(friends_filename) as inf:
      friends = json.load(inf)
  ```

- 可以用Networkx库实现图关系的可视化。

  - ```python
    # 引入Networkx库，创建有向图
    import networkx as nx
    G = nx.DiGraph()
    ```

  - ```python
    # 添加顶点
    main_users = friends.keys()
    G.add_nodes_from(main_users)
    ```

  - ```python
    # 添加边
    for user_id in friends:
        for friend in friends[user_id]:
            if friend in main_users:
               G.add_edge(user_id, friend)
    ```

  - ```python
    # 绘图
    %matplotlib inline
    nx.draw(G)
    ```

  - ```python
    # 使用plt放大图像
    from matplotlib import pyplot as plt
    plt.figure(3,figsize=(40,40))
    nx.draw(G, alpha=0.1, edge_color='b', node_color='g', node_size=2000)
    plt.axis('on')
    plt.xlim(0.45, 0.55)
    plt.ylim(0.45, 0.55)
    ```

- 杰卡德相似系数：两个集合交集的元素数量除以两个集合并集的元素数量，范围为0到1，代表两者的重合比例。

  ```python
  def compute_similarity(friends1, friends2):
      return len(friends1 & friends2) / len(friends1 | friends2)
  ```

- 创建带杰卡德相似系数权重无向图：

  ```python
  def create_graph(followers, threshold=0):
      G = nx.Graph()
      for user1 in friends.keys():
          for user2 in friends.keys():
              if user1 == user2:
                  continue
              weight = compute_similarity(friends[user1], friends[user2])
              if weight >= threshold:
                  G.add_node(user1)
                  G.add_node(user2)
                  G.add_edge(user1, user2, weight=weight)
      return G
  ```

- networkx中布局方式决定顶点和边的位置，常用布局方式有spring_layout，circular_layout，random_layout，shell_layout和spectral_layout。

- 根据布局方式，依次绘制顶点和边，获取权重数据。

  ```python
  plt.figure(figsize=(10,10))
  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos)
  edgewidth = [ d['weight'] for (u,v,d) in G.edges(data=True)]
  nx.draw_networkx_edges(G, pos, width=edgewidth)
  ```

### 7.2 寻找子图

- 聚类分析：找出相似用户群，向他们定向投放广告。

- 聚类分析的复杂之处在于：

  - 缺乏评价结果的标准
  - 没有事先标注的数据进行训练，得到的是近似的分组结果，而不是明确的分类。

- 连通分支是图中由边连接在一起的一组顶点，不要求顶点两两相连，但任意两个顶点之间存在一条路径。连通分支的计算不考虑权重，只考虑边是否存在。

- 用networkx的函数寻找连通分支。`sub_graphs = nx.connected_component_subgraphs(G)`

- 画出连通分支图。

  ```python
  sub_graphs = nx.connected_component_subgraphs(G)
  nx.draw(list(sub_graphs)[6])
  ```

- `n_subgraphs = nx.number_connected_components(G)`计算连通分支数量。

- 画出所有连通分支。

  ```python
  sub_graphs = nx.connected_component_subgraphs(G)
  n_subgraphs = nx.number_connected_components(G)
  fig = plt.figure(figsize=(20, (n_subgraphs * 2)))
  for i, sub_graph in enumerate(sub_graphs):
      ax = fig.add_subplot(int(n_subgraphs / 2), 2, i)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      pos = nx.spring_layout(G)
      nx.draw_networkx_nodes(G, pos, sub_graph.nodes(), ax=ax, node_size=500)
      nx.draw_networkx_edges(G, pos, sub_graph.edges(), ax=ax)
  ```

- 聚类应使得：

  - 同一簇内的个体尽可能相似
  - 不同簇内的个体尽可能不相似

- 轮廓系数：
  $$
  s=\frac{b-a}{max(a,b)}
  $$
  a为簇内距离，表示与簇内其他个体之间的平均距离。b为簇间距离，也就是与最近簇内个体之间的平均距离。

- 总轮廓系数是每个个体轮廓系数的均值。接近1时，表示簇内相似度高，簇间很远；接近0时，表示所有簇重合在一起，簇间距离很小；接近-1时，表示个体分在错误的簇内。

- `from sklearn.metrics import silhouette_score`计算轮廓系数。

- 轮廓函数的定义要求至少有两个顶点，两个连通分支。

- 轮廓系数函数接收距离矩阵，因此要将图转换为距离矩阵。

  ```python
  X = nx.to_scipy_sparse_matrix(G).todense()
  ```

- 对于稀疏矩阵，应使用V-MEASURE或调整互信息进行评价。

- `silhouette_score(X, labels, metric='precomputed')`指定metric，避免X被认为是特征矩阵，而不是距离矩阵。

- 使用`from scipy.optimize import minimize`自动调整参数优化。要求变量在其他参数前面，即：

  ```python
  def compute_silhouette(threshold, friends):
      G = create_graph(friends, threshold=threshold)
      if len(G.nodes()) == 0:
          return -99  # Invalid graph
      sub_graphs = nx.connected_component_subgraphs(G)
      if not (2 <= nx.number_connected_components(G) < len(G.nodes()) - 1):
          return -99  # Invalid number of components, Silhouette not defined
      label_dict = {}
      for i, sub_graph in enumerate(sub_graphs):
          for node in sub_graph.nodes():
              label_dict[node] = i
      labels = np.array([label_dict[node] for node in G.nodes()])
      X = nx.to_scipy_sparse_matrix(G).todense()
      X = 1 - X
      return silhouette_score(X, labels, metric='precomputed')
  ```

- minimize是调整参数使得函数返回值最小，这里要求轮廓系数最大，因此要取反，将其变为损失函数。

  ```python
  def invert(func):
      def inverted_function(*args, **kwds):
          return -func(*args, **kwds)
      return inverted_function
  ```

- 设定初始阈值为0.1，设定优化方法为下山单纯形法，设定被参数参数为friends字典，设定最大迭代次数为10.

  ```python
  result = minimize(invert(compute_silhouette), 0.1, method='nelder-mead', args=(friends,), options={'maxiter':10, })
  ```

  返回结果中的x的最佳阈值大小。

## 第八章 用神经网络破解验证码

### 8.1 人工神经网络

- 神经网络由一系列相互连接的神经元组成，每个神经元都是一个简单的函数，接收一定输入，给出相应输出。

- 神经元中用于处理数据的标准函数被称为激活函数。

- 激活函数应是可导和光滑的。常用的激活函数，如逻辑斯蒂函数：
  $$
  f(x)=\frac{L}{1+e^{-k(x-x_0)}}
  $$

- 全连接层：上一层每个神经元的输出都输入到下一层的所有神经元。
- 边的权重开始时通常是随机选取的，训练过程中再逐步更新。

### 8.2 创建数据集

- 用PIL库的Image初始化图像对象，ImageDraw初始化绘图对象，ImageFont初始化字体对象。用skimage的transform进行图像错切变化。返回归一化结果。

  ```python
  def create_captcha(text, shear=0, size=(100,24)):
      im = Image.new("L", size, "black")
      draw = ImageDraw.Draw(im)
      font = ImageFont.truetype(r"Coval.otf", 22)
      draw.text((2, 2), text, fill=1, font=font)
      image = np.array(im)
      affine_tf = tf.AffineTransform(shear=shear)
      image = tf.warp(image, affine_tf)
      return image / image.max()
  ```

- skimage中的label函数能找出图像中像素值相同且连接在一起的像素块。输入输出均为图像数组，返回的数组中，连接在一起的区域是大于0的值，每个区域的值不同，其他区域为0值。

- skimage的regionprops能抽取连续区域，属性.bbox返回区域的起始结束横纵坐标。

  ```python
  def segment_image(image):
      labeled_image = label(image > 0)
      subimages = []
      for region in regionprops(labeled_image):
          start_x, start_y, end_x, end_y = region.bbox
          subimages.append(image[start_x:end_x, start_y:end_y])
      if len(subimages) == 0:
          return [image,]
      return subimages
  ```

- 利用subplots返回的坐标起点，画图。

  ```python
  f, axes = plt.subplots(1, len(subimages), figsize=(10, 3))
  for i in range(len(subimages)):
      axes[i].imshow(subimages[i], cmap="gray")
  ```

- 使用`from sklearn.utils import check_random_state`随机选取字母和错切值。

  ```python
  def generate_sample(random_state=None):
      random_state = check_random_state(random_state)
      letter = random_state.choice(letters)
      shear = random_state.choice(shear_values)
      return create_captcha(letter, shear=shear, size=(20, 20)), letters.index(letter)
  ```

- 用zip函数将3000次采样数据组合。

  ```python
  dataset, targets = zip(*(generate_sample(random_state) for i in range(3000)))
  ```

- 使用一位有效码编码使得单个神经元有26个输出，为1则是该字母，否则为0。

  ```python
  from sklearn.preprocessing import OneHotEncoder
  onehot = OneHotEncoder()
  y = onehot.fit_transform(targets.reshape(targets.shape[0],1))
  ```

- 用todense将稀疏矩阵转换为密集矩阵。
- 使用`from skimage.transform import resize`改变图像大小，将不规整的图像调到相同大小。
- `X = dataset.reshape((dataset.shape[0], dataset.shape[1] *dataset.shape[2]))`将数组扁平化为二维。

### 8.3 训练和分类

- 使用pybrain进行神经网络的构建，pybrain有自己的数据格式，转换数据格式。

  ```python
  from pybrain.datasets import SupervisedDataSet
  training = SupervisedDataSet(X.shape[1], y.shape[1])
  for i in range(X_train.shape[0]):
      training.addSample(X_train[i], y_train[i])
  ```

- 隐含层的神经元过多会导致过拟合，过少会导致低拟合。

- 导入buildNetwork函数，指定维度，创建神经网络。第一二三个参数分别为三层网络神经元的数量。激活偏执神经元。

  ```python
  from pybrain.tools.shortcuts import buildNetwork
  net = buildNetwork(X.shape[1], 100, y.shape[1], bias=True)
  ```

- 反向传播算法从输出层开始，向上层查找预测错误的神经元，微调这些神经元输入值的权重，以达到修复输出错误的目的。

- 微调的幅度取决于神经元各边权重的偏导数和学习速率。计算出函数误差的梯度乘以学习速率，就是原权重需要下调的幅度。

- 有些情况下，修正的结果仅是局部最优。

- 反向传播算法，限定反向传播次数为20：

  ```python
  from pybrain.supervised import BackpropTrainer
  trainer = BackpropTrainer(net, training, learningrate=0.01, weightdecay=0.01)
  trainer.trainEpochs(epochs=20)
  ```

- 在trainer上调用testOnClassData函数，预测分类结果。

  ```python
  predictions = trainer.testOnClassData(dataset=testing)
  ```

- 输入验证码图片，预测验证码。用activate函数激活神经网络，输入化为一维的子图片数据。

  ```python
  def predict_captcha(captcha_image, neural_network):
      subimages = segment_image(captcha_image)
      predicted_word = ""
      for subimage in subimages:
          subimage = resize(subimage, (20, 20))
          outputs = net.activate(subimage.flatten())
          prediction = np.argmax(outputs)
          predicted_word += letters[prediction]
      return predicted_word
  ```

- 从nltk预料库中下载words语料库，从中找出长度为4的单词，并大写化。

  ```python
  import nltk
  nltk.download('words')
  valid_words = [word.upper() for word in words.words() if len(word) == 4]
  ```

- 用二维混淆矩阵表现预测的正确率和召回率。

  ```python
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(np.argmax(y_test, axis=1), predictions)
  ```

- 画出混淆矩阵，并标出坐标。

  ```python
  plt.figure(figsize=(20,20))
  tick_marks = np.arange(len(letters))
  plt.xticks(tick_marks, letters)
  plt.yticks(tick_marks, letters)
  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  plt.imshow(cm, cmap="Blues")
  ```

### 8.4 用词典提升正确率

- 先检查以下词典内是否包含该单词，包含则直接输出，否则查找相似的单词，作为更新过的预测结果返回。

- 列文斯坦编辑距离适用于确定两个短字符串的相似度，计算一个单词变成另一个单词的步骤数，步骤数越少越相似。以下操作算一步：

  - 在单词的任意位置插入一个字母
  - 从单词中删除任意一个字母
  - 把一个字母替换为另一个字母

- nltk中实现了编辑距离算法。

  ```python
  from nltk.metrics import edit_distance
  steps = edit_distance("STEP", "STOP")
  print("The number of steps needed is: {0}".format(steps))
  ```

- 在字符串等长的情况下，另一种方法是直接计算相同位置不相同的字符数。

  ```python
  def compute_distance(prediction, word):
      return len(prediction) - sum(prediction[i] == word[i] for i in range(len(prediction)))
  ```

- 改进后的预测函数：

  ```python
  from operator import itemgetter
  def improved_prediction(word, net, dictionary, shear=0.2):
      captcha = create_captcha(word, shear=shear)
      prediction = predict_captcha(captcha, net)
      prediction = prediction[:4]
      if prediction not in dictionary:
          distances = sorted([(word, compute_distance(prediction, word))
                              for word in dictionary],
                             key=itemgetter(1))
          best_word = distances[0]
          prediction = best_word[0]
      return word == prediction, word, prediction
  ```


## 第九章 作者归属问题

### 9.1 为作品找作者

- 作者分析的目标是只根据作品内容找出作者独有的特点，作者分析包括以下问题：

  - 作者归属：从一组可能的作者中找到文档真正的主人。
  - 作者画像：根据作品界定作者的年龄、性别或其他特征。
  - 作者验证：根据作者已有作品，推断其他作品是否也是他写的。
  - 作者聚类：用聚类分析方法把作品按照作者进行分类。

- 作者归属问题中，已知一部分作者，训练集为多个作者的作品，目标是确定一组作者不详的作品是谁写的。如果作者恰好是已知作者，叫封闭问题。否则叫开放问题。

- 任何数据挖掘问题，若实际类别不在训练集中，则叫开放问题，要给出不属于任何已知类别的提示。

- 进行作者归属研究，要求：

  - 只能使用作品内容
  - 不考虑作品主题，关注单词用法、标点和其他文本特征。

- 文档中有很多噪音，比如作品前的声明文字，因此要删去这些噪音。

  ```python
  def clean_book(document):
      lines = document.split("\n")
      start= 0
      end = len(lines)
      for i in range(len(lines)):
          line = lines[i]
          if line.startswith("*** START OF THIS PROJECT GUTENBERG"):
              start = i + 1
          elif line.startswith("*** END OF THIS PROJECT GUTENBERG"):
              end = i - 1
      return "\n".join(lines[start:end])
  ```

- 将文档清理并保存到列表中。

  ```python
  def load_books_data(folder=data_folder):
      documents = []
      authors = []
      subfolders = [subfolder for subfolder in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, subfolder))]
      for author_number, subfolder in enumerate(subfolders):
          full_subfolder_path = os.path.join(folder, subfolder)
          for document_name in os.listdir(full_subfolder_path):
              with open(os.path.join(full_subfolder_path, document_name)) as inf:
                  documents.append(clean_book(inf.read()))
                  authors.append(author_number)
      return documents, np.array(authors, dtype='int')
  ```

- 如果数据集过大，无法一次加载到内存中，要每次从一篇或几篇文档中抽取特征，把特征保存在矩阵中。

### 9.2 功能词

- 功能词：指本身具有很少含义，却是组成句子必不可少的成分。如this和which。与功能词相对的是实词。

- 通常来讲，使用越频繁的单词，对于作者分析越能提供更多有价值的信息。

- 功能词的使用通常不是由文档内容而是由作者的使用习惯所决定的，因此可以用来区分作者归属，如美国人在意区分that和which，而澳大利亚人不在意。

- 功能词词汇表：

  ```python
  function_words = ["a", "able", "aboard", "about", "above", "absent",
                    "according" , "accordingly", "across", "after", "against",
                    "ahead", "albeit", "all", "along", "alongside", "although",
                    "am", "amid", "amidst", "among", "amongst", "amount", "an",
                      "and", "another", "anti", "any", "anybody", "anyone",
                      "anything", "are", "around", "as", "aside", "astraddle",
                      "astride", "at", "away", "bar", "barring", "be", "because",
                      "been", "before", "behind", "being", "below", "beneath",
                      "beside", "besides", "better", "between", "beyond", "bit",
                      "both", "but", "by", "can", "certain", "circa", "close",
                      "concerning", "consequently", "considering", "could",
                      "couple", "dare", "deal", "despite", "down", "due", "during",
                      "each", "eight", "eighth", "either", "enough", "every",
                      "everybody", "everyone", "everything", "except", "excepting",
                      "excluding", "failing", "few", "fewer", "fifth", "first",
                      "five", "following", "for", "four", "fourth", "from", "front",
                      "given", "good", "great", "had", "half", "have", "he",
                      "heaps", "hence", "her", "hers", "herself", "him", "himself",
                      "his", "however", "i", "if", "in", "including", "inside",
                      "instead", "into", "is", "it", "its", "itself", "keeping",
                      "lack", "less", "like", "little", "loads", "lots", "majority",
                      "many", "masses", "may", "me", "might", "mine", "minority",
                      "minus", "more", "most", "much", "must", "my", "myself",
                      "near", "need", "neither", "nevertheless", "next", "nine",
                      "ninth", "no", "nobody", "none", "nor", "nothing",
                      "notwithstanding", "number", "numbers", "of", "off", "on",
                      "once", "one", "onto", "opposite", "or", "other", "ought",
                      "our", "ours", "ourselves", "out", "outside", "over", "part",
                      "past", "pending", "per", "pertaining", "place", "plenty",
                      "plethora", "plus", "quantities", "quantity", "quarter",
                      "regarding", "remainder", "respecting", "rest", "round",
                      "save", "saving", "second", "seven", "seventh", "several",
                      "shall", "she", "should", "similar", "since", "six", "sixth",
                      "so", "some", "somebody", "someone", "something", "spite",
                      "such", "ten", "tenth", "than", "thanks", "that", "the",
                      "their", "theirs", "them", "themselves", "then", "thence",
                    "therefore", "these", "they", "third", "this", "those",
  "though", "three", "through", "throughout", "thru", "thus",
  "till", "time", "to", "tons", "top", "toward", "towards",
  "two", "under", "underneath", "unless", "unlike", "until",
  "unto", "up", "upon", "us", "used", "various", "versus",
  "via", "view", "wanting", "was", "we", "were", "what",
  "whatever", "when", "whenever", "where", "whereas",
  "wherever", "whether", "which", "whichever", "while",
                    "whilst", "who", "whoever", "whole", "whom", "whomever",
  "whose", "will", "with", "within", "without", "would", "yet",
  "you", "your", "yours", "yourself", "yourselves"]
  ```

- 使用`CountVectorizer`抽取词频特征。

  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  extractor = CountVectorizer(vocabulary=function_words)
  ```

- 设置支持向量机参数，创建分类器实例。高斯内核如rbf，只适用于特征数小于10000的情况。

  ```python
  from sklearn.svm import SVC
  parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
  svr = SVC()
  ```

- 使用网络搜索法寻找最优参数值。

  ```python
  from sklearn import grid_search
  grid = grid_search.GridSearchCV(svr, parameters)
  ```

- 组建流水线。

  ```python
  pipeline1 = Pipeline([('feature_extraction', extractor),
                        ('clf', grid)
                       ])
  ```

### 9.3 支持向量机

- 支持向量机是一种二类分类器。SVM要做的是找到最佳的一条线，分割开两个类别的数据，让各点到分割线之间的距离最大化，用它做预测。
- 对于多分类问题，就创建多个分类器，最简单的方法是分为1对多，即特定类和其他类。
- `from sklearn.svm import SVC`的参数：
  - C参数：与分类器正确分类的比例有关，过高可能过拟合，过小分类结果可能较差。
  - kernel参数：指定内核函数。如果数据线性不可分，则要加入伪特征将其置入高维空间，直到其线性可分。寻找最佳分割线时，需要计算个体之间的点积，使用点积函数可以创建新特征而无需实际定义这些特征。因此内核函数定义为数据集中两个个体函数的点积。内核有三种：线性内核，多项式内核，高斯内核。

### 9.4 字符N元语法

- N元语法由一系列N个为一组的对象组成，N通常为2到6之间的值。基于字符的N元语法在作者归属问题上效果很好。更常见的是基于单词的N元语法。

- N元语法的特征如`<e t>`是由e、空格和t组成的。

- 字符N元语法的特点是稀疏，但低于基于单词的N元语法。

- 抽取N元语法，analyzer指定了抽取字符，ngram_range指定N的范围，取同样长度的N元语法，则使用相同的值。

  ```python
  CountVectorizer(analyzer='char', ngram_range=(3, 3))
  ```

### 9.5 使用安然公司数据集

- 初始化邮件解析器。

  ```python
  from email.parser import Parser
  p = Parser()
  ```

- 为保证数据集相对平衡，设定发件人最少发件数和最大抽取邮件数。

- 打乱邮箱地址。因listdir每次获取的邮箱顺序不一定相同，所以先排序，再打乱。

  ```python
  email_addresses = sorted(os.listdir(data_folder))
  random_state.shuffle(email_addresses)
  ```

- 解析邮件，获取邮件内容。

  ```python
  contents = [p.parsestr(email)._payload for email in authored_emails]
  documents.extend(contents)
  ```

- 获取安然语料库函数如下：

  ```python
  def get_enron_corpus(num_authors=10, data_folder=data_folder,
                       min_docs_author=10, max_docs_author=100,
                       random_state=None):
      random_state = check_random_state(random_state)
      email_addresses = sorted(os.listdir(data_folder))
      random_state.shuffle(email_addresses)
      documents = []
      classes = []
      author_num = 0
      authors = {}
      for user in email_addresses:
          users_email_folder = os.path.join(data_folder, user)
          mail_folders = [os.path.join(users_email_folder, subfolder)
                          for subfolder in os.listdir(users_email_folder)
                          if "sent" in subfolder]
          try:
              authored_emails = [open(os.path.join(mail_folder, email_filename), encoding='cp1252').read()
                                 for mail_folder in mail_folders
                                 for email_filename in os.listdir(mail_folder)]
          except IsADirectoryError:
              continue
          if len(authored_emails) < min_docs_author:
              continue
          if len(authored_emails) > max_docs_author:
              authored_emails = authored_emails[:max_docs_author]
          contents = [p.parsestr(email)._payload for email in authored_emails]
          documents.extend(contents)
          classes.extend([author_num] * len(authored_emails))
          authors[user] = author_num
          author_num += 1
          if author_num >= num_authors or author_num >= len(email_addresses):
              break
      return documents, np.array(classes), authors
  ```

- 由于回复邮件时会带上别人之前邮件的内容，因此要进行处理。

- 使用`import quotequail`查找邮件中的新内容。

  ```python
  def remove_replies(email_contents):
      r = quotequail.unwrap(email_contents)
      if r is None:
          return email_contents
      if 'text_top' in r:
          return r['text_top']
      elif 'text' in r:
          return r['text']
      return email_contents
  ```

- 线上学习：使用新数据更新训练结果，但不是每次都重新进行训练。

- 输出最佳训练参数。

  ```python
  print(pipeline.named_steps['classifier'].best_params_)
  ```

- 创建混淆矩阵，获取发件人。

  ```python
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(y_pred, y_test)
  cm = cm / cm.astype(np.float).sum(axis=1)
  sorted_authors = sorted(authors.keys(), key=lambda x:authors[x])
  ```

## 第十章 新闻预料分类

### 10.1 获取新闻文章

- 已知目标类别的学习任务叫有监督学习，未知目标类别的学习任务叫无监督学习。

- 使用WEB API采集数据，如使用twitterAPI采集数据，有三个注意事项：

  - 授权方法：是数据提供方用来管理数据采集方的。
  - 采集频率：限制了采集方在约定时间内的最大请求数。
  - API端点：用来抽取信息的实际网址。

- 获取信息时发送HTTP GET请求到指定网址。服务器返回资源信息、信息类型和ID。

- 从reddit上创建script型应用，获得client ID和密钥。

- 设置唯一用户代理，避免与其他API重复，影响采集限制。`USER_AGENT = "python:<unique user agent> (by /u/<reddit username>)"`

- 登录获取令牌。

  ```python
  def login(username, password):
      if password is None:
          password = getpass.getpass("Enter reddit password for username {}:".format(username))
      headers = {"User-Agent": USER_AGENT}
      client_auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
      post_data = {"grant_type": "password", "username":username, "password":password}
      response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers)
      return response.json()
  ```

- 指定reddit栏目搜集信息。设置头部。获取返回的信息。

  ```python
  subreddit = "worldnews"
  url = "https://oauth.reddit.com/r/{}".format(subreddit)
  headers = {"Authorization": "bearer {}".format(token['access_token']), "User-Agent": USER_AGENT}
  response = requests.get(url,headers=headers)
  ```

- 输出每条广播的标题。

  ```python
  result = response.json()
  for story in result['data']['children']:
      print(story['data']['title'])
  ```

- 获取500条广播的标题、连接和喜欢数。因为每页最多是100条广播，因此要用游标，reddit的游标是after。

  ```python
  def get_links(subreddit, token, n_pages=5):
      stories = []
      after = None
      for page_number in range(n_pages):
          headers = {"Authorization": "bearer {}".format(token['access_token']), "User-Agent": USER_AGENT}
          url = "https://oauth.reddit.com/r/{}?limit=100".format(subreddit)
          if after:
              url += "&after={}".format(after)
          response = requests.get(url,headers=headers)
          result = response.json()
          after = result['data']['after']
          sleep(2)
          stories.extend([(story['data']['title'], story['data']['url'], story['data']['score']) for story in result['data']['children']])
      return stories
  ```

### 10.2 从任意网站抽取文本

- 使用中文系统中的txt文件默认编码为gbk，要改成utf8，否则大量英文信息无法正确编码。爬取连接信息时，要加上头部，以免被识别为爬虫而无法正常返回信息。reddit标题不唯一，因此使用md5获取散列值作为文件名，md5在小规模数据中是可靠的。

  ```python
  import os
  import hashlib
  import codecs
  data_floder = "raw/"
  number_errors = 0
  for title, url, score in stories:
      output_filename = hashlib.md5(url.encode()).hexdigest()
      fullpath = os.path.join(data_floder, output_filename + ".txt")
      headers = {
      'User-Agent': 'Mozilla/4.0(compatible; MSIE 5.5; Windows NT)'
      }
      try:
          response = requests.get(url, headers=headers)
          data = response.text
          with codecs.open(fullpath, 'w', 'utf8') as outf:
              outf.write(data)
      except Exception as e:
          number_errors += 1
          print(e)
  ```

- 使用lxml解析HTML文件，lxml的HTML解析器容错能力强，可以处理不规范的HTML代码。

- 文本抽取分三步：

  - 遍历HTML文件的每个节点，抽取其中的文本内容。
  - 跳过JavaScript、样式和注释节点。
  - 确保文本内容长度至少为100个字符。

- 遍历解析树，拼接获取文本。

  ```python
  def get_text_from_node(node):
      if len(node) == 0:
          if node.text and len(node.text) > 100:
              return node.text
          else:
              return ""
      else:
          results = (get_text_from_node(child) for child in node if child.tag not in skip_node_types)
          return "\n".join(r for r in results if len(r) > 1)
  def get_text_from_file(filename):
      with codecs.open(filename, encoding='utf8') as inf:
          html_tree = etree.parse(inf, etree.HTMLParser())
      return get_text_from_node(html_tree.getroot())
  for filename in os.listdir(data_floder):
      text = get_text_from_file(os.path.join(data_floder, filename))
      with codecs.open(os.path.join(text_output_folder, filename), 'w', 'utf8') as outf:
          outf.write(text)
  ```

### 10.3 新闻预料分类

- 聚类算法在学习时没有明确的方向性，根据目标函数而不是数据潜在的含义学习。因此聚类算法选择效果好的特征很重要。有监督学习中，算法会自动降低对分类作用不大的特征的权重，而聚类会综合所有特征给出最后结果。

- k-means聚类算法迭代寻找能够代表数据的聚类质心点。算法开始时使用从训练数据中随机选取的k个数据点作为质心。在迭代一定次数后，质心移动量很小时，可以终止算法的运行。步骤如下：

  - 为每一个数据点分配簇标签，标签根据与各质心的距离选取。
  - 计算各簇内所有数据点均值，更新各簇的质心点。

- `from sklearn.cluster import KMeans`使用kmeans聚类算法。

- `from sklearn.feature_extraction.text import TfidfVectorizer`引入抽取tf-idf特征的向量化工具。

- 封装流水线，设定max_df=0.4忽略在40%以上文档中出现过的词语。

  ```python
  from sklearn.pipeline import Pipeline
  n_clusters = 10
  pipeline = Pipeline([('feature_extraction', TfidfVectorizer(max_df=0.4)),
                       ('clusterer', KMeans(n_clusters=n_clusters))
                       ])
  ```

- 使用Counter函数计算每类数据点个数。

  ```python
  from collections import Counter
  c = Counter(labels)
  for cluster_number in range(n_clusters):
      print("Cluster {} contains {} samples".format(cluster_number, c[cluster_number]))
  ```

- 聚类算法是探索性算法，很难评估算法结果的好坏，评估最直接的方式是根据其学习的标准进行评价。

- 计算kmeans算法的惯性权重即每个数据点到最近质心点的距离，这个值本身没有意义，但可以用来判断分多少簇合适。

- 对于每个簇数计算30次。

  ```
  inertia_scores = []
  n_cluster_values = list(range(2, 20))
  for n_clusters in n_cluster_values:
      cur_inertia_scores = []
      X = TfidfVectorizer(max_df=0.4).fit_transform(documents)
      for i in range(30):
          km = KMeans(n_clusters=n_clusters).fit(X)
          cur_inertia_scores.append(km.inertia_)
      inertia_scores.append(cur_inertia_scores)
  inertia_scores = np.array(inertia_scores)
  ```

- 计算均值和标准差，画出图像。

  ```python
  %matplotlib inline
  from matplotlib import pyplot as plt
  
  inertia_means = np.mean(inertia_scores, axis=1)
  inertia_stderr = np.std(inertia_scores, axis=1)
  
  fig = plt.figure(figsize=(40,20))
  plt.errorbar(n_cluster_values, inertia_means, inertia_stderr, color='green')
  plt.show()
  ```

- 随着簇增加，惯性权重逐渐减少，但当簇数为kt时，惯性权重最后进行了一次大的调整，如同图像的肘部，称为拐点。有的数据集拐点明显，有的数据集则没有拐点。

- 从质心找出特征值最大的5个特征。

  ```python
  print("  Most important terms")
      centroid = pipeline.named_steps['clusterer'].cluster_centers_[cluster_number]
      most_important = centroid.argsort()
      for i in range(5):
          term_index = most_important[-(i+1)]
          print("  {0}) {1} (score: {2:.4f})".format(i+1, terms[term_index], centroid[term_index]))
  ```

- k聚类算法可用来简化特征，其他特征简化方法如主成分分析、潜在语义索引的计算要求很高。使用数据点到质心点的距离作为特征，来简化特征。
- 简化特征后可以进行二次聚类。
- 分类时也可使用聚类来简化特征：
  - 使用标注好的数据选取特征
  - 用聚类方法简化特征
  - 用分类算法对前面处理好的数据分类

### 10.4 聚类融合

- 聚类融合后的算法能够平滑算法多次运行得到的不同结果，也可以减少参数选择对于最终结果的影响。

- 证据累积算法：对数据多次聚类，每次都记录各个数据点的簇标签，计算每两个数据点被分到同一个簇的次数。步骤如下：

  - 使用kmeans等低水平聚类算法对数据集进行多次聚类，记录每一次迭代两个数据点出现在同一簇的频率，将结果保存到共协矩阵。
  - 使用分级聚类对第一步得到的共协矩阵进行聚类分析。分级聚类等价于找到一棵把所有节点连接到一起的树，并把权重低的边去掉。

- `from scipy.sparse import csr_matrix`使用scipy的稀疏矩阵csr_matrix。稀疏矩阵由一系列记录非零值位置的列表组成。

- 创建共协矩阵。

  ```python
  def create_coassociation_matrix(labels):
      rows = []
      cols = []
      unique_labels = set(labels)
      for label in unique_labels:
          indices = np.where(labels == label)[0]
          for index1 in indices:
              for index2 in indices:
                  rows.append(index1)
                  cols.append(index2)
      data = np.ones((len(rows),))
      return csr_matrix((data, (rows, cols)), dtype='float')
  ```

- 分级聚类即找到该矩阵的最小生成树，删除权重低于阈值的边。

- 生成树是所有节点都连接到一起的树。

- 最小生成树是总权重最低的生成树。

- 图中的节点是数据集中的个体，边是被分到同一簇的次数即共协矩阵的值。

- `from scipy.sparse.csgraph import minimum_spanning_tree`使用scipy中的minimum_spanning_tree计算最小生成树。`mst = minimum_spanning_tree(C)`，函数输入为距离，因此要取反。

- 再次遍历，得到第二次聚类的共协矩阵，删除不是两个共协矩阵中都出现的边。

  ```python
  pipeline = Pipeline([('feature_extraction', TfidfVectorizer(max_df=0.4)),
                       ('clusterer', KMeans(n_clusters=3))
                       ])
  pipeline.fit(documents)
  labels2 = pipeline.predict(documents)
  C2 = create_coassociation_matrix(labels2)
  C_sum = (C + C2) / 2
  C_sum.todense()
  mst = minimum_spanning_tree(-C_sum)
  mst.data[mst.data > -1] = 0
  mst.eliminate_zeros()
  ```

- 找到所有连通分支。

  ```python
  from scipy.sparse.csgraph import connected_components
  number_of_clusters, labels = connected_components(mst)
  ```

- kmeans算法假定所有特征取值范围相同，找的是圆形簇。当簇不是圆形的时，用kmeans聚类有难度。

- 证据累积算法把特征重新映射到新空间，证据累积算法只关心数据点之间的距离而不是原先在特征空间的位置。但仍需进行数据规范化。

- 指定n_clusterings次聚类进行融合，删除边的阈值为cut_threshold，每次聚类簇的范围为n_clusters_range。

  ```python
  from sklearn.base import BaseEstimator, ClusterMixin
  
  class EAC(BaseEstimator, ClusterMixin):
      def __init__(self, n_clusterings=10, cut_threshold=0.5, n_clusters_range=(3, 10)):
          self.n_clusterings = n_clusterings
          self.cut_threshold = cut_threshold
          self.n_clusters_range = n_clusters_range
      
      def fit(self, X, y=None):
          C = sum((create_coassociation_matrix(self._single_clustering(X))
                   for i in range(self.n_clusterings)))
          mst = minimum_spanning_tree(-C)
          mst.data[mst.data > -self.cut_threshold] = 0
          mst.eliminate_zeros()
          self.n_components, self.labels_ = connected_components(mst)
          return self
      
      def _single_clustering(self, X):
          n_clusters = np.random.randint(*self.n_clusters_range)
          km = KMeans(n_clusters=n_clusters)
          return km.fit_predict(X)
      
      def fit_predict(self, X):
          self.fit(X)
          return self.labels_
  ```

- 组成流水线。

  ```python
  pipeline = Pipeline([('feature_extraction', TfidfVectorizer(max_df=0.4)),
                       ('clusterer', EAC())
                       ])
  ```

### 10.5 线上学习

- 当没有足够数据用来训练，或内存不能一次装下所有数据，或完成预测后得到了新的数据，此时可以使用线上学习。

- 线上学习是指用新数据增量地改进模型。神经网络是支持线上学习的标准例子。

- 神经网络也支持使用批模式进行训练，每次只使用一组数据进行训练，运行速度快但耗内存多。

- 线上学习与流式学习有关，不同点在于：线上学习能重新评估先前创建模型时所用的数据，但后者的数据只能用一次。

- `from sklearn.cluster import MiniBatchKMeans`支持线上学习，实现了partial_fit函数进行线上学习，而fit函数则会删除之前的训练结果。

  ```python
  mbkm = MiniBatchKMeans(random_state=14, n_clusters=3)
  batch_size = 500
  
  indices = np.arange(0, X.shape[0])
  for iteration in range(100):
      sample = np.random.choice(indices, size=batch_size, replace=True)
      mbkm.partial_fit(X[sample[:batch_size]])
  ```

- 由于TfidfVectorizer不是线上学习算法，所以改用`from sklearn.feature_extraction.text import HashingVectorizer`，使用散列值代替特征名称，记录词袋模型。

- 创建支持线上学习的pipeline类。

  ```python
  class PartialFitPipeline(Pipeline):
      def partial_fit(self, X, y=None):
          Xt = X
          for name, transform in self.steps[:-1]:
              Xt = transform.transform(Xt)
          return self.steps[-1][1].partial_fit(Xt, y=y)
  ```

- 组装成流水线。

  ```python
  pipeline = PartialFitPipeline([('feature_extraction', HashingVectorizer()),
                               ('clusterer', MiniBatchKMeans(random_state=14, n_clusters=3))
                               ])
  ```

- 用批模式训练。

  ```python
  batch_size = 10
  for iteration in range(int(len(documents) / batch_size)):
      start = batch_size * iteration
      end = batch_size * (iteration + 1)
      pipeline.partial_fit(documents[start:end])
  ```


## 第十一章 用深度学习方法为图像中的物体进行分类

### 11.1 应用场景和目标

- 使用CIFAR-10数据集进行训练，所用图像均为numpy数组。

- 图像数据格式为pickle，pickle是保存图形对象的一个库，调用`pickle.load`读取数据。编码设置为Latin，防止不同版本python导致的编码错误。

  ```python
  import pickle
  def unpickle(filename):
      with open(filename, 'rb') as fo:
          return pickle.load(fo, encoding='latin1')
  ```

- 将列表数据转换成能用matplotlib绘制的图像，并旋转图片。

  ```python
  image = image.reshape((32,32, 3), order='F')
  import numpy as np
  image = np.rot90(image, -1)
  ```

### 11.2 深度神经网络

- 至少包含两层隐含层的神经网络被称为深度神经网络。更巧妙的算法能减少实际需要的层数。

- 神经网络接收很基础的特征作为输入，就计算机视觉而言，输入为简单的像素值。经过神经网络，基础的特征组合成复杂的特征。

- 一个神经网络可以用一组矩阵表示，每层增加一个偏置项，永远激活并与下一层的每个神经元都有连接。

- Theano是用来创建和运行数学表达式的工具。和SQL相似，在Theano中只需定义要做什么要不是怎么做。

- Theano用来定义函数，处理标量、数组和矩阵及其他数学表达式。

- 引入张量。定义两个标量数值型输入。构成表达式。定义计算表达式的函数。要注意theano和numpy包的兼容。

  ```python
  import theano
  from theano import tensor as T
  a = T.dscalar()
  b = T.dscalar()
  c = T.sqrt(a ** 2 + b ** 2)
  f = theano.function([a,b], c)
  ```

- Lasagne库基于Theano库，专门用来构建神经网络，使用Theano 进行计算。实现了几种比较新的神经网络层和组成这些层的模块：

  - 内置网络层：这些小神经网络比传统神经网络更容易解释。
  - 删除层：训练过程随机删除神经元，防止产生过拟合问题。
  - 噪音层：为神经元引入噪音，防止过拟合。

- 卷积层使用少量相互连接的神经元，分析有一部分输入值，便于神经网络实现对数据的标准转换。

- 传统神经网络一层所有神经元全都连接到下一层所有神经元。

- 池化层接收某个区域最大输出值，可以降低图像中的微小变动带来的噪音，减少信息量，减少后续各层的工作量。

- Lasagene对数据类型有要求，将数据类型转为32位。

  ```python
  from sklearn.datasets import load_iris
  iris = load_iris()
  X = iris.data.astype(np.float32)
  y_true = iris.target.astype(np.int32)
  ```

- 创建输入层，指定每一批输入数量为10，神经元数量和特征数量相同。

  ```python
  input_layer = lasagne.layers.InputLayer(shape=X_train.shape, input_var=input_val)
  ```

- 创建隐含层，从输入层接收输入，指定神经元数量，使用非线性sigmoid函数。

  ```python
  hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=12, nonlinearity=lasagne.nonlinearities.sigmoid)
  ```

- 创建输出层，输出层共三个神经元与类别数一致，使用非线性softmax函数。

  ```python
  output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)
  ```

- Lasagne中，输入数据先提交到输出层，再向上回溯，直到输入层，将数据交给输入层处理。

- 定义输入、输出、目标数据变量。

  ```python
  import lasagne
  input_val = T.fmatrix("inputs")
  target_val = T.ivector("targets")
  output_val = lasagne.layers.get_output(output_layer)
  ```

- 定义损失函数，训练神经网络时以最小化损失函数为前提。使用交叉熵表示损失，这是一种衡量分类数据分类效果好坏的标准。损失函数表示实际网络输出与期望输出之间的差距。

  ```python
  loss = lasagne.objectives.categorical_crossentropy(output_val, target_val)
  loss = loss.mean()
  ```

- 获取所有参数，调整网络权重，使损失降到最小。

  ```python
  all_params = lasagne.layers.get_all_params(output_layer, trainable=True)
  updates = lasagne.updates.sgd(loss, all_params, learning_rate=0.1)
  ```

- 定义训练函数和获取输出的函数。

  ```python
  train = theano.function([input_val, target_val], loss, updates=updates, allow_input_downcast=True)
  get_output = theano.function([input_val], output_val)
  ```

- 进行1000次迭代，逐渐改进神经网络。

  ```python
  for n in range(1000):
      train(X_train, y_train)
  ```

- 获取测试集的输出结果及各神经元激励作用的大小，找到激励作用最大的神经元，得到预测结果。

  ```python
  y_output = get_output(X_test)
  y_pred = np.argmax(y_output, axis=1)
  from sklearn.metrics import f1_score
  print(f1_score(y_test, y_pred, average='micro'))
  ```

- nolearn对Lasagne实现了封装，可读性更强，更易管理。

- 创建由输入层、密集隐含层和密集输出层组成的层级结构。

  ```python
  from lasagne import layers
  layers=[
      ('input', layers.InputLayer),
      ('hidden', layers.DenseLayer),
      ('output', layers.DenseLayer),
  ]
  ```

- 定义神经网络，输入神经网络参数，定义非线性函数，指定偏置神经元。偏置神经元激活后可以对问题做更有针对性的训练，以消除训练中的偏差。定义神经网络训练方式，这里使用低冲量值和高学习速率。将分类问题定义为回归问题，因为输出是数值，所以定义为回归问题更好。最大训练步数设为1000。

  ```python
  net1 = NeuralNet(layers=layers,
                  input_shape=X.shape,
                  hidden_num_units=100,
                  output_num_units=26,
                  hidden_nonlinearity=sigmoid,
                   output_nonlinearity=softmax,
                   hidden_b=np.zeros((100,), dtype=np.float64),
                   update=updates.momentum,
                   update_learning_rate=0.9,
                   update_momentum=0.1,
                   regression=True,
                   max_epochs=1000,
                  )
  ```

- 在训练集上训练网络。

  ```python
  net1.fit(X_train, y_train)
  ```

- 评估训练得到的网络。

  ```python
  y_pred = net1.predict(X_test)
  y_pred = y_pred.argmax(axis=1)
  assert len(y_pred) == len(X_test)
  if len(y_test.shape) > 1:
      y_test = y_test.argmax(axis=1)
  print(f1_score(y_test, y_pred, average='macro'))
  ```

### 11.3 GPU优化

- 使用稀疏矩阵可以将整个神经网络装进内存。
- 神经网络最核心的计算类型是浮点运算，矩阵操作的大量运算可以并行处理。GPU拥有成千上万个小核，适合并行任务，CPU单核工作速度更快，访问内存效率更高，适合序列化任务。所以用GPU进行计算能够提升训练速度。

### 11.4 应用

- 保留像素结构即行列号，把所有批次图像文件名存储到列表中。

  ```python
  import numpy as np
  batches = []
  for i in range(1, 6):
      batch_filename = os.path.join(data_folder, "data_batch_{}".format(i))
      batches.append(unpickle(batch1_filename))
  ```

- 纵向添加每批次数据。

  ```python
  X = np.vstack([batch['data'] for batch in batches])
  ```

- 像素值归一化，并转化为32位浮点数据。

  ```python
  X = np.array(X) / X.max()
  X = X.astype(np.float32)
  ```

- 纵向添加标签数据，转化为一位有效码。

  ```python
  from sklearn.preprocessing import OneHotEncoder
  y = np.hstack(batch['labels'] for batch in batches).flatten()
  y = OneHotEncoder().fit_transform(y.reshape(y.shape[0],1)).todense()
  y = y.astype(np.float32)
  ```

- 划分训练集、测试集，调整数组形状以保留原始图像的数据结构。

  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
  X_train = X_train.reshape(-1, 3, 32, 32)
  X_test = X_test.reshape(-1, 3, 32, 32)
  ```

- 创建神经网络各层。输入层数据与数据集同型。

  ```python
  from lasagne import layers
  layers=[
          ('input', layers.InputLayer),
          ('conv1', layers.Conv2DLayer),
          ('pool1', layers.MaxPool2DLayer),
          ('conv2', layers.Conv2DLayer),
          ('pool2', layers.MaxPool2DLayer),
          ('conv3', layers.Conv2DLayer),
          ('pool3', layers.MaxPool2DLayer),
          ('hidden4', layers.DenseLayer),
          ('hidden5', layers.DenseLayer),
          ('output', layers.DenseLayer),
          ]
  ```

- 创建神经网络。指定输入数据形状，和数据集形状一致，None表示每次使用默认数量图像数据进行训练。设置卷积层大小及卷积窗口大小。设置池化窗口大小。设置隐含层和输出层大小，输出层大小和类别数量一致。输出层设置非线性函数softmax。设置学习速率和冲量，随着数据量的增加，学习速率应下降。分类问题转换为回归问题。训练步数设置为3以便测试。设置verbose为1，每步输出结果，以便了解模型训练进度。

  ```python
  from nolearn.lasagne import NeuralNet
  from lasagne.nonlinearities import sigmoid, softmax
  nnet = NeuralNet(layers=layers,
                   input_shape=(None, 3, 32, 32),
                   conv1_num_filters=32,
                   conv1_filter_size=(3, 3),
                   conv2_num_filters=64,
                   conv2_filter_size=(2, 2),
                   conv3_num_filters=128,
                   conv3_filter_size=(2, 2),
                   pool1_pool_size=(2,2),
                   pool2_pool_size=(2,2),
                   pool3_pool_size=(2,2),
                   hidden4_num_units=500,
                   hidden5_num_units=500,
                   output_num_units=10,
                   output_nonlinearity=softmax,
                   update_learning_rate=0.01,
                   update_momentum=0.9,
                   regression=True,
                   max_epochs=3,
                   verbose=1)
  ```

- 训练神经网络，进行测试。

  ```python
  nnet.fit(X_train, y_train)
  from sklearn.metrics import f1_score
  y_pred = nnet.predict(X_test)
  print(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='micro'))
  ```

## 第十二章 大数据处理

### 12.1 大数据

- 大数据的特点：
  - 海量：数据量大
  - 高速：数据分析速度快
  - 多样：数据集有多种形式
  - 准确：很难确定采集到的数据是否准确。
- 大数据无法加载到内存中。

### 12.2 大数据应用场景和目标

- 应用场景：
  - 搜索引擎
  - 科学实验
  - 政府数据处理
  - 交通管理
  - 改善客户体验，降低支出
  - 提高公司经营管理的自动化程度，改善产品和服务质量
  - 监测网络流量，寻找大型网络的恶意软件感染。

### 12.3 MapReduce

- 谷歌出于并行计算的需要，提出了MapReduce模型，引入了容错和可伸缩特性，可用于任意大数据集的一般性计算任务。

- MapReduce主要分为映射（Map）和规约（Reduce）两步。

- MapReduce范式还包括排序和合并两步。

- 映射这一步，接收一个函数，用这个函数处理列表中的各个元素，返回和之间列表长度相等的列表，新列表的元素为函数的返回结果。

- 建立sum函数与a之间的映射关系。sums是生成器，在调用前不会计算。

  ```python
  a = [[1,2,1], [3,2], [4,9,1,0,2]]
  sums = map(sum, a)
  # 等效为：
  sums = [] 
  for sublist in a: 
      results = sum(sublist) 
      sums.append(results)
  ```

- 规约需要对返回结果的每一个元素应用一个函数，从初始值开始，对初始值和第一个应用指定函数，得到返回结果，然后再对所得到的结果和下一个值应用指定函数，以此类推。规约函数为`from functools import reduce`三个参数分别为函数的名字，列表和初始值。

  ```python
  def add(a, b):
      return a + b 
  from functools import reduce
  print(reduce(add, sums, 0))
  # 等价于：
  initial = 0
  current_result = initial
  for element in sums:
      current_result = add(current_result, element)
  ```

- 为了实现分布式计算，可以在映射这一步把各个二级列表及函数说明分发到不同的计算机上。计算完成后，各计算机把结果返回主计算机。然后主计算机把结果发送给另一台计算机做规约。大大节省了存储空间。

- 映射函数接收一键值对，返回键值对列表。如接收文档编号文本内容键值对，返回单词词频键值对。

  ```python
  def map_word_count(document_id, document):
      counts = defaultdict(int)
      for word in document.split():
          counts[word] += 1
      for word in counts:
          yield (word, counts[word])
  ```

- 把每个键所有值聚集到一起。

  ```python
  def shuffle_words(results_generators):
      records = defaultdict(list)
      for results in results_generators:
          for word, count in results:
              records[word].append(count)
      for word in records:
          yield (word, records[word])
  ```

- 规约接收一键值对，返回另一键值对。

  ```python
  def reduce_counts(word, list_of_counts):
      return (word, sum(list_of_counts))
  ```

- 获取sklearn的20个新闻语料。

  ```python
  from sklearn.datasets import fetch_20newsgroups
  dataset = fetch_20newsgroups(subset='train')
  documents = dataset.data[:50]
  ```

- 执行映射操作，得到能输出键值对（单词、词频）的生成器。执行shuffle操作，生成单词和该单词在各文档出现次数的列表两项。规约，输出单词和单词在所有文档中的词频。

  ```python
  map_results = map(map_word_count, range(len(documents)), documents)
  shuffle_results = shuffle_words(map_results)
  reduce_results = [reduce_counts(word, list_of_counts) for word, list_of_counts in shuffle_results]
  ```

- Hadoop是一组包括MapReduce在内的开源工具。主要组件为Hadoop MapReduce。其他处理大数据的工具有如下几种：
  - Hadoop分布式文件系统（HDFS）：该文件系统可以将文件保存到多台计算机上，防范硬件故障，提高带宽。
  - YARN：用于调度应用和管理计算机集群。
  - Pig：用于MapReduce的高级语言。
  - Hive：用于管理数据仓库和进行查询。
  - HBase：对谷歌分布式数据库BigTable的一种实现。

### 12.4 应用

- 根据博主用词习惯判断博主性别。

- MapReduce常用映射对列表中的每一篇文档运行预测模型，使用规约来调整预测结果列表，以便把结果和原文档对应起来。

- 测试打开并读取博客内容。设置是否在博客中的标记，找到博客开始标签`<post>`后，将标记设置为True。找到关闭标签`</post>`后，将标记值设置为False。

  ```python
  all_posts = []
  with codecs.open(filename, encoding='utf8') as inf:
      # remove leading and trailing whitespace
      post_start = False
      post = []
      for line in inf:
          line = line.strip()
          if line == "<post>":
              post_start = True
          elif line == "</post>":
              post_start = False
              all_posts.append("\n".join(post))
              post = []
          elif post_start:
              post.append(line)
  ```

- 使用映射规约任务包mrjob。提供了大部分MapReduce任务所需的标准功能，既能在没有安装Hadoop的本地计算机上进行测试，也能在Hadoop服务器上测试。

- 创建MRJob的子类从文件中抽取博客内容。映射函数处理每一行，从文件取一行作为输入，最后生成一篇博客的所有内容，每一行都来自同一任务所在处理的文件。获取以环境变量存储的文件名。获取文件名中的性别信息。使用yield生成器表示博主性别和博客内容，便于mrlib跟踪输出。获取所有以51开始的文件，进行测试。

  ```python
  import os
  import re
  from mrjob.job import MRJob
  word_search_re = re.compile(r"[\w']+")
  class ExtractPosts(MRJob):
      post_start = False
      post = []
      def mapper(self, key, line):
          filename = os.environ["map_input_file"]
          gender = filename.split(".")[1]
          try:
              docnum = int(filename[0])
          except:
              docnum = 8
          if re.match(r"file://blogs\\51.*",filename):
              # remove leading and trailing whitespace
              line = line.strip()
              if line == "<post>":
                  self.post_start = True
              elif line == "</post>":
                  self.post_start = False
                  yield gender, repr("\n".join(self.post))
                  self.post = []
              elif self.post_start:
                  self.post.append(line)
  ```

- 执行MapReduce任务。`python .\extract_posts.py blogs/51* --output-dir=51blogs/blogposts`

  ```python
  if __name__ == '__main__':
      ExtractPosts.run()
  ```

- `from mrjob.step import MRStep`用MRStep管理MapReduce中的每一步操作。任务分为三步：映射、规约、再映射和规约。

- `word_search_re = re.compile(r"[\w']+")`创建用于匹配单词的正则表达式，并对其进行编译，用来查找单词的边界。

- 创建新类，用于训练朴素贝叶斯分类器。

  ```python
  class NaiveBayesTrainer(MRJob):
      # 定义MapReduce任务的各个步骤：第一步抽取单词出现的频率，第二步比较一个单词在男女博主所写博客中出现的概率，旋转较大的作为分类结果，写入输出文件。每一步中定义映射和规约函数。
      def steps(self):
          return [
              MRStep(mapper=self.extract_words_mapping,
                     reducer=self.reducer_count_words),
              MRStep(reducer=self.compare_words_reducer),
              ]
  # 接收一条博客数据，获取里面所有单词，返回1. / len(all_words)，以便后续求词频，输出博主性别。使用eval将字符串转换为列表，但不安全，建议用json。
      def extract_words_mapping(self, key, value):
          tokens = value.split()
          gender = eval(tokens[0])
          blog_post = eval(" ".join(tokens[1:]))
          all_words = word_search_re.findall(blog_post)
          all_words = [word.lower() for word in all_words]
          #for word in all_words:
          for word in all_words:
              #yield "{0}:{1}".format(gender, word.lower()), 1
              #yield (gender, word.lower()), (1. / len(all_words))
              # Occurence probability
              yield (gender, word), 1. / len(all_words)
  # 汇总每个性别使用每个单词的频率，把键改为单词。
      def reducer_count_words(self, key, counts):
          s = sum(counts)
          gender, word = key #.split(":")
          yield word, (gender, s)
  # 数据将作为一致性映射类型直接传入规约函数中，规约函数会将每个单词在所有文章中的出现频率按照性别汇集到一起，输出单词和词频字典。
      def compare_words_reducer(self, word, values):
          per_gender = {}
          for value in values:
              gender, s = value
              per_gender[gender] = s
          yield word, per_gender
  
      def ratio_mapper(self, word, value):
          counts = dict(value)
          sum_of_counts = float(np.mean(counts.values()))
          maximum_score = max(counts.items(), key=itemgetter(1))
          current_ratio = maximum_score[1] / sum_of_counts
          yield None, (word, sum_of_counts, value)
      
      def sorter_reducer(self, key, values):
          ranked_list = sorted(values, key=itemgetter(1), reverse=True)
          n_printed = 0
          for word, sum_of_counts, scores in ranked_list:
              if n_printed < 20:
                  print((n_printed + 1), word, scores)
                  n_printed += 1
              yield word, dict(scores)
  ```

- 运行代码，训练朴素贝叶斯模型。`python .\nb_train.py 51blogs/blogposts/ --output-dir=models/`

  ```python
  if __name__ == '__main__':
      NaiveBayesTrainer.run()
  ```

- 用命令`cat * > model.txt`将数据文件内容追加到model.txt中。

- 重新定义查找单词的正则表达式。`word_search_re = re.compile(r"[\w']+")`

- 声明从指定文件名加载模型的函数。模型是一个值为字典的字典。将模型的每一行分为两部分，用eval函数获得实际的值，它们之前是用repr函数存储的。

  ```python
  def load_model(model_filename):
      model = defaultdict(lambda: defaultdict(float))
      with open(model_filename) as inf:
          for line in inf:
              word, values = line.split(maxsplit=1)
              word = eval(word)
              values = eval(values)
              model[word] = values
      return model
  ```

- 加载实际的模型。中文系统要注意另存为utf8。

  ```python
  model_filename = os.path.join("models", "model.txt")
  model = load_model(model_filename)
  ```

- 创建使用模型做预测的函数。使用log防止下溢，对于模型中不存在的词，给出默认概率1e-5。

  ```python
  def nb_predict(model, document):
      words = word_search_re.findall(document)
      probabilities = defaultdict(lambda : 0)
      for word in set(words):
          probabilities["male"] += np.log(model[word].get("male", 1e-5))
          probabilities["female"] += np.log(model[word].get("female", 1e-5))
      # Now find the most likely gender
      most_likely_genders = sorted(probabilities.items(), key=itemgetter(1), reverse=True)
      return most_likely_genders[0][0]
  ```



