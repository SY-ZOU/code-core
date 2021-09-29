## 计算两个矩阵之间的欧式距离

> 需要计算测试集中每一点到训练集中每一点的欧氏距离，即需要求得两矩阵之间的欧氏距离。
>
> 实现k-NN算法时通常有三种方案，分别是使用两层循环，使用一层循环和不使用循环。

### 1. 使用两层循环

```python
num_test = X.shape[0]
num_train = X_train.shape[0]
dists = np.zeros((num_test, num_train)) 
for i in range(num_test):
  for j in range(num_train):
    dists[i][j] = np.sqrt(np.sum(np.square(X[i] - X_train[j])))
    return dists
```

### 2. 使用一层循环

> 使用矩阵表示训练集的数据，计算测试集中到训练集矩阵的距离，可以对算法优化为只使用一层循环。

```python
def compute_distances_one_loop(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis = 1))
    return dists
```

### 3. 不使用循环

> 使用矩阵运算的方法替代之前的循环操作。但此操作需要我们对矩阵的运算规则非常熟悉。
>
> <img width=350 style="float:left;" src="http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcT26YLfJ4YBHE7mcarQBQZRgXTno16v76u*NlHd.P1ABertbAmVPKfLG1cWOh8.1KR7*XljPsXb6S0kQP4TfQtY!/b&bo=8AMkAQAAAAADF.Q!&rf=viewer_4"/>
>
> <img width=490 style="float:left;" src="http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcT26YLfJ4YBHE7mcarQBQZST6.H6HvxKBEeGUgxe1Vaw6gftSlOUu*fci32EXLhvDdEU2Od82BE50lxKqwmPKTM!/b&bo=ggXcAAAAAAADF2k!&rf=viewer_4"/>
>
> <img width=490 style="float:left;" src="http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcT26YLfJ4YBHE7mcarQBQZRwauJ3LKYIQsgRv.KgPakQ*J*ONSKdkcSZb1qJmfhGmc4iIrdhXdBbsBwxCesBX8k!/b&bo=lAVCAQAAAAADF.A!&rf=viewer_4"/>

```python
def compute_distances_no_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    dists = np.sqrt(-2*np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis = 1) + np.transpose([np.sum(np.square(X), axis = 1)]))
    return dists
```

> pytorch实现

```python
def knn(x,y,k):
    inner = -2*torch.matmul(x.transpose(2, 1), y) # -2*[B,N,D]x[B,D,M]=-2*[B,N,M]
    xx = torch.sum(x**2, dim=1, keepdim=True)
    yy = torch.sum(y**2, dim=1, keepdim=True).transpose(2, 1)
    pairwise_distance = -(xx + inner + yy) #[B,N,M]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx
```

