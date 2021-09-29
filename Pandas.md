# 概述

Df.query('x<2.0&y>1.0)

Pd.todatetime(data['time],.....)

1. 主成分分析
2. 层次分析
3. K均值
4. 三次样条插值
5. 时间序列
6. 神经网络
7. 灰色预测
8. 遗传算法

对于数据类题目，一般的处理方式如下。

1. 预处理，包括缺失值的处理和数据的规范化；
2. PCA，AHP来整合不同指标的影响权重；
3. 构建核心模型 ；
4. 采用时间序列等预测模型预测今后的表现。

从近三年的题目来看，连续三年都运用了数据处理的一些常规方法，不难预测，19年C题肯定如此。

大家可以练习例如：层次分析，主成分分析，模糊综合评判等常规的数据处理方式，同时也要熟悉例如：遗传算法，模拟退火等常见的寻优算法及预测算法。

最后，祝 2019 年选择 C题的同学们都获得理想成绩！

另外，听说点右下角好看的同学都变好看了哟！



作者：弈笙0427
链接：https://www.jianshu.com/p/632cc9a03dd5
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



##一、处理的数据

- 与 SQL 或 Excel 表类似的，含异构列的**表格数据**，有序和无序（非固定频率）的**时间序列数据**，带**行列标签的矩阵数据**，包括同构或异构型数据，**任意其它形式的观测、统计数据集**, 数据转入 Pandas 数据结构时不必事先标记
- **解决了numpy只能处理数值信息**

## 二、特点

- Pandas 所有**数据结构的值都是可变的，但数据结构的大小并非都是可变的**，比如，**Series 的长度不可改变，但 DataFrame 里就可以插入列。**
- Pandas 里，绝大多数方法都**不改变原始的输入数据，而是复制数据，生成新的对象。 **一般来说，原始输入数据**不变**更稳妥。很多函数改变结构需要保存在另一个值。

## 三、迭代

- **不要尝试在迭代时修改任何对象。迭代是用于读取，迭代器返回原始对象(视图)的副本，因此更改将不会反映在原始对象上。**

### 1. Series迭代

```python
for i in s:
   print (i)
```

###2. DataFrame迭代

#### （1）访问列

```python
for col in df:
   print (col)
```

#### （2）访问行

- `iteritems()` - 迭代`(key，value)`对。访问每一列。

```python
df = pd.DataFrame(np.random.randn(4,3),columns=['col1','col2','col3'])
for key,value in df.iteritems():
   print (key,value) #可以value.values
col1 
0    0.802390
1    0.324060
2    0.256811
3    0.839186
Name: col1, dtype: float64
.....
```

- `iterrows()` - 将行迭代为(索引，系列)对。

```python
for index,row in df.iterrows(): 
		print (index,row) #可以value.values
0  
col1    1.529759
col2    0.762811
col3   -0.634691
Name: 0, dtype: float64    
```

- `itertuples()` - 以`namedtuples`的形式迭代行

```python
for row in df.itertuples():
    print (row)

Pandas(Index=0, col1=1.5297586201375899, col2=0.76281127433814944, col3=-0.6346908238310438)
Pandas(Index=1, col1=-0.944087357638086, col2=1.4209186418359423,col3=-0.50789517967096232)
```

## 四、文本字符串

- **序列和索引**包含一些列的字符操作方法，这可以使我们轻易操作数组中的各个元素。
- 最重要的是，这些方法可以自动跳过 缺失/NA 值。
- 这些方法可以在`str`属性中访问到，和python内建的（标量）字符串方法同名。`s.str.method()`。
- 使用`.str[index]`进行索引。

```python
idx = pd.Index([' jack', 'jill ', ' jesse ', 'frank'])
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str[0] #A B C A B nan C d c
```

### 1. 基本函数

- `lower()/upper()` 将`Series/Index`中的字符串转换为小/大写

- `len()` 计算字符串长度
- `islower()` 检查系列/索引中每个字符串中的所有字符是否小写，返回布尔值 
-  `isupper()` 检查系列/索引中每个字符串中的所有字符是否大写，返回布尔值 
-  `isnumeric()` 检查系列/索引中每个字符串中的所有字符是否为数字，返回布尔值
- `swapcase` 变换字母大小写
- `startswith(pattern)` 如果系列/索引中的元素以模式开始，则返回`true`
-  `endswith(pattern)` 如果系列/索引中的元素以模式结束，则返回`true`
-  `get_dummies()` 返回具有单热编码值的数据帧(DataFrame)

-  `contains(pattern)` 如果元素中包含子字符串，则返回每个元素的布尔值`True`，否则为`False`  
-  `repeat(value)` 重复每个元素指定的次数 
-  `count(pattern)` 返回模式中每个元素的出现总数
-  `find(pattern)` 返回模式第一次出现的位置
- `findall(pattern)` 返回模式的所有出现的列表。

###2. 拆分和替换字符串

- `strip()` 帮助从两侧的系列/索引中的每个字符串中删除空格(包括换行符)。 
-  `split(' ')` 用给定的模式拆分每个字符串。切分后的列表中的元素通过 `get` 方法或者 `[]` 方法进行读取。使用`expand`方法可以轻易地将这种返回展开为一个数据表。同样，我们也可以限制切分的次数n。

```python
s2 = pd.Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'])
s2.str.split('_')
    0    [a, b, c]
    1    [c, d, e]
    2          NaN
    3    [f, g, h]
s2.str.split('_').str.get(1) #s2.str.split('_').str[1]
s2.str.split('_', expand=True)
         0    1    2
    0    a    b    c
    1    c    d    e
    2  NaN  NaN  NaN
    3    f    g    h
s2.str.split('_', expand=True, n=1)#a b_c    
```

- `replace(a,b)` 将值`a`替换为值`b`。`replace` 方法默认使用正则表达式

```python
s3 = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca','', np.nan, 'CABA', 'dog', 'cat'])
s3.str.replace('^.a|dog', 'XX-XX ', case=False)
#A B C XX-XX ba XX-XX ca        NaN XX-XX BA XX-XX XX-XX t
#可以使用字典
df.replace({1000:10,2000:60})
```

### 3. 拼接

- `cat(sep=' ')` 使用给定的分隔符连接系列/索引元素。如果没有额外声明，`sep` 即分隔符默认为空字串，即`sep=''`。默认情况下，缺失值会被忽略。使用`na_rep`参数，可以对缺失值进行赋值。

```python
s = pd.Series(['a', 'b', 'c', np,nan , 'd'])
s.str.cat(sep=',')  #'a,b,c,d'
s.str.cat(sep=',', na_rep='-') #'a,b,c,-,d'
```

- 拼接序列和其他类列表型对象为新的序列。第一个参数为类列表对象，但必须要确保长度与`序列`或`索引`相同。任何一端的缺失值都会导致之中结果为缺失值。

```python
s.str.cat(['A', 'B', 'C', 'D']) #aA bB cC dD
```

- 对于拼接`序列`或者`数据表`，我们可以使用 `join`关键字来对齐索引。`join` 的选项为（`'left'`, `'outer'`, `'inner'`, `'right'`）中的一个。按索引来。

```python
s = pd.Series(['a', 'b', 'c', 'd'])
u = pd.Series(['b', 'd', 'a', 'c'], index=[1, 3, 0, 2])
s.str.cat(u, join='left') #aa bb cc dd
```

## 五、处理缺失数据

### 1. 检查缺失值

- `isnull()`和`notnull()`函数，它们也是Series和DataFrame对象的方法 

### 2. 清理缺失的数据

- `fillna(a)`用a替换缺失值
- `pad/fill` 填充方法向前 `bfill/backfill` 填充方法向后。对于时间序列数据，使用填充/填充非常普遍，因此在每个时间点都可以使用“最新值”。

```python
df.fillna(method='pad')
```

- 如果我们只希望连续的间隙填充到一定数量的数据点，则可以使用*limit*关键字

```python
df.fillna(method='pad', limit=1)
```

- 如果只想丢失缺少的值，则使用`dropna`函数和`axis`参数。只要含有Nan就删除。

```python
df.dropna(axis=0)
df.dropna(axis=1)
```

- Series和DataFrame对象都具有在丢失的数据点执行线性插值的功能。`interpolate()`。

```python
ts.interpolate()
df.interpolate()
#可通过`method`关键字使用索引感知
ts2.interpolate(method='time') #索引是时间
ser.interpolate(method='values') #索引是float
```

- 该`method`参数可以访问更高级的插值方法。如果已安装scipy，则可以将1-d插值例程的名称传递给`method`。您需要查阅完整的scipy插值文档和参考指南以获取详细信息。适当的插值方法将取决于您使用的数据类型。

## 六、分组

```python
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings', 'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals','Riders'], 
            'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2], 
            'Year':[2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017], 
            'Points':[876,789,863,673,741,812,756,788,694,701,804,690]} 
df = pd.DataFrame(ipl_data)
```

- 分组

```python
grouped = df.groupby('Team').groups
{
'Devils': Int64Index([2, 3], dtype='int64'), 
'Kings': Int64Index([4, 6, 7], dtype='int64'), 
'Riders': Int64Index([0, 1, 8, 11], dtype='int64'), 
'Royals': Int64Index([9, 10], dtype='int64'), 
'kings': Int64Index([5], dtype='int64')
}
df.groupby(['Team','Year']).groups
```

- 迭代遍历分组

```python
grouped = df.groupby('Year')
for name,group in grouped:
    print (name)
    print (group)
        2014
           Points  Rank    Team  Year
        0     876     1  Riders  2014
        2     863     2  Devils  2014
        4     741     3   Kings  2014
        9     701     4  Royals  2014
        2015
            Points  Rank    Team  Year
        1      789     2  Riders  2015
        3      673     3  Devils  2015
        5      812     4   kings  2015
        10     804     1  Royals  2015 ......
```

- 选择分组`get_group()`

```python
grouped = df.groupby('Year') 
print (grouped.get_group(2014)
```

- 聚合

```python
grouped['Points'].agg(np.mean)
    Year
    2014    795.25
    2015    769.50
    2016    725.00
    2017    739.00
#一次应用多个聚合函数    
agg = grouped['Points'].agg([np.sum, np.mean, np.std])
```

- 查看每个分组的大小的方法是应用`size()`函数 

```python
grouped = df.groupby('Team')
grouped.agg(np.size)
Team                      
    Devils       2     2     2
    Kings        3     3     3
    Riders       4     4     4
    Royals       2     2     2
    kings        1     1     1
```

- 过滤`filter`

```python
filter = df.groupby('Team').filter(lambda x: len(x) >= 3) 
print (filter)
    		Points  Rank    Team  Year
    0      876     1  Riders  2014
    1      789     2  Riders  2015
    4      741     3   Kings  2014
    6      756     1   Kings  2016
    7      788     1   Kings  2017
    8      694     2  Riders  2016
    11     690     2  Riders  2017
```

- 转换`transform`

```python
grouped = df.groupby('Team') 
score = lambda x: (x - x.mean()) / x.std()*10 
print (grouped.transform(score))
```

## 七、合并

```python
left = pd.DataFrame({ 'id':[1,2,3,4,5], 
                      'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
                      'subject_id':['sub1','sub2','sub4','sub6','sub5']}) 
right = pd.DataFrame( {'id':[1,2,3,4,5], 
                       'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
                       'subject_id':['sub2','sub4','sub3','sub6','sub5']}
```

- 在一个键上合并两个数据帧

```python
rs = pd.merge(left,right,on='id')
       Name_x  id subject_id_x Name_y subject_id_y
    0    Alex   1         sub1  Billy         sub2
    1     Amy   2         sub2  Brian         sub4
    2   Allen   3         sub4   Bran         sub3
    3   Alice   4         sub6  Bryce         sub6
    4  Ayoung   5         sub5  Betty         sub5
```

- 多个键上的两个数据框

```python
rs = pd.merge(left,right,on=['id','subject_id'])
       Name_x  id subject_id Name_y
    0   Alice   4       sub6  Bryce
    1  Ayoung   5       sub5  Betty
```

- `how`选项:`left`  `right` `outer`  `inner`

## 八、连接

- `concat()`函数完成了沿轴执行级联操作的所有重要工作。

```python
one = pd.DataFrame({
         'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
         'subject_id':['sub1','sub2','sub4','sub6','sub5'],
         'Marks_scored':[98,90,87,69,78]},
         index=[1,2,3,4,5])
two = pd.DataFrame({
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id':['sub2','sub4','sub3','sub6','sub5'],
         'Marks_scored':[89,80,79,97,88]},
         index=[1,2,3,4,5])
rs = pd.concat([one,two])#索引会重复
```

- 假设想把特定的键与每个碎片的DataFrame关联起来。可以通过使用键参数来实现这一点

```python
rs = pd.concat([one,two],keys=['x','y'])
     Marks_scored    Name subject_id
    x 1            98    Alex       sub1
      2            90     Amy       sub2
      3            87   Allen       sub4
      4            69   Alice       sub6
      5            78  Ayoung       sub5
    y 1            89   Billy       sub2
      2            80   Brian       sub4
      3            79    Bran       sub3
      4            97   Bryce       sub6
      5            88   Betty       sub5
```

- 每个索引重复。如果想要生成的对象必须遵循自己的索引，请将`ignore_index`设置为`True`。

```python
rs = pd.concat([one,two],keys=['x','y'],ignore_index=True)
```

- 使用`append`

```python
rs = one.append([two,one,two]) #可连接多个
```

## 九、通用函数

### 1. 浅拷贝与深拷贝

```python
#深拷贝，改变值不影响
cpys = s.copy(deep=True)
#浅拷贝，改变值影响原来
cpys = s.copy(deep=False)
```

### 2. 改变索引

- 索引变化:`reindex`。如果当前索引原索引没有则没有原数据。

```python
d.reindex([x,y,...], fill_value=NaN) #重返回一个适应新索引的新对象，将缺失值填充为fill_value
d.reindex([x,y,...], method=NaN) #返回适应新索引的新对象，填充方式为method
		#pad/ffill - 向前填充值
  	#bfill/backfill - 向后填充值
    #nearest  - 从最近的索引值填充
d.reindex(columns=[x,y,...]) #对列进行重新索引
df1.reindex_like(df2)#重建索引与其他对象对齐
df2.reindex_like(df1,method='ffill',limit=1)#填充限制，只向前填一列
```

- 改变索引名字:`rename()`方法允许基于一些映射(字典或者系列)或任意函数来重新标记一个轴

```python
data01=data.rename(pd.to_datetime(data['YYYY'])) #
```

### 3. 删除指定行列

```python
#Series 删除行
ds = s.drop('No.1') #s.drop(['No.1', 'No.2'])
#dataframe删除行，列
ds = df.drop('No.1') #删除行
ds = df.drop(['Age'], axis=1)#删除指定列，可以产出多列，序列中指出就可以['Age','Name']
```

### 4. 函数应用

#### （1）元素函数应用

- 在`DataFrame`上的方法`applymap()`和类似于在`Series`上的`map()`接受任何Python函数，并且返回单个值。

```python
#把2010转化为20100101便于以后转化为pd.toDate
def def toDate(a):
    return a+"0101"
data['YYYY']=data['YYYY'].map(toDate)
```

#### （2）行或列函数应用

- 可以使用`apply()`方法沿`DataFrame`的轴应用任意函数，它与描述性统计方法一样，采用可选的`axis`参数。 

```python
data[['TotalDrugReportsCounty','TotalDrugReportsState']].apply(np.mean)
TotalDrugReportsCounty      991.824869
TotalDrugReportsState     56083.943895
```

####（3）表格函数应用

- 可以通过`pipe`将函数和适当数量的参数作为管道参数来执行自定义操作。 

```python
def adder(ele1,ele2):
   return ele1+ele2
df.pipe(adder,2)
```

#### （4）聚合 API

- 聚合 API 可以快速、简洁地执行多个聚合操作。应用单个函数时，该操作与 `apply()`等效。还可以用列表形式传递多个聚合函数。每个函数在输出结果 `DataFrame` 里以行的形式显示，行名是每个聚合函数的函数名。

```python
data[['TotalDrugReportsCounty','TotalDrugReportsState']].agg([np.mean,lambda x: x.mean()])
        TotalDrugReportsCounty	TotalDrugReportsState
mean							9.918249e+02	5.608394e+04
<lambda>					9.918249e+02	5.608394e+04
```

### 5. 排序

#### （1）按索引

- `Series.sort_index(ascending=True,axis=1)`,根据索引返回已排序的新对象。False时降序,`axis`选择轴。

#### （2）按值

- `DataFrame.sort_values(axis=0,by='col1'，kind='mergesort')`，根据值返回已排序的对象，NaN值在末尾。`Dataframe`时用`by=['col1','col2']`。
- `sort_values()`提供了从`mergeesort`，`heapsort`和`quicksort`中选择算法的一个配置。`Mergesort`是唯一稳定的算法。

### 6. 基于dtype选择列

- `select_dtypes()`方法基于 `dtype` 选择列。有两个参数，`include` 与 `exclude`，用于实现“提取这些数据类型的列” （`include`）或 “提取不是这些数据类型的列”（`exclude`）。

```python
df.select_dtypes(include=['number', 'bool'], exclude=['unsignedinteger'])
```

### 7. astype

- `astype()`方法显式地把一种数据类型转换为另一种，默认操作为复制数据，就算数据类型没有改变也会复制数据，`copy=False` 改变默认操作模式。此外，`astype` 无效时，会触发异常。

```python
df3.astype('float32').dtypes
dft[['a', 'b']] = dft[['a', 'b']].astype(np.uint8)
```

### 8. .dt

- `Series` 提供一个可以简单、快捷地返回 `datetime` 属性值的访问器。这个访问器返回的也是 Series，索引与现有的 Series 一样。用下列表达式进行筛选非常方便。

```python
s = pd.Series(pd.date_range('20130101 09:10:12', periods=4))
s.dt.day #hour,second,year,moth,day
s[s.dt.day == 2]
```

- 还可以用 `Series.dt.strftime()` 把 `datetime` 的值当成字符串进行格式化，支持与标准 `strftime()`同样的格式。

```python
s.dt.strftime('%Y/%m/%d')
```

### 9. 选择随机样本

- 使用该`sample()`方法随机选择Series或DataFrame中的行或列。

```python
s.sample()
s.sample(n=3) #3个样本
s.sample(frac=0.5) #0.5个样本大小
#默认情况下，每行具有相同的选择概率，但如果您希望行具有不同的概率，则可以将sample函数采样权重作为 weights。
example_weights = [0, 0, 0.6, 0.2, 0.2]
s.sample(n=3, weights=example_weights)
#应用于DataFrame时，只需将列的名称作为字符串传递,那一列不能为负，其余大小无限定
df.sample(n=3, weights='D')
#sample还允许用户使用axis参数对列而不是行进行采样
df.sample(n=1, axis=1)
#最后，还可以sample使用random_state参数为随机数生成器设置种子。设置之后n一定每次返回一样
df.sample(n=2, random_state=2)
```

### 10. isin()

- `Series`方法`isin()`。

```python
#筛选是否有哪些
s.isin([2, 4, 6])
s[s.isin([2, 4, 6])]
#Index对象可以使用相同的方法，当您不知道哪些搜索标签实际存在时，它们非常有用
s[s.index.isin([2, 4, 6])]
    4    0
    2    2
    dtype: int64
```

- `DataFrame`也有一个`isin()`方法。

```python
df = pd.DataFrame({'vals': [1, 2, 3, 4], 
                   'ids': ['a', 'b', 'f', 'n'],
                   'ids2': ['a', 'n', 'c', 'n']})
#传入列表：匹配所有
values = ['a', 'b', 1, 3]
df.isin(values)
        vals    ids   ids2
    0   True   True   True
    1  False   True  False
    2   True  False  False
    3  False  False  False
#传入字典：将某些值与某些列匹配
values = {'ids': ['a', 'b'], 'vals': [1, 3]}
df.isin(values)    
```

### 11. 删除重复行

- 如果要识别和删除`DataFrame`中的重复行，有两种方法可以提供帮助：`duplicated`和`drop_duplicates`。每个都将用于标识重复行的列作为参数。

- `duplicated` 返回一个布尔向量，其长度为行数，表示行是否重复。
- `drop_duplicates` 删除重复的行。

- 重复集的第一个观察行被认为是唯一的，但每个方法都有一个`keep`参数来指定要保留的目标
  - `keep='first'` （默认值）：标记/删除重复项，第一次出现除外。
  - `keep='last'`：标记/删除重复项，除了最后一次出现。
  - `keep=False`：标记/删除所有重复项。

```python
df2 = pd.DataFrame({'a': ['one', 'one', 'two', 'two', 'two', 'three', 'four'],
                    'b': ['x', 'y', 'x', 'y', 'x', 'x', 'x'],
                    'c': np.random.randn(7)})
df2.duplicated('a') #df2.duplicated('a', keep='last')
    0    False
    1     True
    2    False
    3     True
    4     True
    5    False
    6    False
df2.drop_duplicates('a')
    			 a  b         c
    0    one  x -1.067137
    2    two  x -0.211056
    5  three  x -1.964475
    6   four  x  1.298329
#您可以传递列表列表以识别重复。
df2.duplicated(['a', 'b'])
#要按索引值删除重复项，请使用Index.duplicated然后执行切片。keep参数可以使用相同的选项集。
df3.index.duplicated()
>>>array([False,  True, False, False,  True,  True])
df3[~df3.index.duplicated()]
```

# 数据结构

- Pandas处理以下三个数据结构 - 系列(`Series`)，数据帧(`DataFrame`)，面板(`Panel`)。

- 这些数据结构构建在Numpy数组之上，这意味着它们很快。
- 考虑这些数据结构的最好方法是，较高维数据结构是其较低维数据结构的容器。 例如，`DataFrame`是`Series`的容器，`Panel`是`DataFrame`的容器。

- Pandas 对象（`Index`， `Series`， `DataFrame`）相当于数组的容器，用于存储数据、执行计算。大部分类型的底层数组都是 `numpy.ndarray`

##一、Series

###1. 概念

- 是带标签的一维数组。轴标签统称为**索引**。
- **均匀数据，尺寸大小不变，数据的值可变。**

### 2. 创建Series

```python
 s = pd.Series(data, index, dtype, copy)
```

- `data` 数据采取各种形式，如：`ndarray`，`list`，`dict` 。
-  `index` 索引值必须是唯一的和散列的，**与数据的长度相同**。 默认`np.arange(n)`如果没有索引被传递。 
- `dtype` `dtype`用于数据类型。如果没有，将推断数据类型 。
- `copy` 复制数据，默认为`false` 。

####（1）ndarray/list

- data是多维数组时，**index** 长度必须与 **data** 长度一致**（多维数组长度len(data)把它当成列表，则是最高项的长度，矩阵则为行数）**。没有指定 index参数时，创建数值型索引，即` [0, ..., len(data) - 1]`。
- **Pandas 的索引值可以重复。不支持重复索引值的操作会触发异常。**其原因主要与性能有关，有很多计算实例，比如 GroupBy 操作就不用索引。

```python
s = pd.Series(np.random.randn(3), index=['a', 'b', 'c']) #index=list("abc")
>>> a    0.469112
    b   -0.282863
    c   -1.509059
    dtype: float64
#如果不是一维数组，则为NaN
s=pd.Series(np.random.randn(10).resize(2,5),index=[1,2])
>>> s
1   NaN
2   NaN
dtype: float64
```

#### （2）dict

- `data` 为字典，且未设置 `index` 参数时，`Series` 按字典的插入顺序排序索引。
- 如果设置了 `index` 参数，则按索引标签提取 `data` 里对应的值。Pandas 用 `NaN`表示**缺失数据**。

```python
d = {string.ascii_uppercase[i]:i for i in range(3)}  #{'A': 0, 'B': 1, 'C': 2}
a=pd.Series(d,index=list(string.ascii_uppercase[2:5]))
>>> C    2.0
		D    NaN
		E    NaN
		dtype: float64 #从整数变成float
a.
```

#### （3）标量值

- `data` 是标量值时，**必须提供索引**。`Series` 按**索引**长度重复该标量值。

```python
pd.Series(5., index=['a', 'b'])
>>> a    5.0
    b    5.0
    dtype: float64
```

### 3. Series基本属性/方法

####（1）属性

- `index`返回索引

- `values` 将系列作为`ndarray`返回。 
-  `dtype` 返回对象的数据类型(`dtype`)。
-  `empty` 如果系列为空，则返回`True`。 
- `ndim` 返回底层数据的维数，默认定义：`1`。 
-  `size` 返回基础数据中的元素数。 
- `head()` 返回前`n`行。 
-  `tail()` 返回最后`n`行。
- `name`返回`values`的`name`。

- `array` 属性用于提取 `Index`或 `Series`里的数据。`PandasArray`返回。

- `numpy.asarray()`,`to_numpy()`提取 `NumPy `数组。

#### （2）方法

- `df.argmax()/df.argmin()`，返回含有最大值的索引位置/返回含有最小值的索引位置。

- `pd.isnull(s)`判断series对象是否含有NaN数值。`pd.notnull(s)`。

## 二、DataFrame

###1. 概念

- **DataFrame** 是由多种类型的列构成的二维标签数据结构，类似于 Excel ，或 Series 对象构成的字典。除了数据，还可以有选择地传递 **index**（行标签）和 **columns**（列标签）参数。Series 字典加上指定索引时，**会丢弃与传递的索引不匹配的所有数据。**

### 2. 创建DataFrame

```python
pd.Series( data, index, columns, dtype, copy)
```

- `data` 数据采取各种形式，如:`ndarray`，`series`，`map`，`lists`，`dict`，`constant`和另一个`DataFrame`。 
-  `index` 对于行标签，要用于结果帧的索引是可选缺省值`np.arrange(n)`，如果没有传递索引值。 
-  `columns` 对于列标签，可选的默认语法是 - `np.arange(n)`。 这只有在没有索引传递的情况下才是这样。 
- `dtype` 每列的数据类型。 
-  `copy`如果默认值为`False`，则此命令(或任何它)用于复制数据。 

#### （1）列表/ndarray

```python
#列表
data = [1,2,3,4,5]
df = pd.DataFrame(data)
#ndarray（列表的列表）
data = [
  			['Alex',10],
        ['Bob',12],
        ['Clarke',13]
       ]
df = pd.DataFrame(data,columns=['Name','Age'],dtype=float)
#字典的列表（类似ndarray）
data2 = [
  				{'a': 1, 'b': 2}, 
  				{'a': 5, 'b': 10, 'c': 20}
				]
pd.DataFrame(data2)
>>>    a   b     c
    0  1   2   NaN
    1  5  10  20.0
```

#### （2）Series字典

- **生成的索引是每个 Series 索引的并集，不需要Series都相同长**。先把嵌套字典转换为 Series。

```python
#Series字典
d = {
  	 'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
     'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])
    }
df = pd.DataFrame(d)
>>>    one  two
    a  1.0  1.0
    b  2.0  2.0
    c  3.0  3.0
    d  NaN  4.0
pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three'])#
>>>    two three  #指定了index和columns就按这个顺序，没有的就填NaN    
    d  4.0   NaN
    b  2.0   NaN
    a  1.0   NaN
```

####（3）ndarray字典

- **多维数组的长度必须相同**。**`index` 的长度必须与数组一致。**

```python
data = {
  				'Name':['Tom', 'Jack', 'Steve', 'Ricky'],
  				'Age':[28,34,29,42]
			 }
df = pd.DataFrame(data,index=['rank1','rank2','rank3','rank4'])
>>>          Age    Name
    rank1    28      Tom
    rank2    34     Jack
    rank3    29    Steve
    rank4    42    Ricky
```

### 3. 列操作

#### （1）删除

- del，pop

```python
del df['two']
three = df.pop('three')	
```

#### （2）插入

- 插入Series。标量值以广播的方式填充列。

- 插入与 DataFrame 索引不同的 Series 时，以 DataFrame 的索引为准。多出舍掉，少了NaN。

```python
df['one_trunc'] = df['one'][:2]
>>>    one   flag  foo  one_trunc
    a  1.0  False  bar        1.0
    b  2.0  False  bar        2.0
    c  3.0   True  bar        NaN
    d  NaN  False  bar        NaN
```

- 可以插入原生多维数组，但长度必须与 DataFrame 索引长度一致。默认在 DataFrame 尾部插入列。`insert` 函数可以指定插入列的位置。

```python
df.insert(1, 'bar', [9,9,9,9])
   one  bar   flag  foo  one_trunc
a  1.0  9    False  bar        1.0
b  2.0  9    False  bar        2.0
c  3.0  9     True  bar        NaN
d  NaN  9    False  bar        NaN
```

### 4. 行操作

#### （1）删除

- 使用索引标签从DataFrame中删除或删除行。 如果标签重复，则会删除多行。

```python
df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])
df = df.append(df2)
df = df.drop(0)
```

#### （2）插入

- 使用`append()`函数将新行（DataFrame）添加到DataFrame。

```python
df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])
df = df.append(df2)
```

### 5. 基本功能

- `T` 转置行和列。

-  `axes` 返回一个列，行轴标签和列轴标签作为唯一的成员。 
-  `index`,`columns`返回行，列标签。
-  `dtypes` 返回此对象中的数据类型(`dtypes`)。 
-  `empty` 如果`NDFrame`完全为空[无项目]，则返回为`True`; 如果任何轴的长度为`0`。 
-  `ndim` 轴/数组维度大小。 
-  `shape` 返回表示`DataFrame`的维度的元组。 
-  `size` `NDFrame`中的元素数。 
-  `values` NDFrame的Numpy表示。 
-  `head()` 返回开头前`n`行。 
-  `tail()` 返回最后`n`行。
-  `DataFrame.dtypes.value_counts()` 用于统计 DataFrame 里不同数据类型的列数。





# 索引技巧



## 一、按ndarray方法[ ]

- 使用`[]`进行切片/选择，不管是`Series`还是`DataFrame`都按`list`方法。
- 对于`Series`，和`list`一样，无论按位置还是索引。
- 对于`Dataframe`就没有`[:,:]`,用索引提取列或用位置切片提取行。

###1. Series 

```python
s=pd.Series(10*np.random.randn(5),index=list("abcde"))
    a    16.110235
    b    12.529846
    c     1.040452
    d    -9.592514
    e    14.807942
    dtype: float64
```

- 用索引

```python
s["a"] #提取值 --> 一个标量
s["a":"d":2] #切片 --> 一个序列
s[["a","e"]] #传递列表获取多值 -->一个序列
#上述方法都可以设置值
```

- 用位置

```python
s[1]
s[1:6] #切片，超出部分舍弃
s[[1,2]]
#上述方法都可以设置值
```

- 用属性

```python
s.a 
s.a = 5 #可以设置
```

### 2. DataFrame

```
dates = pd.date_range('1/1/2000', periods=8) #pandas.core.indexes.datetimes.DatetimeIndex
df = pd.DataFrame(np.random.randn(8, 4),index=dates, columns=['A', 'B', 'C', 'D'])
    									 A         B         C         D
    2000-01-01  0.000000  0.000000  0.000000  0.000000
    2000-01-02  1.000000  1.000000  1.000000  1.000000
    2000-01-03  1.182681  0.650956  0.705687  0.210845
    2000-01-04 -0.534640  1.293568 -2.693268 -0.809006
    2000-01-05 -0.080131  0.366861 -1.722215  0.528967
    2000-01-06  0.060695  1.267464 -0.351543 -2.208600
    2000-01-07 -1.076764 -1.013830  0.041624  0.030970
    2000-01-08  0.432933 -1.369993  0.447334 -1.306808
```

- 列：只能通过索引。注意，不能切片，只能选择列

```python
df['A']  #一个Series
df[['A','B']]  #一个dataframe
df.A = list(range(len(df.index)))  
#上述方法都可以设置值
```

- 行：只能通过位置。只能切片或列表。

```python
df[1:2:1]
df[[1,2]]
#上述方法都可以设置值
```

##二、loc（索引）

### 1. Series

```python
s1 = pd.Series(np.random.randn(6), index=list('abcdef'))
s1.loc['b'] #获取值
s1.loc['c':'e':2] #切片
#上述方法都可以设置值
```

###2. DataFrame

```python
df = pd.DataFrame(np.random.randn(6, 4),index=list('abcdef'),columns=list('ABCD'))
df.loc['a', 'A'] #获取值（行，列）
df.loc['a'] #获取行 相当于df.loc['a',:]
df.loc[['a', 'b', 'd']] #获取多行
df.loc[:,'B'] #获取列
df.loc[['a', 'b', 'd'],'B':'D'] #切片
```

##三、iloc（位置）

###1. Series

```python
s1 = pd.Series(np.random.randn(6), index=list('abcdef'))
s1.iloc[3]
s1.iloc[1:4:2]
#上述方法都可以设置值
```

### 2. DataFrame

```python
df = pd.DataFrame(np.random.randn(6, 4),index=list('abcdef'),columns=list('ABCD'))
df.iloc[1] #获取行 相当于df.loc[1,:] 
df.iloc[[1, 3, 5], [1, 3]]
df.iloc[1, 1] #获取值
#没有布尔索引
```

## 四、布尔索引

### 1. Series

```python
s[s > 0]
s[(s < -1) | (s > 0.5)]				 # & | ~
s[~(s < 0)]
```

###2. DataFrame

- 您可以使用与DataFrame**索引长度相同的布尔向量**从DataFrame中选择行。

```python
df[df['A'] > 0]
df[(df.a < df.b) & (df.b < df.c)]

df2 = pd.DataFrame({'a': ['one', 'one', 'two', 'three', 'two', 'one', 'six'],
                     'b': ['x', 'y', 'y', 'x', 'y', 'x', 'x'],
                     'c': np.random.randn(7)})

#列表推导和map系列方法也可用于产生更复杂的标准：
# only want 'two' or 'three'
criterion = df2['a'].map(lambda x: x.startswith('t'))
df2[[x.startswith('t') for x in df2['a']]]
df2[criterion & (df2['b'] == 'x')]
#调用选择
df1.loc[lambda df: df.A > 0 , :]
df1.loc[:, lambda df: ['A', 'B']]
df1.iloc[:, lambda df: [0, 1]]
df1[lambda df: df.columns[0]]
df1.A.loc[lambda s: s > 0] #这些数据中大于0的数据

#loc布尔索引
df.loc[:, df.loc['a'] > 0] #除开BCD列,df1.loc['a'] > 0=[True,False,False,False]
df.loc[:, [True,False,False,False]]
df.loc[df.loc[:,'B']>0,:]
df.loc[ : ,(df.loc['a'] > 0) &(df.loc['b'] > 0)]
df.loc[df.loc[:,'B']>0,df.loc['a'] > 0]
```

- 

























# 描述性统计



##一、描述数据

- 数据帧(DataFrame) - “index”(axis=0，默认)，columns(axis=1)
- 类似于：`sum()`，`cumsum()`函数能与数字和字符(或)字符串数据元素一起工作，不会产生任何错误。字符聚合从来都比较少被使用，虽然这些函数不会引发任何异常。
- 由于这样的操作无法执行，因此，当DataFrame包含字符或字符串数据时，像`abs()`，`cumprod()`这样的函数会抛出异常。

```python
d = {
 'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Minsu','Jack','Lee','David','Gasper','Betina','Andres']),
 'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
 'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
		}
df = pd.DataFrame(d)
        Age  Name   Rating
    0   25   Tom     4.23
    1   26   James   3.24
    2   25   Ricky   3.98
    3   23   Vin     2.56
    4   30   Steve   3.20
    5   29   Minsu   4.60
    6   23   Jack    3.80
    7   34   Lee     3.78
    8   40   David   2.98
    9   30   Gasper  4.80
    10  51   Betina  4.10
    11  46   Andres  3.65
```

- `count()` 非空观测数量 

```python
df.count()
  Name      12
  Age       12
  Rating    12
  dtype: int64
df.count(1)
  0     3
  1     3
  2     3
  3     3
  4     3
  5     3
  6     3
  7     3
  8     3
  9     3
  10    3
  11    3
  dtype: int64
```

-  `sum()` 所有值之和 

```python
df.sum() #每一列求和

Age                                                    382
Name     TomJamesRickyVinSteveMinsuJackLeeDavidGasperBe...
Rating                                               44.92
dtype: object
  
df.sum(1)  #每一行求和
0    29.23
1    29.24
2    28.98
3    25.56
4    33.20
5    33.60
6    26.80
7    37.78
8    42.98
9    34.80
10   55.10
11   49.65
```

-  `mean()` 所有值的平均值 

```python
df.mean()
  Age       31.833333
  Rating     3.743333
  dtype: float64
df.mean(1)
  0     14.615
  1     14.620
  2     14.490
  3     12.780
  4     16.600
  5     16.800
  6     13.400
  7     18.890
  8     21.490
  9     17.400
  10    27.550
  11    24.825
  dtype: float64
```

-  `median()` 所有值的中位数 
-  `mode()` 值的模值 
-  `std()` 值的标准偏差 

```python
df.std()
  Age       9.232682
  Rating    0.661628
  dtype: float64
```

-  `min()` 所有值中的最小值 
-  `max()` 所有值中的最大值 
-  `abs()` 绝对值 
-  `prod()` 数组元素的乘积 
-  `cumsum()` 累计总和

```python
df.cumsum()
                                                 Name  Age  Rating
0                                                 Tom   25    4.23
1                                            TomJames   51    7.47
2                                       TomJamesRicky   76   11.45
3                                    TomJamesRickyVin   99   14.01
4                               TomJamesRickyVinSteve  129   17.21
5                          TomJamesRickyVinSteveMinsu  158   21.81
6                      TomJamesRickyVinSteveMinsuJack  181   25.61
7                   TomJamesRickyVinSteveMinsuJackLee  215   29.39
8              TomJamesRickyVinSteveMinsuJackLeeDavid  255   32.37
9        TomJamesRickyVinSteveMinsuJackLeeDavidGasper  285   37.17
10  TomJamesRickyVinSteveMinsuJackLeeDavidGasperBe...  336   41.27
11  TomJamesRickyVinSteveMinsuJackLeeDavidGasperBe...  382   44.92
```

-   `cumprod()` 累计乘积

## （二）汇总数据

- `describe()`函数是用来计算有关DataFrame列的统计信息的摘要。平均值，标准差和IQR值。

```python
df.describe()
                 Age     Rating
    count  12.000000  12.000000
    mean   31.833333   3.743333
    std     9.232682   0.661628
    min    23.000000   2.560000
    25%    25.000000   3.230000
    50%    29.500000   3.790000
    75%    35.500000   4.132500
    max    51.000000   4.800000
```

- `include`是用于传递关于什么列需要考虑用于总结的必要信息的参数。获取值列表; 默认情况下是”数字值。
  - `object` - 汇总字符串列
  - `number` - 汇总数字列
  - `all` - 将所有列汇总在一起(不应将其作为列表值传递)

```python
df. describe(include='all')
```

## （三）统计函数

### 1. pct_change()

- `DatFrames`和`Panel`都有`pct_change()`。此函数将每个元素与其前一个元素进行比较，并计算变化百分比。
- 默认情况下，`pct_change()`对列进行操作; 如果想应用到行上，那么可使用`axis = 1`参数。

```python
s = pd.Series([1,2,3,4,5,4])
print (s.pct_change())
    0        NaN
    1   1.000000
    2   0.500000
    3   0.333333
    4   0.250000
    5  -0.200000
		dtype: float64
df = pd.DataFrame(np.random.randn(5, 2))
print (df.pct_change())
                0          1
    0         NaN        NaN
    1  -15.151902   0.174730
    2  -0.746374   -1.449088
    3  -3.582229   -3.165836
    4   15.601150  -1.860434
```

### 2. cov()

- 协方差适用于系列数据。`Series`对象有一个方法`cov`用来计算序列对象之间的协方差。`NA`将被自动排除。
- 当应用于`DataFrame`时，协方差方法计算所有列之间的协方差(`cov`)值。

```python
s1 = pd.Series(np.random.randn(10))
s2 = pd.Series(np.random.randn(10))
print (s1.cov(s2))

frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
print (frame['a'].cov(frame['b']))
print (frame.cov())
    -0.406796939839
              a         b         c         d         e
    a  0.784886 -0.406797  0.181312  0.513549 -0.597385
    b -0.406797  0.987106 -0.662898 -0.492781  0.388693
    c  0.181312 -0.662898  1.450012  0.484724 -0.476961
    d  0.513549 -0.492781  0.484724  1.571194 -0.365274
    e -0.597385  0.388693 -0.476961 -0.365274  0.785044
```

### 3. corr()

- 相关性显示了任何两个数值(系列)之间的线性关系。

```python
frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
print (frame['a'].corr(frame['b']))
print (frame.corr())
```

### 4. rank()

- 数据排名为元素数组中的每个元素生成排名。
- `s.rank(method='average', ascending=True, axis=0)`,`Rank`可选地使用一个默认为`true`的升序参数; 当错误时，数据被反向排序，也就是较大的值被分配较小的排序。
- `Rank`支持不同的`tie-breaking`方法，用方法参数指定 -
  - `average` - 并列组平均排序等级
  - `min` - 组中最低的排序等级
  - `max` - 组中最高的排序等级
  - `first` - 按照它们出现在数组中的顺序分配队列

```python
s = pd.Series(np.random.np.random.randn(5), index=list('abcde'))
s['d'] = s['b']
s.rank()
  a    5.0
  b    3.5
  c    2.0
  d    3.5
  e    1.0
  dtype: float64
```

##（四）窗口函数

- 所谓窗口，就是将某个点的取值扩大到包含这个点的一段区间，用区间来进行判断。

- **时间序列平滑**。窗口函数主要用于通过平滑曲线来以图形方式查找数据内的趋势。

  

  







index = pd.date_range('1/1/2020', periods=10)//原文出自【易百教程】，商业转载请联系作者获得授权，非商业转载请保留原文链接：https://www.yiibai.com/pandas/python_pandas_window_functions.html 

columns = ['A', 'B', 'C', 'D'])//原文出自【易百教程】，商业转载请联系作者获得授权，非商业转载请保留原文链接：https://www.yiibai.com/pandas/python_pandas_window_functions.html 

index = pd.date_range('1/1/2020', periods=10), columns = ['A', 'B', 'C', 'D'])//原文出自【易百教程】，商业转载请联系作者获得授权，非商业转载请保留原文链接：https://www.yiibai.com/pandas/python_pandas_window_functions.html

df = pd.DataFrame(np.random.randn(10, 4), index = pd.date_range('1/1/2020', periods=10), columns = ['A', 'B', 'C', 'D']) //原文出自【易百教程】，商业转载请联系作者获得授权，非商业转载请保留原文链接：https://www.yiibai.com/pandas/python_pandas_window_functions.html 