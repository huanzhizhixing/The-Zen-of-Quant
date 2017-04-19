
numpy

一、基本属性

1.显示数据的形状和数据类型。
是自身属性，用 .shape
data.shape,data.dtype

2.创建ndarray
用np.array()函数创建,data为列表。
arr1=np.array(data)

3.零，无意义的初始值
np.zeros((3,6));np.empty((2,3,2))

4.创建由小到大的数组
np.arange(15)

5.转换属性
astype()函数
float=arr.astype(np.float64)

二、基本的索引和切片

1.取某一未知和区域的值
arr=np.arange(10)
arr[5];arr[5:8]

2.不想变动原来数值，显示复制
copy()函数
arr[5:8].copy()

3.二维取某固定值
arr2d[0][2],或arr2d[0,2]

4.切片是沿着某个轴进行取值
切片都是先高维后低维的。
arr2d[:2,1:]

5.在数组中可以用布尔值来结合索引
names == 'Bob'
data[names == 'Bob'

6.python的关键字 and 和 or 在布尔数组中无效，需要用&和|

7.花式索引：用整数数组进行索引。
花式索引跟切片不一样，他总是将数据复制到新数组中。
arr[[4,3,0,6]],效果是把行选出后，并且按照这样进行排列。

8.联立多个轴
注意，reshape后面是两个括号。
arr=np.arange(16).reshape((4,4))

9.一次取多个位点
arr[[1,4,2,3],[3,2,2,1]]

10.一次取矩阵区域
arr[[1,5,2,3]][:,[0,3,1,2]]
含义是先用花式索引把行调整，随后对所有的再按列调整。

11.使用"np.ix_"函数，把两个一维整数数组转换为一个用于选取方形区域的索引器
arr[np.ix_([1,5,7,2],[0,3,1,2])]

三、数据转置和轴对换

1.转置，直接重塑数组。
arr.T

2.计算矩阵内积XTX
np.dot(arr.T,arr)

四、通用计算

1.一元计算
np.sqrt(arr),np.exp(arr)

2.二元计算
np.maximum(x,y)

3.暂时看来，没有多元。然而，通过组合多个二元计算，估计能够达到多元计算的目的。实验了下，可行。
>>> import numpy as np
>>> aaa=np.arange(4)
>>> bbb=np.arange(4)
>>> ccc=np.arange(4)
>>> aaa
array([0, 1, 2, 3])
>>> bbb+=1
>>> ccc+=4
>>> bbb
array([1, 2, 3, 4])
>>> ccc
array([4, 5, 6, 7])
>>> ddd=np.subtract(ccc,np.add(bbb,aaa))
>>> ddd
array([3, 2, 1, 0])
但是，如果是简单的算术运算，可以用加减乘除来做了。

4.利用数组进行数据处理——矢量化运算
z = np.sqrt(xs**2 + ys**2)

五、条件运算
1.条件运算
np.where(cond,xarr,yarr)=[xarr if cond, else y]
np.where(arr>0,2,arr)
把如果大于零，则为2，否则为arr。

2.多个组合的条件运算
原始：
result = []
for i in range(n):
	if cond1[i] and cond2[2]:
		result.append(0)
	elif cond1[i]:
		result.append(1)
	elif cond2[i]:
		result.append(2)
	else:
		resule.append(3)
		
改写
np.where(cond1 & cond2, 0,
			np.where(cond1,1,
				np.where(cond2,2,3)))
(真NB！)

还可以写成
result = 0 * (cond1 & cond2) + 1 * (cond1 -cond2) + 2 * (cond2 & -cond1) + 3 * -(cond1 | cond2)
这个也挺好玩，括号里面都是情况的判断布尔值。

3.布尔值数组计算
arr=randn(100)
(arr>0).sum()#正值的个数

4.any和all计算是否有True
bools = np.array([False,True])
bools.any()
bools.all()

六、聚合运算

1.均值、方差、累计积
arr=np.random.randn(5,4)#正态分布数据。
两种方式聚合，一种是数据名加点，一种是函数加数据名
arr.mean();np.mean(arr)
方差标准差：std，var。
所有元素的累计和，及所有元素的累计积。cumsum，cumprod。

2.排序
arr.sort()
np.sort(arr)
还可以按轴排序，比如二维的按1轴（行）来排
arr.sort(1)

3.唯一化，类似取到集合
np.unique(names)

七、文件输入和输出

1.np.save,保存到以.npy为后缀的文件中
arr=np.arange(10)
np.save('some_array.npy',arr)#前面是名，后面是数据，这里扩展名可以省略。

2.np.load,读取数组
arr1=np.load('some_array.npy')#1.这里扩展名不可省；2.载入时最好指定一个返回值。

3.把多个数组压缩保存
np.savez('array_archive.npz',a=arr,b=arr)#后缀名是.npz

4.如果加载，会得到类似字典的对象，对象会对各数组延迟加载。
arch=np.load('array_archive.npz')
arch['b']来提取延迟加载的数组

八、线性代数

1.x,y相乘
x.dot(y),np.dot(x,y)

2.numpy.linalg是专门进行矩阵分解运算以及求逆和行列式的。
mat=X.T.dot(X）

3.常用函数
diag：以一维数组的形式返回方阵的对角线元素，或将一维数组转换成方阵。（函数提示说没有）
dot：矩阵乘法
trace：计算对角线元素的和
det：计算矩阵行列式
eig：计算方阵的特征值和特征向量
inv：计算方阵的逆
pinv：计算矩阵的Moore-Penrose伪逆
qr：计算QR分解
svd：计算奇异值分解（SVD）
solve：解线性方程组Ax=b，其中A为一个方阵
lstsq：计算Ax=b的最小二乘解。

九、随机数

1.numpy.random
samples=np.random.normal(size=(4,4))
常用函数
seed：确定随机数生成器的种子
permutation：返回一个序列的随机排列或返回一个随机排列的范围
shuffle：对一个序列就地随机排列
rand：产生均匀分布的样本值
randint：从给定的上下限范围内随机选取整数
randn：产生正态分布（平均值为0，标准差为1）的样本值，类似MATLAB接口
binomial：产生二项分布的样本值
normal：产生正态（高斯）分布的样本值
beta：产生Beta分布的样本值
chisquare：产生卡方分布的样本值
gamma：产生Gamma分布的样本值
uniform：产生在[0,1]中均匀分布的样本值。

2.示范随机分布
步长为1000的随机游走。
nsteps=1000
draws=np.random.randint(0,2,size=nsteps)
steps=np.where(draws>0,1,-1)
walk=steps.cumsum()

看漫步的最大值和最小值
walk.min()
walk.max()

3.一次模拟多个随机漫步
nwalks=5000
nsteps=1000
draws=np.random.randint(0,2,size=(nwalk,nsteps))#取0或1
steps=np.where(draws>0,1,-1)
walks=steps.cumsum(1)
walks

计算所有随机漫步过程的最大值和最小值
walks.max()
wakls.min()

计算30和-30的最小穿越时间
#1.首先选出最终绝对值大于30的数组
hits30=(np.abs(walks)>=30).any(1)
hits30.sum()
#2.然后计算整个布尔型数组选出那些绝对值穿越了30的随机漫步行，并调用argmax在轴上获取穿越时间
crossing_time=(np.abs(walks[hits30])>=30).argmax(1)
crossing_time.mean

4.其他随机漫步函数
steps=np.random.normal(loc=0,scale=0.25,size=(nwalks,nsteps))

