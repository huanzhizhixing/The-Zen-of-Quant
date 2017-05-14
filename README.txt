
前半部分为ssd法循环配对上证50股票，后半部分为常用到的命令总结。

————————————————————————————————————————————————————————————————————————————————————————————————————————————————

ssd法配对股票

一、文件及数据下载地址

1.1'zscf1inx_sz50'数据：
链接：http://pan.baidu.com/s/1jIidRSi 密码：jgvr

1.2 shizhan_a1_ssdpeidui.py 文件
https://github.com/huanzhizhixing/The-Zen-of-Quant
同名文件


二、运行结果

2.1开始逐个配对
量化学习1.4：最小距离配对法，代码及数据下载。

2.2逐个配对后，生成ssd参数矩阵

量化学习1.4：最小距离配对法，代码及数据下载。

2.3生成整理后表格
量化学习1.4：最小距离配对法，代码及数据下载。

三、源代码
 
# -*- coding: utf-8 -*-
#1.1 Welcome to my github https://github.com/huanzhizhixing
#1.2 Welcome to my blog http://blog.sina.com.cn/u/6053925177
#请将'zscf1inx_sz50'文件放置于G:/0.1data/5.hssz/
import os
import re
import math
import numpy as np
import pandas as pd
import tushare as ts
import datetime as dt
import statsmodels.api as sm
from arch.unitroot import ADF
import matplotlib.pyplot as plt
#从列表中获得数据位置
zzcflist=['zscf1inx_sz50']
#控制是否打开程序即开始运行，默认1为是，改为0则否。改为0后，只有调用才运行
ssdpei=1
#设定形成期的时间
formPeriod='2016-01-01:2017-01-01'
#设置配对组数量
d=0
e=99
f=e-d+1
#--------------------------------------------------------------------------
#一、class部分，计算ssd。
class PairTrading(object):
    def SSD(self,priceX,priceY):
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        returnX=(priceX-priceX.shift(1))/priceX.shift(1)[1:]
        returnY=(priceY-priceY.shift(1))/priceY.shift(1)[1:]
        standardX=(returnX+1).cumprod()
        standardY=(returnY+1).cumprod()
        SSD=np.sum((standardY-standardX)**2)
        return(SSD)
   
    def SSDSpread(self,priceX,priceY):
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        priceX=np.log(priceX)
        priceY=np.log(priceY)
        retx=priceX.diff()[1:]
        rety=priceY.diff()[1:]
        standardX=(1+retx).cumprod()
        standardY=(1+rety).cumprod()
        spread=standardY-standardX
        return(spread)
#-----------------------------------------------------------------------
#二、S配对方法
#2.1主体部分，循环导入。
def SSDpeidui():
    for aaa in zzcflist:
        num=pd.read_csv('G:/0.1data/5.hssz/'+aaa+'.csv',index_col='date')
        num.index=pd.to_datetime(num.index)
        nonum=not(num.isnull().values.any())
        a=len(num.ix[1,:])
        aa=str((f**2-f)/2)
        ssdwz=str('ssdwz'+aaa)
        ssdwz=np.zeros((a,a))
        if nonum==True:
            for b in range(0,a):
                bb=num.ix[:,b].name
                bbb=[]
                bbb.append(bb)
                for c in range(0,a):
                    cc=num.ix[:,c].name
                    ccc=[]
                    ccc.append(cc)
                    if c>b and c>=d and b>=d and c<=e and b<=e:
                        ssdpeidui(aa,b,bb,c,cc,ssdwz,num)
                    else:
                        pass
        print("ssdwz",ssdwz)
        ssdwwz=pd.DataFrame(ssdwz)
        ssdwwz.to_csv('G:/0.1data/5.hssz/'+aaa+'_ssd'+'.csv',encoding='gbk',date_format=True)
        ssdzl()
#2.2 ssd部分
#处理日期格式及计算ssd
def ssdpeidui(aa,b,bb,c,cc,ssdwz,num):
    priceA=num[bb]
    priceB=num[cc]
    priceAf=priceA[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
    priceBf=priceB[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
    pt=PairTrading()
   
    SSD=pt.SSD(priceAf,priceBf)
    SSD=round(SSD,2)
    SSDSTR=str(SSD)
    ssdwz[b,c]=SSD
    print("\n"+bb+"与"+cc+"的"+"SSD:"+SSDSTR)
    ccc=str(c)
    bbb=str(b)
    print("b:"+bbb+".c:"+ccc+"   "+aa)
#2.3 整理原ssd表格，并添加名称。
def ssdzl():
    for aaa in zzcflist:
        num=pd.read_csv('G:/0.1data/5.hssz/'+aaa+'.csv',index_col='date')
        ssdnum=pd.read_csv('G:/0.1data/5.hssz/'+aaa+'_ssd'+'.csv',index_col=0)
        for i in range(0,len(num.ix[1,:])):
            if i <=999:
                ii=str(i)
                ssdnum[num.ix[:,i].name]=ssdnum[ii]
                del(ssdnum[ii])
        ssdnum=ssdnum.T
        for i in range(0,len(num.ix[1,:])):
            if i <=999:
                ii=str(i)
                ssdnum[num.ix[:,i].name]=ssdnum[i]
                del(ssdnum[i])
        ssdnum=ssdnum.T
        ssdnum.to_csv('G:/0.1data/5.hssz/'+aaa+'_ssd_zl'+'.csv',encoding='gbk',date_format=True)
        ssdxz()
#2.4 寻找合适标的，列出最优配对排名。
def ssdxz():
    for aaa in zzcflist:
        ssdnumb=pd.read_csv('G:/0.1data/5.hssz/'+aaa+'_ssd_zl'+'.csv',index_col=0)
        #防止0知干扰
        ssdnumb=ssdnumb.replace(0,999)
        #用np来定位最小值位置
        ssdnp=np.array(ssdnumb)
        #minre1=np.where(ssdnp==np.min(ssdnp))
        #得到位置后，定位df的相关股票信息
        aa,ab,ac1,ac2,ab1=zhuan(ssdnp,ssdnumb)
       
#2.5反复删去最小值
def zhuan(ssdnp,ssdnumb):
    minre1=np.where(ssdnp==np.min(ssdnp))
    a1=int(minre1[0])
    a2=int(minre1[1])
    print(a1,a2)
    ab1=str(ssdnumb.ix[a1,a2])
    ac1=ssdnumb.ix[a1,:].name
    ac2=ssdnumb.ix[:,a2].name
    print ("配对："+ac1+"/"+ac2+"   SSD:"+ab1)
    return a1,a2,ac1,ac2,ab1

if ssdpei==1:
    SSDpeidui()
 
 
四、评述
利用循环实现了上证50指数内各股的SSD求值，并选出最合适的一个来反馈出配对要求，实现目标要求。


————————————————————————————————————————————————————————————————————————————————————————————————————



数据分析的基本命令



目录
一、pandas	2
二、数据规整化	4
三、数据聚合与分组运算	9
四、时间序列	13
五、金融数据应用	19
六、pandas的绘图	24
七、numpy	25
八、mysql	32
九、自动量化投资者的基本功	36





 

一、pandas

pandas的序列和表格引入
import pandas as pd
import Seires, DataFrame form pandas

一、Series

1.Series创建及属性
#Seires的首字母要大写
obj=Series([4,7,-5,3])
obj.values
obj.index
或
obj2=Series([4,1,5,4],index=['a','b','c','d'])

2.索引选值
选单独一个值
obj2['a']
按顺序选不同的值
obj2[['c','a','d']]#这里是有两个中括号，类似花式索引。

3.基本数值运算，np结合
obj2[obj2>0];obj2*2;
np.exp(obj2)
'b' in obj2

4.检查缺失值
pd.isnull(obj)
pd.notnull(obj)

5.自动根据索引运算
obj3+obj2

6.为列及其索引起名
obj4.name = 'population'
obj.index.name='state'

二、DataFrame
#有两处大写

1.构建DataFrame
数据、列名、索引
frame=DataFrame(data,column=['year','state','pop','debt'],index=['one','two','three','five'])

2.查询列
frame.year
frame['year']

3.属性
frame.column
frame.values返回一个ndarray的二维数组。


4.嵌套字典，外层字典的键作为列，内层字典作为行
pop={'nevada':{2001:2.4,2002:2.9},'ohio':{2000:1.5,2001:2.7}}
frame=DataFrame(pop)

5.命名
frame.index.name='year';frame.column.name='state'

6.添加新的列
为不存在的列赋值会创建出一个新列，关键词del用于删除列。
创建一个叫eastern的列，内容是True和False，判断state里是否严格等于Ohio
frame['eastern'] = frame.state =='Ohio'

三、索引
索引是针对轴标签的，包括列标签和行标签。
构建DataFrame或Series时，所用到的任何数组或其他序列的标签都会被转化为一个Index。
Index
1.







 
二、数据规整化

一、数据库风格的DF合并
1.merge和join通过一个或多个键将行连接起来。
pd.merge(df1,df2,on='key')#如果没有指定，merge就会将重叠列的列名当作主键。最好显式指定一下

2.左右不同时
pd.merge(df1,df2,left_on='lkey',right_on='rkey')
pd.merge(df1,df2,on=['key1','key2'],how='outer')#利用列表来表示
默认情况下，是内部连接，结果是交集。

3.设置连接方式，left，right，outer
pd.merge(df1,df2,on='key',how='left')

4.对重复列名的表示,suffixes
pd.merge(left,right,on='key',suffixes=('_left','_right')

5.连接键位于索引中，left_index=True,right_index=True,来选择
pd.merge(left,right,left_on='key',right_index=True,how='outer')
#把left里的key列和right里的索引合并,合并方式为外联结。

6.如果是层次化索引，需要用列表说明
pd.merge(left,right,left_on=['key1','key2'],right_index=True)

7.如果同时都是索引来联结，则两个都标上。（这时候似乎不用on了）
pd.merge(left,right,left_index=True,right_index=True,how='outer')

8.join函数的方法联结
left.join(right,on='key')

二、轴向连接concat
1.numpy的concatenate
np.concatenate([arr,arr],axis=1)

2.没有重叠索引的Series
pd.concat([s1,s2,s3])
#默认情况下，concat是在axis=0的轴上工作，并最终产生一个新的Series。
#如果把axis设置为1，则生成一个DF
pd.concat([s1,s2,s3],axis=1)

3.通过join_axes来指定要在其他轴上的索引
pd.concat([s1,s4],axis=1,join_axes=[['a','b','c','d']]

4.给每个合并的表分别加入名称以区分
result=pd.concat([s1,s1,s3],keys=['one','two','three'])
result.unstack()

5.对PD的合并
df1=DataFrame(np.arange(6).reshape(3,2),index=['a','b','c'],columns=['one','two'])#1.本身有index；2.columns本身后面有s
df2=DataFrame(np.arange(4).reshape(2,2),index=['a','c'],columns=['three','four'])
pd.concat([df1,df2],axis=1,keys=['level1','level2'],names=['upper','lower'])
#横向合并(添加列)，合并时用index来，并且指定keys来分别对df用level1和level2来标注。upper和lower分别对应level1和level2的层次索引。

6.join='inner'内联结，会去除NaN值

7.与当前工作无关的行索引。ignore_index=True
pd.concat([df1,df2],ignore_index=True)
#合并时会直接照着列排下去

三、合并重叠数据、轴向旋转

1.利用NumPy的where函数
np.where(pd.insnull(a),b,a)
#在a处为Nan的地方放入b，其他地方是a。相当于以a为基础来合并b值。

2.Series里combine_first
b[:-2].combine_first(a[2:])
#以b倒数第二项之前的数据为基础，合并a第三项及之后的数据

3.DataFrame里的combine_first
df1.combine_first(df2)
#以df1为基础，空值合并为df2相关数值。

4.stack将数据列旋转为行；unstack将数据行旋转为列。
对于二维PD，列转为行，会得到一个Series；而对于双重索引的Series，由unstack可以把Series变为二维PD。
resutlt=data.stake()
#默认情况下是对最内层进行旋转，指定层次可以由外层开始旋转。层次的编号是由外向内的。
#例如，state是最外层，number是内层，想要把最外层的state旋转出来
result.unstack(0);result.unstack('state')#指定名比较好，不会乱。

5.如果不是两个Series都有的话，unstack会引入缺失值。而stack时默认会把缺失值删去。
#如果想保留缺失值，则需要设置 dropna=False
data.unstack().stack(dropna=False)

6.对DataFrame进行unstack时，作为旋转轴的级别将会成为结果中的最低级别（最内层）（这里有疑问，因为最内层不是0么？那怎么之前unstack（0）时是对最外层呢？）
df.unstack('state').stack('side')
#state由行专向列的最内层，side由列转向行的最内层。

7.对MySQL里的长格式转为宽格式-pivot
时间序列数据通常以长格式（固定架构：列名和数据关系）储存在数据表中。
固定架构的好处：随着表中数据的添加或减少，item列中的值的种类能够增加或减少。（没明白，其他的就不能增减了？）
缺点：长格式的数据操作起来不轻松。
DataFrame特点：不同的item值分别形成一列，date列中的时间值则用作索引。

8.pivot转MySQL为DF
#原数据表只有三列，data可做行，item做列，最后value是值。
pivoted = Idata.pivot('date','item','value')
#转换后，变为date行，item列，value值的DF

9.对于多个value值
忽略最后一个参数，则value变为列的最外层
pivoted=Idata.pivot('date','item')
其他不同的列形成最外方的索引。


四、数据转换

1.移除重复值drop_duplicates
duplicated,返回布尔值
drop_duplicates,返回一个移除了重复行的DataFrame
#注意这两个命令的拼写，一个是加d，一个是加s。
data.drop_duplicates()
#默认判断全部列，如果只是希望对某一列判断
data.drop_duplicates(['k1'])
#默认是判断和保留第一个，如果是要保留最后一个，则take_last=True
data.drop_duplicates(['k1','k2'],take_last=True)

2.函数映射map，产生新的列
data=DataFrame({'food':['bacon','pulled pork','bacon','Pastrami','corned beef','Bacon','pastrami','honey ham','nova lox'],'ounces':[4,3,12,6,7.5,8,3,5,6]})
data
#添加一个肉类食物来源的种类。这里字典都是x：y，因此，左边是肉，右边是来源值。
meat_to_animal = {'bacon':'pig','pulled pork':'pig','pastrami':'cow','corned beef':'cow','honey ham':'pig','nova lox':'salmon'}
#Series里的map可以接受一个函数或含有映射关系的字典型对象。
#处理时，先转换大小写，然后转换对应值
data['animal']=data['food'].map(str.lower).map(meat_to_animal)
#或者，直接传入一个lambda函数
data['food'].map(lambda x: meat_to_animal[x.lower()])

3.替换值replace，替换现有值
data=Series([1.,-999.,2.,-999.,-1000.,3.])
data.replace([-999,-1000],np.nan)#把-999和-1000都转换成nan值
data.replace([-999,-1000],[np.nan,0])#分别转换为不同的值
#通过字典的方式转换
data.replace({-999:np.nan,-1000:0})

4.对索引轴数据处理map,rename,inplace=True
修改原数据值，map
data=DataFrame(np.arange(12).reshape((3,4)),index=['Ohio','Colorado','New York'],columns=['one','two','three','four'])
data.index.map(str.upper)#并未修改，只是看看
data.index=data.index.map(str.upper)#修改了

5.创建数据集的转换版
data.rename(index=str.title,columns=str.upper)
#这里对索引和列重新命名了，用的是str的内部函数title和upper。
data.rename(index={'Ohio':'Indiana'},columns={'three':'peekaboo'})
#分别对index和columns内的数据进行替换

6.rename操作默认没有直接更改原index，如果要就地修改，需要inplace=True
_ = data.rename(index={'Ohio'='Indiana'},inplace=True)
#这里用'_'能够总是返回DataFrame的引用，挺神奇
#设置inplace=True,就能够就地修改原表值了。

7.对数据分组cut（可应用于通达信强度值）
 7.1#有些MySQL数据库的味道
	ages=[20,22,25,27,21,23,37,31,61,45,41,32]
	bins=[18,25,35,60,100]#设定分组标准
	cats=pd.cut(ages,bins)#对ages数值应用以bins分类标准的cut分组操作
	#检查cats在各个分组中的数目
	pd.value_counts(cats)


 7.2#默认为左开右闭，可修改开闭区间。例如修改为左闭右开，right=False

 7.3#设置分组名称
 group_names = ['Youth','YoungAdult','MiddleAged','Senior']
 pd.cut(ages,bins,label=group_names)#这里是label，而不是lable

8.qcut，利用样本分位数来分组
按照分位数来切割数据，获得大小基本相同的块
data=np.random.randn(1000)#正态分布
cats=pd.qcut(data,4)#按四分位数进行切割
pd.qcut(data,[0,0.1,0.5,0.9,1]#按照自己设置的分位数进行分组

9.过滤异常值基本是数组运算


五、其他

1.利用take来重排
df=DataFrame(np.arange(5*4).reshape(5,4))
sampler=np.random.permutation(5)#生成从（0-4）的随机整数数列
df.take(sampler)#按照列来变换数据顺序
df.take(np.random.permutation(len(df)),[:3])#获取跟表长度值一样的值，重排后只保留前3个。

2.计算哑变量dummy(哑变量是另一矩阵，类似于一种分类
df=DataFrame({'key':['b','b','a','c','a','b'],'data1':range(6)})
pd.get_dummies(df['key'])

3.字符串操作

4.正则表达式











 

三、数据聚合与分组运算
SQL能方便的流行的原因是可对数据进行连接、过滤、转换和聚合，而pandas这方面更强大。

一、基础应用
groupby技术，split-apply-combine,拆分-应用-合并技术。

1.groupby生成的是中间过程数据，并非一些值。
df=DataFrame({'key1':['a','a','b','b','a'],'key2':['one','two','one','two','one'],'data1':np.random.randn(5),'data2':np.random.randn(5)})
grouped=df['data1'].groupby(df['key1'])
#还可以写成
grouped=df.groupby('key1')['data1']#这是上面的简写模式
#这种单独对某一列操作会对大数据表有利。

#按照key1列聚合data1列的值。可以用mean()等函数来看调用过程
grouped.mean()

2.聚合多个函数，并把mean值生成新的数组
means = df['data1'].groupby([df['key1'],df['key2']]).mean()
#简写模式：means = df.groupby(['key1','key2'])['data2']
#生成的means会有key1和key2两个索引，可以用unstack()将之一转换到列上。

3.还可以按照数组名来
states = np.array(['Ohio','California','California','Ohio','Ohio'])
years = np.array([2005,2005,2006,2005,2006])
df['data1'].groupby([states,years]).mean()
#奇怪了，这种怎么就能直接跟数据联系上了。

4.其实第一个不要['data1']也行，这样就是所有数据聚合分析
df.groupby(['key1','key2']).mean()

5.在某列的单独聚合运算时，如对key1值，那么key2就相当于麻烦列，会被排除在外。

6.对列改名，然后对行聚合
mapping = {'a':'red','b':'red','c':'blue','d':'blue','e':'red','f':'orange'}
by_colum = people.groupby(mapping,axis=1)

7.使用多种方法，例如字符长度、外导字典聚合
people.groupby(len).sum()
key_list = ['one','one','one','two','two']
people.groupby([len,key_list]).min()
#由index的本身模长和外导入的key_list来进行聚合分析。

8.根据索引级别分组
涉及level
df.groupby(level='cty',axis=1).count()

9.样本分位数quantile
grouped = df.groupby('key1')
groupby['data1'].quantitle(0.9)


二、高级应用

1.通过定义函数进行分组
#定义函数后，将其传入aggregete或agg即可
def peak_to_peak(arr):
	return arr.max() - arr.min()
	
grouped.agg(peak_to_peak)
#这里可以定义一些复杂的函数，

2.对聚合结果运用describe
groupec.describe()

3.常用聚合函数
count:分组中非NA值的数量
sum，mean，median，std，var，min，max，prod（非NA积），first、last（第一或最后一个非na值）
#自定义聚合函数比常用的要慢

三、案例

1.餐馆小费
tips = pd.read_csv('ch08/tips.csv)
tips['tip_pct']=tips['tip']/tips['total_bill']
grouped=tips.groupby(['sex','smoker'])

2.#从整个groupby里选择出需要的列
grouped_pct=grouped['tip_pct']
grouped_pct.agg(mean)#利用agg来导入均值

3.#求多个值
grouped_pct.agg(['mean','std','peak_to_peak'])

4.#把后面的值名称给前者。
grouped_pct.agg([('foo','mean'),('bar',np.std)])
#或者写成grouped_pct.agg({'foo':mean,'bar':'np.std')#这里挺奇怪，变成由后面的传导给前面的了。

5.#对多列运用多个函数
functions= ['count','mean','max']
result=grouped['tip_pct','total_bill'].agg(functions)

6.无索引聚合as_index=False
tips.groupby(['sex','smoker'],as_index=False).mean()


四、transform及apply的用法

1.为DataFrame添加一个储存平均值的列
k1_means=df.groupby('key1').mean().add_prefix('mean_')
pd.merge(df,k1_means,left_on='key1',right_index=True)

2.people.groupby(key).transform(np.mean)

3.transform会将一个函数应用到各个分组，然后将结果放置到适当的位置上。如果各分组产生的是一个标量值，则该值会被广播出去。
def demean(arr):
	return arr - arr.mean()	
demeaned = people.groupby(key).transform(demean)

4.apply,一般性的‘拆分-应用-合并’
def top(df,n=5,column='tip_pct'):
	return df.sort_index(by=column)[-n:]
top(tips,n=6)
tips.groupby('smoker').apply(top)
tips.groupby(['smoker','day']).apply(top,n=1,column='total_bill')

5.禁止分组索引键
进行groupby，会将分组键和原始对象的索引键共同构成结果对象中的层次化索引，这时需要用group_keys=False
tips.groupby('smoker',group_keys=False).apply(top)

6.分位数和桶分析(???)
frame=DataFrame({'data1':np.random.randn(10000),'data2':np.random.randn(10000)})
factor=pd.cut(frame.data1)


五、透视表和交叉表
透视表根据一个或多个键对数据进行聚合，并根据行和列上的分组键将数据分配到各个矩阵区域中。
pivot_table,pandas.pivot_table
tips.pivot_table(rows=['sex','smoker'])
tips.pivot_table(['tip_pct','size'],rows=['sex','day'],cols='smoker',margins=True)
#首先指定聚合的数据tip_pct和size，行是sex和day，列是smoker，最终是否有全集为是。
#aggfunc=len，也是pivot_table中的一部分，表示可以用模长来聚合。
 

四、时间序列
date time 讲解

一、datetime模块

1.基本组合
from datetime import datetime
now=datetime.now()#获取现在的时间
now.year,now.month,now.day#分别为现在的年、月、日，奇怪的是返回为组合值。

2.时间差delta
delta=datetime(2011,1,7)-datetime(2008,6,24,8,15)
delta.day;delta.time#分别为差值的日、时间值

3.timedelta()日期差函数
start=datetime(2011,1,4)
start+timedelta(12)

4.字符串和日期值互换
#日期转为字符串
stamp=datetime(2001,1,3)
str.stamp#将文本转换为标准字符串，带day和time
stamp.strftime('%Y-%m-%d')#将日期为字符串格式，只有day。注意，这里年大写，月和日小写

#字符串提取为日期格式
value='2011-01-03'
datetime.strptime(value,'%Y-%m-%d')
#字符串转为日期是p，日期转为字符串是f

5.对多个日期转换
datestrs=['7/6/2001','8/6/2011']
[datetime.strptime(x,'%m/%d/%Y') for x in datestrs]

6.parser.parse
from deteutil.parser import parse
#字符串转为日期
parse('2011-01-03')
#dateutil能将几乎所有的字符串转为日期格式
parse('jan 31,1997 10:45 PM')
#可惜，中文不行

7.转换时日在月前
parse('6/12/2011',dayfirst=True)

8.to_datetime处理成组日期
pd.to_datetime(datestrs)

9.datetime函数定义
%Y 4位数年；%y 2位数年；%m 2位数月；%d 2位数日；%F %Y-%m-%d的简写；%D %m/%d/%y的简写


二、时间序列基础

1.构造时间序列
按日期构造1000个时间序列
longer_ts=Series(np.random.randn(1000),index=pd.date_range('1/1/2000',periods=1000))

2.选取某一年或某一月
longer_ts['2001']
longer_ts['2001-05']

3.切片选取
#取2001之后的
ts[datetime(2001,1,7):]
#取两日期之间的，前闭后开
ts['1/6/2011':'1/22/2011']

4.truncate用法
ts.truncate(after='1/9/2001')

5.对datetime操作
dates=pd.date_range('1/1/2000',periods=100,freq='W-WED')
long_df=DataFrame(np.random.randn(100,4),index=dates,columns=['Colorado','Texas','New York','Ohio']
#ix还能这么用！注意，后面的字符串是反过来的
long_df.ix['5-2001']

6.检查是否有重复索引
df.index.is_unique
#查看是哪一个
grouped=df.groupby(level=0)
grouped.count()

7.转换频率
ts.resample('D')
#把所有的‘日’补齐

8.生成日期频率date_range
index = pd.date_range('4/1/2012','6/1/2012')#这里是4月1日到6月1日。月和日的顺序又反过来了，真不习惯
pd.date_range(start='4/1/2012',periods=20)#生成开始后的20期数据
pd.date_range(end='6/1/2012',periods=20)#periods加s，表示复数。

9.生成每月最后一日数据M和最后一个工作日BM
pd.date_range('1/1/2012','12/1/2012',freq='M')
#这样不会出现每月30日和31日的错误，统一归为最后一日。

10.带有分钟的
pd.date_range('5/2/2012 12:34:12',periods=5,normalize=True)


三、时间频率
频率由一个基础频率和一个乘数组成，基础频率通常以一个字符串别名表示，对每个基础频率，都有一个被称为日期偏移量的对象与之对应。

1.M每月，H每小时，前面可以加数字，表示间隔
pd.date_range('1/1/2012','1/3/2012 23:59',freq='4h')
pd.date_range('1/1/2012',periods=10,freq='1h30min')

2.WOM日期：每月第几个星期几
rng=pd.date_range('1/1/2012','9/1/2012',freq='WOM-3FRI')

3.shift向前向后移动
ts.shift(2),ts.shift(-2)
ts.shift(2,freq='M')
ts.shift(3,freq='D'
#等价于 ts.shift(1,freq='3D')

4.时期及算术运算
p=pd.Period(2007,freq-'A-DEC')
p+5;p-2

5.时期的频率转换asfreq
p=pd.Period('2007',freq='A-DEC')
p.asfreq('M',how='start')#将其转换为2007最开始的一个月

6.合成日期index
data.year;date.quarter
index=pd.PeriodIndex(year=data.year,quarter=data.quarter,freq='Q-DEC')
data.index=index
data.infl

7.重采样（resampling）
rng = pd.date_range('1/1/2000',periods=100,freq='D')
ts=Series(randn(len(rng)),index=rng)
ts.resample('M',how='mean')

8.降采样（类似把1f数据规整为5f数据）
rng=pd.date_range('1/1/2000',periods=12,freq='T')
ts=Series(np.arange(12),index=rng)
ts.resample('5min',how='sum',closed='left')
#规整为5s数据，左边闭合（默认是右边闭合）。比如0，1，2，3，4，5，默认是1+2+……+4+5；更改为左边闭合就是0+1+2+3+4

9.OHLC重采样
ts.resample('5min',how='ohlc')

10.升采样
frame=DataFrame(np.random.randn(2,4),index=pd.date_range('1/1/2000',periods=2,freq='W-WED'),columns=['Colorado','Texas','New York','Ohio'])
frame[:5]
df_daily=frame.resample('D')
#填充值
frame.resample('D',fill_method='ffill')

11.在降采样中，目标频率必须是源频率的子时期（subperiod）;在升采样中，目标频率必须是源频率的超时期（superperiod）


四、时间序列绘图

1.基础
close_px_all=pd.read_csv(……)
close_px=close_px_all[['AAPL','MSFT','XOM']]

#对数据按照营业日期来整理，并按照前填充来做。
#疑问，外国的工作日和我国不同，直接用B来做会出错，那应该怎么维护？
#建立一个符合我国节假日的Series表？
close_px=close_px.resample('B',fill_method='ffill')

2.#任取一只来绘图
close_px['AAPL'].plot


3.#选取某一年，绘图三只
close_px.ix['2009'].plot()

4.取某一只的一段时间
close_px['AAPL'].ix['01-2011':'03-2011'].plot()

5.将日数据合并成季度数据plot
app_q = close_px['AAPL'].resample('Q-DEC',fill_method='ffill')
app_q.ix['2009':].plot()

6.移动平均rolling_mean
close_px.AAPL.plot()
pd.rolling_mean(close_px.AAPL,250).plot()
#默认情况下，rolling_mean需要指定数量

7.扩展窗口平均（expanding window mean）(对其概念不了解)
expanding_mean=lambda x: rolling_mean(x,len(x),min_periods=1)
#指定了x，x的长度，还有最小时期？
pd.rolling_mean(close_px,60).plot(logy=True)

8.指数加权窗口
#建立2*1，共享x、y的图，大小是12*7.
fig,axes=plt.subplot(nrows=2,ncols=1,sharex=True,sharey=True,figsize=(12,7))

#分别得出ma60和ewma60的值
aapl_px=close_px.AAPL['2005':'2009']
ma60=pd.rolling_mean(aapl_px,60,min_periods=50)
ewma60=pd.ewma(aapl_px,span=60)

#分别在0，1上画图
aapl_px.plot(style='k-',ax=axes[0])
ma60.plot(style='k--',as=axes[0])
axes[0].set_title('Simple MA')
aapl_px.plot(style='k-',ax=axes[1])
ewma60.plot(style='k--',as=axes[1])
axes[1].set_title('Exponentially-weighted MA')

五、进阶

1.二元移动窗口
计算相关系数
#单只股票
corr=pd.rolling_mean(returns.AAPL,spx_rets,125,min_periods=100)
#多只股票
corr=pd.rolling_mean(returns,spx_rets,125,min_periods=100)

2.自定义窗口
rolling_apply产生自设数组函数，要求为能从数组的各个片段中产生单个值。
计算AAPL2%回报率的百分等级
from scipy.stats import percentileofscore
score_at_2percent = lambda x: percentileofscore(x,0.02)
result=pd.rolling_apply(returns.AAPL,250,score_at_2percent)
result.plot()

3.性能及内存
视图和低频率数据的运算进行了很大的优化，能用pandas尽量用pandas。
 

五、金融数据应用
处理金融数据时，最费神的一个问题是“数据对齐”，MATLAB和R需要花大量时间来对齐数据。
pandas还能在算术中自动对齐数据。

一、相乘和合并
1.两个DF相乘，拥有相同的index和columns
prices * volume

2.通过一组索引不同的Series构建DF
s1,s2,s3分别为Series，index是通过list把face拆开
DataFrame({'one':s1,'two':s2,'three':s3},index=list('face'))

3.Q-DEC\Q-SEP\Q-FEB
Q-DEC,12月底为年度结尾
Q-SEP,9月底为年度结尾
Q-FEB,2月底为年度结尾


二、resample和reindex,at_time
1.频率不同的时间序列运算resample和reindex
resample将数据转换到固定频率
reindex使数据符合一个新的索引

2.resample由低频到高频
ts1=Series(np.random.randn(3),index=pd.date_range('2012-6-13',periods=3,freq='W-WED'))
ts1.resample('B')
ts1.resample('B',fill_method='fill')

3.使用reindex变换一个表的索引
ts2是一个表，里面索引与ts1不一样。想要把ts1索引弄成跟ts2一致。
ts1.reindex(ts2.index,method='ffill')
#含义是ts1索引变换成ts2的，取值是向前填充。
ts2+ts1.reindex(ts2.index,method='ffill')
#随后可以用ts2和ts1运算了

4.转换计算年度数据和季度数据
gdp为季度数据，infl为年度数据，将infl转换为与gdp相同的高频数据
而gdp是Q-SEP格式的，12月底的表示为13Q1
gdp=Series([1.78,1.94,2.08,2.01,2.15,2.31,2.46],index=pd.periods_range('1984Q2',periods=7,freq='Q-SEP'))
infl=Series([0.025,0.045,0.037,0.04],index=pd.periods_range('1984',periods=4,freq='A-DEC'))
#先把infl转换成季度数据，然后再重建索引并前向填充
infl_q=infl.asfreq('Q-SEP',how='end')
infl_q=reindex(gdp.index,method='ffill')

5.抽取固定时间点的数据at_time
from datetime import time
ts[time(10,0)]
ts.at_time(time(10,0))
#between_time，选取两个时间点之间的数据
ts.between_time(time(10,0),time(10,1))

三、拼接多个数据源、收益率

1.在特定时间点上，从一个数据源切换到另一个数据源上.concat
spliced=pd.concat([data1.ix[:'2012-06-14'],data2.ix['2012-06-15':]])

2.用另一个时间序列对当前时间序列中的缺失值“打补丁”。conbime_first
spliced_filled = spliced.conbime_first(data2)
#用data2里的数据填充spliced里的缺失值。

update来就地更新
spliced.update(data2,overwrite=False)

用索引填充
cp_spliced[['a','c']]=data1[['a','c']]

3.收益率和累计收益cumprod
return=price.pct_change()
ret_index=(1+returns).cumprod()
#这里的ret_index并不是index，而是一个Series
ret_index[0]=1
ret_index
#计算制定时期的收益率
m_returns=ret_index.resample('BM',how='last').pct_change()
m_returns['2012']

#或者通过重采样聚合
m_rets=(1+returns).resample('M',how='prod',kind='periods')-1

#将股息率加到每日收益率里
returns[dividend_dates]+=dividend_pcts


四、分组变换和分析
利用数据集进行分组变换分析
生成随机投资组合

1.首先随机生成1000个股票代码
import random ;
#seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，
#如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
#伪随机数？
random.seed(0)
import string

N=1000
def rands(n)
	#choices是字符串的一个组成？
	choices=string.ascii_uppercase
	#怎么就连接起来了？
	return ''.join([random.choice(choices) for _ in xrange(n)])
#生成5位数的代码？
tickers=np.array([rands(5) for _ in xrange(N)])

2.创建一个含有3列的DF来承载数据
M=500
df=DataFrame({'Momentum':np.random.randn(M)/200 + 0.03,
				'Value':np.random.randn(M)/200 + 0.08,
				'ShortInterest':np.random.randn(M) /200 - 0.02},
				index=tickers[:M])

3.创建行业分类
ind_names=np.array(['FINANCE','TECH'])
#从0到N的一个序列，这个len（ind_names）是怎么回事？
sampler=np.random.randint(0,len(ind_names),N)
#生成Series，ind_names里的sampler列？index是股票名，这个列名为industry
industries=Series(ind_names[sampler],index=tickers,name='industry')

4.根据行业进行聚合变换
by_industry=df.groupby(industries)
by_industry.mean()
by_industry.describe()

5.进行行业内标准化处理
def zscore(group):
	return (group - group.mean())/group.std()
df_stand=by_industry.apply(zscore)
这样处理后，各行业的平均值为0，标准差为1.
df_stand.groupby(industries).agg(['mean','std'])
#也可以通过内置变换函数处理
ind_rank=by_industry.rank(ascending=False)
ind_rank.groupby(industries).agg(['min','max'])

6.在股票的投资组合定量分析里，排名和标准化是一种很常见的变换运算组合，可以将rank和zscore合在一起完成。
by_industry.apply(lambda x: zscore(x.rank()))


五、分组因子暴露
将投资组合的持有量和性能（收益与损失）分解为一个或多个表示投资组合权重的因子。比如，beta系数。
案例是一个三因子的ols组合来恢复

1.设置权重构成及加噪声
from numpy.random import rand
fac1,fac2,fac3=np.random.rand(3,1000)#可以一下设置三个随机变量。
#生成1000个？
ticker_subset=tickers.take(np.random.permutation(N)[:1000])
#因子加权及噪声
port=Series(0.7*fac1 - 1.2*fac2 + 0.3*fac3 + rand(1000),index=ticker_subset)
factors=DataFrame({'f1':fac1,'f2':fac2,'f3':fac3},index=ticker_subset)

2.因子分析的方式
#相关性不行
factors.corrwith(port)
#标准方式是最小二乘回归，使用pandas.ols
pd.ols(y=port,x=factors).beta


六、动量交易
股票投资组合的性能可以根据各股的市盈率被划分为四份位，通过pandas.qcut和groupby可以实现。
1.计算收益率，并将收益率变换为趋势信号
px=data['Ady Close']
returns=px.pec_change()

#
def to_index(rets):
	index = (1+rets).cumprod()
	first_loc=max(index.notnull().argmax() - 1,0)
	index.values[first_loc]=1
	return index

#5期内的平均？
def trend_signal(rets,lookback,lag):
	signal=pd.rolling_sum(rets,lookback,min_periods=lookback - 5)
	return signal.shift(lag)

2.创建一个每周五动量信号交易策略
signal = trend_signal(returns,100,3)
trade_friday=signal.resample('W-FRI').resample('B',fill_method='ffill')
trade_rets=trade_friday.shift(1)*returns

to_index(trade_rets).plot()

3.观察什么时候表现最好。
按照不同大小的交易期波幅进行划分，年度标准差是计算波幅的一种方法。
可以通过计算夏普比率来观察不同波动机制下的风险受益。
vol=pd.rolling_std(retruns,250,min_periods=200)*np.sqrt(250)

#设置年度默认值为250，收益的均值和标准差之比
def sharpe(rets,ann=250):
	retrun rets.mean()/rets.std() * np.sqrt(ann)

#用qcut把vol划分为4等份，并用sharpe聚合
trade_rets.groupby(pd.qcut(vol,4)).agg(sharpe)


七、多只股票动量交易策略的投资组合
动量策略各种回顾期和持有期的夏普比率热图
（看不懂，回头再总结。）
 

六、pandas的绘图
pandas给精简了很多参数，直接用

一、线形图
1.Series绘图
s=Series(np.random.randn(10).cumsum(),index=np.arange(0,100,10))
s.plot()#可以直接用了

主要参数：
lable:图例标签； kind：绘图种类line、bar、barh、kde
logy：在y上使用对数  grid：显示轴网格线（默认打开）
xticks：用作x轴刻度的值；xlim：用作x轴的界限（例如[0,10]）
yticks: 用作y轴刻度的值；ylim：用作y轴的界限。

2.DataFrame绘图
df=DataFrame(np.random.randn(10,4).comsum(),
			columns=['A','B','C','D'],
			index=np.arange(0,100,10))			
df.plot()

主要参数：
subplots:将各个DataFrame列绘制到单独的subplot中
sharex：共用一个x轴；sharey：共用y轴
figsize：图像大小
title图像标题字符串
legend：添加一个subplot图例
sort_columns:以字母表顺序绘制各列


二、柱状图
1.Series
kind='bar';kind='barh'
data.plot(kind='bar',ax=axes[0],coler='k',alpha=0.7)

2.DataFrame
df.plot(kind='bar')
#使用stacked=True可为DataFrame生成堆积柱状图
df.plot(kind='barh',stacked=True,alpha=0.5)

3.使用s.value_counts().plot(kind='bar')显示Series中各值出现的频率
party_counts = pd.crosstab(tips.day,tips.size)
party_counts=party_counts.ix[:,2:5]
#注意！这里必须转换成浮点值，以防2.7中的整数除法问题
party_pcts=party_counts.div(party_counts.sum(1).astype(float),axis=0)
party_pcts.plot(kind='bar',stacked=True)


三、直方图和密度图
直方图：对值频率进行离散化显示的柱状图。数据点被拆分到离散的、间隔均匀的面元中，绘制的是各面元中数据点的数量。
tips['tip_pct']=tips['tip']/tips['total_bill']
tips['tip_pct'].hist(bins=50)
密度图：通过计算可能会产生预测数据的连续概率分布的估计而产生。一般是将该分布近似为一组核（如高斯分布等），因此密度图也被成为KDE，核密度估计。
调用时，用plot时，加上 kind=‘kde’
tips['tip_pct'].plot(kind='kde')
这两种图表经常画在一起，直方图以规格化形式给出（一边给出面元化密度），然后再在其上绘制核密度估计。


四、散布图
散布图是观察两个一维数据之间关系的有效手段。
plt.scatter(trans_data['m1'],trans_data['unemp']
plt.title('Changes in log %s vs. log %s' % ('m1','unemp'))


五、绘制地图


七、numpy

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

 

八、mysql
一、
1.重要内容 select from where group by having, order by limit.
2.like,union,视图、储存过程，游标，触发器，
3.连接 mysql， mysql -uroot -p
4.SHOW DATABASES； 显示现有的数据库，SHOW TABLES;显示在某一个库中的表。
5.用USE text 就能进入到数据库里面了。使用数据库 USE crashcourse；只有先打开数据库，才能读取里面的数据。
6.命令需要按 ；或者\g结束，不仅是enter。输入quit和exit能够退出命令行实用程序。
7.SHOW COLUMNS FROM customers；要求返回一个表名（表的基本信息，并非表里的内容。）
8.SHOW GRANTS;显示用户的权限。
9.select……from……；
10.where……in……

二、
1.通配符过滤：like，
2.Concat（）函数能够实现两个列之间的拼接。AS可以实现别名。
SELECT Concat（RTrim（vend_name),‘（‘，Rtrim（vend_country），’）’） AS vend_title
FROM vendors
ORDER BY vend_name
3.MYSQL的时间格式必须是 yyyy-mm-dd。如果使用日期，就用Date（）；时间就用Time（）。
4.某一特定年和月，用Year（）和Month（）来确定。
5.聚集函数，AVG(),返回某列平均值；COUNT(),返回某列行数；MAX(),返回某列最大值，MIN(),返回某列最小值，SUM()，返回某列值之和。
6. SELECT AVG(prod_price) AS avg_price FROM products;
7. GROUP BY vend_id;按照vend_id来实施分组。
8.联结，把两个表按照相同的名称连接起来。
SELECT vend_name,prod_name，prod_price
FROM vendors，products
WHERE vendors.vend.id = products.vend_id
ORDER BY vend_name, prod_name;
联结时，实际上是将第一个表中的每一行和第二个表中的每一行配对。WHERE子句作为过滤条件，它只包含哪些匹配给定条件（这里是联结条件）的行。WHERE子句，第一个表中的每个行将与第二个表中的每个行配对，而不管它们逻辑上是否可以配在一起。
9.自联结
SELECT p1.prod_id,p1.prod_name
FROM products AS p1,products AS p2
WHERE p1.vend_id = p2.vend_id
  AND p2.prod_id = 'DTNTR';
同一个表内连接，DTNTR找到同一个厂商的其他产品。
10.外部联结——能够包含没有关联行的数据。
比如有的人没有订购产品，也得在表中体现出来。
SELECT customers.cust_id,order.order_num
FROM customers INNER JOIN orders
 ON customers.cust_id = orders.cust_id
上面是内部连接

外部联结
SELECT customers.cust_id,order.order_num
FROM customers LEFT OUTER JOIN orders
 ON customers.cust_id = orders.cust_id
以左边的为目录，来建立联结。以右边为目录时，是RIGHT。

三、
1.如果是连接，就不用WHERE来表示了，用ON来表示。
2.UNION可以把多个SELECT语句合成一个结果集。如果有ORDER BY，那最后一个起作用。
3.全文搜索可以用LIKE(),也可以用 Match()   Against()
SELECT note_text
FROM productnotes
WHERE Match(note_text) Against('rabbit');
用where的话只给有rabbit的行，如果在select里面，则给出所有的行。
4.布尔文本搜索
SELECT note_text
FROM productnotes
WHERE Match(note_text) Against('heavy'IN BOOLEAN MODE);
5.搜索时，忽略词中的单引号。例如，don't，所因为dont
6.插入数值。INSERT INTO customers(……） VALUES(……）
7.更新数据，UPDATE。注意，随后是 E 不是A. UPDATE ……SET……
更新某一个数据 
UPDATE customers
SET cust_email = 'elmer@fudd.com'
WHERE cust_id = 10005;
8.删除数据
DELETE FROM customers
WHERE cust_id = 10006;
9.创建表 
CREATE TABLE,新标的名字，在关键字CREATE TABLE 之后给出
mysql> CREATE TABLE mystuff
    -> (
    ->  cust_id      int   NOT NULL   AUTO_INCREMENT,
    ->  cust_name    char(50)  NOT NULL,
    ->  PRIMARY KEY(cust_id)
    -> )ENGINE=InnoDB;
Query OK, 0 rows affected (0.04 sec)
可以给定两个主键，如 PRIMARY KEY (order_num，order_item）
10.MYSQL可以有多个引擎，InnoDB是事务处理引擎，不支持全文搜索。
M有ISAM是性能高的引擎，支持全文本搜索，但是不支持事务处理。事务处理是全都执行或者全都不执行，更稳定一些。因此，一般来说，用InnoDB更适合。

四、

1.更新表：ALTER TABLE，使得表结构改变。这个会常用，比如金融数据里用机器学习时会添加新的列。
ALTER TABLE vendors
ADD vend_phone CHAR(20)
添加一个叫vend_phone的列，并给出了数据类型。
2.ALTER TABLE来定义外键。
ALTER TABLE orderitems
ADD CONSTRAINT fk_orderitems_orders
FOREIGN KEY(order_num) REFERENCES orders (order_num)
3.删除表 DROP TABLE customers2；
4.重命名表 RENAME TABLE customers2 TO customers;
5.视图：虚拟的表，只包含使用时动态检索数据的索引。可以很好的利用重复SQL语句。
6.CREATE VIEW,创建视图；SHOW CREATE VIEW viewname，查看创建视图语句；DROP删除试图，DROP VIEW viewname；更新视图时，可以直接用CREATE OR REPLACE VIEW 来。
7.视图可以用来简化复杂的联结。
8.存储过程，利用多条SQL语句封装。可以接受输入和输出变量。
CALL productpricing(@pricelow,@pricehigh,@priceaverage);
9.创建存储过程
CREATE PROCEDURE productpricing()
BEGIN
   SELECT Avg(prod_price) AS priceaverage
   FROM products;(这里有一个分号)
END;（这里也有一个分号）
10.所有MYSQL变量都必须以@开始，传递到mysql的存储过程中的参数个数必须严格相等。显示参数可以用 select来，eg：select @pricehigh


五、
1.检查存储过程用 SHOW CREATE PROCEDURE
2.游标：游标是一个存储在MYSQL服务器上的数据库查询，它不是一条SELECT语句，而死被该语句检索出来的结果集。在存储了游标之后，应用程序可以根据需要滚动或浏览器中的数据。
3.创建游标
CREATE PROCEDURE processorders()
BEGIN
    DECLARE ordernumbers CURSOR
    FOR
    SELECT order_num FROM orders;
END;
4.打开游标 OPEN CURSOR
5.触发器，是mysql相应DELETE\INSERT\UPDATE语句时自动执行的一条MYSQL语句。使用触发器，把更改（如果需要，甚至还有之前和之后的状态）记录到另一个表非常容易。
6.事务处理：transaction processing.t事务ransaction,一组SQL语句；回退（rollback）撤销指定SQL语句过程；提交（commit）将未存储的SQL语句结果写入数据库表；保留点（savepoint）事务处理中设置的临时占位符（placeholder），可以对它发布回退。
7.每个保留点都有唯一的名字，eg：SAVEPOINT delete1；，ROLLBACK TO delete1；
8.在工作中不要随意使用root，应该创建一系列的帐号，有的用于管理，有的供用户使用，有的供开发人员使用。
9.创建用户。CREATE USER ben IDENTIFIED BY 'p@$$w0rd';显示用户权限，SHOW GRANTS FOR ben；授予搜索权限：GRANT SELECT ON text.* TO ben;更改用户密码; SET PASSWORD FOR ben = Password('n3w p@$$w0rd');
10.备份数据 命令行 mysqldump转储。

六、
1.维护：ANALYZE TABLE;诊断启动服务；查看日志文件；
2.改善性能，存储过程比一条一条执行的快；select用union语句代替；用FULLTEXT而不是LIKE;

 


九、自动量化投资者的基本功

第一部分 编程工具

一、Python
1.Python是基本功，易用、包多、粘合性强。MATLAB并行计算需要加额外的包；R基本是完全封装的，另类策略不好做；C++等做高频时再做。
2.Python基本：1.基本函数用法；2.列表、元组、字典；3.函数；4.类；5.条件及循环控制。
3.Python扩展：1.numpy；2.pandas；3.matplotlib；4.sklearn。

二、PowerShell
1.比cmd强大，而且在linux上很好。
2.安装包和控件会常常用到。
3.需要在里面执行常见命令，如改变目录的CD等。

三、数据库MySQL及Spark
1.MySQL的基本用法：包括连接、建库、提取及储存数据。
2.虽然pandas和MySQL很多命令类似，但是MySQL有保存数据功能。
3.MongoDB在处理期货快速读写数据里可能比MySQL强，但MongoDB的学习应该建立在MySQL已经掌握的基础上，因为MySQL通用性更强也更基础。
4.进阶大数据时，掌握Spark，速度会快很多。


第二部分 数据来源及整理分析

一、获取数据
1.国内以tushare为主，RQalpha据说也可以获取，vnpy获取期货数据。
2.国外很多，比如雅虎金融等。
3.一些量化平台，如米筐、优矿、quanttopian。

二、数据整理及分析
1.运用pandas进行整合。
2.保存到数据库MySQL。


第三部分 策略、回测及资金管理

一、构建策略
1.均值回归策略
2.多因子策略（FAMA三因子等）
3.动量反转
4.波动率策略
5.基本面策略
6.技术面策略
7.轮动策略
8.缠论策略
9.机器学习策略
10.资产定价策略
11.其他策略

二、回测
1.确定回测群体。
2.观察夏普比率、挫跌等信息。
3.优化策略组合。

三、资金管理
1.利用凯利公式进行资金配置，半凯利或更低操作。
2.模拟交易：1.观看是否有用未来信息等错误行为；2.验证策略。

第四部分 实盘

一、实盘平台
1.找到券商API接口实现自动化交易，或者半自动来做中低频的日间交易。easytrader的银河接口，tushare的接口，vnpy的期货接口。
2.小额资金验证策略。
3.参加一些平台的比赛等

二、其他




 


