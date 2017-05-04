# -*- coding: utf-8 -*-

#1.1 Welcome to my github https://github.com/huanzhizhixing
#1.2 Welcome to my blog http://blog.sina.com.cn/u/6053925177
#1.3 'zscf1inx_sz50'及相关资料 于 http://blog.sina.com.cn/s/blog_168d791390102ws32.html 下载
#将'zscf1inx_sz50'文件放置于G:/0.1data/5.hssz/

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
        print('\n'+"1")
        aa,ab,ac1,ac2,ab1=zhuan(ssdnp,ssdnumb)
        print('\n'+"2")
        ssdnp[aa][ab]=999
        aa,ab,bc1,bc2,bb1=zhuan(ssdnp,ssdnumb)
        print('\n'+"3")
        ssdnp[aa][ab]=999
        aa,ab,cc1,cc2,cb1=zhuan(ssdnp,ssdnumb)

        
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


