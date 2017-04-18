# -*- coding: UTF-8 -*-

#下载米筐回测包
#安装米筐回测包
PS C:\Users\Administrator> G:
PS G:\> cd rqalpha-master
PS G:\rqalpha-master> python setup.py install
running install
running bdist_egg
running egg_info
creating rqalpha.egg-info
writing requirements to rqalpha.egg-info\requires.txt


Running line_profiler-2.0\setup.py -q bdist_egg --dist-dir c:\users\admini~1\appdata\local\temp\easy_install-dky33w\line
_profiler-2.0\egg-dist-tmp-5mj6gv
error: Setup script exited with error: Microsoft Visual C++ 9.0 is required. Get it from http://aka.ms/vcpython27

#遇到问题，说VC没安装，很奇怪。先下载vc，安装时说已经有了。后来再看说vc有个转python27的包，应该安装那个。
#按照上面给出的地址，下载vcpython27，估计是把vc改写成python27的一个包。随后再次安装
PS G:\rqalpha-master> python setup.py install

#得到下面信息，就是安装好了
Using g:\programdata\anaconda2\lib\site-packages
Finished processing dependencies for rqalpha==0.3.14

#pip list一下
rqalpha (0.3.14)

#进入python，看能不能用
PS G:\rqalpha-master> python
>>> import rqalpha as ra

#没有报错，可行

#看源码
import string
string.__file__





