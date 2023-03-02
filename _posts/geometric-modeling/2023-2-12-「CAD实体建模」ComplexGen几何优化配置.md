---
layout:     post
title:      "「CAD实体建模」 ComplexGen几何优化配置"
subtitle:   "ComplexGen Geometric Refinement Configuration"
date:       2023-2-12 12:00:00
author:     "Azaan"
header-img: "img/post-bg-geometric-modeling.jpg"
katex: true
tags:
    - CAD实体建模  - Linux系统配置
---





## ComplexGen思路介绍

> ComplexGen: CAD Reconstruction by B-Rep Chain Complex Generation (SIGGRAPH 2022)

ComplexGen是一个从点云数据重建B-Rep格式CAD表示的模型，它的重建过程分为三个步骤：

- 通过深度学习网络从输入点云中提取顶点、边、面的信息
- 通过gurobi约束求解器消除网络预测冲突
- 通过几何参数优化提高结果精度

在ComplexGen的开源代码中，第三部分几何优化只给出了visual studio的代码，意味着需要在windows上配置。但是经过笔者的探索，在windows上配置很难成功，因此笔者转化了思路，在linux上重新编译代码并运行，最终成功实现。

本文介绍如何在linux上配置ComplexGen的几何优化部分，同时也为在linux上跑visual studio项目提供了思路。



## 依赖库准备

### 安装libigl

```shell
git clone https://github.com/libigl/libigl.git
cd libigl
mkdir build
cd build
cmake ..
make
sudo make install
```

在安装过程中可能出现依赖库不存在的问题，按照igl官网的配置安装所有的依赖

```shell
sudo apt-get install git
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install libx11-dev
sudo apt-get install mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get install libxrandr-dev
sudo apt-get install libxi-dev
sudo apt-get install libxmu-dev
sudo apt-get install libblas-dev
sudo apt-get install libxinerama-dev
sudo apt-get install libxcursor-dev
```

### 安装eigen

```shell
sudo apt install libeigen3-dev
```



## GeometricRefine编译

在`GeometricRefine`文件夹下进入终端

```shell
mkdir build
cd build
cmake ..
make
```



### 问题一：`bigobj`

在`ComplexGen/GeometricRefine/GeometricRefine/CmakeList.txt`中注释掉bigobj一行

```cmake
# add_definitions(/bigobj)
```



### 问题二：`#  include "per_face_normals.cpp"`

![image-20230118193633060](https://azaan-zheng.github.io/img/2CAD/ComplexGen/image-20230118193633060.png)

```shell
cp *.cpp usr/local/include/igl
```



### 问题三：`#include <unsupported/Eigen/SparseExtra>`

![](https://azaan-zheng.github.io/img/2CAD/ComplexGen/2023-01-18-20-42-22.png)

需要查看eigen的安装位置，将相对地址改成绝对地址

我的eigen安装位置是 `/usr/include/eigen3/unsupported/Eigen`

因此将diag.cpp中的头文件引用

```c
#include <unsupported/Eigen/SparseEXtra>
```

修改为：

```c
#include </usr/include/eigen3/unsupported/Eigen/SparseEXtra>
```



### 问题四：`#include "raytri.c"`

![image-20230118205705293](https://azaan-zheng.github.io/img/2CAD/ComplexGen/image-20230118205705293.png)

解决方法，将对应文件复制到安装位置

```
cp raytri.c /usr/local/include/igl/
```



### 问题五：`extern"C" FILE * __cdecl __iob_func(void)`

![image-20230118210646191](https://azaan-zheng.github.io/img/2CAD/ComplexGen/image-20230118210646191.png)

解决：注释掉文件`SurfFitter.cpp`25-27行的这个函数

```cpp
extern"C" FILE * __cdecl __iob_func(void)
{
	return _iob;
};
```

通过`grep -rn "__cdecl __iob_func"`命令的输出可以看到，这个函数只在这里定义了一次

![image-20230118211337739](https://azaan-zheng.github.io/img/2CAD/ComplexGen/image-20230118211337739.png)

通过查询可知，该函数与Visual Studio的环境有关，我们这里无需这一项，因此直接注释



### 问题六：`std::sqrt(deltaDiv108)`以及`std::atan2(betaIm, betaRe)`

![image-20230118211649093](https://azaan-zheng.github.io/img/2CAD/ComplexGen/image-20230118211649093.png)

![image-20230118211743184](https://azaan-zheng.github.io/img/2CAD/ComplexGen/image-20230118211743184.png)

这个bug是因为数字类型转换的问题，在文件RootsPolynomial.h中的406-407行中，可以看到如下代码：

```cpp
Rational betaIm = std::sqrt(deltaDiv108);
Rational theta = std::atan2(betaIm, betaRe);
```

通过强制类型转换，替换为以下代码即可

```cpp
Rational betaIm = std::sqrt((double)(deltaDiv108));
Rational theta = std::atan2((double)(betaIm), (double)(betaRe));
```



### 问题七：`inputfile.substr(0, inputfile.find_last_of("."))`

![image-20230118212701419](https://azaan-zheng.github.io/img/2CAD/ComplexGen/image-20230118212701419.png)

字符串处理问题，不是核心代码，直接将其修改为`string`类型

初始469-470行代码为：

```c++
auto& inputfile = result["i"].as<std::string>();
auto& inputprefix = inputfile.substr(0, inputfile.find_last_of("."));
```

将其修改为：

```cpp
std::string inputfile = result["i"].as<std::string>();
std::string inputprefix = inputfile.substr(0, inputfile.find_last_of("."));
```



### 问题八：编译参数报错

![image-20230118213922070](https://azaan-zheng.github.io/img/2CAD/ComplexGen/image-20230118213922070.png)

问题出在cannot find的4个参数

使用`grep -rn "-llegacy_stdio_definitions"` 查看参数所在位置，可以看到位于同级目录下的`link.make`文件中：

![image-20230118214720568](https://azaan-zheng.github.io/img/2CAD/ComplexGen/image-20230118214720568.png)

该文件中的内容如下：

```
/usr/bin/c++ -rdynamic CMakeFiles/GeometricRefine.dir/src/CurveFitter.cpp.o CMakeFiles/GeometricRefine.dir/src/HOptimizer/HLBFGS.cpp.o CMakeFiles/GeometricRefine.dir/src/HOptimizer/HLBFGS_BLAS.cpp.o CMakeFiles/GeometricRefine.dir/src/HOptimizer/HLBFGS_Constraint.cpp.o CMakeFiles/GeometricRefine.dir/src/HOptimizer/HLBFGS_Hessian.cpp.o CMakeFiles/GeometricRefine.dir/src/HOptimizer/HLBFGS_Sparse_Matrix.cpp.o CMakeFiles/GeometricRefine.dir/src/HOptimizer/HOptimizer.cpp.o CMakeFiles/GeometricRefine.dir/src/HOptimizer/HOptimizer_Internal.cpp.o CMakeFiles/GeometricRefine.dir/src/HOptimizer/ICFS.cpp.o CMakeFiles/GeometricRefine.dir/src/HOptimizer/LineSearch.cpp.o CMakeFiles/GeometricRefine.dir/src/Helper.cpp.o CMakeFiles/GeometricRefine.dir/src/Mesh3D.cpp.o CMakeFiles/GeometricRefine.dir/src/NURBSFittingWrapper.cpp.o CMakeFiles/GeometricRefine.dir/src/SurfFitter.cpp.o CMakeFiles/GeometricRefine.dir/src/main.cpp.o -o ../../Bin/GeometricRefine   -L/root/autodl-tmp/ComplexGen/GeometricRefine/GeometricRefine/src  -L/root/autodl-tmp/ComplexGen/GeometricRefine/Lib  -L/root/autodl-tmp/ComplexGen/GeometricRefine/allquadricsdirect/lib/CLAPACK/lib_64  -Wl,-rpath,/root/autodl-tmp/ComplexGen/GeometricRefine/GeometricRefine/src:/root/autodl-tmp/ComplexGen/GeometricRefine/Lib:/root/autodl-tmp/ComplexGen/GeometricRefine/allquadricsdirect/lib/CLAPACK/lib_64 ../allquadricsdirect/liballquadricsdirect.a ../yaml-cpp/libyaml-cpp.a -llegacy_stdio_definitions -llibf2c -lclapack -lBLAS 
```

可以看到后面的四个链接参数，这些参数是windows下使用的，对应的替换为linux下的参数，也即将

```
-llegacy_stdio_definitions -llibf2c -lclapack -lBLAS
```

修改为：

```
-pthread -llapack -lblas
```

此处应当注意，如果未安装对应库应当安装对应的库，否则会出现`cannot find -llapack`类似报错：

```shell
sudo apt-get install liblapack-dev
sudo apt-get install libblas-dev
```



至此，在`Bin`文件夹中编译完成可执行文件`GeometricRefine`。



### 参考文献

1. ComplexGen开源代码： https://github.com/guohaoxiang/ComplexGen
2. ComplexGen原文：https://haopan.github.io/papers/ComplexGen.pdf
