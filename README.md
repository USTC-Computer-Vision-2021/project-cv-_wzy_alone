[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6630551&assignment_repo_type=AssignmentRepo)
# 基于增强现实技术的基础使用

成员及分工
- 王泽雨 PB18061235
  - 单人队伍


## 问题描述

  增强现实 (Augmented Reality, AR) 是目前最热门的应用研究之一。增强现实的概念可以定义为将虚拟信息与真实世界巧妙融合的技术，其中真实世界的视图通过叠加的计算机生成的虚拟元素(例如，图像、视频或 3D 模型等)得到增强。为了叠加和融合数字信息(增强现实)，可以使用不同类型的技术，主要包括基于位置和基于识别的方法。
  选择做这个是对视频马赛克打码感到兴趣，想要复现类似的效果，试图用别的内容替换掉“马赛克”的内容，比如用表情等。

## 原理分析

分析上述抽象出来计算机视觉问题对应的原理，可以结合课件，也可以自己上网学习。不需要千篇一律的说教，可以讲出自己的理解，条理分明，最好做到让一个没学过计算机视觉的人都能大致了解这是怎么做到的。

如果需要在编辑器中插入数学公式，可以使用两个美元符 $$ 包裹 LaTeX 格式的数学公式来实现。如输入：
```
$$
\lim_{x\rightarrow0} \frac{\sin(x)}{x} = 1
$$
```

提交后，配合浏览器插件 [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima) 可以渲染出数学公式。助教端已安装，可以直接渲染，大家直接在报告种插入数学公式代码即可，或者采用别的方式插入公式，比如直接贴图片也行，但需要考虑美观。

$$
\lim_{x\rightarrow 0} \frac{\sin(x)}{x} = 1
$$

## 代码实现

尽量讲清楚自己的设计，以上分析的每个技术难点分别采用什么样的算法实现的，可以是自己写的（会有加分），也可以调包。如有参考别人的实现，虽不可耻，但是要自己理解和消化，可以摆上参考链接，也鼓励大家进行优化和改进。

- 鼓励大家分拆功能，进行封装，减小耦合。每个子函数干的事情尽可能简单纯粹，方便复用和拓展，整个系统功能也简洁容易理解。
- 尽量规范地命名和注释，使代码容易理解，可以自己参考网上教程。

插入算法的伪代码或子函数代码等，能更清晰地说明自己的设计，其中，可以用 markdown 中的代码高亮插入，比如：

```python
data = ["one", "two", "three"]
for idx, val in enumerate(data):
    print(f'{idx}:{val}')

def add_number(a, b):
    return a + b
```


## 效果展示

在这儿可以展示自己基于素材实现的效果，可以贴图，如果是视频，建议转成 Gif 插入，例如：

![AR 效果展示](demo/ar.gif)

如果自己实现了好玩儿的 feature，比如有意思的交互式编辑等，可以想办法展示和凸显出来。

## 工程结构

```text
.
├── code
│   ├── run.py
│   └── utils.py
├── input
│   ├── bar.png
│   └── foo.png
└── output
    └── result.png
```

## 运行说明

在这里，建议写明依赖环境和库的具体版本号，如果是 python 可以建一个 requirements.txt，例如：

```
opencv-python==3.4
Flask==0.11.1
```

运行说明尽量列举清晰，例如：
```
pip install opencv-python
python run.py --src_path xxx.png --dst_path yyy.png
npm run make-es5 --silent
```
