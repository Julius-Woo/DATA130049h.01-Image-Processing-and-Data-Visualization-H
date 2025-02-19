# 黑网吧可视分析
目标：构建一个完整的可交互的多视图的可视分析系统，从网吧上网登记数据中识别不同上网人群，探索上网人群的时空行为特征，检测非法上网行为与团伙上网行为，为寻找黑网吧提供线索。

## 项目结构
- 数据文件夹：`newdata`
  - `intcafe.csv`：处理后的网吧信息表
  - `hydata_swjl_0.csv`等：处理后的上网记录表
  - `provinces.csv`, `cities.csv`, `areas.csv`：区域代码表
  - 其他为各问题处理所需的数据
- 数据处理脚本：
  -`preprocess.py`：数据预处理
  -`prob1.py`：问题1数据处理
  -`prob2.py`：问题2数据处理
  -`prob3.py`：问题3数据处理
- 辅助支持文件：
  -`d3.v7.min.js`: D3库
  -`plot.js`：Observable Plot库
  -`china.json`：中国地图数据
  -`chongqing.json`：重庆地图数据
- 可视系统页面：
  - `prob1.html`：问题一和四
  - `prob2.html`：问题二
  - `prob3.html`：问题三

## 系统使用说明
### 问题一和四
用Live Server打开`prob1.html`，等待数据加载完成后，可以做如下交互：
- 左上方地图可以放大缩小，观察到不同区域的网吧数量分布。鼠标略过某网吧点，可以展示该网吧的名称、上网记录数和接纳未成年人的记录占比，点击该点可以在地图右侧表格区看到此网吧的（疑似）未成年人上网记录数，默认显示为全部数据统计结果，可以翻页。
- 点击地图上某点，会更新下方三个视图，表示该网吧记录中未成年人的画像。其中，中间的折线图可以由鼠标略过某点显示时间和人数。
### 问题二
用Live Server打开`prob2.html`，等待数据加载完成后，可以做如下交互：
- 默认展示全部数据的统计结果，点击左上方地图某省份，展示该省份的结果。
- 鼠标略过日历图的方格，可以看到该日期的统计结果。

### 问题三
用Live Server打开`prob3.html`，等待数据加载完成后，可以做如下交互：
- 鼠标略过点可以看到ID和上网记录数信息。
- 可以拖动点，改变位置。
- 在控制台中打印了社团划分结果。