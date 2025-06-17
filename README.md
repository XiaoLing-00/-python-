# -基于python的股票可视化分析-
一、实验目的
1.	掌握使用 Pandas 进行数据读取、清洗、合并、重塑等核心预处理操作。
2.	掌握利用 Matplotlib 和 Seaborn 库，将处理后的数据转化为信息丰富的可视化图表。
3.	学习并实践三种不同维度的可视化方法：
o	针对单个实体时间序列数据的堆叠柱状图。
o	针对多个实体时间序列数据的3D分层面积图。
o	针对多个实体多时间点截面数据的热力图。
4.	理解数据预处理（特征工程、数据重塑）与数据可视化之间的紧密联系。

二、实验仪器设备或材料
操作系统：Windows / macOS
编程环境：Anaconda3 (内置Jupyter Notebook/JupyterLab, Python 3.x)
核心库：Pandas, NumPy, Matplotlib, Seaborn


三、实验内容与步骤

本次实验分为三个部分，分别从不同维度对给定的股票数据集（存放于E:\code\Jupyter_study\股票画图\股票date文件夹）进行可视化分析。
高低图实验思路
•	目标：为单只股票创建一张图，直观展示其每日价格从最低价到最高价的波动范围。
•	方法：采用堆叠柱状图。底层柱子代表最低价，在其之上堆叠一个代表“价格范围（最高价-最低价）”的柱子。
•	关键技术：
1.	特征工程：从已有的最高价和最低价列计算出新的价格范围列。
2.	堆叠实现：使用Matplotlib的ax.bar()函数，通过设置bottom参数来实现堆叠效果。

3.	核心代码与解析

def create_stacked_bar_chart_yxl(excel_file_path_yxl, output_directory_yxl):
步骤1：数据加载与预处理
df_yxl = pd_yxl.read_excel(excel_file_path_yxl)
# ...（数据清洗代码，如转换日期、处理缺失值等）...
df_yxl['交易时间'] = pd_yxl.to_datetime(df_yxl['交易时间'])
df_yxl.sort_values(by='交易时间', inplace=True)  
  
步骤2：特征工程 - 计算价格范围
df_yxl['价格范围'] = df_yxl['最高价'] - df_yxl['最低价']

步骤3：绘图
fig_yxl, ax_yxl = plt_yxl.subplots(figsize=(12, 7))
x_indices_yxl = range(len(df_yxl)) # 使用数字索引作为X轴

绘制底层柱子（最低价）
ax_yxl.bar(x_indices_yxl, df_yxl['最低价'], color='#7A8B99', label='最低价')

绘制堆叠层（价格范围），关键在于 bottom 参数
ax_yxl.bar(x_indices_yxl, df_yxl['价格范围'], bottom=df_yxl['最低价'], color='#D8AAB7', label='价格范围')

...（图表美化代码，如设置标题、坐标轴、颜色等）...

步骤4：保存与显示
...（保存图片的代码）...
plt_yxl.show()
主程序入口
if name == "main":
file_path_yxl = r'E:\code\Jupyter_study\股票画图\股票date\1_雪人股份.xls'
output_path_yxl = r'E:\code\Jupyter_study\股票画图\output'
create_stacked_bar_chart_yxl(file_path_yxl, output_path_yxl)
•	解析：该代码首先对单个股票数据进行预处理，然后通过计算价格范围创造了用于堆叠的新数据。绘图时，两次调用ax.bar，第二次通过bottom参数将柱子精确地堆叠在第一层之上，清晰地展示了每日价格区间的构成。

________________________________________
3D走势图实验思路
•	目标：在三维空间中，同时展示多只股票的成交量随时间变化的趋势，并形成具有层次感的视觉效果。
•	方法：采用3D面积图（或称Ridgeline Plot）。每一只股票在Y轴上占据一个固定的“轨道”，其成交量在Z轴上以“山脉”的形式呈现。
•	关键技术：
1.	数据管理：遍历文件夹读取所有股票数据，并用字典结构存储，以股票名为键，方便按需调用。
2.	3D坐标映射：将时间（X轴）、不同股票（Y轴）、成交量（Z轴）映射到三维坐标系。
3.	plot_surface的创新应用：利用plot_surface函数绘制“面”的特性，为每只股票构造一个从Z=0平面升起的“立面”，从而模拟出填充面积图的效果。
2. 核心代码与解析
from mpl_toolkits.mplot3d import Axes3D as Axes3D_yxl

def create_3d_area_chart_yxl(data_directory_yxl):

    all_data_yxl = load_stock_data_yxl(data_directory_yxl) # 加载所有股票数据到字典
    
    fig_yxl = plt_yxl.figure(figsize=(30, 21))
    ax_yxl = fig_yxl.add_subplot(111, projection='3d') # 创建3D绘图区

    # 遍历每只股票
    for i_yxl, stock_name_yxl in enumerate(all_data_yxl.keys()):
        df_yxl = all_data_yxl[stock_name_yxl]
        x_data_yxl = range(len(df_yxl))
        y_offset_yxl = i_yxl # Y轴位置
        z_data_yxl = df_yxl['成交量'].values / 1e8 # Z轴高度

 # 步骤1：为plot_surface准备Z轴坐标
        # 构造一个二维数组，第一行是0（底边），第二行是真实成交量（顶边）
        Z_surface_yxl = np_yxl.vstack([np_yxl.zeros_like(z_data_yxl), z_data_yxl])
        X_surface_yxl = np_yxl.vstack([x_data_yxl, x_data_yxl])
        Y_surface_yxl = np_yxl.vstack([np_yxl.full_like(x_data_yxl, y_offset_yxl), np_yxl.full_like(x_data_yxl, y_offset_yxl)])

 # 步骤2：绘制“立面”
        ax_yxl.plot_surface(X_surface_yxl, Y_surface_yxl, Z_surface_yxl, color=colors_yxl[i_yxl], alpha=0.9)
        
  ...（3D图表美化，如调整视角、设置坐标轴标签等）...
    plt_yxl.show()
解析：此代码的核心在于plot_surface的应用。它不是画线，而是画面。通过构造一个下边缘Z坐标为0，上边缘Z坐标为成交量的二维Z_surface，我们巧妙地在3D空间中“画”出了一个2D面积图，并通过Y轴的偏移实现了多图分层排列，视觉效果独特。

________________________________________
热力图实验思路
•	目标：快速概览多只股票在一段时间内的每日涨跌幅表现，一目了然地发现市场的共性和异常点。
•	方法：采用热力图（Heatmap）。行代表股票，列代表日期，格子的颜色深浅代表涨跌幅的大小。
•	关键技术：
1.	数据整合与重塑：这是本实验的灵魂。先将所有股票数据合并成一个长表格，然后使用pivot_table函数将数据重塑为二维网格（矩阵）结构，这正是热力图所需要的输入格式。
2.	自定义颜色映射：为了更直观地反映涨跌情况（如大涨用暖色，大跌用冷色），需要自定义颜色和数值区间的对应关系。


2. 核心代码与解析
def create_heatmap_yxl(data_directory_yxl):
    # 步骤1：调用函数，直接获取重塑好的二维网格数据
    # 假设 prepare_heatmap_data 是一个您已定义的函数
    heatmap_df_yxl = prepare_heatmap_data_yxl(data_directory_yxl, value_column='涨跌幅%')
    
    # 步骤2：定义颜色规则
    bounds_yxl = [-float('inf'), -7, -3, -1, 1, 3, 7, float('inf')]
    colors_yxl = ['#5f7682', '#7a8b99', '#a0b3bf', '#f7f7f7', '#e6cdd2', '#d8aab7', '#c98d9b']
    cmap_yxl = mcolors_yxl.ListedColormap(colors_yxl)
    norm_yxl = mcolors_yxl.BoundaryNorm(bounds_yxl, cmap_yxl.N)

    # 步骤3：使用Seaborn绘制热力图
    plt_yxl.figure(figsize=(20, 10))
    sns_yxl.heatmap(heatmap_df_yxl, 
                annot=True,     # 在格子上显示数字
                fmt=".1f",      # 数字格式
                cmap=cmap_yxl,      # 使用自定义颜色尺
                norm=norm_yxl,      # 使用自定义上色规则
                linewidths=.5)
    
    plt_yxl.show()

# 主程序入口
# ...
•	解析：此代码最关键的一步发生在prepare_heatmap_data_yxl函数内部的pivot_table调用。它将杂乱的多源数据高效地整理成结构化的二维矩阵，为后续seaborn.heatmap的轻松绘制铺平了道路，是“数据决定可视化上限”的典型体现。





四、实验结果与分析

•	图表一（堆叠柱状图）
    成功地展示了指定股票（如“雪人股份”）在一段时间内的每日价格波动区间。通过该图，可以直观地看到价格的振幅变化趋势，例如在某个时间段内波动加剧或趋于平稳。
    ![image](https://github.com/user-attachments/assets/4abec729-febc-4898-821a-3817bd2b2c06)


•	图表二（3D分层面积图）
•	  有效地将多只股票的成交量数据并列呈现在一个三维视图中。通过不同“山脉”的高度和走势，可以快速对比哪些股票是市场热点（成交量高），哪些股票交投清淡，并观察到市场整体的成交量变化趋势。
![image](https://github.com/user-attachments/assets/02b8728a-6a87-498a-88d9-a0da0fdcdd5e)


•	图表三（热力图）
•	  提供了市场的“情绪仪表盘”。通过颜色的分布，可以迅速识别出市场普涨、普跌或分化的交易日，并能快速定位在特定日期表现异常（领涨或领跌）的个股。
![image](https://github.com/user-attachments/assets/7f0dcf9a-eb31-4358-a69d-67107d4e8407)


