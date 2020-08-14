#获取数据
import tushare as ts
start = '2020-01-01'
end = '2020-07-28'
datas = ts.get_hist_data('600519',start=start,end=end)
print(datas)
datas.to_excel('茅台.xlsx')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stock = pd.read_table('data.text'，usecols = range(15),parse_dates=[0],index_col=0)
stock = stock[::-1]
print(stock)

#数据信息
print(stock.info())
print(stock.columns)

#数据清洗
stock.rename(columns={' open':'open'},inplace=True)
print(stock.columns)

#1.股价时间序列图
stock['close'].plot(grid=True,label='close')
plt.legend()
plt.show()


#2.相对变化量
stock['return'] = stock['close']/stock.close.iloc[0] #第一天上市价格
stock['return'].plot(grid=True,label='相对股价')
plt.rcParams['font.sans=serif']='SimHei'
plt.legend()
plt.show()

#3.计算涨跌幅度
stock["p_change"].plot(grid=True).axhlinc(y=0,colot='black',lw=2)
plt.reParams['axex.unicode_minus']=False
plt.show()

#皮尔逊相关系数
cov = np.corrcoef(small.T)
print(cov)
