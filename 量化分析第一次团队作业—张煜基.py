#第一例分析

#8.14东方财富网资金流向
#数据获取
import json
import requests

cookies = {
    'cowCookie': 'true',
    'qgqp_b_id': 'b7bb31619ba2e7b93e05a27bb2052822',
    'st_si': '47664194804802',
    'st_asi': 'delete',
    'waptgshowtime': '2020814',
    'intellpositionL': '546px',
    'st_pvi': '83488985193212',
    'st_sp': '2020-08-13^%^2023^%^3A10^%^3A44',
    'st_inirUrl': 'https^%^3A^%^2F^%^2Fwww.eastmoney.com^%^2F',
    'st_sn': '19',
    'st_psi': '20200814072920304-111000300841-1778311673',
    'intellpositionT': '655px',
}

headers = {
    'Connection': 'keep-alive',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36',
    'Accept': '*/*',
    'Referer': 'http://data.eastmoney.com/zjlx/detail.html',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

params = (
    ('pn', '1^'),
    ('pz', '50^'),
    ('po', '1^'),
    ('np', '1^'),
    ('ut', 'b2884a393a59ad64002292a3e90d46a5^'),
    ('fltt', '2^'),
    ('invt', '2^'),
    ('fid0', 'f4001^'),
    ('fid', 'f62^'),
    ('fs', 'm:0 t:6 f:^!2,m:0 t:13 f:^!2,m:0 t:80 f:^!2,m:1 t:2 f:^!2,m:1 t:23 f:^!2,m:0 t:7 f:^!2,m:1 t:3 f:^!2^'),
    ('stat', '1^'),
    ('fields', 'f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124^'),
    ('rt', '53245379^'),
    #('cb', 'jQuery18308396842823018853_1597361368805^'),
    ('_', '1597361392526'),
)

response = requests.get('http://push2.eastmoney.com/api/qt/clist/get', headers=headers, params=params, cookies=cookies)

#NB. Original query string below. It seems impossible to parse and
#reproduce query strings 100% accurately so the one below is given
#in case the reproduced version is not "correct".
# response = requests.get('http://push2.eastmoney.com/api/qt/clist/get?pn=1^&pz=50^&po=1^&np=1^&ut=b2884a393a59ad64002292a3e90d46a5^&fltt=2^&invt=2^&fid0=f4001^&fid=f62^&fs=m:0+t:6+f:^!2,m:0+t:13+f:^!2,m:0+t:80+f:^!2,m:1+t:2+f:^!2,m:1+t:23+f:^!2,m:0+t:7+f:^!2,m:1+t:3+f:^!2^&stat=1^&fields=f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124^&rt=53245379^&cb=jQuery18308396842823018853_1597361368805^&_=1597361392526', headers=headers, cookies=cookies)

print(response)  # <Response[200]>

#2，数据清洗
resp_dict = json.loads(response.text)
datas = resp_dict.get('data').get('diff')
compaines = []
prices = []

for data in datas:
 company = data.get('f14')

#今日主力净流入
 share_1 = data.get('f184')

#公司当天股价
 price = data.get('f2')

 if share_1 > 15:
    compaines.append(company)
    prices.append(price)

print(compaines)
print(prices)

#数据可视化
from pyecharts.charts import Bar
import pyecharts.options as opts

bar = Bar()
bar.add_xaxis(compaines)
bar.add_yaxis('股价图',prices)

bar.set_global_opts(
    xaxis_opts = opts.AxisOpts(
        axislabel_opts=opts.LabelOpts(rotate=-40),
    ),
    yaxis_opts=opts.AxisOpts(name='价格：（元/股)'),
)

bar.render('股价图.html')


#第二例分析

import pandas as pd
# 从本地导入数据
df = pd.read_csv('data1.csv')
# 查看数据
df.head()

# 剔除缺失数据
df = df.dropna()
df.head()
print(df)


df = df.reset_index().drop(columns='index')
df.head()

# 取出时间
raw_time = pd.to_datetime(df.pop('Unnamed: 0'), format='%Y/%m/%d %H:%M')

from matplotlib import pyplot as plt
import seaborn as sns

# 折线图：股票走势
plt.plot(raw_time, df['close'])
plt.xlabel('Time')
plt.ylabel('Share Price')
plt.title('Trend')
plt.show()

# 散点图：成交量和股价
plt.scatter(df['volume'][:300], df['close'][:300])  # 切片，取前300组数据
plt.xlabel('Volume')
plt.ylabel('Share Price')
plt.title('Volume & Share Price')
plt.show()

# 涨跌幅度
daily_return = df['close'][0::240].pct_change().dropna()
plt.plot(raw_time[0::240][:40], daily_return[:40])
plt.xlabel('Time')
plt.ylabel('Rise and Fall')
plt.show()
