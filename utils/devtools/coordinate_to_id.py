import pandas as pd
import numpy as np
import os
import tqdm
import ast
'''
:['车辆ID', '订单ID', 'GPS时间', '轨迹点经度', '轨迹点纬度']
chengdu:['user_id', 'lat', 'lon', 'passenager', 'timestamps', 'file_num']
Tdrive:['user_id', 'timestamps', 'lat', 'lon']
porto:['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
       'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE']
Geolife:['lat', 'lon', 'useless', 'Altitude ', 'DateInDays', 'Date', 'time',
       'trajectory_id', 'user_id']
beijing:100*100
chengdu:500*500
porto:100*300
'''

def convert_string_to_list(input_str):
    """
    将格式为 "[[x1,y1],[x2,y2]....]" 的字符串转换为 Python 列表
    
    参数:
        input_str (str): 输入字符串
        
    返回:
        list: 转换后的嵌套列表
    """
    # 方法1：使用ast.literal_eval（推荐，安全且处理各种数据类型）
    try:
        return ast.literal_eval(input_str)
    except (SyntaxError, ValueError):
        pass

def main():

    path = ''
    file = 'Geolife'
    full_path = f'{path}{file}/{file}_coordinate.csv'
    df = pd.read_csv(full_path)#.drop_duplicates(ignore_index=True)

    
    lat_key = 'lat'
    lon_key = 'lon'
    lat_num_grid = 100
    lon_num_grid = 300
    before = len(df)
    df = df[(df['lat']>39.83)&(df['lat']<40.05)&(df['lon']>116.17)&(df['lon']<116.62)]
    # df = df[(df['lat']>41.0625)&(df['lat']<41.5623)&(df['lon']>-8.7297)&(df['lon']<-7.5420)]
    after = len(df)
    print(f'keep rate:{after/before}')
    min_lat, min_lon, max_lat, max_lon = df[lat_key].min(), df[lon_key].min(), df[lat_key].max(), df[lon_key].max()
    print(min_lat, min_lon, max_lat, max_lon)

    lat_split = np.arange(min_lat, max_lat, (max_lat-min_lat)/lat_num_grid)
    lon_split = np.arange(min_lon, max_lon, (max_lon-min_lon)/lon_num_grid)
    def to_id(df_ser):
        lat_index = np.searchsorted(lat_split, df_ser[lat_key], side='right')
        lon_index = np.searchsorted(lon_split, df_ser[lon_key], side='right')
        # print("Count: %d\r" % , end="")
        return (lat_index-1) * lat_num_grid + lon_index
    df['POI_id'] = df.progress_apply(to_id, axis=1)
    
    print(f'finish to id, saving.')
    df.to_csv(f'{file}/{file}_poiid.csv', index=False)

if __name__ == '__main__':
    tqdm.tqdm.pandas()
    main()

