
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import koreanize_matplotlib

# 1. 데이터 로드 및 기본 설정
@st.cache_data
def load_data():
    df = pd.read_csv('starbucks_stores/data/starbucks_ai.csv')
    # 'gugun_name' 컬럼의 결측치를 '정보없음'으로 채웁니다.
    df['gugun_name'] = df['gugun_name'].fillna('정보없음')
    return df

df = load_data()

# 사이드바 설정
st.sidebar.title('Starbucks Store Analysis')

# 시도 선택
sido_list = ['전체'] + sorted(df['sido_name'].unique().tolist())
selected_sido = st.sidebar.selectbox('시도 선택', sido_list)

# 시군구 선택
if selected_sido == '전체':
    gugun_list = ['전체']
else:
    gugun_list = ['전체'] + sorted(df[df['sido_name'] == selected_sido]['gugun_name'].unique().tolist())
selected_gugun = st.sidebar.selectbox('시군구 선택', gugun_list)

# 매장명 검색
store_name_search = st.sidebar.text_input('매장명 검색')

# 데이터 필터링
filtered_df = df.copy()
if selected_sido != '전체':
    filtered_df = filtered_df[filtered_df['sido_name'] == selected_sido]
if selected_gugun != '전체':
    filtered_df = filtered_df[filtered_df['gugun_name'] == selected_gugun]
if store_name_search:
    filtered_df = filtered_df[filtered_df['s_name'].str.contains(store_name_search, na=False)]

# 메인 화면
st.title('스타벅스 전국 매장 분석 대시보드')

# 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs(['데이터 개요', '지역별 분석', '전국 매장 지도', '클러스터 맵', '군집화 분석'])

with tab1:
    st.header('데이터 개요')
    st.write(f"전체 데이터 수: {df.shape[0]}행, {df.shape[1]}열")
    st.subheader('데이터 타입')
    st.dataframe(df.dtypes.reset_index().rename(columns={'index': '컬럼명', 0: '데이터 타입'}))
    st.subheader('결측치 요약')
    st.dataframe(df.isnull().sum().reset_index().rename(columns={'index': '컬럼명', 0: '결측치 수'}))
    
    st.subheader('시도별 매장 수')
    st.bar_chart(df['sido_name'].value_counts())
    
    st.subheader('시군구별 매장 수 (상위 20)')
    st.bar_chart(df['gugun_name'].value_counts().head(20))

with tab2:
    st.header('지역별 분석')
    st.write(f"**{selected_sido} {selected_gugun if selected_gugun != '전체' else ''}** 지역의 매장 수: {len(filtered_df)}개")
    
    st.subheader('매장 목록')
    st.dataframe(filtered_df[['s_name', 'sido_name', 'gugun_name', 's_biz_code', 'addr']])
    
    if not filtered_df.empty:
        st.subheader('매장 타입(s_biz_code)별 분포')
        st.bar_chart(filtered_df['s_biz_code'].value_counts())
    else:
        st.write('해당 지역에 매장이 없습니다.')

with tab3:
    st.header('전국 매장 지도')
    
    # 지도 초기 중심 설정
    map_center = [36.5, 127.5]
    zoom_start = 7
    
    if not filtered_df.empty:
        # 위도, 경도 컬럼에 결측치가 없는 데이터만 사용
        map_df = filtered_df.dropna(subset=['lat', 'lot'])
        if not map_df.empty:
            map_center = [map_df['lat'].mean(), map_df['lot'].mean()]
            if selected_sido != '전체':
                zoom_start = 10
            if selected_gugun != '전체':
                zoom_start = 12

    m = folium.Map(location=map_center, zoom_start=zoom_start)

    # 위도, 경도 컬럼에 결측치가 없는 데이터만 사용
    map_df = filtered_df.dropna(subset=['lat', 'lot'])
    for idx, row in map_df.iterrows():
        folium.Marker(
            location=[row['lat'], row['lot']],
            popup=f"<strong>{row['s_name']}</strong><br>주소: {row['addr']}<br>타입: {row['s_biz_code']}",
            tooltip=row['s_name']
        ).add_to(m)
        
    st_folium(m, width=725, height=500)

with tab4:
    st.header('클러스터 맵')
    
    m_cluster = folium.Map(location=map_center, zoom_start=zoom_start)
    
    # 위도, 경도 컬럼에 결측치가 없는 데이터만 사용
    map_df = filtered_df.dropna(subset=['lat', 'lot'])
    marker_cluster = MarkerCluster().add_to(m_cluster)

    for idx, row in map_df.iterrows():
        folium.Marker(
            location=[row['lat'], row['lot']],
            popup=f"<strong>{row['s_name']}</strong><br>주소: {row['addr']}<br>타입: {row['s_biz_code']}",
            tooltip=row['s_name']
        ).add_to(marker_cluster)
        
    st_folium(m_cluster, width=725, height=500)

with tab5:
    from sklearn.cluster import KMeans
    import numpy as np

    st.header('군집화 분석')
    
    # 클러스터 개수 선택
    n_clusters = st.slider('클러스터 개수 선택', min_value=2, max_value=20, value=5, step=1)
    
    cluster_df = filtered_df.dropna(subset=['lat', 'lot'])

    if not cluster_df.empty and len(cluster_df) > n_clusters:
        # K-means 모델 적용
        coords = cluster_df[['lat', 'lot']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(coords)
        cluster_df['cluster'] = kmeans.labels_

        # 군집화 결과 시각화
        map_clusters = folium.Map(location=map_center, zoom_start=zoom_start)
        
        # 색상 팔레트
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

        for idx, row in cluster_df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lot']],
                radius=5,
                color=colors[row['cluster'] % len(colors)],
                fill=True,
                fill_color=colors[row['cluster'] % len(colors)],
                popup=f"Cluster: {row['cluster']}",
                tooltip=row['s_name']
            ).add_to(map_clusters)
            
        st_folium(map_clusters, width=725, height=500)

        st.subheader('클러스터별 매장 수')
        st.bar_chart(cluster_df['cluster'].value_counts())

        st.subheader('클러스터별 특성')
        for i in range(n_clusters):
            st.write(f"---")
            st.write(f"**클러스터 {i}**")
            c_df = cluster_df[cluster_df['cluster'] == i]
            st.write(f"매장 수: {len(c_df)}개")
            if not c_df.empty:
                st.write(f"주요 시도: {c_df['sido_name'].value_counts().idxmax()}")
                st.write(f"주요 매장 타입: {c_df['s_biz_code'].value_counts().idxmax()}")
            st.dataframe(c_df[['s_name', 'sido_name', 'gugun_name']].head())

    else:
        st.warning('군집화를 수행하기에 데이터가 부족합니다. 필터 설정을 변경해주세요.')

st.sidebar.info('This dashboard displays Starbucks store information across South Korea.')
st.sidebar.markdown('[View on GitHub](https://github.com/corazzon/pycli)')
