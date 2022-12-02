import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
#import lightgbm as lgb


from time import sleep
st.set_page_config(
    page_icon="🍓",
    page_title="KOREATECH 2022-2 Cloud Computing, AI Service",
    layout="wide",
)

st.header ("🍓영화 관객수 예측 프로그램🍓에 오신 것을 환영합니다.")
st.subheader ("#1 데이터 전처리") 


train = pd.read_csv('./movies_train.csv') # 모델 학습을 위한 데이터
test = pd.read_csv('./movies_test.csv') # train 데이터와 다르게 관객수가 없음
submission = pd.read_csv('./submission.csv') # 제출을 위한 폼을 제공하는 데이터로 test 데이터와 index가 같음

st.write("🐥 train 하위 5개 데이터 출력")
st.write(train.tail())

st.write("🐥 test 하위 5개 데이터 출력")
st.write(test.tail())

st.write("🐥 submission 하위 5개 데이터 출력")
st.write(submission.tail()) # 관객수만을 출력

st.info("title : 영화의 제목, distributor : 배급사, genre : 장르, release_time : 개봉일, time : 상영시간(분), screening_rat : 상영등급 ,director : 감독이름, dir_prev_bfnum : 해당 감독이 이 영화를 만들기 전 제작에 참여한 영화에서의 평균 관객수(단 관객수가 알려지지 않은 영화 제외), dir_prev_num : 해당 감독이 이 영화를 만들기 전 제작에 참여한 영화의 개수(단 관객수가 알려지지 않은 영화 제외), num_staff : 스텝수, num_actor : 주연배우수, box_off_num : 관객수")

st.write("🐥 각 데이터의 column과 row수 알아보기")
st.write("train: ", train.shape)
st.write("test: ", test.shape)
st.write("submission: ", submission.shape)

st.write("🐥 인기 장르와 비인기 장르 알아보기")

pd.options.display.float_format = '{:.1f}'.format
st.write(train[['genre', 'box_off_num']].groupby('genre').mean().sort_values('box_off_num'))
# train 데이터에서 장르와 관객수를 선택하고 장르별 관객수를 더해서 나타냄

sns.heatmap(train.corr(), annot=True)