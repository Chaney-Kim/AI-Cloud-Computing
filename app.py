import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
#import lightgbm as lgb


from time import sleep
st.set_page_config(
    page_icon="π",
    page_title="KOREATECH 2022-2 Cloud Computing, AI Service",
    layout="wide",
)

st.header ("πμν κ΄κ°μ μμΈ‘ νλ‘κ·Έλ¨πμ μ€μ  κ²μ νμν©λλ€.")
st.subheader ("#1 λ°μ΄ν° μ μ²λ¦¬") 


train = pd.read_csv('./movies_train.csv') # λͺ¨λΈ νμ΅μ μν λ°μ΄ν°
test = pd.read_csv('./movies_test.csv') # train λ°μ΄ν°μ λ€λ₯΄κ² κ΄κ°μκ° μμ
submission = pd.read_csv('./submission.csv') # μ μΆμ μν νΌμ μ κ³΅νλ λ°μ΄ν°λ‘ test λ°μ΄ν°μ indexκ° κ°μ

st.write("π₯ train νμ 5κ° λ°μ΄ν° μΆλ ₯")
st.write(train.tail())

st.write("π₯ test νμ 5κ° λ°μ΄ν° μΆλ ₯")
st.write(test.tail())

st.write("π₯ submission νμ 5κ° λ°μ΄ν° μΆλ ₯")
st.write(submission.tail()) # κ΄κ°μλ§μ μΆλ ₯

st.info("title : μνμ μ λͺ©, distributor : λ°°κΈμ¬, genre : μ₯λ₯΄, release_time : κ°λ΄μΌ, time : μμμκ°(λΆ), screening_rat : μμλ±κΈ ,director : κ°λμ΄λ¦, dir_prev_bfnum : ν΄λΉ κ°λμ΄ μ΄ μνλ₯Ό λ§λ€κΈ° μ  μ μμ μ°Έμ¬ν μνμμμ νκ·  κ΄κ°μ(λ¨ κ΄κ°μκ° μλ €μ§μ§ μμ μν μ μΈ), dir_prev_num : ν΄λΉ κ°λμ΄ μ΄ μνλ₯Ό λ§λ€κΈ° μ  μ μμ μ°Έμ¬ν μνμ κ°μ(λ¨ κ΄κ°μκ° μλ €μ§μ§ μμ μν μ μΈ), num_staff : μ€νμ, num_actor : μ£Όμ°λ°°μ°μ, box_off_num : κ΄κ°μ")

st.write("π₯ κ° λ°μ΄ν°μ columnκ³Ό rowμ μμλ³΄κΈ°")
st.write("train: ", train.shape)
st.write("test: ", test.shape)
st.write("submission: ", submission.shape)

st.write("π₯ μΈκΈ° μ₯λ₯΄μ λΉμΈκΈ° μ₯λ₯΄ μμλ³΄κΈ°")

pd.options.display.float_format = '{:.1f}'.format
st.write(train[['genre', 'box_off_num']].groupby('genre').mean().sort_values('box_off_num'))
# train λ°μ΄ν°μμ μ₯λ₯΄μ κ΄κ°μλ₯Ό μ ννκ³  μ₯λ₯΄λ³ κ΄κ°μλ₯Ό λν΄μ λνλ

sns.heatmap(train.corr(), annot=True)