import streamlit as st
from PIL import Image  
import io  
from test import judge

st.title("AI合成图片检测系统") 

upload_file = st.file_uploader("上传你要检测的图片", type=['jpg', 'png', 'jpeg'])

if upload_file:
    bytes_data = upload_file.read()  
    image = Image.open(io.BytesIO(bytes_data))  
    st.image(image, caption='待检测的图片', use_column_width=True)
    #st.image(upload_file, caption='上传的图片', use_column_width=True)
    st.write('上传成功！')

    if st.button('开始识别'):
        st.write('识别中...')
        if judge(image):
            #st.write('AI合成图片！')
            st.markdown("## 该图片是AI合成图片！")  
        else:
            #st.write('不是AI合成图片！')
            st.markdown("## 该图片不是AI合成图片！")  
#streamlit run test2.py