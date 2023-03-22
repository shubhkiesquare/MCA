import streamlit as st
from PIL import Image
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
import subprocess
import os






def save_uploadedfile(uploadedfile):
    with open(os.path.join("C:/Users/SHUBHAM AGNIHOTRI/Desktop/MCA_GUI/model", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to model".format(uploadedfile.name))

def run_model():
    df=load_data(r"C:\Users\SHUBHAM AGNIHOTRI\Desktop\MCA_GUI\Masala_Oats_Dataset.xlsx", sheet_name = 'Dataset')
    #df = pd.read_excel(r"C:\Users\SHUBHAM AGNIHOTRI\Desktop\MCA_GUI\Masala_Oats_Dataset.xlsx", sheet_name = 'Dataset')
    df1=create_lag(df)
    df2 = create_features(df1)
    coeff=built_coefficients(df2)
    return("model ran successfully")
def main():

    col1, col2 ,col3= st.beta_columns((1,2,1))
    img = Image.open("new.jpeg")
    col3.image(img,width=100)

    st.text("Upload Latest Data Set")
    if st.button("Upload Dataset"):
        dataset_file = st.file_uploader("Upload Dataset File", type=['xlsx'])
        st.write(dataset_file)
        with open(os.path.join("model", dataset_file.name), "wb") as f:
            f.write(dataset_file.getbuffer())




    


    if st.button("Check Cofficient File"):

        filelist = []
        for root, dirs, files in os.walk("coeff"):
            for file in files:
                filename = os.path.join(root, file)
                filelist.append(filename)
        choice = st.selectbox('SELECT FILE', filelist)
        df_coeff_gui=pd.read_csv(choice,encoding="utf-8")
        st.dataframe(df_coeff_gui)



if __name__ == '__main__':
	main()
