import streamlit as st
import streamlit.components.v1 as stc
import os
# File Processing Pkgs
import pandas as pd
from PIL import Image 
import subprocess
@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')
def main():
	col1, col2, col3 = st.columns(3)
	img = Image.open("new.jpeg")
	col3.image(img,width=100)







	st.title("MCA ANALYTICS")

	menu = ["Select","Dataset","Model","coeff"]
	choice = st.sidebar.selectbox("INPUTS",menu)

	


	if choice == "Dataset":
		st.subheader("Dataset")
		data_file = st.file_uploader("Upload dataset",type=['csv','xlsx'])
		if st.button("Process"):
			if data_file is not None:
				file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
				st.write(file_details)
				with (tempfile.TemporaryFile()) as f:
					f.write(data_file.getbuffer())
				st.success("dataset uploaded")
				df = pd.read_csv(data_file)
				st.dataframe(df)
				


	elif choice == "Model":
		st.subheader("Model Files")
		docx_file = st.file_uploader("Upload Model File", type=['py'])
		if st.button("Process"):
			if docx_file is not None:
				file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
				st.write(file_details)
				# Check File Type
				with open(os.path.join("model", docx_file.name), "wb") as f:
					f.write(docx_file.getbuffer())
				st.success("model uploaded")
	else:
		st.write("Select any process")
					    
	menu_2=["select","run model to get coefficient","check coefficient output"]			
	choice_2 = st.sidebar.selectbox("COEFF",menu_2)
	if choice_2 == "run model to get coefficient":
		st.subheader("Running Model To Get Coefficient")
		subprocess.run(["python", "model/model_1.py"])
		st.success("Output Model Ran Succesfully")
	elif choice_2=="check coefficient output":
		st.subheader("Displaying Coefficient Output")
		filelist = []
		for root, dirs, files in os.walk("C:/Users/SHUBHAM AGNIHOTRI/Desktop/MCA_GUI/coeff"):
			for file in files:
				filename = os.path.join(root, file)
				filelist.append(filename)
				choice_3 = st.selectbox('SELECT FILE', filelist)
				df_coeff_gui=pd.read_csv(choice_3)
				st.dataframe(df_coeff_gui)
				st.balloons()
	else:
		st.write("")

	menu_3=["select","ROIs, Base & Contribution, Elasticity"]			
	choice_3 = st.sidebar.selectbox("OUTPUT MODELS",menu_3)
	if choice_3 == "ROIs, Base & Contribution, Elasticity":
		st.subheader("Running Model To Get ROIs, Base & Contribution, Elasticity")
		subprocess.run(["python", "mcroe.py"])
		st.success("Model Ran Succesfully")
		menu_4=["Check Output","Download Output"]			
		choice_4 = st.selectbox("Output Process",menu_4)
		if choice_4=="Check Output":
			df_output=pd.read_csv("results/roi_elasticity.csv")
			st.dataframe(df_output)
		elif choice_4=="Download Output":
			df_output=pd.read_csv("results/roi_elasticity.csv")
			csv = convert_df(df_output)
			st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)
		else:
			st.write("")
	



if __name__ == '__main__':
	main()