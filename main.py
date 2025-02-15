import os
import PyPDF2
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text[:1000]
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

readable_files = [".pdf", ".txt", ".docx", ".xml"]

path = input("Select the desired directory to be organized")

files_data = []

os.chdir(path)

for file_path in os.listdir():
  if(not os.path.isfile(file_path)):
    file_data = {
      "Path": file_path,
      "Extensions": None,
      "Content": None
    }
  else:
    _, ext = os.path.splitext(file_path)
    if (ext in readable_files):
      content = read_pdf(file_path)
      file_data = {
      "Path": file_path,
      "Extensions": ext.lower(),
      "Content": content
    }
    else:
      file_data = {
      "Path": file_path,
      "Extensions": ext.lower(),
      "Content": None
    }
  files_data.append(file_data)
  df = pd.DataFrame(files_data)

print(df)

#Extract keywords/labels using dependecies or ML modelm
#Train model with labels/keywords (maybe)

# Analyze data and separate files into folders
  # Make the ML model analyze/predict the category for the file
    #For file in folder:
      # Predict file category
      # If category does not already exist :
        # Create category folder
      # Move respective file to category folder 


# Labels/Categories dynamically generated based on files contents (name, content, extension) 
  # Create a finite amount of labels that are equal
    ## Read every single file and then create labels (?) OR read each file and then create a label based on that file
    ## Put every file in the respective folder based on name, content and extension
    ### The label creation can't be too specific so that every file has its own folder, but it can't be too general so that all files are in one folder