# Data Science Project Portfolio

This repository presents a collection of *personal data science projects* in the form of iPython Notebook. All data set were retrieved from public sources and cited. The topics and datasets being explored in this portfolio are chosen based on my own interest, the primary focus of the projects is to employ various approaches and tools of data analysis/modelling to extract buried stories of the datasets on hand. This portfolio will also look into deep learning project using GPU.

Any questions or feedback regarding this portfolio can be kindly directed to the author, Kyle Lee, at _***kylelee417@gmail.com***_.

## Projects

#### *tools: keras, tensorflow-gpu, scikit-learn, Pandas, Matplotlib, Seaborn, Plotly, Numpy*

* **[Master's Program Admission][1]**: Designed a customized ***Logistic Regression and SVM *** model and achieved 90% accuracy in classifying whether an aplicant would likely be admitted or not admitted using 7 different features. ***PCA*** was also performed during the EDA process; however, the information loss was more than 20%. Therefore, it was found that using a simple logictic regression or classification model such as SVM would give the best fit model considering the dataset and variables were small.

<p align="center">
  <img src="Master's Program Admission/cm_LR.png" width="35%" class="center">
  <img src="Master's Program Admission/plot1.png" width="55%" class="center">
  </p>

## Mini Capstone Projects

#### *tools: scikit-learn, Pandas, Matplotlib, Seaborn, Plotly, Numpy, Folium*
* **[2017 911 Responses in Toronto][2]**: This project mainly focused on data visualization using location data provided from Toronto City Open Data. The data frame was reorganized and clustered based on the top 10 call reason for each intersection in downtown Toronto using ***K-mean Clustering***. Then, each data were visualized on the folium map. For the future reference, it will be great to see the top 5 fire stations recieved 911 calls and to show thier coverage within x km radius on the same map. 

<p align="center">
  <img src="2017 911 Responses Toronto/toronto map.png" width="55%" class="center">
  <img src="2017 911 Responses Toronto/map legend.png" width="45%" class="center">
  </p>


[1]:https://github.com/kylelee417/CollabProject/blob/master/project_notebook.ipynb
[2]:https://github.com/kylelee417/Capstone-Project
