# Data Science Project Portfolio

This repository presents a collection of *personal data science projects* in the form of iPython Notebook. All data set were retrieved from public sources and cited. The topics and datasets being explored in this portfolio are chosen based on my own interest, the primary focus of the projects is to employ various approaches and tools of data analysis/modelling to extract buried stories of the datasets on hand. This is an on-going portfolio which will more focus on deep learning project using GPU.

Any questions or feedback regarding this portfolio can be kindly directed to the author, Kyle Lee, at _***kylelee417@gmail.com***_.

## Projects

#### *tools: Keras, TensorFlow-gpu, scikit-learn, Pandas, Matplotlib, Seaborn, Plotly, Numpy*

* **[Master's Program Admission][1]**: Designed a customized ***Logistic Regression and SVM*** model and achieved 90% accuracy in classifying whether an aplicant would likely be admitted or not admitted using 7 different features. ***PCA*** was also performed during the EDA process; however, the information loss was more than 20%. Therefore, it was found that using a simple logictic regression or classification model such as SVM would give the best fit model considering the dataset and variables were small.

<p align="center">
  <img src="Master's Program Admission/cm_LR.png" width="35%" class="center">
  <img src="Master's Program Admission/plot1.png" width="55%" class="center">
  </p>
  
## Data Visualization Focused Projects
#### *tools required: scikit-learn, dtreeviz, graphviz, ipywidgets*
##### Unfortunately, ipywidgets do not render on Github or nbviewer. You can still view it through Google Colab or run locally to get the access
* **[Wisconsin Breast Cancer Detection ver. 1.0][4]**: Breast cancer is one of the well-known diseases for female, as well as for male, has been studied years. As far as a tumor type is concerned, early detection with a great precision and accuracy helps much better in developinsg a treating process for both patients and physicians. For this project, **decision tree** model was mainly used but in 3 different structures. 699 samples were used with 9 features to determine whether a sample is likely classified a binary target class as **Benign (non-invasive)** or **Malignant (invasive)** cancer type. 

The Area Under the Curve (AUC) was used to measure the classifier's skill in ranking a set of patterns accornding to the degree to which they belong to the positive class. However, this model is not mainly focused on performance assessment or parameter tuning. It was primarily focused on the model visualization when the true prediction (**Benign type**) is maximized for better and easier understanding.

I have found it gives eaiser interpretation using ***dtreeviz package*** than the graph created using ***graphviz***. **Random Forest** was used to overcome some overfitting problem in a single decision tree; however, the data set is too small to have comparable results. Another downside is, it does not give as much detailed interpretation as the decision tree classifier. I have also simply tried a regression tree model because the binary classification for this cancer detection can be vary depending on the random values from each feature (believing in those characteristics driven from unexpected tumor cells division during the mitosis). I used the most 3 important feature outcomes from Random Forest to construct a 3D graph and a 2D heat map; however, it did not visually show correlations among the features. Therefore, it definitely needs some optimization works or another approach like the multivariate regression model as a next goal.

**Acknowldegement:**

* Decision Tree [Concepts][7]
* Breast cancer databases: University of Wisconsin Hospitals, Madison from Dr. William H. Wolberg.
* Interactive Decision Trees Jupyter Widgets Resources, [Dafni Sidiropoulou Velidou][5]'s Blog
* More ipywidgets [Contents][6]

<p align="center">
  <img src="Breast_Cancer/dtree1.png" width="55%" class="center">
  <img src="Breast_Cancer/dtree2.png" width="35%" class="center">
  
  graphiz - **left** ,    interactive decision tree - **right**
  <img src="Breast_Cancer/dtree3.1.png" width="65%" class="center">
  **dtreeviz-visualization**
  
  <img src="Breast_Cancer/dtree4.1.png" width="55%" class="center"> **Bivariate Regression tree in 3D**
  
  </p>


## Mini Capstone Projects
* **[Mini MNIST Project ver. 1.0][3]**: A simple mini project to create and test a deep learning model using MNIST data from Keras. 80:20 test split was done out of 60k dataset using the ***TensorFlow***. The model was created and tested using Google Colaboratory, GPU method. No optimization process was done because this was only meant to learn how to create a model. the loss vs. accuracy graph was created by running 20 epochs:

<p align="center">
  <img src="TensorFlow_miniproj/epoch_train_val.png" width="100%" class="center">
</p>

#### *tools: scikit-learn, Pandas, Matplotlib, Seaborn, Plotly, Numpy, Folium*
* **[2017 911 Responses in Toronto][2]**: This project mainly focused on data visualization using location data provided from Toronto City Open Data. The data frame was reorganized and clustered based on the top 10 call reason for each intersection in downtown Toronto using ***K-mean Clustering***. Then, each data were visualized on the folium map. For the future reference, it will be great to see the top 5 fire stations recieved 911 calls and to show thier coverage within x km radius on the same map. 

<p align="center">
  <img src="2017 911 Responses Toronto/toronto map.png" width="55%" class="center">
  <img src="2017 911 Responses Toronto/map legend.png" width="45%" class="center">
  </p>


[1]:https://github.com/kylelee417/CollabProject/blob/master/project_notebook.ipynb
[2]:https://github.com/kylelee417/Capstone-Project
[3]:https://github.com/kylelee417/Data-Science_Portfolio/blob/master/TensorFlow_miniproj/tensorflow_miniproj.ipynb
[4]:https://nbviewer.jupyter.org/github/kylelee417/Data-Science_Portfolio/blob/master/Breast_Cancer/breast_cancer.ipynb
[5]:https://towardsdatascience.com/interactive-visualization-of-decision-trees-with-jupyter-widgets-ca15dd312084
[6]:https://ipywidgets.readthedocs.io/en/stable/
[7]:http://dkopczyk.quantee.co.uk/tree-based/
