<details>
<summary>Making environment more useful</summary>

**Enable conda in cmd**

- add this in system environment path `C:\ProgramData\Anaconda3\Scripts`

**Run Jupyter Notebook instantly from cmd**

- run `jupyter notebook`

**List of running Jupyter Notebook**

- in cmd run `jupyter notebook list`

**Stop running notebook**

- `jupyter notebook stop`

**Update conda** 

- open anaconda shell as administrator **(also in cmd if enabled)** and run `conda update --all`

**If any error related to Module not found** 

- name of the module e.g: yapf; run : `pip install yapf` or `conda install yapf`

**Adding Anaconda in environment path**
> Note: my anaconda setup was done for user not for all user in my system
- Goto `Edit the system environment variables` and add those in path:
  - `C:\Users\Tansen\anaconda3`
  - `C:\Users\Tansen\anaconda3\Library\mingw-w64\bin`
  - `C:\Users\Tansen\anaconda3\Library\usr\bin`
  - `C:\Users\Tansen\anaconda3\Library\bin`
  - `C:\Users\Tansen\anaconda3\Scripts`

**Enable Jupyter Notebook extension (If not present in notebook)**

> If bellow 2 process doesn't work there is a problem related to version.

- Install by one command in `Anaconda Prompt` (open as admin for write permission)
  - `conda install -c conda-forge jupyter_contrib_nbextensions`
- Descriptive way:
- `pip install jupyter_contrib_nbextensions`
- Now type and enter:
- `pip install jupyter_nbextensions_configurator`
- Type and enter:
- `jupyter contrib nbextension install --user`
- Type and enter:
- `jupyter nbextensions_configurator enable --user`
- Enable/Disable extension by command:
- `jupyter nbextension enable/disable highlighter/highlighter`
- Now open Jupyter Notebook `Nbextensions` will be there now enable required extension

**Disable warning**

- Create `disable-warnings.py` at location `C:\Users\Tansen\.ipython\profile_default\startup`
- in `disable-warnings.py`
    
    ```python
    import warnings
    warnings.filterwarnings('ignore')
    ```
- Also this line can be written in notebook to disable warning.

**Virtual Environment Setup and Package installation:**

- In project directory open cmd type and enter:
    
    `conda create --prefix ./env jupyter`
    
- Now activate conda:
    
    `conda activate {"location path of ./env"}`
    
- Now let’s install a package called `pyresparser`
    
    install one by one :
    
    - `pip install nltk`
    - `pip install spacy==2.3.8 --no-cache-dir --only-binary :all:`
    - `pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz`
    - `pip install pyresparser`
</details>

<details>
<summary>Machine Learning Study Roadmap</summary>

- **Supervised Learning**
    - Supervised Learning Algorithm
        - SVM
        - RF (Random Forest)
        - Decision Tree
        - Linear Regression
        - Naive Bayes
        - Neural Network
- **Unsupervised Learning**
    - Unsupervised Learning Algorithm
        - K-mean clustering
        - K-nearest neighbor
- Semi-Supervised Learning
- Reinforcement
- Batch and online learning
- Dataset visualization, Analysis
- Data Cleaning, Preprocessing
- Feature Extraction
- Binary and Multilevel classification
- Confusion Matrix
- ROC and AOC Curve
- Errors
- Dimensionality Reduction

</details>

<details>
<summary>Machine Learning 101</summary>

### **What is Machine Learning?**

> A subset of AI that focuses on building systems that can learn from data and make predictions or decisions without being explicitly programmed.
> 

### AI and Machine Learning

- **AI**—>A machine that acts like a human
    - **Machine Learning**—> a subset of AI
        - An approach to achieve artificial intelligence through systems that can find patterns in a set of data.
        - Stanford university describe machine learning as the science of getting computers to act without being explicitly programmed. Which means we do not need to tell the machine do this , do that, if then , if this then that…etc.
            - **Deep Learning**—>a subset of Machine Learning
                - one of the techniques for implementing machine learning
    - **Data Science**—>Analyzing data and then doing something. It include both **Machine Learning & Deep Learning.**
- **Narrow AI**—>A machine that acts like a human for a specific task
    - Detecting heart disease from images
    - Game of go or chess or Star craft and other video games
    - Only work on single task
- **General AI**: A machine that acts like a human with multiple abilities

### Why Machine Learning

```mermaid
%%{init: {'theme': 'dark', "flowchart" : { "curve" : "basis" } } }%%
graph LR
A[Spreadsheets] -->|Then we move to| B[Relational DB - MySQL]
B -->|Then we move to| C[Big Data - NoSQL]
C -->|Finally| D[Machine Learning]
```

### Machine Learning Project Steps

```mermaid
%%{init: {'theme': 'dark', "flowchart" : { "curve" : "basis" } } }%%
graph LR
A[Data Collection] --> B[Data Preprocessing]
B -->|What problem we are tring to solve?| C[Problem Definition]
C -->|What data do we have?| D[Data]
D -->|What Defines success?| E[Evaluation]
E -->|What features</br>should we model?| F[Features]
F -->|What kind of</br> model should we use?| G[Modelling]
G -->|What have we tried </br>/what else can we try?| H[Experiments]

```

- Data collection (Hardest part)
    - How to clean noisy data?
    - What can we grab data from?
    - How do we find data?
    - How do we clean it so we can actually learn from it?
    - How to turn data from useless to useful?
- Data modelling
    - Problem definition: What problem are we trying to solve?
    - Data: What data do we have?
    - Evaluation: What defines success?
    - Features: What features should we model?
    - Modelling: What kind of model should we use?
    - Experiments: What have we tried / What else can we try?

**Playground**

- [Teachable Machine](https://teachablemachine.withgoogle.com/) by Google(easy)
- [ML-Playground](https://ml-playground.com/)(easy)
- [ML Playground](https://mlplaygrounds.com/) by Mrityunjay Bhardwaj (advanced)
- [ML Playground](https://playground.tensorflow.org/) by TensorFlow (More advanced)

### Types of Machine Learning

1. **Classical Learning**
    1. **Supervised**
        1. Classification
            - K-NN
            - Naive Bayes
            - SVM
            - Decision Tress
            - Logistic Regression
        2. Regression
            - Linear Regression
            - Polynomial Regression
            - Ridge/Lasso Regression
    2. **Unsupervised**
        1. Clustering
            - Fuzzy C-Means
            - Mean-Shift
            - K-Means
            - DBSCAN
            - Agglomerative
        2. Pattern Search
            - Eclat
            - Apriori
            - FP-Growth
        3. Dimension Reduction (Generalization)
            - T-SNE
            - PCA
            - LSA
            - SVD
            - LDA
2. **Reinforcement Learning**
    - Genetic Algorithm
    - A3C
    - SARSA
    - Q-Learning
    - Deep Q-Network (DQN)
3. **Neural Networks and Deep learning**
    1. Convolutional Neural Networks (CNN)
        - DCNN
    2. Recurrent Neural Networks (RNN)
        - LSM
        - LSTM
        - GRU
    3. Generative Adversarial Networks (GAN)
        - Vanilla GAN
        - Super Resolution GAN (SRGAN)
    4. Autoencoders
        - Seq2seq
    5. Perceptrons (MLP)
4. **Ensemble Methods**
    1. Stacking
    2. Bagging
        - Random Forest
    3. Boosting
        - AdaBoost
        - CatBoost
        - LightGBM
        - XGBoost
<p align="left"><a href="https://i.imgur.com/WDwwkSm.jpg" target="_blank">Types of Machine Learning Diagram</a></p>

**Types Simplified**

- Predict results based on incoming data
- Supervised: Data are labeled into categories
    - Classification: is this an apple or is this a pear?
    - Regression: based on input to predict stock prices
- Unsupervised: Data don't have labels
    - Clustering: machine to create these groups
    - Association rule learning: associate different things to predict what a customer might buy in the future
- Reinforcement: teach machines through trial and error
- Reinforcement: teach machines through rewards and punishment
    - Skill acquisition
    - Real time learning

**What Is Machine Learning Part 02**

- Now: Data -> machine learning algorithm -> pattern
- Future: New data -> Same algorithm (model) -> More patterns
- Normal algorithm: Starts with inputs and steps -> Makes output
- Machine learning algorithm
    - Starts with inputs and output -> Figures out the steps
- Data analysis is looking at a set of data and gain an understanding of it by comparing different examples, different features and making visualizations like graphs
- Data science is running experiments on a set of data with the hopes of finding actionable insights within it
    - One of these experiments is to build a machine learning model
- Data Science = Data analysis + Machine learning

**Section Review**

- Machine Learning lets computers make decisions about data
- Machine Learning lets computers learn from data and they make predictions and decisions
- Machine can learn from big data to predict future trends and make decision

</details>

<details>
<summary>Machine Learning and Data Science Framework</summary>

### Introducing Framework

**Steps to learn machine learning**

- Create a framework
- Which match to data science and machine learning tools
- Learn by doing

****[6 Step Field Guide for Building Machine Learning Projects](https://www.mrdbourke.com/a-6-step-field-guide-for-building-machine-learning-projects/)****

1. **Problem definition** — What business problem are we trying to solve? How can it be phrased as a machine learning problem?
    - Supervised or Unsupervised ?
    - Classification or Regression ?
2. **Data** — what data we have? How does it match the problem definition?
    - Structured or Unstructured?
    - Static or streaming?
3. **Evaluation** — What defines success? Is a 95% accurate machine learning model good enough?
    - Different type of matrices
    - Predicted price vs Actual price
4. **Features** — What parts of our data are we going to use for our model? How can what we already know influence this?
    - Example: Heart disease
    - Turn features such as weight, gender, BP, chest pain into patterns to make predictions whether a patient has heart disease?
5. **Modelling** — Which model should you choose based on your problem and data? How can you improve it? How do you compare it with other models?
    - Different problem —> Different type of model/algorithm
6. **Experimentation** — What else could we try? Does our deployed model do as we expected? How do the other steps change based on what we’ve found? How could we improve it?

### Framework we created

```mermaid
%%{init: {'theme': 'dark', "flowchart" : { "curve" : "basis" } } }%%
graph LR
A[Data Collection] --> B[Data Preprocessing]
B -->|What problem we are tring to solve?| C[Problem Definition]
C -->|What data do we have?| D[Data]
D -->|What Defines success?| E[Evaluation]
E -->|What features</br>should we model?| F[Features]
F -->|What kind of</br> model should we use?| G[Modelling]
G -->|What have we tried </br>/what else can we try?| H[Experiments]

```

### 1. Types of Machine Learning Problems

When shouldn't you use machine learning?

- if a simple hand-coded instruction based system work

Main types of machine learning

- Supervised Learning (Common)
- Unsupervised Learning (Common)
- Transfer Learning (Common)
- Reinforcement Learning (Uncommon)

**Supervised Learning**

- Data + labels —> make prediction
    - **Classification**
        - Both Binary & Multiclass Classification
            - Binary Classification = 2 Options
                - Based on 2 option choose 1
                - Example : Heart disease or no heart disease?
            - Multiclass Classification = More than 2 Options
                - Example : Predict dog breed based on photos in images
    - **Regression**
        - It is used to predict number
        - It is also preferred as continuous number
            - A number that goes up or down
        - Classical Regression problem
            - Predict sell price of a house based on number of rooms, area, etc.
            - How many People will buy this app based on clicks

**Unsupervised Learning**

- There is data but no labels
- **Example Scenario**
    - Marketing team want to send out promotion for next summer
    - Here is the table
    
    | Customer ID | Purchase 1 | Purchase 2 |
    | --- | --- | --- |
    | 1 | Summer Cloth | Summer Cloth |
    | 2 | Winter cloth | Winter cloth |
    | 3 | Summer Cloth | Summer Cloth |
    - Now you have to find out which customer is interested for summer cloth from this store
    - To solve this you make a `group` of customer who purchase only in summertime and a `group` of customer who purchase only in wintertime
    - Which is Cluster 1 (`summer`) and Cluster 2 (`winter`)
    - Now label them in `summer` and `winter` list
    - This is called Clustering

**Transfer Learning**

- It leverages what one machine learning model has learned in another machine learning model
- Example—>Predict what dog breed appears in a photo
- Already created existing model —> Car model identify model
- Now use that foundational patterns to apply in dog breed problem

**Reinforcement Learning**

a computer program perform some actions within a defined space and rewarding it for doing it well or punishing it for doing poorly

- Example —> Teach a machine learning algorithm to play chess

**Recap**

Matching your problem

- Supervised Learning: I know my inputs and outputs
- Unsupervised Learning: I am not sure of the outputs but I have inputs
- Transfer Learning: I think my problem may be similar to something else

### 2. Different types of data

- Structured Data —> Rows Columns is structured
- Unstructured Data —> Videos, Photos, Audio files
    - We have to structured this by converting it to number
- Static data —> doesn't change over time. e.g : csv file
- Streaming Data —> which is constantly changed over time
    - Example : predict how a stock price will change based on news headlines
    - News headlines are being updated constantly you'll want to see how they change stocks

**A data science workflow**

- open csv file in jupyter notebook (a tool to build machine learning project)
- perform data analysis with panda (a python library for data analysis)
- make visualizations such as graphs and comparing different data points with Matplotlib
- build machine learning model on the data using scikit learn to predict using these patterns
    
    ```mermaid
    %%{init: {'theme': 'dark', "flowchart" : { "curve" : "basis" } } }%%
    graph LR
    A[CSV file] -->|Open|B[Jupyter Notebook]
    B -->|Data analysis|C[Pandas]
    B -->|Visualization|D[Matplotlib]
    C -->|Scikit Learn|E[Machine Learning Model]
    D -->|Scikit Learn|E
    E --> F[Hearth Disease or not?]
    ```
    

### 3. Evaluation (Matrices): What defines success for us?

Example

- if your problem is to use patient medical records to classify whether someone has heart disease or not you might start by saying for this project to be valuable we need a machine learning model with over 99% accuracy
    
    ```mermaid
    %%{init: {'theme': 'dark', "flowchart" : { "curve" : "basis" } } }%%
    graph LR
    A[Data] -->B[Machine Learning Model]
    B -->C[Heart Disease or not?]
    C -->D[Accuracy ]
    ```
    
- This type of problem required highly accurate model

**Different types of metrices**

| Classification | Regression | Recommendation |
| --- | --- | --- |
| Accuracy | Mean Absolute Error (MAE) | Precision at K |
| Precision | Mean Squared Error (MSE) |  |
| Recall | Root Mean Squared (RMSE) |  |

### 4. **Features : What do we already know about the data?**

It is another word for `different forms of data`

- Structured or Unstructured data
- Feature variables — Target variable

Example

| ID | Weight | Gender | Heart Rate | Chest Pain | Heart Disease |
| --- | --- | --- | --- | --- | --- |
| 1 | 110kg | M | 81 | 4 | Yes |
| 2 | 64kg | F | 61 | 1 | No |
| 3 | 51kg | M | 57 | 0 | No |
- Weight, Gender, Heart Rate, Chest Pain —>Feature variables
- Heart Disease —> Target variable

**Different features of data**

- Numerical features : number; like body weight
- Categorical features : One thing or another ; like gender or whether a patient is a smoker or not etc
- Derived features : looks at different features of data and creates a new feature / alter existing feature
    - Example: look at someone's hospital visit history timestamps and if they've had a visit in the last year you could make a categorical feature called visited in last year. If someone had visited in the last year they would get true.
    - feature engineering: process of deriving features
        
        
        | ID | Weight | Gender | Heart Rate | Chest Pain | Heart Disease | Visited Last Year |
        | --- | --- | --- | --- | --- | --- | --- |
        | 1 | 110kg | M | 81 | 4 | Yes | Yes |
        | 2 | 64kg | F | 61 | 1 | No | Yes |
        | 3 | 51kg | M | 57 | 0 | No | No |
- Unstructured data has features too
    - images of dog
    - look for different shape in images
    - look for similarity
    - Eyes, leg, tail etc
    - `machine learning algorithm` figure out what features are there on its own
- What features should you use?
    - A machine learning algorithm learns best when all samples have similar information
    - Feature coverage: process of ensuring all samples have similar information
    - Every field have values , at least —> Want > 10% Coverage

### 5.1. Modelling Part 1 - 3 sets

**Based on our problem and data, what model should we use?**

3 parts to modelling

- Choosing and training a model
- Tuning a model
- Model comparison

**The most important concept in machine learning**

- The training
- Validation
- Test sets or 3 sets
    - Data split into 3 sets
        - Training set: train your model on this
        - Validation set: tune your model on this
        - Test set: test and compare on this
    - Example (University)
        
        ```mermaid
        %%{init: {'theme': 'dark', "flowchart" : { "curve" : "basis" } } }%%
        graph LR
        A[Course Materials<->Training] -->B[Practice Exam<->Validation]
        B -->C[Final Exam<->Test Set]
        ```
        
        This process is referred as Generalization in Machine Learning
        
        Generalization: The ability for a machine learning model to perform well on data it has not seen before
        
        Then things goes wrong
        
        - Everyone participate practice exam
        - Everyone did good
        - Now for Final exam , Professor give the same question
        - Everyone get top mark
        - Now this looks good but did the student learn anything?
    - This scenario need to avoid in machine learning by following:
        - split 100 patient records
            - training split: 70 patient records (70-80%)
            - validation split: 15 patient records (10-15%)
            - test split: 15 patient records (10-15%)
        - After training on training set
        - Validation split will be used to improve which is called tuning
        - Next use the improve result to use it with test split

### 5.2. Modelling Part 2 - Choosing

3 parts to modelling

- Choosing and training a model—>training data
- Tuning a model—>validation data
- Model comparison—>test data

Choosing a model

- Based on data choose a model
- Structured Data
    - [CatBoost](https://catboost.ai/)
    - [XGBoost](https://github.com/dmlc/xgboost)
    - [Random Forest](https://www.stat.berkeley.edu/~breiman/RandomForests/)
- Unstructured Data
    - Deep Learning
    - Transfer Learning
- Training a model
    
    ```mermaid
    %%{init: {'theme': 'dark', "flowchart" : { "curve" : "basis" } } }%%
    graph LR
    A[X - Data] -->|Inputs|B[Model]
    B -->|Predict|C[Y - label]
    ```
    
    - X is Feature variable and Y is Target variable
    - Different machine learning algorithm (Model) had different way of doing this
- Goal—>minimize time between experiments
    - If dataset include 100 thousand example
    - start with first 10 thousand and see how it goes
    - start with less complicated model (algorithm)
        
        
        | Experiment | Model | Accuracy | Training Time |
        | --- | --- | --- | --- |
        | 1 | 1 | 87.5% | 3 min |
        | 2 | 2 | 91.3% | 92 min |
        | 3 | 3 | 94.7% | 176 min |
- **Things to remember**
    - Some models work better than others and different problems
    - Don't be afraid to try things
    - Start small and build up (add complexity) as you need.

### 5.3. Modelling Part 3 - Tuning

Example

- Random Forest
    - adjust number of trees
- Neural Networks
    - adjust number of layers

Things to remember

- Machine learning models have hyper parameters you can adjust
- A model first results are not it's last
- Tuning can take place on training or validation data sets

### 5.4. Modelling Part 4 - Comparison

If the split of data goes well it will indicate how well it will perform

**Testing a model**

This is alright : ✅

| Data Set | Performance |
| --- | --- |
| Training | 98% |
| Test | 96% |
- Balanced (Goldilocks zone)

Underfitting (Potential) ❌

| Data Set | Performance |
| --- | --- |
| Training | 64% |
| Test | 47% |

Overfitting (Potential)❌

| Data Set | Performance |
| --- | --- |
| Training | 93% |
| Test | 99% |

In simple way

| Underfitting | Overfitting |
| --- | --- |
| Data mismatch | Data leakage |
| Test Data is different to Training Data | Training Data overlap Test Data |

Fixes for underfitting

- Try a more advanced model
- Increase model hyperparameters
- Reduce amount of features
- Train longer

Fixes for overfitting

- Collect more data
- Try a less advanced model

**No Model is perfect so check good result as much as you check poor result**

| Experiment | Model | Accuracy | Training Time | Prediction Time |
| --- | --- | --- | --- | --- |
| 1 | 1 | 87.5% | 3 min | 0.5 sec |
| 2 | 2 | 91.3% | 92 min | 1 sec |
| 3 | 3 | 94.7% | 176 min | 4 sec |

**Things to remember**

- Want to avoid overfitting and underfitting (head towards generality)
- Keep the test set separate at all costs
- Compare apples to apple
    - Model 1 on dataset 1
    - Model 2 on dataset 1
- One best performance Metric does not equal the best model

### **Experimentation**

How could we improve / what can we try next?

- Start with a problem
- Data Analysis: Data, Evaluation, Features
- Machine learning modelling: Model 1
- Experiments: Try model 2 then 3

**6 Step Machine Learning Framework questions**

- Problem definition: What kind of problem ?
- Data: What type of data ?
- Evaluation: What do you measure ?
- Features: What are features of your problems ?
- Modelling: What was the last thing you testing ability on?

### Tools We Will Use

- Data Science: 6 Step Machine Learning Framework
- Data Science: [Anaconda](https://www.anaconda.com/), [Jupyter Notebook](https://jupyter.org/)
- Data Analysis: Data, Evaluation and Features
- Data Analysis: [pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [NumPy](https://numpy.org/)
- Machine Learning: Modelling
- Machine Learning: [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.ai/), [CatBoost](https://catboost.ai/)
    
    <a href="https://i.imgur.com/rir7VpO.png" target="_blank">Machine Learning Tools</a>

</details>

<details>
<summary>Data Science Environment Setup</summary>

### Tools we are going to use

**Steps to learn machine learning**

- Create a framework (we created in previous section)
- Match to data science and machine learning tools
- Learn by doing

**Machine Learning Tools**

- [Anaconda](https://www.anaconda.com/): Hardware Store = 7.53GB (443 packs v2022.10 with update 30 Jan 2023)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html): Workbench = 200 MB
- Choosing Anaconda vs Miniconda
    
    
    | Anaconda | Miniconda |
    | --- | --- |
    | New to conda or python | familiar with conda and python |
    | Preinstalled Packages | can install individual packages |
    | Have the time and disk space | Not enough disk space |
- [Conda](https://docs.conda.io/en/latest/): Package Manager which is use to setup the rest of tools
    - Data Analysis: [pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [NumPy](https://numpy.org/)
    - Machine Learning: [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.ai/), [CatBoost](https://catboost.ai/)
- `Note`: miniconda required conda to install tools. Anaconda come with full packages but need to update packages (mentioned earlier how to update) . So either Miniconda+conda or Anaconda

### Jupyter Notebook

| Command Mode (press Esc to enable) | Edit Mode (press Enter to enable) |
| --- | --- |
| H: get full list of shortcuts | Shift + Enter: run the current cell and move to the next one. |
| Esc: enter command mode. | Ctrl + Enter: run the current cell and keep it selected. |
| A: insert a new cell above the current cell. | Ctrl + ]: indent the current block. |
| B: insert a new cell below the current cell. | Ctrl + [: un-indent the current block. |
| C: copy the current cell. | Ctrl + A: select all text in the current cell. |
| V: paste cells below the current cell. | Ctrl + Z: undo. |
| D, D: delete the current cell. | Ctrl + Y: redo. |
| Shift + J or Shift + Down: select the next cell in the same column. | Ctrl + Home: go to the beginning of the cell. |
| Shift + K or Shift + Up: select the previous cell in the same column. | Ctrl + End: go to the end of the cell. |
| Ctrl + Shift + -: split the current cell at the cursor. | Ctrl + Left: go one word to the left. |
| Z: undo cell deletion. | Ctrl + Right: go one word to the right. |
| X: cut the current cell. | Tab: indent the current line. |
| Shift + M: merge selected cells. | Shift + Tab: un-indent the current line. |
| M: markdown , Y: Code | More added manually by editing |

### Sample Project

```python
import pandas as pd
df = pd.read_csv("heart-disease.csv")
df.head(10)
df.target.value_counts().plot(kind="bar")
```

**If the above code does not work we need to import matplotlib**

```python
import matplotlib.pyplot as plt
```

**Opening a csv file**

- pd.read_csv("file.csv")

**Data frame row**

Pandas data frame `df` where we use `df.target.value_counts().plot(kind="bar")` here `target` is a column name where we use `value_counts()` to count & `plot(kind="bar")` to make a bar graph

**Image in markdown**

- ![](img location or img link)

</details>

<details>
<summary>Pandas Data Analysis</summary>

### Pandas Introduction

**Why pandas?**

- Simple to use
- Integrated with many other data science & ML Python Tools
- Helps you get your data ready for machine learning

**Learning on this section**

- Most useful functions
- pandas Datatypes
- Importing & exporting data
- Describing data
- Viewing & Selecting data
- Manipulating data

**Two Main Datatype**

- Series is 1D and similar to list in python
- DataFrame is 2D and similar to  Dictionary in python
    
    ```python
    # series = 1 dimentional
    series = pd.Series(['BMW','Toyota','Honda'])
    colours = pd.Series(['Red','Blue','White'])
    
    # DataFrame = 2 dimentional
    car_data = pd.DataFrame({'Car make':series, 'Colour':colours})
    ```
    
- Import & export to csv
    
    ```python
    # import data
    car_sales = pd.read_csv('car-sales.csv')
    # Exporting to csv
    car_sales.to_csv('exported-car-sales.csv',index=False) #index won't counted
    ```
    
- Import & export to excel
    
    ```python
    # import data
    car_sales = pd.read_csv("car-sales.csv")
    # Exporting to excel
    car_sales.to_excel("exported-car-sales.xlsx", index=False) #index won't counted
    export_car_sales = pd.read_excel("exported-car-sales.xlsx")
    ```
    

### Anatomy of a Dataframe

![Anatomy of a Dataframe](https://images2.imgbox.com/6c/eb/LfMv4qeh_o.png)

### Describe Data

```python
#An Attribute doesn't have bracket "()" only Function contain bracket"()"
#Attribute -- dtypes
car_sales.dtypes

#Function -- to_csv()
car_sales.to_csv()

car_sales.dtypes #get data types
car_sales.columns #get columns names
car_sales.index #get index range start,stop,step
car_sales.describe() #get statistics info of numeric columns
car_sales.info() #get more details similar to .dtypes
car_sales.mean(numeric_only=True) #get mean
#custom created series mean
car_prices = pd.Series([300,1500,111250])
car_prices.mean()

car_sales.sum() #get all column sum
car_sales['Doors'].sum() #get Door column sum
len(car_sales) #get length
car_sales #get first 10 column
```

### Viewing and selecting data

```python
car_sales.head() #get top 5 rows
car_sales.head(7) #get top 7 rows
car_sales.tail() #get bottom 5 rows

# .loc = index location & .iloc = position
animals = pd.Series(['cat','dog','bird','panda','snake'])

# Custom index
animals = pd.Series(['cat','dog','bird','panda','snake'],index=[9,3,6,2,3])
animals.loc[3] #index location
animals.iloc[3] #position

# Get first four row
car_sales.loc[:3]
car_sales.head(4)

# Selecting individual columns
car_sales['Make']
car_sales.Make
# If column name contain spaces it won't work in dot way
car_sales['Odometer (KM)']
# car_sales.Odometer (KM) <-- This will give error

# Filtering
car_sales[car_sales['Make']=='Toyota'] # This will show only Toyota data from Make column
car_sales[car_sales['Odometer (KM)']>100000]

# Crossover
pd.crosstab(car_sales['Make'], car_sales['Doors'])

# more useful as crossover is Groupby
car_sales.groupby(['Make','Colour','Price']).mean()

# Fixing the Price columns $4,000.00
car_sales["Price"] = car_sales["Price"].str.replace('[\$\,]', '',regex=True).astype(float)
```

### Data Manipulation

```python
# Every data on "Make" column to lowercase
car_sales['Make']=car_sales['Make'].str.lower()

# Working with Missing data filling Odometer missing value with mean of Odometer
car_sales_missing['Odometer'].fillna(car_sales_missing['Odometer'].mean())

# Filling value in datatset 2 ways 
# assigning way:
car_sales_missing['Odometer']=car_sales_missing['Odometer'].fillna(car_sales_missing['Odometer'].mean())
# inplace way:
car_sales_missing['Odometer'].fillna(car_sales_missing['Odometer'].mean(),inplace = True)

# Dropping missing value
car_sales_missing_dropped = car_sales_missing.dropna()

# Capitalize car names
car_sales['Make']=car_sales['Make'].str.capitalize()

# Column from series
seats_column = pd.Series([5,5,5,5,5])
car_sales['Seats']=seats_column

# Column from python list
fuel_economy = [7.5,9.2,5.0,9.6,8.7,4.7,7.6,8.7,3.0,4.5]
car_sales['Fuel per 100KM']=fuel_economy
car_sales

# Calculation
car_sales['Total fuel used (L)']=car_sales['Odometer (KM)']/100 *car_sales['Fuel per 100KM']
car_sales

# Creating a boolean column
car_sales['Passed road saftry']=True

# creating a column from single value
car_sales['Number of wheels']=4
car_sales

# creating a dummy column to drop
dump=pd.Series([5,3,3,4,5,3,7,3,2,4])
car_sales['Dump']=dump
# Droping the dummy column
car_sales=car_sales.drop('Dump',axis=1)

# Using lambda to convert Miles to KM
car_sales['Odometer (KM)'] = car_sales['Odometer (KM)'].apply(lambda x:x/1.6)
```

</details>