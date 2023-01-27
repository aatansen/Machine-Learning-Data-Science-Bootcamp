<details>
<summary>Enable Jupyter Notebook extension and Code formatter</summary>

- pip install jupyter_contrib_nbextensions Now type and enter:
- jupyter contrib nbextension install --user Now let's enable the extension:
- jupyter nbextension enable
- e.g-->C:\ProgramData\Anaconda3\Lib\site-packages\jupyter_contrib_nbextensions\nbextensions\codefolding\main Now open Jupyter Notebook
- 'Nbextensions' will be there now enable required extension
- Done!!!
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

**What is Machine Learning?**

> A subset of AI that focuses on building systems that can learn from data and make predictions or decisions without being explicitly programmed.
> 

**AI and Machine Learning**

- **AI**—>A machine that acts like a human
    - **Machine Learning—>**a subset of AI
        - An approach to achieve artificial intelligence through systems that can find patterns in a set of data.
        - Stanford university describe machine learning as the science of getting computers to act without being explicitly programmed. Which means we do not need to tell the machine do this , do that, if then , if this then that…etc.
            - **Deep Learning**—>a subset of Machine Learning
                - one of the techniques for implementing machine learning
    - **Data Science—>**Analyzing data and then doing something. It include both **Machine Learning & Deep Learning.**
- **Narrow AI**—>A machine that acts like a human for a specific task
    - Detecting heart disease from images
    - Game of go or chess or Star craft and other video games
    - Only work on single task
- **General AI**: A machine that acts like a human with multiple abilities

**Why Machine Learning**

```mermaid
%%{init: {'theme': 'dark', "flowchart" : { "curve" : "basis" } } }%%
graph LR
A[Spreadsheets] -->|Then we move to| B[Relational DB - MySQL]
B -->|Then we move to| C[Big Data - NoSQL]
C -->|Finally| D[Machine Learning]
```

**Machine Learning Project Steps**

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
- [ML Playground](https://mlplaygrounds.com/) by ***Mrityunjay Bhardwaj*** (advanced)
- [ML Playground](https://playground.tensorflow.org/) by TensorFlow (More advanced)

**Types of Machine Learning**

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
- Machine can learn from big data to predict future trends and make business decision

</details>