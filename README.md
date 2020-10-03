# clickbait_identification
An application of NLP techniques in order to identify clickbaits 

![Image of Clickbait](https://github.com/jyotiyadav99111/clickbait_identification/blob/main/images/clickbait-advertising-spam-icons-mobile-phone-screen-internet-business-concept-153257728.jpg)

[Clickbait](https://en.wikipedia.org/wiki/Clickbait) is very common and people tend to fall a victim very easily. They are merely to grab attention and make users click on the link. Information present in on such links are often misleading and deceptive.

#### Example: Tempering election results
This can potentially create very huge and silent issues in the Economy. As we all are aware of the fact that Indian Economy is highly driven by sentiments. These sentiments can be altered by these clickbaits by providing false information to the user. For example, election results can be totally manipulated by infusing wrong facts about the politicians to change people's opinion. 


# Model Content Description
Problem |	Data |	Methods |	Libs
--------|------|----------|-----
Classification, Text analysis | Clickbait/non clickbait headings |	Tokenization, Bidirectional LSTM, Sigmoid | Numpy, Pandas, Matplotlib, Tensorflow

In this project, we need to predict the probability of any article being a clickbait given the heading of the article. The underlying data is balanced therefore, accuracy has been used as the measuring stick to compare teh model. In order to prevent model from overfitting, Gaussian Noise has been used in the model. Bidirectional LSTM has been used for text context analysis. 

# Stats from the model
AS every pucntuation or stopwords are important for distinguishing clickbaits, no treatment has been performed for this. 

The distribution of top most 30 words is presented below:
##### 1. Clickbait
![link of clickbait_freq](https://github.com/jyotiyadav99111/clickbait_identification/blob/main/images/clickbait.samples.png)

##### 2. Non-clickbaits
![link of non_clickbait_freq](https://github.com/jyotiyadav99111/clickbait_identification/blob/main/images/non_clickbait_samples.png)

Stats clearly depict that top 30 words contain quotation(") and integrogative words(what, how, who) in order to make headlines attractive. This is one of the distinguished factor analysed by the model for classification. 

# How to run the model

1. Clone the repository
2. Install all the requirements using the following command in terminal:
``` pip install -r requirements.txt```
3. Once the required packages are installed. Run the following file through terminal:
```python predict.py```
