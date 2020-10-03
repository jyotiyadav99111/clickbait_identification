# clickbait_identification
An application of NLP techniques in order to identify clickbaits 

![Image of Clickbait](https://github.com/jyotiyadav99111/clickbait_identification/blob/main/images/clickbait-advertising-spam-icons-mobile-phone-screen-internet-business-concept-153257728.jpg)

[Clickbait](https://en.wikipedia.org/wiki/Clickbait) is very common and people tend to fall a victim very easily. They are merely to grab attention and make users click on the link. Information present in on such links are often misleading and deceptive.

## Example: Tempering election results
This can potentially create very huge and silent issues in the Economy. As we all are aware of the fact that Indian Economy is highly driven by sentiments. These sentiments can be altered by these clickbaits by providing false information to the user. For example, election results can be totally manipulated by infusing wrong facts about the politicians to change people's opinion. 


# Model Content Description
Problem |	Data |	Methods |	Libs
--------|------|----------|-----
Classification, Text analysis | Clickbait/non clickbait headings |	Tokenization, Bidirectional LSTM, Sigmoid | Numpy, Pandas, Matplotlib, Tensorflow

In this project, we need to predict the probability of any article being a clickbait given the heading of the article. The underlying data is balanced therefore, accuracy has been used as the measuring stick to compare teh model. In order to prevent model from overfitting, Gaussian Noise has been used in the model. Bidirectional LSTM has been used for text context analysis. 

# Stats from the model
