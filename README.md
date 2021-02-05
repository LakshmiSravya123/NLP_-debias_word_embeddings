# NLP_-debias_word_embeddings
Word Embedding converts a word to an n-dimensional vector. The related words are mapped to similar n-dimensional vector, whereas the dissimilar words have different vectors. This method's main advantage is that the model trained on one word can react to a similar word even though the word is new for the model. In this way, the 'meaning' of the word is reflected as embedding, and a model can use this information to learn the relationship between words.
For example, take four words represented by vectors. Since 'Dog' and 'Cat' are closer to each other as they are animals there word vectors are relative, whereas 'Apple' and 'Mango' are closer to each other as they represent fruits. On the contrary, both groups are far from each other as they are not similar.

Some words are gender-neutral in our vocabulary, for example, 'football’ and ‘receptionist,' whereas some words are gender-specific, like 'brother’ and ‘father.' Through various studies, it is understood that the gender-neutral word embedding acquire stereotypes and bias. The possible solution to this problem is to Neutralize and Equalize.

# Gender Space
I have identified a gender subspace where i can identify the dimensions where the bias is captured in the embedding. I first take the difference between sets of word embedding that define the concept of gender (e.g., 'male’ & ‘female,' 'he’ and ‘she,' etc.). Then, the bias subspace is calculated by taking the SVD of these differences

# Neutralize
After obtaining the bias direction,i removed the bias components from all gender-neutral words like receptionist and surgeon by subtracting the embedding’s projection onto the bias axis

# Equalize
For every gender-specific word, i equalized their vector lengths such that the gender component is preserved with equal strength in all pairs of words. Furthermore, it enforces that all gender-neutral words are equidistant from gender-specific words. E.g., the receptionist is equidistant from boy and girl.

# Datsets Used
- gender.csv (blog data set)
http://www.cs.uic.edu/~liub/FBS/blog-gender-dataset.rar
- gender-classifier-DFE-791531 (twitter data set)
https://www.kaggle.com/crowdflower/twitter-user-gender-classification

# Feature Design
- TF-IDF 
- Word Embeddings 

# Machine learning Classifiers

- Multinomial Naive Bayes
- Random Forest 
- SVM
- Voting Classifier

# Methodology 
Train-Test split: The training set is divided into 80% of training data and 20% as a test data set. After training, the whole test data is passed to the classifier for predictions.

# Hyperparameters
The hyper parameters are tuned using the most popular ways, using grid search and random search. In grid search, every combination of values of hyper parameters is taken into consideration, whereas in a random search, random combinations are considered. In our case, we have used a grid search for MultinomialNB and SVM, whereas for the random forest, the random search is being used. The random search is taken lesser time comparatively and very likely to be optimized.

# Gender Classification Results 
 ## Twitter dataset
Classfiers|TFIDF after preprocessing accuracy (%)| Word Embedding before bias removal accuracy(%)|Word Embedding after bias removal accuracy (%)|
|------|------|------|-----|
Random Forest | 70.4|65.8|68.3|
Multinomial Naive Bayes|67.5|-|-|
SVM |70.6|67.3|67.4|
Voting| 70.2|67.38|67.6|
## Blog data set
Classfiers|TFIDF after preprocessing accuracy (%)| Word Embedding before bias removal accuracy(%)|Word Embedding after bias removal accuracy (%)|
|------|------|------|-----|
Random Forest | 68.1|61.7|64.8|
Multinomial Naive Bayes|65.7|-|-|
SVM |61.6|63.5|65.7|
Voting| 65.5|62.8|64.0|

# Discussions 
|Methods|Advantages| Disadvantages|
|-----|-----|------|
TF-IDF| - Trained without external data <br> - Applied on the training data at once <br> - Less memory intensive | Catures no meaning
Word Embedding | - Captured word meaning and relative words| - Trained in extensive external data <br> - Applied to each word individually <br> - More memory  intensive



# Languages
Pythong (NLP word embeddings)
# References
https://github.com/Dalia-Sh/Debiasing-Word-Embeddings/blob/master/Code/project.ipynb
