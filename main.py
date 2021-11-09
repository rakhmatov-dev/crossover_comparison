import pandas as pd
import openpyxl
from gensim.models import Word2Vec

df = pd.read_csv('C:/Users/rakhm/Desktop/didox goods total/crossover_comparison/data.csv')
# print(df.head())

# We should prapare list of list in order to use Word2Vec
# Create a new column for Make Model
df['Maker_Model'] = df['Make'] + " " + df['Model']
# print(df['Maker_Model'])

#----------------- Generate a format of 'list of lists' for each Make Model with the following features: -----------------#
# - Engine Fuel Type
# - Transmission Type
# - Driven_Wheels
# - Market Category
# - Vehicle Size
# - Vehicle Style

# Select features from original dataset to form a new dataframe
df1 = df[['Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style', 'Maker_Model']]
# print(df1.head())

# For each row, combine all the columns into one column
df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)
# print(df2.head())

# Store them in a pandas dataframe
df_clean = pd.DataFrame({'clean': df2})
# print(df_clean.head())

# Create the list of list format of the custom corpus for gensim modeling
sent = [row.split(',') for row in df_clean['clean']]

# Show the example of list of list format of the custom corpus for gensim modeling
print(sent[:2])

#-------------------------------------------------------------------------------------#

#----------------- Train the gensim word2vec model with our own custom corpus -----------------#

model = Word2Vec(
    sent,
    min_count=1,
    vector_size=50,
    workers=3,
    window=3,
    sg=1
)

# Lets's try to understand the hyperparameters of this model
# vector_size - the number of dimensions of the embeddings, default = 100
# window - the maximum distance between a target word and words around the target word, default = 5
# min_count - the minimum count of words to consider when training the model; words with occurrence less than
#               this count will be ignored, default = 5
# workers - the number of partitions during training, default = 3
# sg - the training algorithm, either CBOW (0) or skip gram (1), default = 0

model.build_vocab(sent)
model.train(sent, total_examples=len(sent), epochs=30)


# After training the word2vec model, we can obtain the word embedding directly from the training model as following:
print(model.wv.__getitem__("Toyota Camry"))

#-------------------------------------------------------------------------------------#

#----------------- Compare Similarities -----------------#

# Now we could even use Word2vec to compute similarity between two make model in the vocabulary
print(model.wv.similarity('Porsche 718 Cayman', 'Nissan Van'))
print(model.wv.similarity('Porsche 718 Cayman', 'Mercedes-Benz SLK-Class'))

# From the above example, we can tell that Porsche 718 Cayman is more similar with Mercedes-Benz SLK-Class than Nissan Van

# Show the most similar vehicles for Mercedes-Benz SLK-Class : Default by eculidean distance
print(model.wv.most_similar('Mercedes-Benz SLK-Class')[:5])

# Show the most similar vehicles for Toyota Camry : Default by eculidean distance
print(model.wv.most_similar('Toyota Camry')[:5])






