
import pandas as pd
# import seaborn as sns

import nltk;
from nltk.tokenize import word_tokenize
from nltk import pos_tag,ne_chunk
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm

#imports for task-2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

#from bertopic import BERTopic
import re
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')




def find_top_websites(data,url_column='url',top=10):
    """
        this function will get the top [top] websites with highest article counts
    """
    data['domain'] = data[url_column].apply(lambda x: x.split('/')[2])

    #count occurences of each domain
    domain_counts = data['domain'].value_counts()

    top_domains = domain_counts.head(top)
    return top_domains

def find_high_traffic_websites(data,top=10):
    """
    this function will return websites with high reference ips(assuming the ips are the number of traffic)
    this should include/join the dataset in news
    """

    traffic_per_domain = data.groupby(['Domain'])['RefIPs'].sum()
    traffic_per_domain = traffic_per_domain.sort_values(ascending=False)
    return traffic_per_domain.head(top)

def find_countries_with_most_media(data,top=10):
    """
    this function will return the top countires with the most media outlets
    it simply groups the data by the country field and counts them
    """
    media_per_country = data['Country'].value_counts()
    media_per_country = media_per_country.sort_values(ascending=False)
    return media_per_country.head(top)


"""
the following set of functions will create
analysis function using nltk(an NLP libraries) to calculate
how many news stories are about a specific countries
"""
#first download required nltk packages

def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

count = 0

def extract_countries_from_article_content(article):
    index,row = article
    text = row['content']
    #tokenize every text into words
    words = word_tokenize(text)
    #part of speech tagging; this means tag every word; with noun,verb ...
    tagged_words = pos_tag(words)
    #get named entities from the tagged lists
    named_entities = ne_chunk(tagged_words)
    #GPE stands for 'Geopolitical Entity' in our case country name
    countries = [chunk[0] for chunk in list(named_entities) if hasattr(chunk,'label') and chunk.label() == 'GPE']

    
    return countries

def find_popular_articles(popular_countries_data):
    print('downloading nltk resources ...')
    download_nltk_resources()
    print('finished downloading resources ...')
    print('loading data ...')
    df = popular_countries_data
    print('starting processing this might take a while ...')
    #since we have a lot of data we need to parallize the process

    # Maximum number of rows to process
    max_rows = len(df)
    print(f'max rows is: {max_rows}')
    processed_count = 0
    # Apply function to each article in parallel with tqdm for progress bar
    with Pool() as pool:
        results = []
        for countries in tqdm(pool.imap(extract_countries_from_article_content, df.iterrows()), total=len(df)):
            # Append the results
            results.append(countries)
            
            # Increment processed_count
            processed_count += 1
            
            # Check if maximum number of rows processed
            if processed_count >= max_rows:
                print("Maximum number of rows processed. Stopping pool.")
                break
    print('done processing!')
    # Flatten the list of results
    all_countries = [country for countries in results for country in countries]
    
   

    # Count occurrences of each country
    print("debug printing count...")
    country_counts = Counter(all_countries)
    print(country_counts.most_common(3))
    return country_counts.most_common(10)


def webiste_sentiment(data):
    sentiment_counts = data.groupby(['source_name','title_sentiment']).size().unstack(fill_value = 0)
    return sentiment_counts

def website_sentiment_distribution(data):
    sentiment_counts=data.groupby(['source_name','title_sentiment']).size().unstack(fill_value = 0)
    sentiment_counts['Total'] = sentiment_counts.sum(axis=1)

    # Calculate mean and median sentiment counts for each domain
    sentiment_counts['Mean'] = sentiment_counts[['Positive', 'Neutral', 'Negative']].mean(axis=1)
    sentiment_counts['Median'] = sentiment_counts[['Positive', 'Neutral', 'Negative']].median(axis=1)

    # Display the sentiment counts along with mean and median
    print("Sentiment counts with mean and median:")
    print(sentiment_counts)
    return sentiment_counts


#task two topic modeling and sentiment analysis


def keyword_extraction_and_analysis(news_data):

    """
        this function will perform key word extraction usig tf-idf
        used 10 keyword accross header and title
        and it needs to occur at least once on both header and content to be considered

        min_words_threshold will limit the analysis for articles with at least 5 words
        
    """
    #nltk.download('stopwords')
    # Define stop words (English in this example)
    stop_words = set(stopwords.words('english'))

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=10, min_df=1)  # Adjust max_features for desired number of keywords

    # Initialize lists to store keywords and similarities
    title_keywords_list = []
    content_keywords_list = []
    similarity_list = []

    MIN_WORDS_THRESHOLD = 5  # Adjust as needed
    article_count = 0 #perform for the first 10 articles for now
    # Process each news item
    for index, row in news_data.iterrows():
        # Preprocess text (lowercase, remove punctuation, tokenize)

        if(article_count == 101): break
        
        title_text = row['title']
       
        # Preprocess text (lowercase, remove punctuation, tokenize)
        processed_title = [word.lower() for word in word_tokenize(title_text) if word.lower() not in stop_words and word.isalpha()]
        content_text = row['content']
       
        processed_content = [word.lower() for word in word_tokenize(content_text) if word.lower() not in stop_words and word.isalpha()]

        if len(processed_title) < MIN_WORDS_THRESHOLD and len(processed_content) < MIN_WORDS_THRESHOLD:
            continue

        # Combine title and content for TF-IDF analysis
        combined_text = ' '.join(processed_title + processed_content)
       
        # Fit vectorizer to combined text
        vectorizer.fit([combined_text])

        # Extract TF-IDF scores for the current article
        tfidf_scores = vectorizer.transform([combined_text]).toarray()[0]
        
        
        
        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()

        # Sort keywords by TF-IDF scores (descending order) and select top 5
        top_keywords_title = [keyword for keyword, _ in sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:5]]
        top_keywords_content = [keyword for keyword, _ in sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[5:10]]

       
        if(len(tfidf_scores) >= 10):

             # Append top keywords to lists
            title_keywords_list.append(top_keywords_title)
            content_keywords_list.append(top_keywords_content)

            # Calculate cosine similarity between title and content keywords
            title_tfidf_scores = tfidf_scores[:5]
            content_tfidf_scores = tfidf_scores[5:10]

            

            similarity = 1 - cosine(title_tfidf_scores, content_tfidf_scores)
            
            similarity_list.append(similarity)
            article_count = article_count+1
        else:
            print('can not caluclate similarty on unbalanced keywords')
        
        
    return title_keywords_list, content_keywords_list, similarity_list


#bert topic modeling
# Define your own categories
own_categories = {
    0: 'Breaking News',
    1: 'Politics',
    2: 'World News',
    3: 'Business/Finance',
    4: 'Technology',
    5: 'Science',
    6: 'Health',
    7: 'Entertainment',
    8: 'Sports',
    9: 'Environment',
    10: 'Crime',
    11: 'Education',
    12: 'Weather',
    
}

def clean_text(text):
    clean_text = re.sub('<.*?>','',text)
    clean_text = re.sub(r'[^\w\s]','',text)
    return clean_text

def preprocess_text(text):
    #tokenization and removal of stop words
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    words = [word for word in text if word not in stop_words]

    #lemmetize : meaning reducing the word to its root form
    lemmetizer = WordNetLemmatizer()
    words = [lemmetizer.lemmatize(word) for word in words]
    return ' '.join(words)



    

    

        
