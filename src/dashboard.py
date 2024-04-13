



import streamlit as st
from utils import  find_top_websites,find_high_traffic_websites,find_countries_with_most_media,find_popular_articles,webiste_sentiment,keyword_extraction_and_analysis
from loader import NewsDataLoader

import numpy as np
_loader = NewsDataLoader()





st.title('News Analysis Dashboard')


loader = NewsDataLoader()
news_data = _loader.load_data('../data/data.csv/rating.csv')
traffic_data = _loader.load_data('..//data/traffic.csv')
domain_data = _loader.load_data('../data/domains_location.csv')


with st.spinner('loading top news websites...'):
    top_news_websites= find_top_websites(news_data)
    st.bar_chart(top_news_websites)

with st.spinner('key word xtraction...'):
    title_vector, content_vector,cosine_similarity = keyword_extraction_and_analysis(news_data)
    print(title_vector)
    print(content_vector)
    print(cosine_similarity)
    # Create scatter plot
    st.title('Keyword Extraction and Analysis')
    st.subheader('Scatter Plot of Title Vector vs. Content Vector')

    # Display scatter plot
    scatter_data = np.column_stack((title_vector, content_vector, cosine_similarity))
    
    st.scatter_chart(range(1, len(cosine_similarity)+1), color='#ffaa00', size=None, width=0, height=0, use_container_width=True)

with st.spinner('sentiment...'):
    st.subheader('website sentiment analysis')
    web_sentiment = webiste_sentiment(news_data)
    st.bar_chart(web_sentiment)

with st.spinner('Loading High Traffic Websites...'):
    st.subheader('High traffic websites')
    top_traffic_websites = find_high_traffic_websites(traffic_data)
    st.line_chart(top_traffic_websites)

with st.spinner('countries with the most media presense...'):
    st.subheader('Countires with most article written about them')
    top_traffic_websites = find_countries_with_most_media(domain_data)
    st.line_chart(top_traffic_websites)

# with st.spinner('Popular articles...'):
#     top_traffic_websites = find_popular_articles(news_data)
#     st.line_chart(top_traffic_websites)







