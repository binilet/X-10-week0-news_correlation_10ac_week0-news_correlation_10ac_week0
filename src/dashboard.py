import streamlit as st
from utils import  find_top_websites,find_high_traffic_websites,find_countries_with_most_media,find_popular_articles
from loader import NewsDataLoader
_loader = NewsDataLoader()

def main_dashboard():
    st.title('News Analysis Dashboard')
    st.sidebar.title('Options')

    df = _loader.load_data('../data/data.csv/traffic.csv')

    function_options = {
        'Top Websites by Article Count': find_top_websites,
        'High Traffic Websites': find_high_traffic_websites,
        'Countries with Most Media Outlets': find_countries_with_most_media,
        'Popular Articles by Country': find_popular_articles
    }

    selected_function = st.sidebar.selectbox('Select Function', list(function_options.keys()))
    
    top = 10

    if selected_function == 'Popular Articles by Country':
        st.info("This function might take a while to run due to data processing.")
        popular_countries_data = df[['index_column', 'content']]  # Replace 'index_column' with your actual index column name
        top_countries = find_popular_articles(popular_countries_data)
        st.write("Top 10 Countries with Most Articles:")
        st.write(top_countries)
    else:
        #top = st.sidebar.number_input('Top', min_value=1, value=10)

        if selected_function == 'Top Websites by Article Count':
            top_websites = find_top_websites(df, top=top)
            st.write("Top Websites by Article Count:")
            st.write(top_websites)

        elif selected_function == 'High Traffic Websites':
            high_traffic_websites = find_high_traffic_websites(df, top=top)
            st.write("High Traffic Websites:")
            st.write(high_traffic_websites)

        elif selected_function == 'Countries with Most Media Outlets':
            countries_with_most_media = find_countries_with_most_media(df, top=top)
            st.write("Countries with Most Media Outlets:")
            st.write(countries_with_most_media)

    if __name__ == "__main__":
        main_dashboard()