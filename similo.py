import streamlit as st
import pandas as pd
from urllib.request import urlopen
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json
import requests
from streamlit_lottie import st_lottie

#Layout
st.set_page_config(
    page_title="SimiLo",
    layout="wide",
    initial_sidebar_state="expanded")

#Data Pull and Functions
st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

@st.cache_data
def pull_clean():
    master=pd.read_csv('/Users/kevinsoderholm/Desktop/similo/data/MASTER_ZIP.csv')
    return master

#col1,col2,col3=st.columns(3)
#col2.markdown('<p class="big-font">SimiLo</p>', unsafe_allow_html=True)

#Options Menu
with st.sidebar:
    selected = option_menu('SimiLo', ["Intro", 'Search','About'], 
        icons=['play-btn','search','info-circle'],menu_icon='intersect', default_index=0)
    lottie = load_lottiefile("similo3.json")
    st_lottie(lottie,key='loc')

#Intro Page
if selected=="Intro":
    st.title('Welcome to SimiLo')
    st.subheader('*A new tool to find similar locations across the United States.*')

    st.divider()

    with st.container():
        col1,col2=st.columns(2)
        with col1:
            st.header('Use Cases')
            st.write('_Remote work got you thinking about relocation?_')
            st.write('_Looking for a new vacation spot?_')
            st.write('_Conducting market research for product expansion?_')
            st.write('_Just here to play and learn?_')
        with col2:
            lottie2 = load_lottiefile("place2.json")
            st_lottie(lottie2,key='place',height=300,width=300)

    st.divider()

    st.header('Tutorial Video')
    
#Search Page
if selected=="Search":
    master=pull_clean()
    master['ZIP'] = master['ZCTA5'].astype(str).str.zfill(5)
    st.subheader('Start with a zip code you know (required):')
    zip_select = st.selectbox('',['']+list(master['ZIP'].unique()))
    st.subheader('Filter results by state (optional)')
    col1,col2=st.columns(2)
    states=sorted(list(master['STATE_LONG'].astype(str).unique()))
    csas=sorted(list(master['CSA'].astype(str).unique()))
    state_select=col1.multiselect('State',states)

    if zip_select != '':
        #Benchmark
        selected_record = master[master['ZIP']==zip_select].reset_index()
        selected_zip=selected_record['ZIP'][0]
        selected_county=selected_record['County Title'][0]

        if len(state_select)>0:
            master=master[master['STATE_LONG'].isin(state_select)]

        #Columns for scaling
        PeopleCols_sc=['MED_AGE_sc', 'MED_HH_INC_sc', 'PCT_POVERTY_sc','PCT_BACH_MORE_sc']
        ProximityCols_sc=['POP_DENSITY_sc','Metro_Index_sc']
        HomeCols_sc=['HH_SIZE_sc','PCT_OWN_sc','MED_HOME_sc','PCT_UNIT1_sc']
        WorkCols_sc=['MEAN_COMMUTE_sc','PCT_WC_sc']
        EnvironmentCols_sc=['Pct_Water_sc','Env_Index_sc','Pct_ToPark_OneMile_sc']

        # Calculate the euclidian distance between the selected record and the rest of the dataset
        People_dist             = euclidean_distances(master.loc[:, PeopleCols_sc], selected_record[PeopleCols_sc].values.reshape(1, -1))
        Proximity_dist          = euclidean_distances(master.loc[:, ProximityCols_sc], selected_record[ProximityCols_sc].values.reshape(1, -1))
        Home_dist               = euclidean_distances(master.loc[:, HomeCols_sc], selected_record[HomeCols_sc].values.reshape(1, -1))
        Work_dist               = euclidean_distances(master.loc[:, WorkCols_sc], selected_record[WorkCols_sc].values.reshape(1, -1))
        Environment_dist        = euclidean_distances(master.loc[:, EnvironmentCols_sc], selected_record[EnvironmentCols_sc].values.reshape(1, -1))

        # Create a new dataframe with the similarity scores and the corresponding index of each record
        df_similarity = pd.DataFrame({'PEOPLE_SIM': People_dist [:, 0],'PROX_SIM': Proximity_dist [:, 0],'HOME_SIM': Home_dist [:, 0],'WORK_SIM': Work_dist [:, 0],'ENV_SIM': Environment_dist [:, 0], 'index': master.index})
        df_similarity['OVERALL_SIM']=df_similarity[['PEOPLE_SIM', 'PROX_SIM', 'HOME_SIM', 'WORK_SIM', 'ENV_SIM']].mean(axis=1)
        people_max=df_similarity['PEOPLE_SIM'].max()
        prox_max=df_similarity['PROX_SIM'].max()
        home_max=df_similarity['HOME_SIM'].max()
        work_max=df_similarity['WORK_SIM'].max()
        env_max=df_similarity['ENV_SIM'].max()
        overall_max=df_similarity['OVERALL_SIM'].max()

        df_similarity['PEOPLE_SCALE']  = 100 - (100 * df_similarity['PEOPLE_SIM'] / people_max)
        df_similarity['PROX_SCALE']    = 100 - (100 * df_similarity['PROX_SIM'] / prox_max)
        df_similarity['HOME_SCALE']    = 100 - (100 * df_similarity['HOME_SIM'] / home_max)
        df_similarity['WORK_SCALE']    = 100 - (100 * df_similarity['WORK_SIM'] / work_max)
        df_similarity['ENV_SCALE']     = 100 - (100 * df_similarity['ENV_SIM'] / env_max)
        df_similarity['OVERALL_SCALE'] = 100 - (100 * df_similarity['OVERALL_SIM'] / overall_max)

        # Sort the dataframe by the similarity scores in descending order and select the top 10 most similar records
        df_similarity = df_similarity.sort_values(by='OVERALL_SIM', ascending=True).head(11)

        # Merge the original dataframe with the similarity dataframe to display the top 10 most similar records
        df_top10 = pd.merge(df_similarity, master, left_on='index', right_index=True).reset_index(drop=True)
        df_top10=df_top10.loc[1:10]
        df_top10['Rank']=list(range(1,11))

        st.header('Top 10 Most Similar Locations')
        #st.write('You selected zip code '+zip_select+' from '+selected_record['County Title'][0])
        # CSS to inject contained in a string
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        col1,col2=st.columns(2)
        with col1:
            st.table(df_top10[['Rank','OVERALL_SIM','OVERALL_SCALE','ZIP','County Title']])
        with col2:
            latcenter=df_top10['LAT'].mean()
            loncenter=df_top10['LON'].mean()
            #map token for additional map layers
            token = "pk.eyJ1Ijoia3NvZGVyaG9sbTIyIiwiYSI6ImNsZjI2djJkOTBmazU0NHBqdzBvdjR2dzYifQ.9GkSN9FUYa86xldpQvCvxA" # you will need your own token
            fig1 = px.scatter_mapbox(df_top10, lat='LAT',lon='LON',center=go.layout.mapbox.Center(lat=latcenter,lon=loncenter),
                                     color="Rank", color_continuous_scale=px.colors.sequential.ice, hover_name='ZIP', hover_data=['Rank','County Title'],zoom=3)
            fig1.update_traces(marker={'size': 15})
            fig1.update_layout(mapbox_style="mapbox://styles/mapbox/light-v11",mapbox_accesstoken=token)
            fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig1,use_container_width=True)

        st.divider()

        st.header('Location Deep Dive')
        rank_select=st.selectbox('From rankings above, which do you want to compare to?',list(df_top10['Rank']))
        if rank_select:
            compare_record=df_top10[df_top10['Rank']==rank_select].reset_index(drop=True)
            compare_zip=compare_record['ZIP'][0]
            compare_county=compare_record['County Title'][0]
            st.write('Comparing '+selected_zip+' in '+selected_county+' to '+compare_zip+' in '+compare_county)
            tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['Overall','People','Proximity','Home','Work','Environment'])
            with tab1:
                st.subheader('Overall and Category Similarity Scores')
                col1,col2,col3=st.columns(3)
                col1.metric('Overall',compare_record['OVERALL_SCALE'][0].round(2))
                col1.progress(compare_record['OVERALL_SCALE'][0]/100)
                col1.metric('People',compare_record['PEOPLE_SCALE'][0].round(2))
                col1.progress(compare_record['PEOPLE_SCALE'][0]/100)
                col2.metric('Proximity',compare_record['PROX_SCALE'][0].round(2))
                col2.progress(compare_record['PROX_SCALE'][0]/100)
                col2.metric('Home',compare_record['HOME_SCALE'][0].round(2))
                col2.progress(compare_record['HOME_SCALE'][0]/100)
                col3.metric('Work',compare_record['WORK_SCALE'][0].round(2))
                col3.progress(compare_record['WORK_SCALE'][0]/100)
                col3.metric('Environment',compare_record['ENV_SCALE'][0].round(2))
                col3.progress(compare_record['ENV_SCALE'][0]/100)
            with tab2:
                selected_record['PCT_18_65']=selected_record['PCT_OVER_18']-selected_record['PCT_OVER_65']
                compare_record['PCT_18_65']=compare_record['PCT_OVER_18']-compare_record['PCT_OVER_65']
                st.subheader('People Metrics')
                col1,col2=st.columns(2)
                col1.metric('Pct Minors',selected_record['PCT_UNDER_18'][0].round(2))
                col2.metric('Pct Minors',compare_record['PCT_UNDER_18'][0].round(2))
                col1.metric('Pct Adults',selected_record['PCT_18_65'][0].round(2))
                col2.metric('Pct Adults',compare_record['PCT_18_65'][0].round(2))
                col1.metric('Pct Seniors',selected_record['PCT_OVER_65'][0].round(2))
                col2.metric('Pct Seniors',compare_record['PCT_OVER_65'][0].round(2))
                col1.metric('Median Age',selected_record['MED_AGE'][0].round(2))
                col2.metric('Median Age',compare_record['MED_AGE'][0].round(2))
                col1.metric('Median HH Income',selected_record['MED_HH_INC'][0].round(2))
                col2.metric('Median HH Income',compare_record['MED_HH_INC'][0].round(2))
                col1.metric('Pct in Poverty',selected_record['PCT_POVERTY'][0].round(2))
                col2.metric('Pct in Poverty',compare_record['PCT_POVERTY'][0].round(2))
                col1.metric('Pct with Bach Degree',selected_record['PCT_BACH_MORE'][0].round(2))
                col2.metric('Pct with Bach Degree',compare_record['PCT_BACH_MORE'][0].round(2))
            with tab3:
                st.subheader('Proximity Metrics')
                col1,col2=st.columns(2)
                col1.metric('Population Density',selected_record['POP_DENSITY'][0].round(2))
                col2.metric('Population Density',compare_record['POP_DENSITY'][0].round(2))
                col1.metric('Metropolitan Index',selected_record['Metro_Index'][0].round(2))
                col2.metric('Metropolitan Index',compare_record['Metro_Index'][0].round(2))
            with tab4:
                st.subheader('Home Metrics')
                col1,col2=st.columns(2)
                col1.metric('Avg HH Size',selected_record['HH_SIZE'][0].round(2))
                col2.metric('Avg HH Size',compare_record['HH_SIZE'][0].round(2))
                col1.metric('Avg Family Size',selected_record['FAM_SIZE'][0].round(2))
                col2.metric('Avg Family Size',compare_record['FAM_SIZE'][0].round(2))
                col1.metric('Pct Own Home',selected_record['PCT_OWN'][0].round(2))
                col2.metric('Pct Own Home',compare_record['PCT_OWN'][0].round(2))
                col1.metric('Pct Rent',selected_record['PCT_RENT'][0].round(2))
                col2.metric('Pct Rent',compare_record['PCT_RENT'][0].round(2))
                col1.metric('Median Home Price',selected_record['MED_HOME'][0].round(2))
                col2.metric('Median Home Price',compare_record['MED_HOME'][0].round(2))
                col1.metric('Median Rent Price',selected_record['MED_RENT'][0].round(2))
                col2.metric('Median Rent Price',compare_record['MED_RENT'][0].round(2))
                col1.metric('Pct 1-unit Property',selected_record['PCT_UNIT1'][0].round(2))
                col2.metric('Pct 1-unit Property',compare_record['PCT_UNIT1'][0].round(2))
            with tab5:
                st.subheader('Work Metrics')
                col1,col2=st.columns(2)
                col1.metric('Pct Working',selected_record['PCT_WORKING'][0].round(2))
                col2.metric('Pct Working',compare_record['PCT_WORKING'][0].round(2))
                col1.metric('Avg Commute Time',selected_record['MEAN_COMMUTE'][0].round(2))
                col2.metric('Avg Commute Time',compare_record['MEAN_COMMUTE'][0].round(2))
                col1.metric('Pct Service Jobs',selected_record['PCT_SERVICE'][0].round(2))
                col2.metric('Pct Service Jobs',compare_record['PCT_SERVICE'][0].round(2))
                col1.metric('Pct Blue Collar Jobs',selected_record['PCT_BC'][0].round(2))
                col2.metric('Pct Blue Collar Jobs',compare_record['PCT_BC'][0].round(2))
                col1.metric('Pct White Collar Jobs',selected_record['PCT_WC'][0].round(2))
                col2.metric('Pct White Collar Jobs',compare_record['PCT_WC'][0].round(2))
            with tab6:
                st.subheader('Environmental Metrics')
                col1,col2=st.columns(2)
                col1.metric('Pct Area is Water',selected_record['Pct_Water'][0].round(2))
                col2.metric('Pct Area is Water',compare_record['Pct_Water'][0].round(2))
                col1.metric('Environmental Quality Index',selected_record['Env_Index'][0].round(2))
                col2.metric('Environmental Quality Index',compare_record['Env_Index'][0].round(2))
                col1.metric('Pct within 0.5 mile to Park',selected_record['PCT_SERVICE'][0].round(2))
                col2.metric('Pct within 0.5 mile to Park',compare_record['PCT_SERVICE'][0].round(2))
                col1.metric('Pct within 1 mile to Park',selected_record['Pct_ToPark_HalfMile'][0].round(2))
                col2.metric('Pct within 1 mile to Park',compare_record['Pct_ToPark_OneMile'][0].round(2))

                                 

#About Page
if selected=='About':
    st.title('Data')
    st.subheader('All data for this project was publicly sourced from U.S. Government organizations shown below:')
    
    with st.container():
        col1,col2=st.columns([1,2])
        col1.image('census_graphic.png',width=150)
        col2.write('Demographic, housing, and industry data was sourced from the U.S. Census Bureau.')
        col2.write('American Community Survey, 5-Year Profiles, 2021, datasets DP02 - DP05')
        col2.write('See: https://data.census.gov/')
    
    with st.container():
        col1,col2=st.columns([1,2])
        col1.image('cdc.png',width=150)
        col2.write('Environmental location data was sourced from the Centers for Disease Control and Prevention (CDC).')
        col2.write('See: https://data.cdc.gov/')
    
    with st.container():
        col1,col2=st.columns([1,2])
        col1.image('hud.png',width=150)
        col2.write('Crosswalk files to connect zip codes, counties, MSAs, and States was sourced from the U.S. Department of Housing and Urban Development (HUD).')
        col2.write('See: https://www.huduser.gov/portal/datasets/usps_crosswalk.html')
    st.divider()
    
    st.title('Creator')
    st.write('**Name:**    Kevin Soderholm')
    st.write('**Education:**    M.S. Applied Statistics')
    st.write('**Experience:**    8 YOE in Data Science across Banking, Financial Technology, and Retail')
    st.write('**Contact:**    kevin.soderholm@gmail.com or https://www.linkedin.com/in/kevin-soderholm-67788829/')
    st.write('**Thanks for stopping by!**')
