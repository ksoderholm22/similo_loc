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
import pydeck as pdk

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
    master=pd.read_csv('MASTER_ZIP.csv')
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
            st.markdown(
                """
                - _Remote work got you thinking about relocation?_
                - _Looking for a new vacation spot?_
                - _Conducting market research for product expansion?_
                - _Just here to play and learn?_
                """
                )
        with col2:
            lottie2 = load_lottiefile("place2.json")
            st_lottie(lottie2,key='place',height=300,width=300)

    st.divider()

    st.header('Tutorial Video')
    
#Search Page
if selected=="Search":
    st.subheader('Enter Zip (required):')
    master=pull_clean()
    master['ZIP'] = master['ZCTA5'].astype(str).str.zfill(5)
    zip_select = st.selectbox(label='zip',options=['Zip']+list(master['ZIP'].unique()),label_visibility='collapsed')
    with st.expander('Advanced Settings'):

        st.subheader('Filter Results (optional):')
        col1,col2=st.columns(2)
        states=sorted(list(master['STATE_LONG'].astype(str).unique()))
        state_select=col1.multiselect('Filter Results by State(s)',states)
        count_select=col2.number_input(label='How Many Results?',min_value=5,max_value=25,value=10,step=5)
        st.subheader('Data Category Importance (optional):')
        col1,col2,col3=st.columns([1,.7,.7])
        col1.caption('0.5 = Less Important')
        col2.caption('1.0 = Default')
        col3.caption('1.5 = More Important')
        people_select=st.slider(label='People',min_value=0.5, max_value=1.5, step=0.1, value=1.0)
        home_select=st.slider(label='Home',min_value=0.5, max_value=1.5, step=0.1, value=1.0)
        work_select=st.slider(label='Work',min_value=0.5, max_value=1.5, step=0.1, value=1.0)
        environment_select=st.slider(label='Environment',min_value=0.5, max_value=1.5, step=0.1, value=1.0)

    filt_master=master
    if len(state_select)>0:
        filt_master=master[master['STATE_LONG'].isin(state_select)]
    #Benchmark
    if zip_select != 'Zip':
        selected_record = master[master['ZIP']==zip_select].reset_index()
        selected_zip=selected_record['ZIP'][0]
        selected_county=selected_record['County Title'][0]

        #Columns for scaling
        PeopleCols_sc=['MED_AGE_sc','PCT_UNDER_18_sc','MED_HH_INC_sc', 'PCT_POVERTY_sc','PCT_BACH_MORE_sc']
        HomeCols_sc=['HH_SIZE_sc','PCT_OWN_sc','MED_HOME_sc','PCT_UNIT1_sc','PCT_UNIT24_sc']
        WorkCols_sc=['MEAN_COMMUTE_sc','PCT_WC_sc','PCT_WORKING_sc','PCT_SERVICE_sc','PCT_BC_sc']
        EnvironmentCols_sc=['Pct_Water_sc','Env_Index_sc','Pct_ToPark_OneMile_sc','POP_DENSITY_sc','Metro_Index_sc']

        # Calculate the euclidian distance between the selected record and the rest of the dataset
        People_dist             = euclidean_distances(filt_master.loc[:, PeopleCols_sc], selected_record[PeopleCols_sc].values.reshape(1, -1))
        Home_dist               = euclidean_distances(filt_master.loc[:, HomeCols_sc], selected_record[HomeCols_sc].values.reshape(1, -1))
        Work_dist               = euclidean_distances(filt_master.loc[:, WorkCols_sc], selected_record[WorkCols_sc].values.reshape(1, -1))
        Environment_dist        = euclidean_distances(filt_master.loc[:, EnvironmentCols_sc], selected_record[EnvironmentCols_sc].values.reshape(1, -1))

        # Create a new dataframe with the similarity scores and the corresponding index of each record
        df_similarity = pd.DataFrame({'PEOPLE_SIM': People_dist [:, 0],'HOME_SIM': Home_dist [:, 0],'WORK_SIM': Work_dist [:, 0],'ENV_SIM': Environment_dist [:, 0], 'index': filt_master.index})

        #df_similarity['OVERALL_SIM']=df_similarity['PEOPLE_SIM','HOME_SIM','WORK_SIM','ENV_SIM'].mean(axis=1)
        weights=[people_select,home_select,work_select,environment_select]
        # Multiply column values with weights
        df_weighted = df_similarity.loc[:, ['PEOPLE_SIM', 'HOME_SIM', 'WORK_SIM','ENV_SIM']].mul(weights)
        df_similarity['OVERALL_W']=df_weighted.sum(axis=1)/sum(weights)

        people_max=df_similarity['PEOPLE_SIM'].max()
        home_max=df_similarity['HOME_SIM'].max()
        work_max=df_similarity['WORK_SIM'].max()
        env_max=df_similarity['ENV_SIM'].max()
        overall_max=df_similarity['OVERALL_W'].max()

        df_similarity['PEOPLE']  = 100 - (100 * df_similarity['PEOPLE_SIM'] / people_max)
        df_similarity['HOME']    = 100 - (100 * df_similarity['HOME_SIM'] / home_max)
        df_similarity['WORK']    = 100 - (100 * df_similarity['WORK_SIM'] / work_max)
        df_similarity['ENVIRONMENT']     = 100 - (100 * df_similarity['ENV_SIM'] / env_max)
        df_similarity['OVERALL'] = 100 - (100 * df_similarity['OVERALL_W'] / overall_max)

        # Sort the dataframe by the similarity scores in descending order and select the top 10 most similar records
        df_similarity = df_similarity.sort_values(by='OVERALL_W', ascending=True).head(count_select+1)

        # Merge the original dataframe with the similarity dataframe to display the top 10 most similar records
        df_top10 = pd.merge(df_similarity, filt_master, left_on='index', right_index=True).reset_index(drop=True)
        df_top10=df_top10.loc[1:count_select]
        df_top10['Rank']=list(range(1,count_select+1))
        df_top10['Ranking']=df_top10['Rank'].astype(str)+' - Zip Code '+df_top10['ZIP']+' from '+df_top10['County Title']
        df_top10['LAT_R']=selected_record['LAT'][0]
        df_top10['LON_R']=selected_record['LON'][0]
        df_top10['SAVE']=False

        st.header('Top '+'{}'.format(count_select)+' Most Similar Locations')
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
        tab1,tab2=st.tabs(['Map','Data'])
        with tab2:
            with st.expander('Expand for Table Info'):
                st.markdown(
                """
                - The values for OVERALL, PEOPLE, HOME, WORK, and ENVIRONMENT in the table below are similarity scores for the respective categories from 0-100 with 100 representing a perfect match.
                - Locations are ranked by their OVERALL score, which is a weighted average of the individual category scores.  The category scores contribute equally to the OVERALL score unless you specified otherwise in the Advanced section above.
                - The column SAVE allows you to check which locations you want to save for more research after you are done using SimiLo.
                - The CSV Download button will only save the selections you checked in the SAVE column.
                """
                )
            @st.cache_data
            def convert_df(df):
                return df.to_csv().encode('utf-8')
            cols=['Rank','OVERALL','PEOPLE','HOME','WORK','ENVIRONMENT','ZIP','County Title']
            df=df_top10[cols+['SAVE']]
            df=df.set_index('Rank')
            edited_df=st.experimental_data_editor(df)
            save=edited_df[edited_df['SAVE']==True]
            save=save.reset_index()
            csv = convert_df(save[cols])
            st.download_button(label="Download Selections as CSV",data=csv,file_name='SIMILO_SAVED.csv',mime='text/csv',)
        with tab1:
            latcenter=df_top10['LAT'].mean()
            loncenter=df_top10['LON'].mean()
            #map token for additional map layers
            token = "pk.eyJ1Ijoia3NvZGVyaG9sbTIyIiwiYSI6ImNsZjI2djJkOTBmazU0NHBqdzBvdjR2dzYifQ.9GkSN9FUYa86xldpQvCvxA" # you will need your own token
            #mapbox://styles/mapbox/streets-v12
            fig1 = px.scatter_mapbox(df_top10, lat='LAT',lon='LON',center=go.layout.mapbox.Center(lat=latcenter,lon=loncenter),
                                    color="Rank", color_continuous_scale=px.colors.sequential.ice, hover_name='ZIP', hover_data=['Rank','County Title'],zoom=3,)
            fig1.update_traces(marker={'size': 15})
            fig1.update_layout(mapbox_style="mapbox://styles/mapbox/satellite-streets-v12",
                               mapbox_accesstoken=token)
            fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig1,use_container_width=True)

        st.divider()

        st.header('Location Deep Dive')
        rank_select=st.selectbox('From rankings above, which one do you want to investigate?',list(df_top10['Ranking']))
        if rank_select:
            compare_record=df_top10[df_top10['Ranking']==rank_select].reset_index(drop=True)
            compare_zip=compare_record['ZIP'][0]
            compare_county=compare_record['County Title'][0]
            compare_state=compare_record['STATE_SHORT'][0].lower()
            #st.write(selected_zip+' in '+selected_county+' VS '+compare_zip+' in '+compare_county)
            tab1,tab2,tab3,tab4,tab5 = st.tabs(['Overall','People','Home','Work','Environment'])
            with tab1:
                col1,col2=st.columns(2)
                col1.subheader('Selected')
                col1.write(selected_zip+' in '+selected_county)
                col2.subheader('Similar')
                col2.write(compare_zip+' in '+compare_county)
                st.divider()
                st.subheader('Similarity Scores')
                col1,col2,col3,col4,col5=st.columns(5)
                col1.metric('Overall',compare_record['OVERALL'][0].round(2))
                col1.progress(compare_record['OVERALL'][0]/100)
                col2.metric('People',compare_record['PEOPLE'][0].round(2))
                col2.progress(compare_record['PEOPLE'][0]/100)
                col3.metric('Home',compare_record['HOME'][0].round(2))
                col3.progress(compare_record['HOME'][0]/100)
                col4.metric('Work',compare_record['WORK'][0].round(2))
                col4.progress(compare_record['WORK'][0]/100)
                col5.metric('Environment',compare_record['ENVIRONMENT'][0].round(2))
                col5.progress(compare_record['ENVIRONMENT'][0]/100)
                df_long = pd.melt(compare_record[['OVERALL','PEOPLE','HOME','WORK','ENVIRONMENT']].reset_index(), id_vars=['index'], var_name='Categories', value_name='Scores')
                fig = px.bar(df_long, x='Categories', y='Scores', color='Scores', color_continuous_scale='blues')
                fig.update_layout(xaxis_title='Categories',
                  yaxis_title='Similarity Scores')
                st.plotly_chart(fig,use_container_width=True)
            with tab2:
                selected_record['PCT_18_65']=selected_record['PCT_OVER_18']-selected_record['PCT_OVER_65']
                compare_record['PCT_18_65']=compare_record['PCT_OVER_18']-compare_record['PCT_OVER_65']
                dif_cols=['MED_AGE','MED_HH_INC','PCT_POVERTY','PCT_BACH_MORE','POP_DENSITY','Metro_Index',
                        'HH_SIZE','FAM_SIZE','MED_HOME','MED_RENT','PCT_UNIT1','PCT_WORKING',
                        'MEAN_COMMUTE','Pct_Water','Env_Index','Pct_ToPark_HalfMile','Pct_ToPark_OneMile']
                dif_record=compare_record[dif_cols]-selected_record[dif_cols]
                st.write(
                """
                <style>
                [data-testid="stMetricDelta"] svg {
                display: none;
                }
                </style>
                """,
                unsafe_allow_html=True,
                )
                col1,col2=st.columns(2)
                col1.subheader('Selected')
                col1.write(selected_zip+' '+selected_county)
                col2.subheader('Similar')
                col2.write(compare_zip+' '+compare_county)
                st.divider()
                col1,col2=st.columns(2)
                fig = px.pie(selected_record, values=[selected_record['PCT_UNDER_18'][0], selected_record['PCT_18_65'][0], selected_record['PCT_OVER_65'][0]],names=['< 18','18-65','> 65'])
                fig.update_layout(legend={'title': {'text': 'Age Distribution'}})
                col1.caption('Selected')
                col1.plotly_chart(fig,use_container_width=True)
                fig = px.pie(compare_record, values=[compare_record['PCT_UNDER_18'][0], compare_record['PCT_18_65'][0], compare_record['PCT_OVER_65'][0]],names=['< 18','18-65','> 65'])
                fig.update_layout(legend={'title': {'text': 'Age Distribution'}})
                col2.caption('Similar')
                col2.plotly_chart(fig,use_container_width=True)
                st.divider()
                col1,col2=st.columns(2)
                col1.metric('Median Age',selected_record['MED_AGE'][0].round(2))
                col2.metric('Median Age',compare_record['MED_AGE'][0].round(2),delta=dif_record['MED_AGE'][0].round(2))
                st.divider()
                col1,col2=st.columns(2)
                col1.metric('Median Household Income','${:,.0f}'.format(selected_record['MED_HH_INC'][0].round(2)))
                col2.metric('Median Household Income','${:,.0f}'.format(compare_record['MED_HH_INC'][0].round(2)),delta='${:,.0f}'.format(dif_record['MED_HH_INC'][0].round(2)))
                st.divider()
                col1,col2=st.columns(2)
                col1.metric('Percent in Poverty','{:.1%}'.format(selected_record['PCT_POVERTY'][0].round(2)/100))
                col2.metric('Percent in Poverty','{:.1%}'.format(compare_record['PCT_POVERTY'][0].round(2)/100),delta='{:.1%}'.format(dif_record['PCT_POVERTY'][0].round(2)/100))
                st.divider()
                col1,col2=st.columns(2)
                col1.metric('Percent with Bachelors Degree or More','{:.1%}'.format(selected_record['PCT_BACH_MORE'][0].round(2)/100))
                col2.metric('Percent with Bachelors Degree or More','{:.1%}'.format(compare_record['PCT_BACH_MORE'][0].round(2)/100),delta='{:.1%}'.format(dif_record['PCT_BACH_MORE'][0].round(2)/100))    
            with tab3:
                col1,col2=st.columns(2)
                col1.subheader('Selected')
                col1.write(selected_zip+' '+selected_county)
                col2.subheader('Similar')
                col2.write(compare_zip+' '+compare_county)
                st.divider()
                col1,col2=st.columns(2)
                fig = px.pie(selected_record, values=[selected_record['PCT_OWN'][0], selected_record['PCT_RENT'][0]],names=['Percent Own Home','Percent Renting'])
                fig.update_layout(legend={'title': {'text': 'Home Ownership'}})
                col1.plotly_chart(fig,use_container_width=True)
                fig=px.pie(selected_record, values=[compare_record['PCT_OWN'][0], compare_record['PCT_RENT'][0]],names=['Percent Own Home','Percent Renting'])
                fig.update_layout(legend={'title': {'text': 'Home Ownership'}})
                col2.plotly_chart(fig,use_container_width=True)
                st.divider()
                col1,col2=st.columns(2)
                col1.metric('Avg HH Size','{:,.1f}'.format(selected_record['HH_SIZE'][0].round(2)))
                col2.metric('Avg HH Size','{:,.1f}'.format(compare_record['HH_SIZE'][0].round(2)),delta='{:,.1f}'.format(dif_record['HH_SIZE'][0].round(2)))
                st.divider()
                col1,col2=st.columns(2)        
                col1.metric('Avg Family Size','{:,.1f}'.format(selected_record['FAM_SIZE'][0].round(2)))
                col2.metric('Avg Family Size','{:,.1f}'.format(compare_record['FAM_SIZE'][0].round(2)),delta='{:,.1f}'.format(dif_record['FAM_SIZE'][0].round(2)))
                st.divider()
                col1,col2=st.columns(2)        
                col1.metric('Median Home Price','${:,.0f}'.format(selected_record['MED_HOME'][0].round(2)))
                col2.metric('Median Home Price','${:,.0f}'.format(compare_record['MED_HOME'][0].round(2)),delta='${:,.0f}'.format(dif_record['MED_HOME'][0].round(2)))
                st.divider()
                col1,col2=st.columns(2)            
                col1.metric('Median Rent Price','${:,.0f}'.format(selected_record['MED_RENT'][0].round(2)))
                col2.metric('Median Rent Price','${:,.0f}'.format(compare_record['MED_RENT'][0].round(2)),delta='${:,.0f}'.format(dif_record['MED_RENT'][0].round(2)))
                st.divider()
                col1,col2=st.columns(2)            
                col1.metric('Pct Single Family Residential','{:.1%}'.format(selected_record['PCT_UNIT1'][0].round(2)/100))
                col2.metric('Pct Single Family Residential','{:.1%}'.format(compare_record['PCT_UNIT1'][0].round(2)/100),delta='{:.1%}'.format(dif_record['PCT_UNIT1'][0].round(2)/100))
            with tab4:
                col1,col2=st.columns(2)
                col1.subheader('Selected')
                col1.write(selected_zip+' '+selected_county)
                col2.subheader('Similar')
                col2.write(compare_zip+' '+compare_county)
                st.divider()
                col1,col2=st.columns(2)
                fig = px.pie(selected_record, values=[selected_record['PCT_SERVICE'][0], selected_record['PCT_BC'][0],selected_record['PCT_WC'][0]],names=['Percent Service','Percent Blue Collar','Percent White Collar'])
                fig.update_layout(legend={'title': {'text': 'Occupation Type'}})
                col1.plotly_chart(fig,use_container_width=True)
                fig = px.pie(compare_record, values=[compare_record['PCT_SERVICE'][0], compare_record['PCT_BC'][0],compare_record['PCT_WC'][0]],names=['Percent Service','Percent Blue Collar','Percent White Collar'])
                fig.update_layout(legend={'title': {'text': 'Occupation Type'}})
                col2.plotly_chart(fig,use_container_width=True)
                st.divider()
                col1,col2=st.columns(2)   
                col1.metric('Pct Working','{:.1%}'.format(selected_record['PCT_WORKING'][0].round(2)/100))
                col2.metric('Pct Working','{:.1%}'.format(compare_record['PCT_WORKING'][0].round(2)/100),delta='{:.1%}'.format(dif_record['PCT_WORKING'][0]/100))
                st.divider()
                col1,col2=st.columns(2)   
                col1.metric('Avg Commute Time',selected_record['MEAN_COMMUTE'][0].round(2))
                col2.metric('Avg Commute Time',compare_record['MEAN_COMMUTE'][0].round(2),delta='{:,.1f}'.format(dif_record['MEAN_COMMUTE'][0]))
            with tab5:
                col1,col2=st.columns(2)
                col1.subheader('Selected')
                col1.write(selected_zip+' '+selected_county)
                col2.subheader('Similar')
                col2.write(compare_zip+' '+compare_county)
                st.divider()
                col1,col2=st.columns(2)
                col1.write('Location Type')
                col1.write(selected_record['Metropolitan'][0])
                col2.write('Location Type')
                col2.write(compare_record['Metropolitan'][0])
                st.divider()
                col1,col2=st.columns(2)   
                col1.metric('Population Density','{:,.0f}'.format(selected_record['POP_DENSITY'][0].round(2)))
                col2.metric('Population Density','{:,.0f}'.format(compare_record['POP_DENSITY'][0].round(2)),delta='{:.0f}'.format(dif_record['POP_DENSITY'][0]))
                st.divider()
                col1,col2=st.columns(2)  
                col1.metric('Pct Area is Water','{:.2%}'.format(selected_record['Pct_Water'][0]))
                col2.metric('Pct Area is Water','{:.2%}'.format(compare_record['Pct_Water'][0]),delta='{:.2%}'.format(dif_record['Pct_Water'][0]))
                st.divider()
                col1,col2=st.columns(2)  
                col1.metric('Environmental Quality Index','{:.2f}'.format(selected_record['Env_Index'][0].round(2)))
                col2.metric('Environmental Quality Index','{:.2f}'.format(compare_record['Env_Index'][0].round(2)),delta='{:.2f}'.format(dif_record['Env_Index'][0]))
                st.divider()
                col1,col2=st.columns(2)  
                col1.metric('Pct within 0.5 mile to Park','{:.1%}'.format(selected_record['Pct_ToPark_HalfMile'][0].round(2)/100))
                col2.metric('Pct within 0.5 mile to Park','{:.1%}'.format(compare_record['Pct_ToPark_HalfMile'][0].round(2)/100),delta='{:.1%}'.format(dif_record['Pct_ToPark_HalfMile'][0]/100))
                st.divider()
                col1,col2=st.columns(2)  
                col1.metric('Pct within 1 mile to Park','{:.1%}'.format(selected_record['Pct_ToPark_OneMile'][0].round(2)/100))
                col2.metric('Pct within 1 mile to Park','{:.1%}'.format(compare_record['Pct_ToPark_OneMile'][0].round(2)/100),delta='{:.1%}'.format(dif_record['Pct_ToPark_OneMile'][0]/100))
                                 

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

