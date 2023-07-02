import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import altair as alt

def userWeather():
    temp_input = st.number_input("Temperature", value=0.0)
    RH_input = st.number_input("Relative humidity", value=0.0)
    wind_input = st.number_input("Wind speed", value=0.0)
    rain_input = st.number_input("Rain", value=0.0)
    
    user_inputs = pd.DataFrame({
        'temp': [temp_input],
        'RH': [RH_input],
        'wind' :[wind_input],
        'rain': [rain_input]
    })
        
    return user_inputs

def userFWI():
    FFMC_input = st.number_input("FFMC", value=0.0)
    DMC_input = st.number_input("DMC", value=0.0)
    DC_input = st.number_input("DC", value=0.0)
    ISI_input = st.number_input("ISI", value=0.0)
            
    user_inputs = pd.DataFrame({
        'FFMC': [FFMC_input],
        'DMC': [DMC_input],
        'DC' :[DC_input],
        'ISI': [ISI_input]
    })
    
    return user_inputs

st.sidebar.write('**The Intensity of Forest Fires Throughout the Year**')
st.sidebar.write('TDS2101 Data Science Fundamentals Project')
st.sidebar.write('Anis Hazirah binti Mohamad Sabry')
st.sidebar.write('Student ID: 1211300373')

st.title("**The Intensity of Forest Fires Throughout the Year**")

with st.expander("**Project Objectives:**"):
    st.write("- Finding the relation between weather conditions and the area of the forest that was burned","\n",
    "- Finding the relationship between the FWI, the area burned, and the month of the year","\n",
    "- Understanding the relationship between weather conditions, the area burned and the month of the year","\n",
    "- Ranking the intensity of a forest fire based on the FWI and the area burned","\n",
    "- Analyzing the area of a forest burned and the intensity of a forest fire")
    
with st.expander("**Expected Output:**"):
    st.write("- Finding out which month of the year produces the most intense forest fires","\n",
    "- Showing the relation between natural occurrences, the FWI, and the intensity of a fire","\n",
    "- Forecasting when a forest fire would occur based on certain weather conditions","\n",
    "- Show prediction on weather conditions that would result in a high FWI","\n",
    "- Provide information on the amount of forest that is burned due to the intensity of the fire.")

tab1, tab2, tab3, tab4, tab5= st.tabs(["Database", "Exploratory Data Analysis", "Data Modelling", "Conclusions", "Make Predictions"])

with tab1:
    # Intro -------------------------------------------------------------------------
    st.title("🌳 Forest Fires Database")
    
    st.header('Original Database')
    df_original = pd.read_csv('forestfires.csv')
    st.dataframe(df_original)
    
    # Data cleanup ------------------------------------------------------------------
    df = df_original.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    df.dropna()
    df.reset_index(drop=True, inplace=True)

    st.header('Database of non-zero \'area\' and additional \'logarea\' attribute')
    df_log = df[df['area'] != 0].copy()
    df_log.reset_index(drop=True, inplace=True)
    df_log['logarea'] = np.log(df_log['area'])
    st.dataframe(df_log)

with tab2:
    # EDA ---------------------------------------------------------------------------
    st.header('🔎 Exploratory Data Analysis')

    st.subheader('Brief description of data')
    st.table(df.describe())
    
    tabEDA1, tabEDA2 = st.tabs(['Feature Understanding', 'Feature Relationship'])

    with tabEDA1:
        st.title('Feature understanding')
        sns.set_palette("viridis")
        sns.set_style('darkgrid')
        colours_list = ["viridis", "rocket", 'mako']
        alt.themes.enable('dark')

        st.header('Area burned per month')
        # fires per month -----------------------------------------------------------------
        fires_per_month = df[df['area'] > 0]['month'].value_counts().reset_index()
        fires_per_month.columns = ['Month', 'Number of Fires']
        chart = alt.Chart(fires_per_month).mark_bar().encode(
            x=alt.X('Number of Fires:Q'),
            y=alt.Y('Month:N', sort=None),
            tooltip=['Month', 'Number of Fires'])
        st.altair_chart(chart, use_container_width=True)
        
        # tooltip makes this redundant
        # st.write('Exact count of fires per month')
        # fireMonthsCount = {'Month': ['August', 'September', 'July', 'March', 'February', 'Dec', 'June', 'October', 'April', 'May'],
        # 'Number of Fires': [99, 97, 18, 18, 10, 9, 8, 5, 4, 1]}
        # fireMonthsCount_df = pd.DataFrame(fireMonthsCount)
        # st.table(fireMonthsCount_df)

        # log area burned
        st.subheader('Log area burned')
        area_burnt = df[df['area'] != 0].copy()
        chart = alt.Chart(df_log).mark_bar().encode(
            x=alt.X('logarea:Q', bin=alt.Bin(maxbins=30), title='Logarea'),
            y=alt.Y('count()', stack=None),
            tooltip='count()')
        st.altair_chart(chart, use_container_width=True)
        
        with st.expander("Why log area?"):
            st.write("In the original dataset, the output \'area\' had undergone a ln(x+1) transformation function. For the purpose of this project, the exact transformation was applied.")
            st.write("By creating a histogram of log (area), we can see that the data is normally distributed.")

        # area burned months
        st.subheader("Area burned throughout the year")
        # Create an Altair chart
        chart = alt.Chart(area_burnt).mark_boxplot().encode(
            x=alt.X('month:N', title='Month'),
            y=alt.Y('area:Q', title='Area burned'),
            tooltip=['month', 'area']
        ).properties(
            height=900,
            title='Area burned throughout the year'
        )
        st.altair_chart(chart, use_container_width=True)
        
        # weather and fwi ----------------------------------------------------------------
        st.header("FWI indices and weather conditions for each month")
        # fwi ----------------------------------------------------------------------------
        # FWI indices plot
        fwi_indices = ['FFMC', 'DMC', 'DC', 'ISI']
        fwi_charts = []

        for index in fwi_indices:
            chart = alt.Chart(df).mark_bar().encode(
                x='month',
                y=alt.Y(index, title='Index'),
                tooltip=index
            ).properties(
                width=200,
                title=f'{index} Index'
            )
            fwi_charts.append(chart)

        # Environmental conditions plot
        conditions = ['temp', 'RH', 'wind', 'rain']
        condition_charts = []

        for condition in conditions:
            chart = alt.Chart(df).mark_boxplot().encode(
                x='month',
                y=alt.Y(condition, title=condition),
                tooltip=[condition]
            ).properties(
                width=200,
                title=condition.capitalize()
            )
            condition_charts.append(chart)

        fwi_combined_chart = alt.concat(*fwi_charts, columns=2)
        condition_combined_chart = alt.concat(*condition_charts, columns=2)

        st.subheader('FWI: Forest Fire Weather Index statistics')
        st.altair_chart(fwi_combined_chart, use_container_width=True)
        st.subheader('Environmental conditions throughout the year')
        st.altair_chart(condition_combined_chart, use_container_width=True)
        
        st.write('It is worth noting that seasonal changes affect the environmental conditions and the FWI indices, potentially leading to more outbreaks of forest fires.')

    with tabEDA2:
        st.title('Feature relationship')
        
        st.subheader('Statistical summary of non-zero \'area\' and additional \'logarea\' database')
        st.table(df_log.describe())
        
        EDA1, EDA2, EDA3 = st.tabs(['Correlation Metric', 'Relation between the variables', 'Grading Fire Intensity'])

        with EDA1:
            st.header('Analysing the relationship between the FWI, environmental factors, and the area burned')

            st.subheader('Correlation analysis')
            variables = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area', 'logarea']
            df_corr = df_log[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area', 'logarea']].dropna().corr()
            
            pairplot = sns.pairplot(df_log, vars=variables, hue='month')
            st.pyplot(pairplot.fig)
            plt.close(pairplot.fig)  # closing the pairplot to avoid duplicate display
            
            # Correlation Summary
            st.subheader('Correlation summary')
            st.dataframe(df_corr)

            # Heatmap of Correlation
            heatmap = alt.Chart(df_corr.reset_index().melt('index')).mark_rect().encode(
                x='index:O',
                y='variable:O',
                color='value:Q'
            ).properties(
                width=500,
                height=400,
                title='Heatmap of Correlation'
            )
            st.altair_chart(heatmap, use_container_width=True)
                    
        
        with EDA2:
            st.header('Observing the relationship between forest fire counts within a month and the enviromental conditions and FWI index')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            fig.suptitle('Analysis of burned area, environmental conditions, and FWI averages by month')

            month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            monthly_avg = df.copy()

            monthly_avg['month'] = pd.Categorical(monthly_avg['month'], categories=month_order, ordered=True)
            monthly_avg = monthly_avg.sort_values('month')

            # Melt the dataframe to long format
            melted_df = monthly_avg.melt('month', var_name='Variable', value_name='Mean Value')

            # Line chart for Average values by Month
            line_chart = alt.Chart(melted_df).mark_line(point=True).encode(
                x=alt.X('month:N', title='Month'),
                y=alt.Y('mean(Mean Value):Q', title='Mean Value'),
                color=alt.Color('Variable:N', title='Variable'),
                tooltip=['month:N', 'Variable:N', alt.Tooltip('mean(Mean Value):Q')]
            ).properties(
                height=600,
                title='Average Values by Month'
            )

            st.altair_chart(line_chart, use_container_width=True)
            st.write('We can conclude that enviromental conditions and FWI metrics play a role in the occurrence and severity of forest fires, ',
                     'with specific months showing distinct patterns in relation to burned areas.')
        
        with EDA3:
            # FWI intensity ranking ---------------------------------------------------------------
            st.header('Grading forest fire intensity by the FWI')
            fwi = df_log.copy()

            fwi['FWI_score'] = fwi['FFMC'] + fwi['DMC'] + fwi['DC'] + fwi['ISI']
            fwi['FWI_rank'] = fwi.groupby('FWI_score')['area'].rank()

            scatterplot = alt.Chart(fwi).mark_circle().encode(
                x='FWI_score',
                y='logarea',
                color='FWI_rank:N'
            ).properties(
                width=600,
                height=800
            ).interactive()

            st.altair_chart(scatterplot, use_container_width=True)

            st.write("Analysis of fire intensity, their FWI scoring, and the area burned")
            fwi_stats = fwi.groupby('FWI_rank').agg({'FWI_score': ['min', 'max', 'mean'], 'area': ['min', 'max', 'mean', 'count']})

            table_data = []
            for rank, stats in fwi_stats.iterrows():
                fwi_min = stats[('FWI_score', 'min')]
                fwi_max = stats[('FWI_score', 'max')]
                fwi_mean = stats[('FWI_score', 'mean')]
                area_min = stats[('area', 'min')]
                area_max = stats[('area', 'max')]
                area_mean = stats[('area', 'mean')]
                fwi_count = stats[('area', 'count')]
                
                table_data.append([rank, fwi_min, fwi_max, fwi_mean, area_min, area_max, area_mean, fwi_count])

            column_names = ['Rank', 'Min FWI', 'Max FWI', 'Mean FWI', 'Min Area', 'Max Area', 'Mean Area', 'Fire Counts']
            table_df = pd.DataFrame(table_data, columns=column_names)

            st.table(table_df)
                
            with st.expander('Conclusion:'):
                st.write('Higher FWI scoring do not necesarily attribute to a high intensity fire.')
                st.write('Most scores of high intensity fires are between 800 and 1000.')
                
            st.subheader('Maximum intensity ranking recorded for each month')
            max_rank = fwi.groupby('month')['FWI_rank'].max().reset_index()
            st.dataframe(max_rank)


with tab3:
    st.title('🧪 Data Modelling')
    DM1, DM2, DM3, DM4 = st.tabs(["Fire prediction based on weather", "Fire prediction based on FWI", "Weather and FWI score", "Prediction of forest fires in a month"])
    
    # data prep
    features = ['month', 'FFMC', 'DMC', 'DC', 'ISI', 'FWI_score', 'FWI_rank', 'temp', 'RH', 'wind', 'rain', 'area', 'logarea']
    modelling = fwi[features].copy()
    
    encoder = LabelEncoder()
    fires =  modelling[['FFMC', 'DMC', 'DC', 'ISI',  'FWI_score', 'FWI_rank', 'temp', 'RH', 'wind', 'rain', 'area', 'logarea']].copy()
    fires['month_encoded'] = encoder.fit_transform(modelling['month'])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(fires)
    
    with DM1:
        st.header('Predicting if a fire will start based on environmental factors')
        env_factors = ['temp', 'RH', 'wind', 'rain']
        X = fires[env_factors]
        y = fires['area']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        DM1reg = LinearRegression().fit(X_train, y_train)
        y_pred = DM1reg.predict(X_test)

        st.write('A linear regression model with 70% training and 30% testing was used. ',
                 'The X variables in this model were temperature, RH, wind, and rain. While our target Y variable is area.')
        with st.expander('Model results:'):
            st.write('Mean absolute error:', mean_absolute_error(y_test, y_pred))
            st.write('Coefficients (temperature, RH (reslative humidity), wind, rain):', DM1reg.coef_)
            st.write('Intercept:', DM1reg.intercept_)
            st.write('Determination coefficient: ', DM1reg.score(X_test, y_test))
        
        y_test_log = np.log(y_test)
        y_pred_log = np.log(y_pred)

        st.subheader('Model predictions vs actual results')
        
        scatter_data = pd.DataFrame({'y_test_log': y_test_log, 'y_pred_log': y_pred_log})
        scatterplot = alt.Chart(scatter_data).mark_circle().encode(
            x='y_test_log',
            y='y_pred_log'
        ).properties(
            width=600,
            height=600,
            title='Actual vs Predicted Burned LogArea by weather'
        ).interactive()
        st.altair_chart(scatterplot, use_container_width=True)
        st.write('Weather may not hold much of an influence on forest fires')
        
        st.header('Using K-Means')
        KMEnv = fires[['temp', 'RH', 'wind', 'rain', 'logarea']]
        scaler = StandardScaler()
        KMEnvScaled = scaler.fit_transform(KMEnv)
        
        # elbow
        st.subheader('Finding ideal cluster count')
        distortions = []
        for i in range(2, 15):
            km = KMeans(
                n_clusters=i, 
                n_init="auto",
                max_iter=300,
                tol=1e-04, random_state=0
            )
            km.fit(KMEnvScaled)
            distortions.append(km.inertia_)

        # plot
        line_data = pd.DataFrame({'Number of clusters': range(2, 15), 'Distortion': distortions})
        lineplot = alt.Chart(line_data).mark_line().encode(
            x='Number of clusters',
            y='Distortion'
        ).properties(
            width=600,
            height=400,
            title='Cluster count based off distortions'
        )
        # Add markers to the line plot
        markerplot = alt.Chart(line_data).mark_point().encode(
            x='Number of clusters',
            y='Distortion'
        )
        combined_plot = lineplot + markerplot
        st.altair_chart(combined_plot, use_container_width=True)
        st.write('Here, we will use K=5.')
        
        k5 = KMeans(n_clusters = 5, random_state=1).fit(KMEnvScaled)
        KMEnv['K5'] = k5.labels_
        
        st.subheader('Cluster distribution analsysi')
        scatter_data = pd.DataFrame({'FWI_score': fwi['FWI_score'], 'logarea': fwi['logarea'], 'K5': KMEnv['K5']})

        # Scatter plot
        scatterplot = alt.Chart(scatter_data).mark_circle(size=50).encode(
            x='FWI_score',
            y='logarea',
            color='K5:N',
            tooltip=['FWI_score', 'logarea', 'K5']
        ).properties(
            width=500,
            height=400,
            title='KMeans of Weather conditions with FWI scores and LogArea burned'
        ).interactive()

        # Histogram
        histogram = alt.Chart(KMEnv).mark_bar().encode(
            x=alt.X('K5:O', title='Cluster'),
            y=alt.Y('count()', title='Count')
        ).properties(
            width=300,
            height=400,
            title='Histogram of cluster counts'
        )

        combined_plot = alt.hconcat(scatterplot, histogram)
        st.altair_chart(combined_plot, use_container_width=True)

        st.write('We can make the following assumptions:')
        st.write('- Clusters 1-2 are of normal weather conditions','\n',
                 '- Clusters 3-4 are of moderate weather conditions','\n',
                 '- Cluster 5 is of intense weather conditions')
        st.write('Clusters of a more moderate weather condition tend to have a higher FWI score, and a higher area burned,',
                 'indicting that theses conditions have a higher probability of having a forest fire.','\n',
                 'More intense weather conditions can be found having a lower FWI score, and lower area burned.')
        
        
        st.subheader('Breakdown of each clusters')
        cluster_centers = k5.cluster_centers_
        scatter_data = []

        for i, center in enumerate(cluster_centers):
            cluster_data = KMEnv[KMEnv['K5'] == i]
            X = cluster_data[['temp', 'RH', 'wind', 'rain']]
            y = cluster_data['logarea']
            model = LinearRegression()
            model.fit(X, y)
            prediction = model.predict(X)
            
            cluster_scatter = pd.DataFrame({'Cluster': [f'Cluster {i+1}'] * len(y),
                                            'Actual Area': y,
                                            'Predicted Area': prediction})
            scatter_data.append(cluster_scatter)

        scatterplot = (
            alt.Chart(pd.concat(scatter_data))
            .mark_circle(size=50)
            .encode(
                x='Actual Area',
                y='Predicted Area',
                tooltip=['Cluster', 'Actual Area', 'Predicted Area'],
                color=alt.Color('Cluster:N', scale=alt.Scale(scheme='category10'))
            )
            .properties(
                width=200,
                height=200,
                title='Actual Area vs Predicted Area by Cluster'
            )
            .facet(
                facet='Cluster:N',
                columns=5
            ).interactive()
        )

        st.altair_chart(scatterplot, use_container_width=True)

        center_data = {'Cluster': [], 'temp': [], 'RH': [], 'wind': [], 'rain': [], 'logarea': []}
        for i, center in enumerate(cluster_centers):
            center_data['Cluster'].append(f'Cluster {i+1}')
            center_data['temp'].append(center[0])
            center_data['RH'].append(center[1])
            center_data['wind'].append(center[2])
            center_data['rain'].append(center[3])
            center_data['logarea'].append(center[4])
        center_df = pd.DataFrame(center_data)
        st.write('Cluster Centers:')
        st.table(center_df)
        
        st.write('Based on the combination of KMeans and linear regression,',
                 'we can conclude forest fires with a large burned area tends to happen within weather conditions of clusters 1, 2, and 3')
        
    with DM2:
        st.header('Predicting area burned based on FWI indices')
        FWI_index = ['FFMC', 'DMC', 'DC', 'ISI']
        X = fires[FWI_index]
        y = fires['logarea']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        
        DM2reg = LinearRegression().fit(X_train, y_train)

        y_pred_reg = DM2reg.predict(X_test)
        
        st.write('A linear regression model was used with 70% training and 30% testing. ',
                 'Variables FFMC, DMC, DC, and ISI were set as our X variables while logarea was set as our target Y variable.')
        with st.expander('Model results:'):
            st.write('Mean absolute error:', mean_absolute_error(y_test, y_pred_reg))
            st.write('Coefficients (temperature, RH (relative humidity), wind, rain):', DM2reg.coef_)
            st.write('Intercept:', DM2reg.intercept_)
            st.write('Determination coefficient: ', DM2reg.score(X_test, y_test))

        st.subheader('Model predictions vs actual results')
        scatter_data = pd.DataFrame({'Actual LogArea': y_test,
                                    'Predicted LogArea': y_pred})

        scatterplot = alt.Chart(scatter_data).mark_circle().encode(
            x='Actual LogArea',
            y='Predicted LogArea',
            tooltip=['Actual LogArea', 'Predicted LogArea']
        ).properties(
            title='Actual vs Predicted Burned LogArea by FWI'
        ).interactive()
        st.altair_chart(scatterplot, use_container_width=True)
        
        st.header('Using K-Means')
        KMFWI = fires[['FFMC', 'DMC', 'DC', 'ISI', 'logarea']]
        scaler = StandardScaler()
        KMFWIScaled = scaler.fit_transform(KMFWI)

        st.subheader('Finding ideal cluster count')
        distortions = []
        for i in range(2, 15):
            km = KMeans(
                n_clusters=i, 
                n_init="auto",
                max_iter=300,
                tol=1e-04, random_state=0
            )
            km.fit(KMFWIScaled)
            distortions.append(km.inertia_)
            
        line_data = pd.DataFrame({'Number of clusters': range(2, 15), 'Distortion': distortions})
        lineplot = alt.Chart(line_data).mark_line().encode(
            x='Number of clusters',
            y='Distortion'
        ).properties(
            width=600,
            height=400,
            title='Cluster count based off distortions'
        )
        # Add markers to the line plot
        markerplot = alt.Chart(line_data).mark_point().encode(
            x='Number of clusters',
            y='Distortion'
        )
        combined_plot = lineplot + markerplot
        st.altair_chart(combined_plot, use_container_width=True)
        st.write('We will be using K=6.')
        
        k6 = KMeans(n_clusters = 6, random_state=1).fit(KMFWIScaled)
        KMFWI['K6'] = k6.labels_
        
        st.subheader('Cluster distribution analysis')
        scatter_data = pd.DataFrame({'FWI_score': fwi['FWI_score'],
                                    'LogArea': fwi['logarea'],
                                    'Cluster': KMFWI['K6']})
        histogram_data = pd.DataFrame({'Cluster': KMFWI['K6']})

        scatterplot = alt.Chart(scatter_data).mark_circle().encode(
            x='FWI_score',
            y='LogArea',
            color=alt.Color('Cluster:N', legend=alt.Legend(title='Cluster')),
            tooltip=['FWI_score', 'LogArea', 'Cluster']
        ).properties(
            title='KMeans of FWI with FWI scores and LogArea burned'
        ).interactive()

        # Histogram
        histogram = alt.Chart(histogram_data).mark_bar().encode(
            x=alt.X('Cluster:O', title='Cluster'),
            y=alt.Y('count()', title='Count')
        ).properties(
            title='Histogram of cluster counts'
        )

        chart = alt.hconcat(scatterplot, histogram)
        st.altair_chart(chart, use_container_width=True)

        st.subheader('Breakdown of cluster distribution')
        cluster_centers = k6.cluster_centers_
        breakdown_data = pd.DataFrame()

        for i, center in enumerate(cluster_centers):
            cluster_data = KMFWI[KMFWI['K6'] == i]
            
            X = cluster_data[['FFMC', 'DMC', 'DC', 'ISI']]
            y = cluster_data['logarea']
            
            model = LinearRegression()
            model.fit(X, y)
            
            prediction = model.predict(X)
            
            cluster_breakdown = pd.DataFrame({'Actual Area': y, 'Predicted Area': prediction, 'Cluster': f'Cluster {i+1}'})
            breakdown_data = pd.concat([breakdown_data, cluster_breakdown], ignore_index=True)

        # Scatter plot
        scatterplot = alt.Chart(breakdown_data).mark_circle().encode(
            x='Actual Area',
            y='Predicted Area',
            color=alt.Color('Cluster:N', legend=alt.Legend(title='Cluster')),
            tooltip=['Actual Area', 'Predicted Area', 'Cluster']
        ).properties(
            width=200,
            height=200,
            title='Breakdown of cluster distribution'
        ).interactive()

        chart = scatterplot.facet(
            column='Cluster',
            columns=6
        ).resolve_scale(y='shared')

        st.altair_chart(chart, use_container_width=True)

        center_columns = ['FFMC', 'DMC', 'DC', 'ISI', 'logarea']
        center_data = []

        for i, center in enumerate(cluster_centers):
            center_data.append(list(center))

        st.write("Cluster Centers")
        st.table(pd.DataFrame(center_data, columns=center_columns))
        
        st.write('To conclude, clusters with higher FWI indices tend to be associated with larger areas burned. ',
                 'Cluster 2 stands out with unusually low center values compared to other clusters. ',
                 'Negative ISI values along with positive values in FFMC, DMC, and DC indicate a potential for larger burned areas, as observed in clusters 5 and 6.')

    with DM3:
        st.header('Does the weather influence the FWI score?')
        env_factors = ['temp', 'RH', 'wind', 'rain']
        X = fires[env_factors]
        y = fires['FWI_score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        DM3reg = LinearRegression().fit(X_train, y_train)
        
        st.write('A linear regression test was done with temp, RH, wind, and rain as the X variables and FWI score as our target Y variable.',
                 'The dataset was split wtih 80% training and 20% testing.')

        with st.expander('Model results'):
            st.write('Coefficients (temperature, RH (relative humidity), wind, rain):', DM3reg.coef_)
            st.write('Intercept:', DM3reg.intercept_)
            st.write('Determination coefficient: ', DM3reg.score(X_test, y_test))
            y_pred = DM3reg.predict(X_test)
            model_mae = mean_absolute_error(y_test, y_pred)
            st.write('Mean absolute error:', model_mae)
            
        st.subheader('Model predictions vs actual results')
        scatter_data = pd.DataFrame({'Actual FWI score': y_test, 'Predicted FWI score': y_pred})
        scatterplot = alt.Chart(scatter_data).mark_circle().encode(
            x='Actual FWI score',
            y='Predicted FWI score',
            tooltip=['Actual FWI score', 'Predicted FWI score']
        ).properties(
            height=500,
            title='Actual vs Predicted FWI score based on weather conditions with Linear Regression'
        ).interactive()
        st.altair_chart(scatterplot, use_container_width=True)

        
        st.header('Using random Forest model')
        randforest = RandomForestRegressor()
        randforest.fit(X_train, y_train)
        
        y_pred = randforest.predict(X_test)

        st.write('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

        scatter_data = pd.DataFrame({'Actual FWI score': y_test, 'Predicted FWI score': y_pred})
        scatterplot = alt.Chart(scatter_data).mark_circle().encode(
            x='Actual FWI score',
            y='Predicted FWI score',
            tooltip=['Actual FWI score', 'Predicted FWI score']
        ).properties(
            height=500,
            title='Actual vs Predicted FWI score based on weather conditions with Random Forest'
        ).interactive()
        st.altair_chart(scatterplot, use_container_width=True)

    with DM4:
        st.header('Fire predictions by the month')
        st.write('Logistic regression and gradient boosting classifier was used to predict which month would produce a forest fire. ',
                 'Both share the same X variables of FFMC, DMC, DC, ISI, FWI_score, FWI_rank, temperature, wind, rain, and area. ',
                 'They share the same Y variables of month, with month being encoded for the gradient boosting classifier. ')
        st.write('For both models, the dataset was split by 70% training and 30% testing.')
        
        st.subheader('Logistic Regression Model')
        features = ['FFMC', 'DMC', 'DC', 'ISI',  'FWI_score', 'FWI_rank', 'temp', 'RH', 'wind', 'rain', 'area']
        X = modelling[features].copy()
        y = modelling['month'].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)

        y_prob = logreg.predict_proba(X_test)
        max_prob_indices = y_prob.argmax(axis=1)
        predicted_months = logreg.classes_[max_prob_indices]
        predicted_months_counts = pd.Series(predicted_months).value_counts()

        st.write('Accuracy score:', logreg.score(X_test, y_test))
        st.write('Predicted month counts:')
        st.table(predicted_months_counts)

        st.subheader('Gradient Boosting Classification Model')
        gb_classifier = GradientBoostingClassifier()

        features = ['FFMC', 'DMC', 'DC', 'ISI', 'FWI_score', 'FWI_rank', 'temp', 'RH', 'wind', 'rain', 'area']
        X = fires[features].copy()
        y = fires['month_encoded'].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        gb_classifier.fit(X_train, y_train)

        y_pred = gb_classifier.predict(X_test)

        y_prob = gb_classifier.predict_proba(X_test)
        most_likely_month_indices = np.argmax(y_prob, axis=1)
        month_labels = encoder.classes_
        predicted_months = month_labels[most_likely_month_indices]
        predicted_month_counts = pd.Series(predicted_months).value_counts()

        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Predicted month counts:')
        st.table(predicted_month_counts)
        
        st.write('Based on the results, we observe that both August and September are likely candidates for months with a high occurrence of forest fires. ',
                 'This is not surprising considering the extreme weather conditions and the FWI scoring during those months.')
        st.write('In both of the models, each of them predicted a month that was not present in the other prediction. The linear regression predicted the month of October, which does not appear in the gradient boosting classification. ',
                 'While the gradient boosting classification predicted the month of June. ',
                 'Neither of these predictions are entirely false as both of these months have had an instance of a forest fire occurring.')

with tab4:
    st.write('Through analysing the forest fires dataset, we find that the months of August and September are candidates with high forest fire occurrences. ',
             'The months of October, April, and May on the other hand, have the lowest fire occurrence count. Higher scores of FWI were shown in the months of August, September, June, and July.')
    st.write('Correlation between each variable and burned area indicates that most of the factors, apart from relative humidity and rain, play a role in influencing the chances of a forest fire occurring. ',
             'In trying to predict the intensity of a forest fire, all the variables must be taken into account as relying on FWI scores alone shows us that not all high FWI scores translates into fires of high intensity.')
    st.write('Various models including linear regression, random forest, and K-Means clustering were used as an approach to finding the relationship between the variables and area burned, ',
             'along with predicting the likelihood of fires happening and the extent of the damage that would occur. ',
             'Our findings suggest that weather conditions are not a reliable predictor in trying to predict a forest fire. ',
             'Through K-Means, we discover that most fires tend to take place in moderate weather conditions, ',
             'as intense weather conditions of rain and RH dampen the chances of a forest fire becoming more intense.')
    st.write('A similar procedure was done with FWI indices in place. From this, we uncovered a more linear relationship between the FWI indices and the log area burned. ',
             'The results of the clustering from K-Means is also similar to the rankings of fire intensity, indicating that they have an impact on the level of intensity a fire is.')
    st.write('As FWI indices are based on weather conditions, we can draw a conclusion that periods of more rain and RH may indicate lesser FWI scoring. ',
             'Thus, reducing the intensity of possible forest fires that might occur. ',
             'It is advised that additional support and observation for forest fire prevention is required during periods of hotter temperatures and the months following them.')
    st.write('Although seasons were not a factor with great emphasis in this project, a conclusion can still be drawn based on the months most often associated with those seasons. ',
             'From this study, we find that the seasons of summer and autumn have a higher potential than other seasons to attract forest fires.')
    st.write('To conclude, our research finds that both weather conditions and FWI indices play a significant role in the count of fire occurrences and the intensity of a forest fire. ',
             'However, neither can be used as a standalone measure, as both are required in order to make a more accurate judgement and prediction.')


with tab5:
    st.title('🔥 Predict a forest fire! 🔥')
    
    col1, col2 = st.columns(2)
    with col1:
        user_inputs = userWeather()
        
        if st.button("Predict by weather"):
            predictions = DM1reg.predict(user_inputs)

            if predictions[0] > 0:
                st.write("There is a high likelihood of fire occurrence.")
                st.write('Potential area burned: ', predictions)
            else:
                st.write("There is a low likelihood of fire occurrence.")

    with col2:
        user_inputs = userFWI()
            
        if st.button("Predict by FWI indices"):    
            predictions = DM2reg.predict(user_inputs)

            if predictions[0] > 0:
                st.write("There is a high likelihood of fire occurrence.")
                st.write('Potential area burned: ', np.exp(predictions))
            else:
                st.write("There is a low likelihood of fire occurrence.")