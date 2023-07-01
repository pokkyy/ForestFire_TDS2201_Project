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

tab1, tab2, tab3 = st.tabs(["Database", "Exploratory Data Analysis", "Data Modelling"])

with tab1:
    # Intro -------------------------------------------------------------------------
    st.title("Forest Fires Database")
    
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
    st.header('Exploratory Data Analysis')

    st.subheader('Brief description of data')
    st.table(df.describe())
    
    tabEDA1, tabEDA2 = st.tabs(['Feature Understanding', 'Feature Relationship'])

    with tabEDA1:
        st.title('Feature understanding')
        sns.set_palette("viridis")
        sns.set_style('darkgrid')
        colours_list = ["viridis", "rocket", 'mako']

        st.header('Area burned per month')
        # fires per month
        fires_per_month = df[df['area'] > 0]['month'].value_counts()
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.barh(fires_per_month.index, fires_per_month.values)
        ax.set_xlabel('Number of fires')
        ax.set_ylabel('Month')
        ax.set_title('Number of fires per month')
        st.pyplot(fig)
        plt.close(fig)
        
        st.write('Exact count of fires per month')
        fireMonthsCount = {'Month': ['August', 'September', 'July', 'March', 'February', 'Dec', 'June', 'October', 'April', 'May'],
        'Number of Fires': [99, 97, 18, 18, 10, 9, 8, 5, 4, 1]}
        fireMonthsCount_df = pd.DataFrame(fireMonthsCount)
        st.table(fireMonthsCount_df)

        # log area burned
        st.subheader('Log area burned')
        area_burnt = df[df['area'] != 0].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(area_burnt['area'], log_scale=True, kde=True, ax=ax)
        ax.set_xlabel('Log area')
        ax.set_title('Log area burned')
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("Why log area?"):
            st.write("In the original dataset, the output \'area\' had undergone a ln(x+1) transformation function. For the purpose of this project, the exact transformation was applied.")
            st.write("By creating a histogram of log (area), we can see that the data is normally distributed.")

        # area burned months
        st.subheader("Area burned throughout the year")
        fig, ax = plt.subplots(figsize=(8, 15))
        sns.boxplot(data=area_burnt, x='month', y='area', ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Area burned')
        ax.set_title('Area burned throughout the year')
        st.pyplot(fig)
        plt.close(fig)
        
        # weather and fwi ----------------------------------------------------------------
        st.header("FWI indices and weather conditions for each month")
        # fwi ----------------------------------------------------------------------------
        fig, ax = plt.subplots(1, 4, figsize=(20, 8))
        fig.suptitle('FWI: Forest Fire Weather Index statistics')
        
        sns.barplot(x='month', y='FFMC', data=df, ax=ax[0])
        sns.barplot(x='month', y='DMC', data=df, ax=ax[1])
        sns.barplot(x='month', y='DC', data=df, ax=ax[2])
        sns.barplot(x='month', y='ISI', data=df, ax=ax[3])

        ax[0].set_xlabel("Month")
        ax[0].set_ylabel("Index")
        ax[0].set_title("FFMC Index")

        ax[1].set_xlabel("Month")
        ax[1].set_ylabel("DMC Index")
        ax[1].set_title("DMC Index")

        ax[2].set_xlabel("Month")
        ax[2].set_ylabel("Index")
        ax[2].set_title("DC Index")

        ax[3].set_xlabel("Month")
        ax[3].set_ylabel("Index")
        ax[3].set_title("ISI Index")
        st.pyplot(fig)
        plt.close(fig)

        # weather ------------------------------------------
        fig, ax = plt.subplots(1,4, figsize=(20,8))
        fig.suptitle('Enviromental conditions throughout the year')

        sns.boxplot(x='month', y='temp', data=df, ax=ax[0])
        sns.boxplot(x='month', y='RH', data=df, ax=ax[1])
        sns.boxplot(x='month', y='wind', data=df, ax=ax[2])
        sns.boxplot(x='month', y='rain', data=df, ax=ax[3])

        ax[0].set_xlabel("Month")
        ax[0].set_ylabel("degree Celcius")
        ax[0].set_title("Temperature in Celcius degrees")

        ax[1].set_xlabel("Month")
        ax[1].set_ylabel("RH")
        ax[1].set_title("Relative humidity")

        ax[2].set_xlabel("Month")
        ax[2].set_ylabel("Speed")
        ax[2].set_title("Wiind speed in km/h")

        ax[3].set_xlabel("Month")
        ax[3].set_ylabel("mm/m2")
        ax[3].set_title("Outside rain")
        st.pyplot(fig)
        plt.close(fig)
        
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
            pairplot = sns.pairplot(df_log, vars=variables, hue='month')
            st.pyplot(pairplot.fig)
            plt.close(pairplot.fig)  # closing the pairplot to avoid duplicate display
            
            df_corr = df_log[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area', 'logarea']].dropna().corr()
            st.write('Correlation summary')
            st.table(df_corr)
            
            st.write('Heatmap of correlation')
            heatmap = sns.heatmap(df_corr, annot=True)
            st.pyplot(heatmap.figure)
            plt.close(heatmap.figure)            
        
        with EDA2:
            st.header('Observing the relationship between forest fire counts within a month and the enviromental conditions and FWI index')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            fig.suptitle('Analysis of burned area, environmental conditions, and FWI averages by month')

            month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            monthly_avg = df.copy()

            monthly_avg['month'] = pd.Categorical(monthly_avg['month'], categories=month_order, ordered=True)
            monthly_avg = monthly_avg.sort_values('month')

            ax1.plot(month_order, monthly_avg.groupby('month')['area'].mean(), marker='o', label='Burned Area')
            ax1.plot(month_order, monthly_avg.groupby('month')['temp'].mean(), marker='o', label='Temperature')
            ax1.plot(month_order, monthly_avg.groupby('month')['RH'].mean(), marker='o', label='Relative Humidity')
            ax1.plot(month_order, monthly_avg.groupby('month')['wind'].mean(), marker='o', label='Wind')
            ax1.plot(month_order, monthly_avg.groupby('month')['rain'].mean(), marker='o', label='Rain')

            ax1.set_xlabel('Month')
            ax1.set_ylabel('Average Value')
            ax1.set_title('Average Burned Area and Environmental Conditions by Month')
            ax1.legend()
            ax1.set_xticklabels(month_order, rotation=45)

            ax2.plot(month_order, monthly_avg.groupby('month')['area'].mean(), marker='o', label='Burned Area')
            ax2.plot(month_order, monthly_avg.groupby('month')['FFMC'].mean(), marker='o', label='FFMC')
            ax2.plot(month_order, monthly_avg.groupby('month')['DMC'].mean(), marker='o', label='DMC')
            ax2.plot(month_order, monthly_avg.groupby('month')['DC'].mean(), marker='o', label='DC')
            ax2.plot(month_order, monthly_avg.groupby('month')['ISI'].mean(), marker='o', label='ISI')

            ax2.set_xlabel('Month')
            ax2.set_ylabel('Average Value')
            ax2.set_title('Average Burned Area and FWI Metrics by Month')
            ax2.legend()
            ax2.set_xticklabels(month_order, rotation=45)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.write('We can conclude that enviromental conditions and FWI metrics play a role in the occurrence and severity of forest fires, with specific months showing distinct patterns in relation to burned areas.')
            
        
        with EDA3:
            # FWI intensity ranking ---------------------------------------------------------------
            st.header('Grading forest fire intensity by the FWI')
            fwi = df_log.copy()

            fwi['FWI_score'] = fwi['FFMC'] + fwi['DMC'] + fwi['DC'] + fwi['ISI']
            fwi['FWI_rank'] = fwi.groupby('FWI_score')['area'].rank()

            fig, ax = plt.subplots(figsize=(8, 8))
            scatterplot = sns.scatterplot(x='FWI_score', y='logarea', hue='FWI_rank', data=fwi, ax=ax)

            scatterplot.set_xlabel('FWI score')
            scatterplot.set_ylabel('Area burned')
            scatterplot.set_title('Fire Intensity vs. Log Area Burned')

            st.pyplot(fig)
            plt.close(fig)

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
    st.title('Data Modelling')
    DM1, DM2, DM3, DM4 = st.tabs(["Fire prediction based on weather", "Fire prediction based on FWI", "Area burned prediction based on FWI", "Prediction of forest fires in a month"])
    
    # data prep
    features = ['month', 'FFMC', 'DMC', 'DC', 'ISI', 'FWI_score', 'FWI_rank', 'temp', 'RH', 'wind', 'rain', 'area', 'logarea']
    modelling = fwi[features].copy()
    
    encoder = LabelEncoder()
    fires =  modelling[['FFMC', 'DMC', 'DC', 'ISI',  'FWI_score', 'FWI_rank', 'temp', 'RH', 'wind', 'rain', 'area', 'logarea']].copy()
    fires['month_encoded'] = encoder.fit_transform(modelling['month'])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(fires)
    
    with DM1:
        st.header('Finding relation between environmental factors and the area burned, and predicting if a fire will start based on environmental factors')
        env_factors = ['temp', 'RH', 'wind', 'rain']
        X = fires[env_factors]
        y = fires['area']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        st.write('A linear regression model with 70% training and 30% testing was used. ',
                 'The X variables in this model were temperature, RH, wind, and rain. While our target Y variable is area.')
        with st.expander('Model results:'):
            st.write('Mean absolute error:', mean_absolute_error(y_test, y_pred))
            st.write('Coefficients (temperature, RH (relative humidity), wind, rain):', reg.coef_)
            st.write('Intercept:', reg.intercept_)
            st.write('Determination coefficient: ', reg.score(X_test, y_test))
        
        y_test_log = np.log(y_test)
        y_pred_log = np.log(y_pred)

        st.subheader('Model predictions vs actual results')
        fig = plt.figure()
        plt.scatter(y_test_log, y_pred_log)
        plt.xlabel('LogArea')
        plt.ylabel('Predicted LogArea')
        plt.title('Actual vs Predicted Burned LogArea by weather')
        st.pyplot(fig)
        plt.close(fig)
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
        fig = plt.figure()
        plt.plot(range(2, 15), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('Cluster count based off distortions')
        st.pyplot(fig)
        plt.close(fig)
        st.write('Here, we will use K=5.')
        
        k5 = KMeans(n_clusters = 5, random_state=1).fit(KMEnvScaled)
        KMEnv['K5'] = k5.labels_
        
        st.subheader('Cluster distribution analsysi')
        fig, ax = plt.subplots(1,2, figsize=(15,8))
        fig.suptitle('Analysis of cluster distribution')
        plt.figure(figsize=(8,8))
        sns.scatterplot(x=fwi['FWI_score'], y=fwi['logarea'], hue=KMEnv['K5'], ax=ax[0])
        sns.histplot(KMEnv['K5'], bins=5, ax=ax[1])

        ax[0].set_xlabel('FWI score')
        ax[0].set_ylabel('logarea')
        ax[0].set_title('KMeans of Weather conditions with FWI scores and LogArea burned')
        ax[0].legend()

        ax[1].set_xlabel('Cluster')
        ax[1].set_ylabel('Count')
        ax[1].set_title('Histogram of cluster counts')

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.write('We can make the following assumptions:')
        st.write('- Clusters 1-2 are of normal weather conditions','\n',
                 '- Clusters 3-4 are of moderate weather conditions','\n',
                 '- Cluster 5 is of intense weather conditions')
        st.write('Clusters of a more moderate weather condition tend to have a higher FWI score, and a higher area burned,',
                 'indicting that theses conditions have a higher probability of having a forest fire.','\n',
                 'More intense weather conditions can be found having a lower FWI score, and lower area burned.')
        
        
        st.subheader('Breakdown of each clusters')
        cluster_centers = k5.cluster_centers_

        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(16, 8))

        for i, center in enumerate(cluster_centers):
            cluster_data = KMEnv[KMEnv['K5'] == i]
            X = cluster_data[['temp', 'RH', 'wind', 'rain']]
            y = cluster_data['logarea']
            model = LinearRegression()
            model.fit(X, y)
            prediction = model.predict(X)
            
            axs[i].scatter(y, prediction)
            axs[i].set_xlabel('Actual Area')
            axs[i].set_ylabel('Predicted Area')
            axs[i].set_title(f'Cluster {i+1}')
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

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
        
        reg = LinearRegression().fit(X_train, y_train)

        y_pred_reg = reg.predict(X_test)
        
        st.write('A linear regression model was used with 70% training and 30% testing. ',
                 'Variables FFMC, DMC, DC, and ISI were set as our X variables while logarea was set as our target Y variable.')
        with st.expander('Model results:'):
            st.write('Mean absolute error:', mean_absolute_error(y_test, y_pred_reg))
            st.write('Coefficients (temperature, RH (relative humidity), wind, rain):', reg.coef_)
            st.write('Intercept:', reg.intercept_)
            st.write('Determination coefficient: ', reg.score(X_test, y_test))

        st.subheader('Model predictions vs actual results')
        plt.scatter(y_test, y_pred)
        plt.xlabel('LogArea')
        plt.ylabel('Predicted LogArea')
        plt.title('Actual vs Predicted Burned LogArea by FWI')
        st.pyplot(plt)
        plt.close()
        
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
            
        plt.plot(range(2, 15), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('Cluster count based off distortions')
        st.pyplot(plt)
        plt.close()
        
        st.write('We will be using K=6.')
        k6 = KMeans(n_clusters = 6, random_state=1).fit(KMFWIScaled)
        KMFWI['K6'] = k6.labels_
        
        st.subheader('Cluster distribution analysis')
        fig, ax = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle('Analysis of cluster distribution')

        sns.scatterplot(x=fwi['FWI_score'], y=fwi['logarea'], hue=KMFWI['K6'], ax=ax[0])
        sns.histplot(KMFWI['K6'], bins=6, ax=ax[1])

        ax[0].set_xlabel('FWI score')
        ax[0].set_ylabel('logarea')
        ax[0].set_title('KMeans of FWI with FWI scores and LogArea burned')
        ax[0].legend()

        ax[1].set_xlabel('Cluster')
        ax[1].set_ylabel('Count')
        ax[1].set_title('Histogram of cluster counts')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader('Breakdown of cluster distribution')
        cluster_centers = k6.cluster_centers_

        fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(16, 8))

        for i, center in enumerate(cluster_centers):
            cluster_data = KMFWI[KMFWI['K6'] == i]
            
            X = cluster_data[['FFMC', 'DMC', 'DC', 'ISI']]
            y = cluster_data['logarea']
            
            model = LinearRegression()
            model.fit(X, y)
            
            prediction = model.predict(X)
            
            axs[i].scatter(y, prediction)
            axs[i].set_xlabel('Actual Area')
            axs[i].set_ylabel('Predicted Area')
            axs[i].set_title(f'Cluster {i+1}')
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
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
        
        reg = LinearRegression().fit(X_train, y_train)
        
        st.write('A linear regression test was done with temp, RH, wind, and rain as the X variables and FWI score as our target Y variable.',
                 'The dataset was split wtih 80% training and 20% testing.')

        with st.expander('Model results'):
            st.write('Coefficients (temperature, RH (relative humidity), wind, rain):', reg.coef_)
            st.write('Intercept:', reg.intercept_)
            st.write('Determination coefficient: ', reg.score(X_test, y_test))
            y_pred = reg.predict(X_test)
            model_mae = mean_absolute_error(y_test, y_pred)
            st.write('Mean absolute error:', model_mae)
            
        st.subheader('Model predictions vs actual results')
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual FWI score')
        plt.ylabel('Predicted FWI score')
        plt.title('Actual vs Predicted FWI score based on weather conditions with Linear Regression')

        st.pyplot(plt)
        plt.close()

        
        st.header('Using random Forest model')
        randforest = RandomForestRegressor()
        randforest.fit(X_train, y_train)
        
        y_pred = randforest.predict(X_test)

        st.write('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual FWI score')
        plt.ylabel('Predicted FWI score')
        plt.title('Actual vs Predicted FWI score based on weather conditions with Random Forest')

        st.pyplot(plt)
        plt.close()

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
