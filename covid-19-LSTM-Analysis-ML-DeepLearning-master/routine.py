
#import libs
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
import plotly as py
import plotly.express as px

pd.set_option('display.max_rows', 500)
c_df = pd.read_csv('./covid_19_data_2.csv')#R:\Covid-19 genesis analyser\covid-19-LSTM-Analysis-ML-DeepLearning-master/covid_19_data_2.csv
df = c_df.copy()
# print(df.head())

print(df.shape)

print(df.info())

#EDA on dataset
df.isnull().sum()

nan_states_df = df[df['Province/State'].isnull()]

print('nan_states_df shape is : '+ str(nan_states_df.shape))
print('nan_states_df has got : '+ str(nan_states_df['Country/Region'].nunique()) + ' unique Country/Region values')

nan_states_df = nan_states_df[['ObservationDate','Country/Region','Confirmed','Deaths','Recovered']]
nan_states_df.head()


states_df = df[df['Province/State'].notnull()]

print('states_df shape is : '+ str(states_df.shape))
print('states_df has got : '+ str(states_df['Province/State'].nunique()) + ' unique Province/State values')
 
states_df = states_df[['ObservationDate','Province/State','Country/Region','Confirmed','Deaths','Recovered']]
states_df.head()

concentrated_states_df= states_df.groupby(['ObservationDate','Country/Region'])[['Confirmed','Deaths','Recovered']].sum().reset_index()
concentrated_states_df.head()

full_countries_df = pd.concat([nan_states_df, concentrated_states_df], axis=0).reset_index()
full_countries_df.head()

lastest_full_countries_df = full_countries_df.groupby(['Country/Region'])[['ObservationDate','Confirmed','Deaths','Recovered']].max().reset_index()
lastest_full_countries_df.head()

china_df = states_df[states_df['Country/Region']=='Mainland China'] 
china_df.head()

lastest_china_df = china_df.groupby(['Province/State']).max().reset_index()
lastest_china_df.head()

print('Total countries affected by covid virus: ' + str(lastest_full_countries_df['Country/Region'].nunique()) + '\n' + 'That countries are : ' +'\n'+str(lastest_full_countries_df['Country/Region'].unique()) )

print('Worldwide Confirmed Cases: ',lastest_full_countries_df['Confirmed'].sum())
print('Worldwide Deaths: ',lastest_full_countries_df['Deaths'].sum())
print('Worldwide Recovered Cases: ',lastest_full_countries_df['Recovered'].sum())

lastest_full_countries_df.sort_values(by='Confirmed', ascending=False)

sorted_lastest_full_countries_df = lastest_full_countries_df.sort_values(by='Confirmed', ascending=False)
sorted_lastest_full_countries_df[:10]

other_countries_df = full_countries_df[~(full_countries_df['Country/Region']=='Mainland China')]
other_countries_df.head()

lastest_other_countries_df = other_countries_df.groupby('Country/Region')[['Confirmed','Deaths','Recovered']].max().reset_index()


sorted_lastest_other_countries_df = lastest_other_countries_df.sort_values(by='Confirmed', ascending=False)

lastest_other_countries_df.head()

f, ax = plt.subplots(figsize=(12, 30))

sns.set_color_codes("pastel")
sns.barplot(x="Confirmed", y="Country/Region", data=sorted_lastest_other_countries_df[:],
            label="Confirmed", color="b")

sns.set_color_codes("pastel")
sns.barplot(x="Recovered", y="Country/Region", data=sorted_lastest_other_countries_df[:],
            label="Recovered", color="g")

sns.set_color_codes("pastel")
sns.barplot(x="Deaths", y="Country/Region", data=sorted_lastest_other_countries_df[:],
            label="Deaths", color="k")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 5000), ylabel="",
       xlabel="Stats")
sns.despine(left=True, bottom=True)

fig = px.pie(sorted_lastest_other_countries_df, values = 'Confirmed',names='Country/Region', height=600)
fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))

fig.show()

sorted_lastest_full_countries_df[(sorted_lastest_full_countries_df['Confirmed'] == sorted_lastest_full_countries_df['Recovered'])]

sorted_lastest_full_countries_df[(sorted_lastest_full_countries_df['Confirmed'] == sorted_lastest_full_countries_df['Deaths'])]

sorted_lastest_full_countries_df[(sorted_lastest_full_countries_df['Recovered'] < sorted_lastest_full_countries_df['Deaths'])]



lastest_full_countries_df['Treatment'] = (lastest_full_countries_df['Confirmed']-(lastest_full_countries_df['Recovered']+lastest_full_countries_df['Deaths']))

Iran = lastest_full_countries_df[lastest_full_countries_df['Country/Region']=='Iran'][['Treatment','Recovered','Deaths']].iloc[0]
Spain = lastest_full_countries_df[lastest_full_countries_df['Country/Region']=='Spain'][['Treatment','Recovered','Deaths']].iloc[0]
SouthKorea = lastest_full_countries_df[lastest_full_countries_df['Country/Region']=='South Korea'][['Treatment','Recovered','Deaths']].iloc[0]
Italy = lastest_full_countries_df[lastest_full_countries_df['Country/Region']=='Italy'][['Treatment','Recovered','Deaths']].iloc[0]

fig, axes = plt.subplots(
                     ncols=2,
                     nrows=2,
                     figsize=(15, 15))

ax1, ax2, ax3, ax4 = axes.flatten()

colors = ['b','g','k']
ax1.pie(Italy
           , colors=colors
           , autopct='%1.1f%%' # adding percentagens
           , labels=['Treatment','Recovered','Deaths']
           , shadow=True
           , startangle=140)
ax1.set_title("Italy Cases Distribution")

ax2.pie(Iran
           , colors=colors
           , autopct='%1.1f%%' # adding percentagens
           , labels=['Treatment','Recovered','Deaths']
           , shadow=True
           , startangle=140)
ax2.set_title("Iran Cases Distribution")

ax3.pie(SouthKorea
        , colors=colors
        , autopct='%1.1f%%' # adding percentagens
        , labels=['Treatment','Recovered','Deaths']
        , shadow=True
        , startangle=140)
ax3.set_title("South Korea Cases Distribution")



ax4.pie(Spain
           , colors=colors
           , autopct='%1.1f%%' # adding percentagens
           , labels=['Treatment','Recovered','Deaths']
           , shadow=True
           , startangle=140)
ax4.set_title("Spain Cases Distribution")

fig.legend(['Treatment','Recovered','Deaths']
           , loc = "upper right"
           , frameon = True
           , fontsize = 15
           , ncol = 2 
           , fancybox = True
           , framealpha = 0.95
           , shadow = True
           , borderpad = 1)

# plt.show();



fig = px.choropleth(full_countries_df, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="Confirmed",
                    #color_continuous_scale='Rainbow',
                    hover_name="Country/Region", 
                    animation_frame="ObservationDate"
                   )

fig.update_layout(
    title_text = 'Spread of Coronavirus',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
# fig.show()

temp = full_countries_df.groupby('ObservationDate')['Country/Region'].nunique().reset_index()
temp.columns = ['ObservationDate','CountOfCountry']

fig = px.bar(temp, x='ObservationDate', y='CountOfCountry')
fig.update_layout(
    title_text = 'Number Of Countries With Cases',
    title_x = 0.5)
# fig.show()

line_data = full_countries_df.groupby('ObservationDate').sum().reset_index()

line_data = line_data.melt(id_vars='ObservationDate', 
                 value_vars=['Confirmed',
                             'Deaths',
                             'Recovered', 
                             ], 
                 var_name='Ratio', 
                 value_name='Value')

fig = px.line(line_data, x="ObservationDate", y="Value", color='Ratio', 
              title='Confirmed cases, Recovered cases, and Death Over Time')
# fig.show()

#lstm thing

import pandas as pd
import numpy as np

df_patient = pd.read_csv('./covid_19_data_2.csv')

daily_count = df_patient.groupby('ObservationDate').Confirmed.count()
daily_count = pd.DataFrame(daily_count)
print(daily_count)



data = daily_count.cumsum()
# print(data) 



dataset = data.iloc[14:]
# print(dataset)
dataset.columns = ['Confirmed']
print("len of the dataset::"+str(len(dataset)))
data = np.array(dataset).reshape(-1, 1)
train_data = dataset[:len(dataset)-5]#0..35
test_data = dataset[len(dataset)-5:]#35..40

# print(train_data)



# print(test_data)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)
scaled_train_data

# print(scaled_test_data)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
n_input =3
n_features =1
                             
generator = TimeseriesGenerator(scaled_train_data,scaled_train_data, length=n_input, batch_size=1)

lstm_model = Sequential()
lstm_model.add(LSTM(19, activation='relu', input_shape = (n_input, n_features)))
lstm_model.add(Dense(10))
lstm_model.add(Dense(5))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

lstm_model.fit_generator(generator, epochs=28)

# import matplotlib.pyplot as plt
# losses_lstm = lstm_model.history.history['loss']
# plt.figure(figsize = (12,4))
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.xticks(np.arange(0,21,1))
# plt.plot(range(len(losses_lstm)), losses_lstm)




lstm_predictions_scaled = []

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)


# As you know we scaled our data that’s why we have to inverse it to see true predictions.
# print(lstm_predictions_scaled)

lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled) 
# print(lstm_predictions)

test_data['LSTM_Predictions'] = lstm_predictions
# print(test_data)

# test_data.plot()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print('MAE of LSTM Model ',mean_absolute_error(test_data['Confirmed'], test_data['LSTM_Predictions']))

print('MSE of LSTM Model ',mean_squared_error(test_data['Confirmed'], test_data['LSTM_Predictions']))
