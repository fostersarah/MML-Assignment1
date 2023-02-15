import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pygal

#import data frames
df1 = pd.read_csv('articleInfo.csv')
df2 = pd.read_csv('authorInfo.csv')

#merge data based on article no.
df3 = pd.merge(df1, df2, on="Article No.")

#fill all NaN values with 0
df3 = df3.fillna(0)

#Plot 1 - yearly_publication: x-year, y-number of articles during that year
plot1 = df3.groupby(['Year'])['Year'].count().reset_index(name="Number of Articles Published")

year_list = plot1['Year'].values.tolist()
article_count = plot1['Number of Articles Published'].values.tolist()

plt.plot(year_list, article_count)
plt.title("yearly_publication")
plt.xlabel("Year")
plt.ylabel("Number of Articles Published")
plt.show()


#Plot 2 - yearly_citation: x-year, y-number of citations during that year
plot2 = df3.groupby(['Year'])['Citation'].sum().reset_index(name="Number of Citations")

year_list = plot2['Year'].values.tolist()
citation_count = plot2['Number of Citations'].values.tolist()

plt.plot(year_list, citation_count)
plt.title("yearly_citation")
plt.xlabel("Year")
plt.ylabel("Number of Citations")
plt.show()


#Plot 3 - num citations per country
plot3 = df3.groupby(['Country'])['Citation'].sum().reset_index(name="Number of Citations")
country_codes = pd.read_csv('country_codes.csv')

plot3 = pd.merge(plot3, country_codes, on="Country")

country_list = plot3['code'].values.tolist()
citation_count = plot3['Number of Citations'].values.tolist()

worldmap_chart = pygal.maps.world.World()
worldmap_chart.title = 'Citations per Country'

worldmap_chart.add('Citation Count', {
    'au' : 325,
    'ca' : 322,
    'cl' : 107,
    'cn' : 1290,
    'cy' : 2687,
    'cz' : 12,
    'fr' : 81,
    'de' : 303,
    'gr' : 20,
    'hk' : 16,
    'in' : 3,
    'ie' : 60,
    'it' : 254,
    'kg' : 41,
    'li' : 96,
    'my' : 12,
    'mx' : 3,
    'nz' : 284,
    'no' : 96,
    'pk' : 12,
    'sk' : 45,
    'za' : 34,
    'es' : 105,
    'ch' : 5,
    'ua' : 6,
    'ae' : 662,
    'gb' : 468,
    'us' : 1478,
    'ps' : 0,
    'ru' : 0,
    'kr' : 7,
})
worldmap_chart.render_in_browser()

#Plot 4: top 5 institutions with most published articles in this area

#group by institution and num articles
plot4 = df3.groupby(['Author Affiliation'])['Citation'].count().reset_index(name="Number of Articles")

plot4 = plot4.sort_values(by=['Number of Articles'])

plot4 = plot4.iloc[-5:]

print(plot4)
print("\n")

#Plot 5: top 5 researchers with most h-index

plot5 = df3.groupby(['Author Name'])['h-index'].sum().reset_index(name="sum h-index")

plot5 = plot5.sort_values(by=['sum h-index'])

plot5 = plot5.iloc[-5:]

print(plot5)