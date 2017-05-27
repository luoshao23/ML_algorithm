import pandas as pd
import plotly.plotly as py
import plotly

df2 = pd.read_csv('input.csv', header=0)
df2['promo_dep15'].astype(float)
color = pd.Series(['rgb(100,100,100)', 'rgb(38,17,235)', 'rgb(17,93,235)', 'rgb(17,235,220)',
                   'rgb(49,235,17)', 'rgb(188,235,17)', 'rgb(235,202,17)', 'rgb(235,115,17)', 'rgb(255,0,0)'])

# cities = []

# for i in range(9):

#     df_sub = df2[df2['cat_tot'] == i]
#     city = dict(
#         type='scattergeo',
#         locationmode='USA-states',
#         lon=df_sub['long'],
#         lat=df_sub['lat'],
#         text=df_sub['store_nbr'],
#         marker=dict(
#             size=(df_sub['cat_tot'] + 1) * 5,
#             color=color[df_sub['cat_tot']],
#             line=dict(width=0.5, color='rgb(40,40,40)'),
#             sizemode='area'
#         ),
#         name='Cluster %d' % i)
#     cities.append(city)

# layout = dict(
#     title='Sale cluster',
#     showlegend=True,
#     geo=dict(
#         scope='usa',
#         projection=dict(type='albers usa'),
#         showland=True,
#         landcolor='rgb(217, 217, 217)',
#         subunitwidth=1,
#         countrywidth=1,
#         subunitcolor="rgb(255, 255, 255)",
#         countrycolor="rgb(255, 255, 255)"
#     ),
# )

# fig = dict(data=cities, layout=layout)
# # py.plot( fig, validate=False, filename='store_cluster' )
# plotly.offline.plot(fig, validate=False, filename='store_cluster')

promo = []

for i in range(9):

    df_sub = df2[df2['cat_tot'] == i]
    city = dict(
        type='scattergeo',
        locationmode='USA-states',
        lon=df_sub['long'],
        lat=df_sub['lat'],
        text=df_sub['store_nbr'],
        marker=dict(
            size=(df_sub['promo_dep15'] *100) ,
            color=color[df_sub['cat_tot']],
            line=dict(width=0.5, color='rgb(40,40,40)'),
            sizemode='area'
        ),
        name='Cluster %d' % i)
    promo.append(city)

layout = dict(
    title='Promotion depth cluster',
    showlegend=True,
    geo=dict(
        scope='usa',
        projection=dict(type='albers usa'),
        showland=True,
        landcolor='rgb(217, 217, 217)',
        subunitwidth=1,
        countrywidth=1,
        subunitcolor="rgb(255, 255, 255)",
        countrycolor="rgb(255, 255, 255)"
    ),
)

fig = dict(data=promo, layout=layout)
# py.plot( fig, validate=False, filename='store_cluster' )
plotly.offline.plot(fig, validate=False, filename='store_cluster_promo')