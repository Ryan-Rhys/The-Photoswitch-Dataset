# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2021
# Author: Ryan-Rhys Griffiths
"""
Script for plotting marginal box plots.
"""

import plotly.graph_objects as go

if __name__ == '__main__':

    fig = go.Figure()
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")

    # Defining x axis

    # x = ['$\mathrm{\huge{E-Isomer}} \: \huge{\pi - \pi^*}$', '$\mathrm{\huge{E-Isomer}} \: \huge{\pi - \pi^*}$',
    #      '$\mathrm{\huge{E-Isomer}} \: \huge{\pi - \pi^*}$', '$\mathrm{\huge{E-Isomer}} \: \huge{\pi - \pi^*}$',
    #      '$\mathrm{\huge{E-Isomer}} \: \huge{n - \pi^*}$', '$\mathrm{\huge{E-Isomer}} \: \huge{n - \pi^*}$',
    #      '$\mathrm{\huge{E-Isomer}} \: \huge{n - \pi^*}$', '$\mathrm{\huge{E-Isomer}} \: \huge{n - \pi^*}$',
    #      '$\mathrm{\huge{Z-Isomer}} \: \huge{\pi - \pi^*}$', '$\mathrm{\huge{Z-Isomer}} \: \huge{\pi - \pi^*}$',
    #      '$\mathrm{\huge{Z-Isomer}} \: \huge{\pi - \pi^*}$', '$\mathrm{\huge{Z-Isomer}} \: \huge{\pi - \pi^*}$',
    #      '$\mathrm{\huge{Z-Isomer}} \: \huge{n - \pi^*}$', '$\mathrm{\huge{Z-Isomer}} \: \huge{n - \pi^*}$',
    #      '$\mathrm{\huge{Z-Isomer}} \: \huge{n - \pi^*}$', '$\mathrm{\huge{Z-Isomer}} \: \huge{n - \pi^*}$']

    x = ['$\mathrm{\Huge{E}} \:\:\: \mathrm{\Huge{\pi - \pi^*}}$', '$\mathrm{\Huge{E}} \:\:\: \Huge{\pi - \pi^*}$',
         '$\mathrm{\Huge{E}} \:\:\: \Huge{\pi - \pi^*}$', '$\mathrm{\Huge{E}} \:\:\: \Huge{\pi - \pi^*}$',
         '$\mathrm{\Huge{E}} \:\:\: \Huge{n - \pi^*}$', '$\mathrm{\Huge{E}} \:\:\: \Huge{n - \pi^*}$',
         '$\mathrm{\Huge{E}} \:\:\: \Huge{n - \pi^*}$', '$\mathrm{\Huge{E}} \:\:\: \Huge{n - \pi^*}$',
         '$\mathrm{\Huge{Z}} \:\:\: \Huge{\pi - \pi^*}$', '$\mathrm{\Huge{Z}} \:\:\: \Huge{\pi - \pi^*}$',
         '$\mathrm{\Huge{Z}} \:\:\: \Huge{\pi - \pi^*}$', '$\mathrm{\Huge{Z}} \:\:\: \Huge{\pi - \pi^*}$',
         '$\mathrm{\Huge{Z}} \:\:\: \Huge{n - \pi^*}$', '$\mathrm{\Huge{Z}} \:\:\: \Huge{n - \pi^*}$',
         '$\mathrm{\Huge{Z}} \:\:\: \Huge{n - \pi^*}$', '$\mathrm{\Huge{Z}} \:\:\: \Huge{n - \pi^*}$']

    # x = ['$\mathrm{\Huge{E-Isomer}} \: \Huge{\pi - \pi^*}$', '$\mathrm{\Huge{E-Isomer}} \: \Huge{\pi - \pi^*}$',
    #      '$\mathrm{\Huge{E-Isomer}} \: \Huge{\pi - \pi^*}$', '$\mathrm{\Huge{E-Isomer}} \: \Huge{\pi - \pi^*}$',
    #      '$\mathrm{\Huge{E-Isomer}} \: \Huge{n - \pi^*}$', '$\mathrm{\Huge{E-Isomer}} \: \Huge{n - \pi^*}$',
    #      '$\mathrm{\Huge{E-Isomer}} \: \Huge{n - \pi^*}$', '$\mathrm{\Huge{E-Isomer}} \: \Huge{n - \pi^*}$',
    #      '$\mathrm{\Huge{Z-Isomer}} \: \Huge{\pi - \pi^*}$', '$\mathrm{\Huge{Z-Isomer}} \: \Huge{\pi - \pi^*}$',
    #      '$\mathrm{\Huge{Z-Isomer}} \: \Huge{\pi - \pi^*}$', '$\mathrm{\Huge{Z-Isomer}} \: \Huge{\pi - \pi^*}$',
    #      '$\mathrm{\Huge{Z-Isomer}} \: \Huge{n - \pi^*}$', '$\mathrm{\Huge{Z-Isomer}} \: \Huge{n - \pi^*}$',
    #      '$\mathrm{\Huge{Z-Isomer}} \: \Huge{n - \pi^*}$', '$\mathrm{\Huge{Z-Isomer}} \: \Huge{n - \pi^*}$']

    # x = ['$\huge{E-Isomer}} \: \huge{\pi - \pi^*}$', '$\huge{E-Isomer} \: \huge{\pi - \pi^*}$',
    #      '$\huge{E-Isomer}} \: \huge{\pi - \pi^*}$', '$\huge{E-Isomer} \: \huge{\pi - \pi^*}$',
    #      '$\huge{E-Isomer}} \: \huge{n - \pi^*}$', '$\huge{E-Isomer} \: \huge{n - \pi^*}$',
    #      '$\huge{E-Isomer}} \: \huge{n - \pi^*}$', '$\huge{E-Isomer} \: \huge{n - \pi^*}$',
    #      '$\huge{Z-Isomer}} \: \huge{\pi - \pi^*}$', '$\huge{Z-Isomer} \: \huge{\pi - \pi^*}$',
    #      '$\huge{Z-Isomer}} \: \huge{\pi - \pi^*}$', '$\huge{Z-Isomer} \: \huge{\pi - \pi^*}$',
    #      '$\huge{Z-Isomer}} \: \huge{n - \pi^*}$', '$\huge{Z-Isomer} \: \huge{n - \pi^*}$',
    #      '$\huge{Z-Isomer}} \: \huge{n - \pi^*}$', '$\huge{Z-Isomer} \: \huge{n - \pi^*}$']
    


    fig.add_trace(go.Box(

        # defining y axis in corresponding
        # to x-axis
        y=[16.4, 17.3, 17.6, 17.4, 8.5, 8.6, 8.8, 9.4, 12.2, 11.5, 12.0, 12.3, 9.0, 8.2, 8.3, 8.9],
        x=x,
        name='Fragments',
        marker_color='paleturquoise',
        boxpoints='outliers'
    ))

    fig.add_trace(go.Box(
        y=[15.5, 15.2, 15.3, 17.9, 7.3, 8.4, 8.6, 10.1, 10.1, 9.8, 11.9, 10.0, 6.6, 6.9, 7.0, 7.2],
        x=x,
        name='Morgan',
        marker_color='darksalmon',
        boxpoints='outliers'
    ))

    fig.add_trace(go.Box(
        y=[13.9, 13.3, 13.5, 18.1, 7.7, 8.2, 8.3, 8.6, 10.0, 9.8, 10.2, 10.4, 6.8, 7.1, 7.1, 7.0],
        x=x,
        name='Fragprints',
        marker_color='sienna',
        boxpoints='outliers'
    ))

    fig.update_layout(

        # group together boxes of the different
        # traces for each value of x
        yaxis_title="MAE (nm)",
        font=dict(
            family="arial",
            size=40),
        boxmode='group',
        legend=dict(font=dict(family="arial", size=50, color="black")),
        boxgap=0.3,
        boxgroupgap=0.1
    )

    # fig.update_layout(font=dict(size=50))
    fig.update_xaxes(title_font_family="arial", title_font_size=40)
    fig.update_traces(marker_line_width=50, selector=dict(type='box'))
    fig.update_traces(marker_size=50, selector=dict(type='box'))
    fig.update_traces(line=dict(width=4), selector=dict(type='box'))
    fig.show()
