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
    x = ['$\mathrm{\LARGE{E-Isomer}} \: \LARGE{\pi - \pi^*}$', '$\mathrm{\LARGE{E-Isomer}} \: \LARGE{\pi - \pi^*}$',
         '$\mathrm{\LARGE{E-Isomer}} \: \LARGE{\pi - \pi^*}$', '$\mathrm{\LARGE{E-Isomer}} \: \LARGE{\pi - \pi^*}$',
         '$\mathrm{\LARGE{E-Isomer}} \: \LARGE{n - \pi^*}$', '$\mathrm{\LARGE{E-Isomer}} \: \LARGE{n - \pi^*}$',
         '$\mathrm{\LARGE{E-Isomer}} \: \LARGE{n - \pi^*}$', '$\mathrm{\LARGE{E-Isomer}} \: \LARGE{n - \pi^*}$',
         '$\mathrm{\LARGE{Z-Isomer}} \: \LARGE{\pi - \pi^*}$', '$\mathrm{\LARGE{Z-Isomer}} \: \LARGE{\pi - \pi^*}$',
         '$\mathrm{\LARGE{Z-Isomer}} \: \LARGE{\pi - \pi^*}$', '$\mathrm{\LARGE{Z-Isomer}} \: \LARGE{\pi - \pi^*}$',
         '$\mathrm{\LARGE{Z-Isomer}} \: \LARGE{n - \pi^*}$', '$\mathrm{\LARGE{Z-Isomer}} \: \LARGE{n - \pi^*}$',
         '$\mathrm{\LARGE{Z-Isomer}} \: \LARGE{n - \pi^*}$', '$\mathrm{\LARGE{Z-Isomer}} \: \LARGE{n - \pi^*}$']

    fig.add_trace(go.Box(

        # defining y axis in corresponding
        # to x-axis
        y=[16.4, 17.3, 17.2, 17.4, 8.5, 8.6, 8.9, 9.4, 12.2, 11.5, 11.9, 12.3, 9.0, 8.2, 8.5, 8.9],
        x=x,
        name='Fragments',
        marker_color='paleturquoise',
        boxpoints='outliers'
    ))

    fig.add_trace(go.Box(
        y=[15.5, 15.2, 14.4, 17.9, 7.3, 8.4, 8.5, 10.1, 10.1, 9.8, 9.6, 10.0, 6.6, 6.9, 6.9, 7.2],
        x=x,
        name='Morgan',
        marker_color='darksalmon',
        boxpoints='outliers'
    ))

    fig.add_trace(go.Box(
        y=[13.9, 13.3, 13.1, 18.1, 7.7, 8.2, 8.3, 8.6, 10.0, 9.8, 8.8, 10.4, 6.8, 7.1, 7.1, 7.0],
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
            family="roman",
            size=40),
        boxmode='group',
        legend=dict(font=dict(family="roman", size=50, color="black")),
        boxgap=0.2,
        boxgroupgap=0.1,
        yaxis=dict(tickfont=dict(size=30)),
        xaxis=dict(tickfont=dict(size=50)
        )
    )
    fig.update_xaxes(title_font_family="Arial")
    fig.show()
