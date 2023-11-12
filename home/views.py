from django.shortcuts import render
import pandas as pd
from .models import PasienLama
from datetime  import datetime, time, timedelta
import numpy as np
import sklearn
import sklearn
from sklearn.cluster import KMeans

from sklearn.cluster import KMeans
import plotly.graph_objs as go

# Create your views here.
from django.http import HttpResponse


def home(request):

    queryset = PasienLama.objects.all()

    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(list(queryset.values()))
    print(df)
    print(df.DURASI)
    ###########
    #K-MEANS CODE
    durations = df.DURASI

    def extract_date_from_string(date_str):
        day = int(date_str[2:4])
        month = int(date_str[4:6])
        year = int('20' + date_str[6:8])  # Assuming the year is in the 21st century
        return pd.to_datetime(f"{year}-{month:02d}-{day:02d}")

    
    def parse_time_duration(duration_str):
        hours, minutes, seconds = map(int, duration_str.split(':'))
        return hours * 3600 + minutes * 60 + seconds
    
    def parse_time_duration_mnt(duration_str):
        hours, minutes, seconds = map(int, duration_str.split(':'))
        minutes = (hours * 3600 + minutes * 60 + seconds)/3600
        return 

    def combine_date_and_time(row):
        time_str = row['JAM_DAFTAR'].strftime('%H:%M:%S')
        return row['Date'] + pd.to_timedelta(time_str)

    # Use the function to convert the list of duration strings to integers
    durations_in_seconds = [parse_time_duration(str(duration)) for duration in durations]

    # Convert the list of duration data to a NumPy array
    X = np.array(durations_in_seconds).reshape(-1, 1)

    # Choose the number of clusters (k)
    k = 5

    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)

    # Get the cluster assignments for each data point
    cluster_labels = kmeans.labels_

    # Get the cluster centers (in seconds)
    cluster_centers = kmeans.cluster_centers_

    df["cluster"] = cluster_labels
    #df["centers"] = cluster_centers

    # Print cluster assignments and cluster centers
    for i, label in enumerate(cluster_labels):
        print(f'Duration: {durations[i]}, Cluster: {label + 1}')

    print("Cluster Centers (in seconds):")
    for i, center in enumerate(cluster_centers):
        print(f'Cluster {i + 1}: {center[0]} seconds')

    df['Date'] = df['TGL_PELAYANAN'].apply(extract_date_from_string)

    # Combine 'Date' and 'JAM_DAFTAR' to create a new datetime column
    df['datetime'] = df.apply(combine_date_and_time, axis=1)

    # K-Means clustering code...

    # Set the x-axis range dynamically based on the minimum and maximum date in the dataset
    xaxis_range = [df['Date'].min(), df['Date'].max()]
    default_range = ['06:00:00', '16:00:00']

    # Plotly scatter plot
    trace = go.Scatter(
        x=df['datetime'],
        y=df['DURASI'],
        mode='markers',
        marker=dict(color=cluster_labels, size=10, colorscale='Viridis'),
        showlegend=True,
        name='Clusters'
    )

    layout = go.Layout(
        
        xaxis=dict(title='Datetime', range=['2021-08-10 06:00:00', '2021-08-10 14:00:00']),
        yaxis=dict(title='Duration (seconds)'),
        showlegend=False
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig['layout']['height'] = 600

    # Convert the Plotly figure to JSON to be passed to the template
    plot_json = fig.to_json()

    context = {
        'data': df,
        'plot_json': plot_json,
    }




    # Return a "created" (201) response code.
    return render(request, 'index.html', context)