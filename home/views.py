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
    
    def parse_time_duration(duration_str):
        hours, minutes, seconds = map(int, duration_str.split(':'))
        return hours * 3600 + minutes * 60 + seconds
    
    def parse_time_duration_mnt(duration_str):
        hours, minutes, seconds = map(int, duration_str.split(':'))
        minutes = (hours * 3600 + minutes * 60 + seconds)/3600
        return 

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

    list_minutes = df.JAM_DAFTAR
    durations_in_minutes = [parse_time_duration(str(duration)) for duration in list_minutes]

    #Divide all values by 3600 using a loop
    minutes = []
    for value in durations_in_minutes:
        minutes.append(value)

    # Alternatively, use a list comprehension
    #minutes = [value / 3600 for value in minutes]

    # Convert 'JAM_DAFTAR' to total minutes
    df['JAM_DAFTAR_MINUTES'] = minutes

    # Plotly scatter plot
    trace = go.Scatter(
        x=df['DURASI'],
        y=df['JAM_DAFTAR_MINUTES'],
        mode='markers',
        marker=dict(color=cluster_labels, size=10, colorscale='Viridis'),
        showlegend=True,
        name='Clusters'
    )

    layout = go.Layout(
        title='K-Means Clustering',
        xaxis=dict(title='DURASI'),
        yaxis=dict(title='JAM_DAFTAR (minutes)'),
        showlegend=True
    )

    fig = go.Figure(data=[trace], layout=layout)

    # Convert the Plotly figure to JSON to be passed to the template
    plot_json = fig.to_json()

    # Convert the Plotly figure to JSON to be passed to the template
    plot_json = fig.to_json()

    context = {
        'data': df,
        'plot_json': plot_json,
    }


    # Return a "created" (201) response code.
    return render(request, 'index.html', context)