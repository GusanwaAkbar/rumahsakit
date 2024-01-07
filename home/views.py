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

import plotly.express as px
from dash import dcc, html
#from plotlyapp.dash_apps import my_plot

import plotly
import dash

from django.shortcuts import  redirect
from django.http import JsonResponse
from .models import PasienLama
from .forms import PasienLamaForm

from kneed import KneeLocator





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
    k = 4

    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)

    # Get the cluster centers (in seconds)
    cluster_centers = kmeans.cluster_centers_

    # Print the centroids of each cluster
    print("===========CLUSTER CENTER================")
    for i, centroid in enumerate(cluster_centers):
        print(f"Centroid of Cluster {i+1}: {centroid[0]} seconds")



    # Get the cluster assignments for each data point
    cluster_labels = kmeans.labels_

    # Get the cluster centers (in seconds)
    cluster_centers = kmeans.cluster_centers_

    df["cluster"] = cluster_labels
    #df["centers"] = cluster_centers

    # Order clusters based on cluster centers
    cluster_order = np.argsort(cluster_centers[:, 0])

    # Create a mapping from cluster label to ordered cluster number
    cluster_mapping = {cluster_order[i]: i + 1 for i in range(k)}

    # Map the cluster labels to the ordered cluster numbers in the DataFrame
    df["ordered_cluster"] = df["cluster"].map(cluster_mapping)

    df["cluster"] = df["ordered_cluster"]

    # Print ordered cluster assignments and cluster centers
    
    # Print cluster assignments and cluster centers
    #for i, label in enumerate(cluster_labels):
    #    print(f'Duration: {durations[i]}, Cluster: {label + 1}')

    #print("Cluster Centers (in seconds):")
    #for i, center in enumerate(cluster_centers):
    #    print(f'Cluster {i + 1}: {center[0]} seconds')

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

    def generate_plot(data):
    # Your Plotly code using the provided data
    # Example:
        fig = px.bar(data, x='POLI_TUJUAN', y='DURASI_seconds', color='ordered_cluster', barmode='group',
                    labels={'DURASI': 'Total Duration (seconds)', 'POLI_TUJUAN': 'POLI TUJUAN'},
                    title='Top 5 POLI TUJUAN with Highest Durations for Each Cluster')

        return dcc.Graph(figure=fig)

    def layout():
        return html.Div(children=[generate_plot()])


    cluster_order = np.argsort(cluster_centers[:, 0])
    cluster_mapping = {cluster_order[i]: i + 1 for i in range(k)}
    #df["ordered_cluster"] = df["cluster"].map(cluster_mapping)



    df['DURASI_seconds'] = df['DURASI'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

    top_polis = df.groupby(['ordered_cluster', 'POLI_TUJUAN'])['DURASI_seconds'].sum().reset_index()
    top_polis = top_polis.groupby('ordered_cluster').apply(lambda x: x.nlargest(5, 'DURASI_seconds')).reset_index(drop=True)

    # Generate Plotly visualization
    fig = px.bar(top_polis, x='POLI_TUJUAN', y='DURASI_seconds', color='ordered_cluster', barmode='group',
                 labels={'DURASI_seconds': 'Total Duration (seconds)', 'POLI_TUJUAN': 'POLI TUJUAN'},
                 title='Top 5 POLI TUJUAN with Highest Durations for Each Cluster')

    # Convert Plotly figure to JSON
    plot_bad = fig.to_json()

    
    # Filter data for Cluster 5
    # Get the top 5 POLI_TUJUAN for Cluster 5
    # Get the top 5 POLI_TUJUAN for Cluster 5
    # Get the count of POLI_TUJUAN for each cluster
    count_by_cluster = df.groupby(['POLI_TUJUAN', 'ordered_cluster']).size().reset_index(name='Count')

    # Filter counts for Cluster 5
    count_cluster_5 = count_by_cluster[count_by_cluster['ordered_cluster'] == 4]

    # Get the top POLI_TUJUAN based on the sum of counts in Cluster 5
    top_poli_tujuan = count_cluster_5.groupby('POLI_TUJUAN')['Count'].sum().nlargest(5).index

    # Filter counts for the top POLI_TUJUAN in Cluster 5
    count_by_cluster_top_5 = count_by_cluster[count_by_cluster['POLI_TUJUAN'].isin(top_poli_tujuan)]

    # Order the x-axis based on the sum of counts in Cluster 5
    ordered_poli_tujuan = count_by_cluster_top_5.groupby('POLI_TUJUAN')['Count'].sum().sort_values(ascending=False).index

    # Generate Plotly visualization
    fig = px.bar(count_by_cluster_top_5, x='POLI_TUJUAN', y='Count', color='ordered_cluster',
                 text='Count', title='Top 5 POLIKLINIK with most Cluster 5 Data',
                 category_orders={'POLI_TUJUAN': ordered_poli_tujuan, 'ordered_cluster': [1, 2, 3, 4]})

    # Convert Plotly figure to JSON
    #plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)



    plot_bad2 = fig.to_json()

    

    count_by_cluster2 = df.groupby(['POLI_TUJUAN', 'ordered_cluster']).size().reset_index(name='Count')

    count_by_cluster2 = df.groupby(['POLI_TUJUAN', 'ordered_cluster']).size().reset_index(name='Count')

    # Get the top POLI_TUJUAN values for all clusters
    top_poli_tujuan_all_clusters = count_by_cluster2.groupby('POLI_TUJUAN')['Count'].sum().nlargest(5).index

    # Filter counts for the top POLI_TUJUAN in all clusters
    count_by_cluster_top_5_all_clusters = count_by_cluster2[count_by_cluster2['POLI_TUJUAN'].isin(top_poli_tujuan_all_clusters)]

    # Order the x-axis based on the sum of counts in Cluster 1
    ordered_poli_tujuan_all_clusters = count_by_cluster_top_5_all_clusters.groupby('POLI_TUJUAN')['Count'].sum().sort_values(ascending=False).index

    # Generate Plotly visualization for all clusters
    fig_all_clusters = px.bar(count_by_cluster_top_5_all_clusters, x='POLI_TUJUAN', y='Count', color='ordered_cluster',
                            text='Count', title='Top 5 POLIKLINIK with most Cluster 1 Data',
                            category_orders={'POLI_TUJUAN': ordered_poli_tujuan_all_clusters, 'ordered_cluster': [1, 2, 3, 4]})

    # Convert Plotly figure to JSON
    plot_cluster_1 = fig_all_clusters.to_json()





    context = {
        'data': df,
        'plot_json': plot_json,
        'plot_bad': plot_bad,
        'plot_bad2': plot_bad2,
        'plot_good':plot_cluster_1,
    }




    # Return a "created" (201) response code.
    return render(request, 'index.html', context)


def table(request):

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
    k = 4

    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)

    # Get the cluster centers (in seconds)
    cluster_centers = kmeans.cluster_centers_

    # Print the centroids of each cluster
    print("===========CLUSTER CENTER================")
    for i, centroid in enumerate(cluster_centers):
        print(f"Centroid of Cluster {i+1}: {centroid[0]} seconds")



    # Get the cluster assignments for each data point
    cluster_labels = kmeans.labels_

    # Get the cluster centers (in seconds)
    cluster_centers = kmeans.cluster_centers_

    df["cluster"] = cluster_labels
    #df["centers"] = cluster_centers

    # Order clusters based on cluster centers
    cluster_order = np.argsort(cluster_centers[:, 0])

    # Create a mapping from cluster label to ordered cluster number
    cluster_mapping = {cluster_order[i]: i + 1 for i in range(k)}

    # Map the cluster labels to the ordered cluster numbers in the DataFrame
    df["ordered_cluster"] = df["cluster"].map(cluster_mapping)

    df["cluster"] = df["ordered_cluster"]

    context = {
        'data': df
    }

    




    # Return a "created" (201) response code.
    return render(request, 'table.html', context)


from datetime import datetime, timedelta, time

def tambah_pasien(request):
    if request.method == 'POST':
        form = PasienLamaForm(request.POST)
        if form.is_valid():
            # Extract the values from the form
            jam_daftar_str = str(form.cleaned_data['JAM_DAFTAR'])
            jam_map_tersedia_str = str(form.cleaned_data['JAM_MAP_TERSEDIA'])

            # Parse string representations of time to datetime objects
            jam_daftar_dt = datetime.strptime(jam_daftar_str, '%H:%M:%S')
            jam_map_tersedia_dt = datetime.strptime(jam_map_tersedia_str, '%H:%M:%S')

            # Extract time component from datetime objects
            jam_daftar = jam_daftar_dt.time()
            jam_map_tersedia = jam_map_tersedia_dt.time()

            # Calculate the duration in hours
            duration_seconds = (datetime.combine(datetime.today(), jam_map_tersedia) - 
                                datetime.combine(datetime.today(), jam_daftar)).total_seconds()

            # Convert duration to hours
            duration_hours = duration_seconds / 3600

            # Assign the duration value to the form instance
            form.instance.DURASI = duration_hours

            # Save the form
            form.save()
            
            return redirect('table')  # Redirect to your list view
        else:
            return redirect('home')
    else:
        form = PasienLamaForm()
        return redirect('home')


    
