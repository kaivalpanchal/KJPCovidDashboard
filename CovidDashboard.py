# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 02:08:10 2021

@author: kaiva_ukyplg4
"""

import pandas as pd 
import numpy as np 
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly 
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import cufflinks as cf
import chart_studio.plotly as py
import seaborn as sns
import plotly.express as px


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected = True)
cf.go_offline()
from covid import Covid #JohnsHopkins API 
from dash.dependencies import Input, Output,State

# loading data right from the source:
death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
country_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
USconfirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
USdeaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

#function input states and outputs County Time Series Data 
def MakeData(State):
    StateName = USconfirmed_df[USconfirmed_df['Province_State'] == State]
    Sum = StateName.groupby('Admin2').sum().reset_index()
    return Sum 

NJ = MakeData('New Jersey')
NJ = NJ.drop([15,20])
PA = MakeData('Pennsylvania')
NY = MakeData('New York')
CT = MakeData('Connecticut')
MI = MakeData('Michigan')

# function inputs State and outputs Total Cases information by county 
def getTotals(State):
    Totals = State.iloc[:,-1:]
    Totals = Totals.rename(columns={ Totals.columns[0]: "Totals" })
    Totals['Counties'] = State['Admin2']
    
    return Totals

NJTotals = getTotals(NJ)
NYTotals = getTotals(NY)
PATotals = getTotals(PA)
CTTotals = getTotals(CT)
MITotals = getTotals(MI)

#function to Prepare Time Series Data based on County for each state for graphing 

def PrepTSData(State):
    x=[]
    for i in State['Admin2']:
        x.append(i)
    d ={}
    for counties in x:
        countyname = State[State['Admin2'] == counties]
        countyname = countyname.T.reset_index()
        countyname.columns = ['Date','Cases']
        countyname = countyname.drop([0,1,2,3,4,5],axis = 0).reset_index()
        countyname["Delta"] = countyname['Cases'].diff()
        countyname = countyname.dropna()
        d[counties] = pd.DataFrame(countyname)
            
    return d
    

NJdata = PrepTSData(NJ)


# function to make a dictironary of graphs for each county for each State.
def Makegraphs(State):
    x=[]
    graphs = {}
    for county in State.keys():
        x.append(county)
    for counties in x:
           graphs[counties] =  px.line(State[counties],x='Date',y='Delta',
                                     labels = {'Delta': 'Cases'}, title = (str(counties) + ' County Cases'))
    return graphs
            

NJGraphs = Makegraphs(NJdata)

NJGraphs



#function to create dataframe to calculale latest change in cases for choropleth map. 
#INput data fram state from above outpud
#output data frame of county name fips code, yesterdays and todays cases and diff. 

def getdiff(state):
    Statelate = state[['Admin2',"FIPS"]]
    Statelate['Today'] = state.iloc[:,-1:]
    Statelate['Yesterday'] = state.iloc[:,-2]
    Statelate['Diff'] = Statelate['Today']-Statelate['Yesterday']
    return Statelate


NJLate = getdiff(NJ)
PALate = getdiff(PA)
CTLate = getdiff(CT)
NYLate = getdiff(NY)
MILate = getdiff(MI)
CTLate = CTLate.drop([6,8])
CTLate['FIPS'] = CTLate['FIPS'].astype(str)
CTLate['FIPS'] = CTLate['FIPS'].str[0:4]
CTLate['FIPS']= str(0) + CTLate['FIPS']


Tristate = NJLate.append(PALate)
Tristate = Tristate.append(CTLate)
Tristate = Tristate.append(NYLate)
Tristate = Tristate.append(MILate)



i = []
for data in NJ.iloc[:,-1:]:
    i.append(data)
    
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    

USChoro = px.choropleth_mapbox(Tristate, geojson=counties, locations='FIPS', color="Diff",
                           color_continuous_scale="reds",mapbox_style="carto-positron",
                           center = {"lat": 39.8399, "lon": -99.8915},zoom = 2.75,hover_name ='Admin2',
                           labels={'Diff':'Cases'},title = ("United States Covid Heat Map ")
                          )

USChoro.update_layout(margin={"r":2,"t":2,"l":2,"b":2})
USChoro.show()

StateGB = USconfirmed_df.groupby('Province_State').sum().T.reset_index()
StateGB = StateGB.drop([0,1,2,3,4,5],axis = 0).reset_index()

StateGB['USConfirmed_Total'] = StateGB.iloc[:,3:57].sum(axis = 1)

USDeath = USdeaths_df.groupby('Province_State').sum().T.reset_index()
USDeath = USDeath.drop([0,1,2,3,4,5],axis = 0).reset_index()

USDeath['Death_Total'] = USDeath.iloc[:,3:57].sum(axis = 1)


new_df = pd.merge(StateGB,
                 USDeath[['index', 'Death_Total']], 
                 on='index')
new_df['Recovered_Total'] = new_df['USConfirmed_Total'] - new_df['Death_Total']

new_df



UStot =  px.line(new_df,x='index',y=['USConfirmed_Total','Death_Total','Recovered_Total'],
                                    labels = {'index': 'Date', 'value':'Cases'}, title = ("United States Covid 19 Cases Time Series")
                                    )

UStot


States = USconfirmed_df.groupby('Province_State').sum().T.reset_index()
States = States.drop([0,1,2,3,4,5],axis = 0).reset_index()
States



app = dash.Dash()


app.layout = html.Div([
    # Title Div
    html.Div([
        html.H1(' United States Covid-19 Dashboard',style={'margin-bottom':'50px', 'text-align':'center'}),
        html.H2('by Kaival Panchal',style={'margin-bottom':'50px', 'text-align':'center'}),
        html.H1(" ------------------------------------------------------------------------------------------------------------------------------",
               style={'margin-bottom':'50px', 'text-align':'center'}),
        
    
    ]),
    
    # Heat Map and US Total Graphs
    html.H3('United States Heat Map'),
    
    html.Div(
        [
            
            dcc.Dropdown(
                id='SelState',
                options= [{'label': k, 'value': k} for k in np.unique(USconfirmed_df['Province_State'])],
                value="New Jersey",
                className='six columns',
                style=dict(
                    width='50%',
                    
                )
            )

        ],className="row",
         style={'margin-left': '1000px',
                                 'margin-bottom': '10px',
                                 'verticalAlign': 'left'}
    ), 
    
    html.Div([
        html.Div(
          dcc.Graph(id='g1', 
                    figure=USChoro), 
                    className="six columns",
                    style={"width":700, "margin": 0, 'display': 'inline-block'}
                ),
        html.Div(
          dcc.Graph(id='g2', 
                    figure={}), 
                    className="six columns",
                    style={"width":785, "margin": 0, 'display': 'inline-block'}
                ),
    ], className="row"),
    
    
    
    
    

                
    
    
    # Bar Graph with all County Totals by choosing a state in dropdown
    
    html.H3('Choose State Below to See Total Cases by County '),
    
    
    html.Label(['Select State',
        dcc.Dropdown(
                id='State-dropdown',
                options=[{'label': k, 'value': k} for k in np.unique(USconfirmed_df['Province_State'])],
                value='New Jersey'
        )]),
                
    dcc.Graph(id='Bargraph-container',figure = {}),
    
    
    # Time Series Graph for each County by choosing a state and County in dropdown
    
    html.H3('Choose State and County Below to see Cases over Time'),
    
    html.Label(["Select State", 
                dcc.Dropdown(
                    id='State-dropdown2',
                    options=[{'label': k, 'value': k} for k in np.unique(USconfirmed_df['Province_State'])],
                    value='New Jersey',  #default value to show
                    
    ),
               ]),
    
    
    html.Label([ "Select County",
                dcc.Dropdown(id="County-dropdown",value= "Bergen"),
            ]),
    
   dcc.Graph(id='Timeseries-container',figure = {}),
    
    
])


# Callbacks for all dropdowns. 

@app.callback(
    dash.dependencies.Output('Bargraph-container', 'figure'),
    [dash.dependencies.Input('State-dropdown', 'value')])

def update_output(value):
    StateName = USconfirmed_df[USconfirmed_df['Province_State'] == value]
    State = StateName.groupby('Admin2').sum().reset_index()
    Totals = State.iloc[:,-1:]
    Totals = Totals.rename(columns={ Totals.columns[0]: "Totals" })
    Totals['Counties'] = State['Admin2']
    fig = px.bar(Totals,x='Counties',y='Totals',labels = {'Totals': 'Total # of Cases'})
    return fig


@app.callback(
    dash.dependencies.Output('County-dropdown', 'options'),
    [dash.dependencies.Input('State-dropdown2', 'value')])

def getCounty(value):# value is state name 
    state = USconfirmed_df[USconfirmed_df['Province_State'] == value]
    countyname = np.unique(state['Admin2'])
    county =  [{'label': i, 'value': i} for i in countyname]
    return county 
    
@app.callback(
    dash.dependencies.Output('Timeseries-container', 'figure'),
    [dash.dependencies.Input('State-dropdown2', 'value')],
    [dash.dependencies.Input('County-dropdown', 'value')])

def MakeTimeSeries(State,County):
    state = USconfirmed_df[USconfirmed_df['Province_State'] == State]
    countydata = state.groupby('Admin2').sum().reset_index()
    countyname = countydata[countydata['Admin2'] == County]
    countyname = countyname.T.reset_index()
    countyname.columns = ['Date','Cases']
    countyname = countyname.drop([0,1,2,3,4,5],axis = 0).reset_index()
    countyname["Delta"] = countyname['Cases'].diff()
    countyname = countyname.dropna()
    d = {}
    d[County] = pd.DataFrame(countyname)
    graphs = px.line(d[County],x='Date',y='Delta',
                                     labels = {'Delta': 'Cases'}, title = (str(County) + ' County Cases'))
    return graphs
    
@app.callback(
    dash.dependencies.Output('g2', 'figure'),
    [dash.dependencies.Input('SelState', 'value')])

def getlineGraph(value): # value is state name 
    fig = px.line(new_df,x='index',y=value,
                                    labels = {'index': 'Date'}, title = (str(value) +' Total Cases over Time'))
    return fig

    
if __name__ == '__main__':
    app.run_server(port =4051 )    


