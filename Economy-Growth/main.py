
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("C:/Users/HP/Desktop/Book1.csv")
data.dropna(inplace=True)

def plotting(x_axis,y_axis):
    x=data[x_axis].values.reshape(-1,1)
    y=data[y_axis]

    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=2529)
    x_train.shape, x_test.shape, y_train.shape, y_test.shape

    model=LinearRegression()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    viz_train = plt
    viz_train.scatter(x_train, y_train, color='red')
    viz_train.plot(x_train, model.predict(x_train), color='blue')
    viz_train.title('Economic growth in India')
    viz_train.xlabel(x_axis)
    viz_train.ylabel(y_axis)
    viz_train.savefig("graph.png")
    f=Image.open("graph.png")
    return f


import gradio as gr
from PIL import Image

app = gr.Interface(
    plotting,
    [
    gr.Radio(['Year',
       'Birth rate, crude (per 1,000 people)',
       'Death rate, crude (per 1,000 people)',
       'Electric power consumption (kWh per capita)', 'GDP (USD)',     
       'GDP per capita (USD)',
       'Individuals using the Internet (% of population)',
       'Infant mortality rate (per 1,000 live births)',
       'Life expectancy at birth (years)',
       'Population density (people per sq. km of land area)',
       'Unemployment (% of total labor force) (modeled ILO estimate)'], label="x-axis", info="Choose any one to be represented as abscissa"),
    gr.Radio(['Year',
       'Birth rate, crude (per 1,000 people)',
       'Death rate, crude (per 1,000 people)',
       'Electric power consumption (kWh per capita)', 'GDP (USD)',     
       'GDP per capita (USD)',
       'Individuals using the Internet (% of population)',
       'Infant mortality rate (per 1,000 live births)',
       'Life expectancy at birth (years)',
       'Population density (people per sq. km of land area)',
       'Unemployment (% of total labor force) (modeled ILO estimate)'], label="y-axis", info="Choose any one to be represented as ordinate")],
    outputs=gr.Image(label="Output Image", type="pil", height=500, width=580,show_download_button=True),
    title="Plotting through Linear Regression",
    description="Determining economic growth in India over years using Linear Regression.",
    allow_flagging="never"
)

app.launch(share=True)
