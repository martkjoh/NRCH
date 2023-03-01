import numpy as np
import plotly.offline as ply
import plotly.graph_objs as go

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from base64 import b64encode

app = Dash(__name__)


k = 1
N = 1000
x0 = np.linspace(-k, k, N)
r0 = np.linspace(-k, 0.001, N)
x, r = np.meshgrid(x0, r0)
a = np.sqrt(x**2**2 - (r + 2*x**2)**2 + 0j).real
# a[0:2]=None
a[0:2]=0
a

edge = np.zeros_like(a, dtype=bool)
for i in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 2), (0, -2), (2, 0), (-2, 0)):
    edge = edge | np.roll(a!=0, i, (0, 1))

a[edge==0]=None
a[0:2]=None


def get_fig():
    fig = go.Figure()
    fig.add_surface(
        x=x, y=r, z=a,
        surfacecolor=a,
        showscale=False, 
        colorscale="Blues",
        contours = {
            # "x": {"show": True, "start": -k, "end": k, "size": 0.1},
            "y": {"show": True, "start": -k, "end": 0, "size": 0.1},
            "z": {"show": True, "start":  0, "end": k, "size": 0.1}
        }
    )
    fig.update_layout(
        xaxis_title="...",
        width=2000, 
        height=1000,
        font = {"family" : "Droid Serif"}
    )
    return fig




app.layout = html.Div([
    html.H4('Rendering options of plots in Dash '),
    html.P("Choose render option:"),
    dcc.RadioItems(
        id='render-option',
        options=['interactive', 'image'],
        value='image'
    ),
    html.Div(id='output'),
])


@app.callback(
    Output("output", "children"), 
    Input('render-option', 'value'))
def display_graph(render_option):
    
    fig = get_fig()
    # ply.plot(fig)

    if render_option == 'image':
        img_bytes = fig.to_image(format="png")
        encoding = b64encode(img_bytes).decode()
        img_b64 = "data:image/png;base64," + encoding
        return html.Img(src=img_b64, style={'height': '500px'})
    else:
        return dcc.Graph(figure=fig)


app.run_server()