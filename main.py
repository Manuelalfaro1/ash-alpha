import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go
import pandas as pd
import os
import numpy as np
import socket
import webbrowser
from threading import Timer
import signal
import sys
import subprocess

# Bruk et eksternt stylesheet fra Bootswatch (Lux-tema)
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/lux/bootstrap.min.css']

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    addr, port = s.getsockname()
    s.close()
    return port

def get_csv_files():
    downloads_folder = os.path.expanduser('~/Downloads')
    files = [f for f in os.listdir(downloads_folder) if f.endswith('.csv')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(downloads_folder, x)), reverse=True)
    return [os.path.join(downloads_folder, f) for f in files]

def open_browser(port):
    webbrowser.open_new(f"http://127.0.0.1:{port}")

def signal_handler(sig, frame):
    print('Shutting down server...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
server = app.server

# Legg din tilpassede CSS direkte inn i appens index_string:
custom_css = '''
.custom-button {
    background-color: rgb(43, 108, 74);
    color: white;
    border: none;
    font-weight: light;
    outline: none;
    transition: background-color 0.2s ease;
}
.custom-button:hover {
    background-color: rgb(80, 140, 100);
}
.custom-button:active {
    background-color: grey;
}
.custom-button:focus {
    outline: none;
    box-shadow: none;
}
'''

# Endre index_string for å injisere CSS-en i <head>
app.index_string = f'''<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{custom_css}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>'''

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='file-index', data=0),
    # Hovedcontainer med to kolonner via flex
    html.Div(
        children=[
            # Venstre: Dash-innhold som okkuperer all plass minus 200px for iframen
            html.Div(
                id='page-content',
                children=[
                    html.Div([
                        html.H1("Real-Time Force Data", style={
                            'color': '#000000',
                            'fontFamily': 'Futura',
                            'textAlign': 'center',
                            'fontSize': 40,
                            'marginTop': '20px'
                        }),
                        # Øverste knapp-panel (Remove, Previous, Next, Exit)
                        html.Div([
                            html.Button('Remove', id='remove-button', n_clicks=0, className='custom-button'),
                            html.Button('Previous', id='prev-button', n_clicks=0, className='custom-button'),
                            html.Button('Next', id='next-button', n_clicks=0, className='custom-button'),
                            html.Button('Exit', id='exit-button', className='custom-button')
                        ], style={
                            'position': 'absolute',
                            'top': '40px',
                            'right': '10px',
                            'display': 'flex',
                            'flexDirection': 'column',
                            'gap': '10px',
                            'z-index': '1000'
                        }),
                    ]),
                    html.Div(id='force-at-100ms-display', style={
                        'textAlign': 'left',
                        'paddingTop': '10px',
                        'paddingLeft': '50px'
                    }),
                    # Grafen er pakket inn i en boks med svak lysegrå bakgrunn
                    html.Div(
                        dcc.Graph(id='live-graph'),
                        style={
                            'border': '1px solid #ccc',
                            'padding': '10px',
                            'margin': '20px 50px',
                            'borderRadius': '5px',
                            'boxShadow': '0px 4px 8px rgba(0,0,0,0.2)',
                            'backgroundColor': '#F8F8F8'
                        }
                    ),
                    # Container for de to nederste knappene
                    html.Div([
                        html.Button('Position Analysis', id='position-analysis-button', n_clicks=0, className='custom-button'),
                        html.Button('ASH Report', id='ash-report-button', n_clicks=0, className='custom-button')
                    ], style={
                        'textAlign': 'center',
                        'marginTop': '10px'
                    }),
                    dcc.Interval(
                        id='interval-component',
                        interval=1000,  # 1 sekund
                        n_intervals=0
                    ),
                    html.Div(id='position-analysis-output', style={
                        'textAlign': 'center',
                        'paddingTop': '20px'
                    }),
                    html.Div(id='ash-report-output', style={
                        'textAlign': 'center',
                        'paddingTop': '20px'
                    })
                ],
                style={
                    'width': 'calc(100vw - 200px)',  # Reservasjon av 200px til iframen
                    'height': '100vh',
                    'overflowY': 'auto',
                    'position': 'relative',
                    'paddingTop': '20px'
                }
            ),
            # Høyre: Iframe fast til høyre
            html.Iframe(
                src="https://pwr-staging.alphatek.no/csv",
                style={
                    "width": "360px",
                    "height": "100vh",
                    "border": "none",
                    "margin": "0",
                    "padding": "0",
                    "zIndex": "999"
                }
            )
        ],
        style={
            'display': 'flex',
            'width': '100vw',
            'height': '100vh'
        }
    )
], style={
    'backgroundColor': 'white',
    'margin': '0',
    'padding': '0',
    'overflow': 'hidden'
})

@app.callback(
    [Output('live-graph', 'figure'),
     Output('force-at-100ms-display', 'children'),
     Output('file-index', 'data')],
    [Input('interval-component', 'n_intervals'),
     Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks'),
     Input('remove-button', 'n_clicks')],
    [State('file-index', 'data')]
)
def update_graph_live(n, prev_clicks, next_clicks, remove_clicks, current_index):
    files = get_csv_files()
    if not files:
        return go.Figure(), "No files available.", 0

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'interval-component'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'prev-button' and current_index < len(files) - 1:
        current_index += 1
    elif button_id == 'next-button' and current_index > 0:
        current_index -= 1
    elif button_id == 'remove-button' and files:
        os.remove(files[current_index])
        del files[current_index]
        current_index = min(current_index, len(files) - 1)

    if files:
        current_file = files[current_index]
    else:
        return go.Figure(), "No files available after deletion.", current_index

    df = pd.read_csv(current_file, header=20)
    Resultant = np.array((((df.iloc[:, 0] + df.iloc[:, 1] + df.iloc[:, 2] + df.iloc[:, 3])) * 9.81))
    Time = np.linspace(0, len(Resultant) / 434, len(Resultant))
    avg_initial_force = np.mean(Resultant[:214])
    net_force = Resultant - avg_initial_force
    ten_n_indices = np.where(net_force > 10)[0]
    if len(ten_n_indices) > 0:
        ten_n_index = ten_n_indices[0]
    else:
        return go.Figure(), "No force increase detected.", current_index

    start_index = ten_n_index
    while start_index > 0 and net_force[start_index] > 5:
        start_index -= 1
    start_index = max(0, start_index)

    time_100ms_index = start_index + int(0.1 * 434)
    if time_100ms_index >= len(net_force):
        return go.Figure(), "Insufficient data for 100 ms calculation.", current_index

    force_at_100ms = net_force[time_100ms_index]
    max_force = np.max(net_force)
    max_force_index = np.argmax(net_force)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Time, y=net_force, mode='lines', name='Net Force',
        line=dict(color="rgb(43, 108, 74)")
    ))
    fig.add_shape(dict(
        type="line",
        xref="x",
        x0=Time[max_force_index],
        x1=Time[max_force_index],
        yref="paper",
        y0=0,
        y1=1,
        line=dict(color="black", width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=[Time[max_force_index]], y=[max_force],
        mode='markers', marker=dict(size=10, color='blue'), showlegend=False
    ))
    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Net Force (N)',
        height=400,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', family='Futura', size=12),
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )

    force_at_100ms_display = html.Div([
        html.Div([
            html.Span(f"{max_force:.1f} ", style={'fontSize': 100}),
            html.Span("- N ", style={'fontSize': 30})
        ], style={'color': '#000000', 'fontWeight': 'lighter', 'fontFamily': 'Futura'}),
        html.Br(),
        html.Div([
            html.Span(f"{force_at_100ms:.1f} ", style={'fontSize': 60}),
            html.Span("- N at 100ms", style={'fontSize': 20})
        ], style={'color': '#000000', 'fontWeight': 'lighter', 'fontFamily': 'Futura'})
    ])

    return fig, force_at_100ms_display, current_index

@app.callback(Output('page-content', 'children'),
              [Input('exit-button', 'n_clicks')])
def shutdown_server(n_clicks):
    if n_clicks is not None:
        shutdown_layout = html.Div(
            children=[
                html.H1("Now you can close the browser", style={
                    'color': '#000000',
                    'fontFamily': 'Futura',
                    'textAlign': 'center',
                    'fontSize': 50,
                    'paddingTop': '200px'
                })
            ],
            style={
                'backgroundColor': 'white',
                'height': '100vh',
                'width': '100vw',
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center'
            }
        )
        Timer(2, os.kill, args=[os.getpid(), signal.SIGINT]).start()
        return shutdown_layout
    return dash.no_update

@app.callback(
    Output('position-analysis-output', 'children'),
    Input('position-analysis-button', 'n_clicks')
)
def run_position_analysis(n_clicks):
    if n_clicks and n_clicks > 0:
        subprocess.Popen([sys.executable, 'position_analysis.py'])
        return html.Div("Position Analysis started.", style={
            'fontFamily': 'Futura',
            'fontSize': '20px',
            'color': 'black'
        })
    return ""

@app.callback(
    Output('ash-report-output', 'children'),
    Input('ash-report-button', 'n_clicks')
)
def run_ash_report(n_clicks):
    if n_clicks and n_clicks > 0:
        subprocess.Popen([sys.executable, 'ASH_report.py'])
        return html.Div("ASH Report started.", style={
            'fontFamily': 'Futura',
            'fontSize': '20px',
            'color': 'black'
        })
    return ""

if __name__ == '__main__':
    port = find_free_port()
    Timer(3, open_browser, args=[port]).start()
    app.run_server(debug=False, use_reloader=False, port=port)
