import os
import sys
import signal
import socket
import webbrowser
from threading import Timer

import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from flask import Flask

# ================================
# 1. Hjelpefunksjoner og "Utility"
# ================================

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    addr, port = s.getsockname()
    s.close()
    return port

def open_browser(port):
    webbrowser.open_new(f"http://127.0.0.1:{port}")

def signal_handler(sig, frame):
    print('Shutting down server...')
    sys.exit(0)

def read_latest_files(n=6):
    downloads_folder = os.path.expanduser('~/Downloads')
    files = [f for f in os.listdir(downloads_folder) if f.endswith('.csv') and f.startswith('ASH')]
    # Sorter slik at de nyeste filene kommer først (reverse=True)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(downloads_folder, x)), reverse=True)
    # Ta de 6 første, og sorter oppover (eldst til nyest)
    latest_files = files[:n]
    latest_files.sort(key=lambda x: os.path.getmtime(os.path.join(downloads_folder, x)))
    return [os.path.join(downloads_folder, file) for file in latest_files]

def slice_and_compute_net_force(df):
    avg_initial_force = np.mean(df.iloc[:214, :4].sum(axis=1) * 9.81)
    df['Netto_Kraft'] = df.iloc[:, 0:4].sum(axis=1) * 9.81 - avg_initial_force

    ten_n_indices = np.where(df['Netto_Kraft'] > 10)[0]
    if len(ten_n_indices) == 0:
        return None

    ten_n_index = ten_n_indices[0]
    start_index = ten_n_index
    while start_index > 0 and df['Netto_Kraft'][start_index] > 2:
        start_index -= 1
    start_index = max(0, start_index)

    end_index = ten_n_indices[-1]
    end_index = min(end_index, df.shape[0] - 1)

    if start_index - 43 >= 0:
        df_sliced = df.iloc[start_index - 43:end_index].reset_index(drop=True)
    else:
        df_sliced = df.iloc[:end_index].reset_index(drop=True)

    sampling_rate = 434.02777
    n_samples = df_sliced.shape[0]
    time_array = np.linspace(-43/sampling_rate, n_samples/sampling_rate, n_samples)
    df_sliced['Time_s'] = time_array

    return df_sliced

def compute_key_metrics(df_sliced):
    CM_Jump = df_sliced['Netto_Kraft'].values
    Time = df_sliced['Time_s'].values
    if len(CM_Jump) == 0:
        return None

    max_force = CM_Jump.max()
    max_force_index = CM_Jump.argmax()
    time_to_max_force = Time[max_force_index]

    idx_100ms = np.where(Time >= 0.1)[0]
    if len(idx_100ms) == 0:
        return None
    i_100ms = idx_100ms[0]
    force_100ms = CM_Jump[i_100ms]

    rfd = np.gradient(CM_Jump, Time)
    max_rfd_100 = rfd[:i_100ms+1].max() if i_100ms >= 0 else None

    return {
        "max_force": max_force,
        "force_at_100ms": force_100ms,
        "max_rfd_100": max_rfd_100,
        "time_to_max_force": time_to_max_force
    }

def process_file(file_path, side):
    try:
        df = pd.read_csv(file_path)
        df_sliced = slice_and_compute_net_force(df)
        if df_sliced is None:
            return None
        metrics = compute_key_metrics(df_sliced)
        if metrics is None:
            return None
        metrics["side"] = side
        metrics["file"] = file_path
        return metrics
    except Exception as e:
        print(f"Feil under behandling av {file_path}: {e}")
        return None

def pick_best_trials(metrics_list):
    df = pd.DataFrame(metrics_list)
    if df.empty:
        return {}

    left_df = df[df.side == "Left"]
    right_df = df[df.side == "Right"]
    if left_df.empty or right_df.empty:
        return {}

    best_left_force = left_df.loc[left_df['max_force'].idxmax()].to_dict()
    best_right_force = right_df.loc[right_df['max_force'].idxmax()].to_dict()
    best_left_100ms = left_df.loc[left_df['force_at_100ms'].idxmax()].to_dict()
    best_right_100ms = right_df.loc[right_df['force_at_100ms'].idxmax()].to_dict()

    return {
        'best_left_force':  best_left_force,
        'best_right_force': best_right_force,
        'best_left_100ms':  best_left_100ms,
        'best_right_100ms': best_right_100ms
    }

# ================================
# 2. Funksjoner for å lage figurer
# ================================

def create_figure_force(best_left_force, best_right_force):
    fig = go.Figure()

    def add_side_trace(trial_data, color):
        file_path = trial_data["file"]
        df = pd.read_csv(file_path)
        df_sliced = slice_and_compute_net_force(df)
        if df_sliced is None:
            return

        Time = df_sliced['Time_s'].values
        Force = df_sliced['Netto_Kraft'].values
        max_force_idx = Force.argmax()

        idx_100ms = np.where(Time >= 0.1)[0]
        i_100ms = idx_100ms[0] if len(idx_100ms) else None

        side_name = trial_data["side"]
        fig.add_trace(go.Scatter(
            x=Time, y=Force,
            mode='lines',
            name=f'{side_name} Force',
            line=dict(color=color)
        ))

        if max_force_idx < len(Time):
            fig.add_shape(
                dict(
                    type="line",
                    xref="x",
                    x0=Time[max_force_idx],
                    x1=Time[max_force_idx],
                    yref="paper",
                    y0=0,
                    y1=1,
                    line=dict(color=color, width=2, dash="dash"),
                )
            )

        if i_100ms is not None and i_100ms < len(Time):
            t_100ms = Time[i_100ms]
            fig.add_shape(
                dict(
                    type="line",
                    xref="x",
                    x0=t_100ms,
                    x1=t_100ms,
                    yref="paper",
                    y0=0,
                    y1=1,
                    line=dict(color='dimgray', width=2, dash="dash"),
                )
            )

    if best_left_force is not None:
        add_side_trace(best_left_force, "salmon")
    if best_right_force is not None:
        add_side_trace(best_right_force, "dodgerblue")

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name='100 ms',
        line=dict(color='dimgray', width=2, dash="dash")
    ))

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Force (N)',
        title='Force Over Time',
        height=400,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', family='Futura', size=12),
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=14, family='Futura')
        )
    )
    return fig

def create_figure_limited_force(best_left_100ms, best_right_100ms):
    fig = go.Figure()

    def add_side_trace(trial_data, color):
        file_path = trial_data["file"]
        df = pd.read_csv(file_path)
        df_sliced = slice_and_compute_net_force(df)
        if df_sliced is None:
            return
        Time = df_sliced['Time_s'].values
        Force = df_sliced['Netto_Kraft'].values

        idx_51 = np.where(Time <= 0.51)[0]
        if len(idx_51) == 0:
            return
        last_idx = idx_51[-1]

        side_name = trial_data["side"]
        fig.add_trace(go.Scatter(
            x=Time[:last_idx+1]*1000,
            y=Force[:last_idx+1],
            mode='lines',
            name=f'{side_name} Force',
            line=dict(color=color)
        ))

        # Time 0-linje
        fig.add_shape(
            dict(
                type="line",
                xref="x",
                x0=0,
                x1=0,
                yref="paper",
                y0=0,
                y1=1,
                line=dict(color='dimgray', width=2, dash="dot"),
            )
        )

        # 100 ms-linje
        idx_100ms = np.where(Time >= 0.1)[0]
        if len(idx_100ms) > 0:
            i_100ms = idx_100ms[0]
            if i_100ms < len(Time):
                fig.add_shape(
                    dict(
                        type="line",
                        xref="x",
                        x0=Time[i_100ms]*1000,
                        x1=Time[i_100ms]*1000,
                        yref="paper",
                        y0=0,
                        y1=1,
                        line=dict(color='dimgray', width=2, dash="dash"),
                    )
                )

    if best_left_100ms is not None:
        add_side_trace(best_left_100ms, "salmon")
    if best_right_100ms is not None:
        add_side_trace(best_right_100ms, "dodgerblue")

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines', name='Time 0',
        line=dict(color='dimgray', width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines', name='100 ms',
        line=dict(color='dimgray', width=2, dash="dash")
    ))

    fig.update_layout(
        xaxis_title='Time (ms)',
        yaxis_title='Force (N)',
        title='Limited Force Over Time',
        height=400,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', family='Futura', size=12),
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=14, family='Futura')
        )
    )
    return fig

def create_figure_rfd(best_left_100ms, best_right_100ms):
    fig = go.Figure()

    def add_side_trace(trial_data, color):
        file_path = trial_data["file"]
        df = pd.read_csv(file_path)
        df_sliced = slice_and_compute_net_force(df)
        if df_sliced is None:
            return

        Time = df_sliced['Time_s'].values
        Force = df_sliced['Netto_Kraft'].values
        if len(Force) < 2:
            return
        rfd = np.gradient(Force, Time)

        idx_05 = np.where(Time <= 0.5)[0]
        if len(idx_05) == 0:
            return
        last_idx = idx_05[-1]

        side_name = trial_data["side"]
        fig.add_trace(go.Scatter(
            x=Time[:last_idx+1],
            y=rfd[:last_idx+1],
            mode='lines',
            name=f'{side_name} RFD',
            line=dict(color=color)
        ))

        # Time 0-linje
        fig.add_shape(
            dict(
                type="line",
                xref="x",
                x0=0,
                x1=0,
                yref="paper",
                y0=0,
                y1=1,
                line=dict(color='dimgray', width=2, dash="dot"),
            )
        )

        # 100 ms-linje
        idx_100ms = np.where(Time >= 0.1)[0]
        if len(idx_100ms) > 0:
            i_100ms = idx_100ms[0]
            if i_100ms < len(Time):
                fig.add_shape(
                    dict(
                        type="line",
                        xref="x",
                        x0=Time[i_100ms],
                        x1=Time[i_100ms],
                        yref="paper",
                        y0=0,
                        y1=1,
                        line=dict(color='dimgray', width=2, dash="dash"),
                    )
                )

    if best_left_100ms is not None:
        add_side_trace(best_left_100ms, "salmon")
    if best_right_100ms is not None:
        add_side_trace(best_right_100ms, "dodgerblue")

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines', name='Time 0',
        line=dict(color='dimgray', width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines', name='100 ms',
        line=dict(color='dimgray', width=2, dash="dash")
    ))

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='RFD (N/s)',
        title='Rate of Force Development',
        height=400,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', family='Futura', size=12),
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=14, family='Futura')
        )
    )
    return fig

def create_bar_chart(best_trials, injured_side, bodyweight):
    best_right_force = best_trials['best_right_force']
    best_left_force  = best_trials['best_left_force']
    best_right_100ms = best_trials['best_right_100ms']
    best_left_100ms  = best_trials['best_left_100ms']

    right_data = [
        best_right_force['max_force'],
        best_right_100ms['force_at_100ms'] / 0.1,
        (best_right_100ms['force_at_100ms'] / best_right_force['max_force']) * 100
    ]
    left_data = [
        best_left_force['max_force'],
        best_left_100ms['force_at_100ms'] / 0.1,
        (best_left_100ms['force_at_100ms'] / best_left_force['max_force']) * 100
    ]

    bw_val = None
    if bodyweight is not None:
        try:
            bw_val = float(bodyweight)
            if bw_val <= 0:
                bw_val = None
        except:
            bw_val = None

    # Legger til Max Force / BW (N/kg) hvis bodyweight er gyldig
    if bw_val is not None:
        right_data.append(best_right_force['max_force'] / bw_val)
        left_data.append(best_left_force['max_force'] / bw_val)

    if injured_side == 'Right':
        base_data    = left_data
        injured_data = right_data
    else:
        base_data    = right_data
        injured_data = left_data

    bar_data = {
        'Max Force (N)': [
            (right_data[0]/base_data[0])*100,
            (left_data[0]/base_data[0])*100
        ],
        'RFD 100 ms (N/s)': [
            (right_data[1]/base_data[1])*100,
            (left_data[1]/base_data[1])*100
        ],
        '% of Max at 100 ms (%)': [
            (right_data[2]/base_data[2])*100,
            (left_data[2]/base_data[2])*100
        ],
    }

    if bw_val is not None:
        bar_data['Max Force / BW (N/kg)'] = [
            (right_data[3]/base_data[3])*100,
            (left_data[3]/base_data[3])*100
        ]

    left_texts  = [f"{val:.1f}" for val in left_data]
    right_texts = [f"{val:.1f}" for val in right_data]

    bar_chart = go.Figure(data=[
        go.Bar(
            name='Left',
            x=list(bar_data.keys()),
            y=[vals[1] for vals in bar_data.values()],
            marker_color='salmon',
            text=left_texts,
            textposition='inside',
            insidetextanchor="start",
            textfont=dict(size=26)
        ),
        go.Bar(
            name='Right',
            x=list(bar_data.keys()),
            y=[vals[0] for vals in bar_data.values()],
            marker_color='dodgerblue',
            text=right_texts,
            textposition='inside',
            insidetextanchor='start',
            textfont=dict(size=26)
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='lines', 
            name='90% Threshold',
            line=dict(color='black', width=1, dash="dash")
        )
    ])

    bar_chart.update_layout(
        barmode='group',
        yaxis_title='Percentage of Non-Injured Side (%)',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', family='Futura', size=14),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=14, family='Futura')
        ),
        title=''
    )

    bar_chart.add_shape(
        dict(
            type="line",
            xref="paper",
            x0=0, x1=1,
            yref="y",
            y0=90, y1=90,
            line=dict(color="black", width=1, dash="dash"),
            name='90% Threshold'
        )
    )

    metrics_names = list(bar_data.keys())
    for i, metric_name in enumerate(metrics_names):
        for side_name in ['Right', 'Left']:
            value = right_data[i] if side_name=='Right' else left_data[i]
            ratio = (value / base_data[i]) * 100
            if (injured_side=='Right' and side_name=='Right') or (injured_side=='Left' and side_name=='Left'):
                percentage_diff = ratio - 100
                if -10 <= percentage_diff <= 10:
                    color = 'green'
                elif -15 <= percentage_diff < -10:
                    color = '#FFA500'
                elif 10 < percentage_diff <= 15:
                    color = 'green'
                elif -20 <= percentage_diff < -15:
                    color = '#FF4500'
                elif 15 < percentage_diff <= 20:
                    color = 'green'
                elif percentage_diff > 20:
                    color = 'green'
                else:
                    color = 'red'

                bar_chart.add_annotation(
                    x=metric_name,
                    y=ratio + 10 if ratio>100 else 110,
                    text=f"{percentage_diff:.1f}%",
                    showarrow=False,
                    font=dict(color=color, size=32),
                    xanchor='center',
                    yanchor='bottom'
                )

    return bar_chart

# ================================
# 3. Dash-app og layout
# ================================
signal.signal(signal.SIGINT, signal_handler)

server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.config.suppress_callback_exceptions = True

EMPTY_FIG = go.Figure()
EMPTY_STYLE = {'display': 'none'}

metrics_cache = {}

app.layout = html.Div([
    # Logo-ikon
    html.Img(
        src='/assets/le.png',
        style={
            'position': 'absolute',
            'top': '10px',
            'left': '10px',
            'height': '100px'
        }
    ),

    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=[

        # Overskrift
        html.Div([
            html.H1(
                "Athletic Shoulder Test",
                style={
                    'color': '#000000',
                    'fontFamily': 'Futura',
                    'textAlign': 'center',
                    'fontSize': 33,
                    'paddingTop': '100px'
                }
            ),
        ], style={'paddingTop': '20px'}),

        # Tom div for max_force_display
        html.Div(
            id='max-force-display',
            style={
                'textAlign': 'left',
                'paddingTop': '20px',
                'paddingLeft': '50px'
            },
            children=""
        ),

        # Tre bokser ved siden av hverandre
        html.Div([
            # Boks 1: Injured Side
            html.Div([
                html.Label(
                    "Select Injured Side:",
                    style={'fontFamily': 'Futura', 'fontSize': '20px'}
                ),
                dcc.RadioItems(
                    id='injured-side-radio',
                    options=[
                        {'label': 'Left', 'value': 'Left'},
                        {'label': 'Right', 'value': 'Right'}
                    ],
                    value=None,
                    labelStyle={
                        'display': 'inline-block',
                        'marginTop': '5px',
                        'fontFamily': 'Futura'
                    }
                )
            ], style={
                'border': '2px solid lightgrey',
                'padding': '20px',
                'borderRadius': '10px',
                'backgroundColor': '#f9f9f9',
                'boxShadow': '0px 4px 8px rgba(0,0,0,0.1)',
                'textAlign': 'center',
                'width': '220px'
            }),

            # Boks 2: Test Orientation
            html.Div([
                html.Label(
                    "Test Orientation:",
                    style={'fontFamily': 'Futura', 'fontSize': '20px'}
                ),
                dcc.RadioItems(
                    id='test-orientation-radio',
                    options=[
                        {'label': 'I', 'value': 'I'},
                        {'label': 'Y', 'value': 'Y'},
                        {'label': 'T', 'value': 'T'}
                    ],
                    value=None,
                    labelStyle={
                        'display': 'inline-block',
                        'marginTop': '5px',
                        'fontFamily': 'Futura'
                    }
                )
            ], style={
                'border': '2px solid lightgrey',
                'padding': '20px',
                'borderRadius': '10px',
                'backgroundColor': '#f9f9f9',
                'boxShadow': '0px 4px 8px rgba(0,0,0,0.1)',
                'textAlign': 'center',
                'width': '220px'
            }),

            # Boks 3: Bodyweight
            html.Div([
                html.Label(
                    "Bodyweight (kg):",
                    style={
                        'fontFamily': 'Futura',
                        'fontSize': '20px',
                        'display': 'inline-block'
                    }
                ),
                dcc.Input(
                    id='bodyweight-input',
                    type='number',
                    placeholder='E.g. 75',
                    style={
                        'fontFamily': 'Futura',
                        'marginTop': '5px',
                        'width': '80px',
                        'display': 'inline-block'
                    }
                )
            ], style={
                'border': '2px solid lightgrey',
                'padding': '20px',
                'borderRadius': '10px',
                'backgroundColor': '#f9f9f9',
                'boxShadow': '0px 4px 8px rgba(0,0,0,0.1)',
                'textAlign': 'center',
                'width': '220px'
            })

        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'justifyContent': 'center',
            'alignItems': 'flex-start',
            'gap': '20px',
            'paddingBottom': '20px'
        }),

        # Stolpediagram
        dcc.Graph(id='bar-chart', figure=go.Figure(), style={'display': 'none'}),
        html.Div(style={'height': '100px'}),

        # Linjediagram
        dcc.Graph(id='live-graph', figure=go.Figure()),
        dcc.Graph(id='limited-force-graph', figure=go.Figure()),
        dcc.Graph(id='rfd-graph', figure=go.Figure()),

        dcc.Interval(id='interval-component', interval=1000, n_intervals=1),

        # Tabell
        html.Div(
            id='reps-table',
            style={
                'paddingTop': '20px',
                'paddingLeft': '50px',
                'paddingRight': '50px'
            }
        ),

        # Logo nederst
        html.Img(
            src='/assets/Logo.png',
            style={
                'position': 'relative',
                'height': '150px',
                'paddingTop': '50px',
                'display': 'block',
                'margin': '0 auto'
            }
        )
    ])
], style={
    'backgroundColor': 'white',
    'height': '100vh',
    'width': '100vw'
})


# ================================
# 4. Callback
# ================================
@app.callback(
    [
        Output('live-graph', 'figure'),
        Output('limited-force-graph', 'figure'),
        Output('rfd-graph', 'figure'),
        Output('max-force-display', 'children'),
        Output('reps-table', 'children'),
        Output('bar-chart', 'figure'),
        Output('bar-chart', 'style')
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('injured-side-radio', 'value'),
        Input('bodyweight-input', 'value'),
    ]
)
def update_graph_live(n, injured_side, bodyweight):
    from dash import callback_context as ctx

    if not ctx.triggered:
        # Ingen handlinger
        return (EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, "", "", EMPTY_FIG, EMPTY_STYLE)

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    files = read_latest_files()
    if not files:
        return (EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, "", "", EMPTY_FIG, EMPTY_STYLE)

    # Cache-håndtering
    valid_paths = set(files)
    to_del = [k for k in metrics_cache.keys() if k not in valid_paths]
    for k in to_del:
        del metrics_cache[k]

    metrics_list = []
    for i, fp in enumerate(files):
        if injured_side == 'Right':
            # Injured side er Right, dermed testes venstre først
            side = "Left" if i < 3 else "Right"
        elif injured_side == 'Left':
            # Injured side er Left, dermed testes høyre først
            side = "Right" if i < 3 else "Left"
        else:
            # Standard fordeling
            side = "Left" if i < 3 else "Right"

        if fp not in metrics_cache:
            m = process_file(fp, side)
            metrics_cache[fp] = m
        else:
            metrics_cache[fp]["side"] = side
        if metrics_cache[fp] is not None:
            metrics_list.append(metrics_cache[fp])

    if not metrics_list:
        return (EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, "", "", EMPTY_FIG, EMPTY_STYLE)

    best_dict = pick_best_trials(metrics_list)

    fig_force = create_figure_force(
        best_dict.get('best_left_force'),
        best_dict.get('best_right_force')
    )
    fig_limited = create_figure_limited_force(
        best_dict.get('best_left_100ms'),
        best_dict.get('best_right_100ms')
    )
    fig_rfd = create_figure_rfd(
        best_dict.get('best_left_100ms'),
        best_dict.get('best_right_100ms')
    )

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics[['side','max_force','force_at_100ms','max_rfd_100','time_to_max_force']].copy()
    for col in ['max_force','force_at_100ms','max_rfd_100','time_to_max_force']:
        df_metrics[col] = df_metrics[col].round(2)

    df_metrics.rename(columns={
        'side': 'Side',
        'max_force': 'Max Force (N)',
        'force_at_100ms': 'Force at 100ms (N)',
        'max_rfd_100': 'Max RFD (N/s) [0-100ms]',
        'time_to_max_force': 'Time to Max Force (s)'
    }, inplace=True)

    # Konstruer en enkel HTML-tabell
    table_header = [html.Th(col) for col in df_metrics.columns]
    table_body = []
    for i in range(len(df_metrics)):
        row = [html.Td(df_metrics.iloc[i][col]) for col in df_metrics.columns]
        table_body.append(html.Tr(row))

    table = html.Table(
        [html.Thead(html.Tr(table_header))] + [html.Tbody(table_body)],
        style={
            'fontFamily': 'Futura',
            'color': '#000000',
            'fontSize': 14,
            'width': '100%',
            'textAlign': 'left',
            'border': '1px solid black',
            'borderCollapse': 'collapse'
        }
    )

    max_force_display = ""

    bar_fig = EMPTY_FIG
    bar_style = EMPTY_STYLE
    if len(df_metrics['Side'].unique()) == 2 and injured_side in ['Left','Right']:
        bar_fig = create_bar_chart(best_dict, injured_side, bodyweight)
        bar_style = {'display': 'block'}

    return (
        fig_force,
        fig_limited,
        fig_rfd,
        max_force_display,
        table,
        bar_fig,
        bar_style
    )

# ================================
# 5. Kjør appen
# ================================
if __name__ == '__main__':
    port = find_free_port()
    Timer(3, open_browser, args=[port]).start()
    # Deaktiver auto-reloaderen for å unngå at Timer kjøres to ganger
    app.run_server(debug=False, use_reloader=False, port=port)

