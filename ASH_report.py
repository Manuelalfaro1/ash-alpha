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
# 1. Hjelpefunksjoner og Utility
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

def read_latest_files(n=18):
    downloads_folder = os.path.expanduser('~/Downloads')
    files = [f for f in os.listdir(downloads_folder)
             if f.endswith('.csv') and f.startswith('ASH')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(downloads_folder, x)), reverse=True)
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
        'best_left_force': best_left_force,
        'best_right_force': best_right_force,
        'best_left_100ms': best_left_100ms,
        'best_right_100ms': best_right_100ms
    }

def process_all_positions(injured_side):
    files = read_latest_files(n=18)
    if len(files) < 18:
        print("Ikke nok filer funnet!")
        return {}
    
    pos_labels = ["I", "Y", "T"]
    position_results = {}
    for idx, pos in enumerate(pos_labels):
        group_files = files[idx*6:(idx+1)*6]
        metrics_list = []
        for i, fp in enumerate(group_files):
            if injured_side == 'Right':
                side = "Left" if i < 3 else "Right"
            elif injured_side == 'Left':
                side = "Right" if i < 3 else "Left"
            else:
                side = "Left" if i < 3 else "Right"
            m = process_file(fp, side)
            if m is not None:
                metrics_list.append(m)
        best_trials = pick_best_trials(metrics_list)
        if best_trials:
            position_results[pos] = best_trials
    return position_results

# Definer standard stil for posisjonbokser og grønn stil
default_box_style = {
    'position': 'relative',
    'border': '2px solid lightgrey',
    'padding': '5px',
    'borderRadius': '10px',
    'backgroundColor': 'white',
    'boxShadow': '0px 0px 15px rgba(0,0,0,0.1)',
    'marginBottom': '150px',
    'maxWidth': '1200px',
    'margin': 'auto'
}

green_box_style = {
    'position': 'relative',
    'border': '3px solid rgba(0, 128, 0, 1)',
    'padding': '5px',
    'borderRadius': '10px',
    'backgroundColor': 'white',
    'boxShadow': '0px 0px 20px rgba(0, 128, 0, 1)',
    'marginBottom': '150px',
    'maxWidth': '1200px',
    'margin': 'auto'
}

def create_bar_chart_for_position(best_trials, injured_side, bodyweight, pos_label):
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

    if bw_val is not None:
        right_data.append(best_right_force['max_force'] / bw_val)
        left_data.append(best_left_force['max_force'] / bw_val)

    if injured_side == 'Right':
        base_data = left_data
    else:
        base_data = right_data

    bar_data = {
        'Max Force <br>(N)': [
            (right_data[0]/base_data[0])*100,
            (left_data[0]/base_data[0])*100
        ],
        'RFD 100 ms <br>(N/s)': [
            (right_data[1]/base_data[1])*100,
            (left_data[1]/base_data[1])*100
        ],
        '% of Max at 100 ms <br>(%)': [
            (right_data[2]/base_data[2])*100,
            (left_data[2]/base_data[2])*100
        ],
    }
    if bw_val is not None:
        bar_data['Max Force / BW <br>(N/kg)'] = [
            (right_data[3]/base_data[3])*100,
            (left_data[3]/base_data[3])*100
        ]

    left_texts  = [f"{val:.1f}" for val in left_data]
    right_texts = [f"{val:.1f}" for val in right_data]

    fig = go.Figure(data=[
        go.Bar(
            name='Left',
            x=list(bar_data.keys()),
            y=[vals[1] for vals in bar_data.values()],
            marker_color='salmon',
            text=left_texts,
            textposition='inside',
            insidetextanchor="start",
            textfont=dict(size=20, family='Futura')
        ),
        go.Bar(
            name='Right',
            x=list(bar_data.keys()),
            y=[vals[0] for vals in bar_data.values()],
            marker_color='dodgerblue',
            text=right_texts,
            textposition='inside',
            insidetextanchor='start',
            textfont=dict(size=20, family='Futura')
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name='90% Threshold',
            line=dict(color='black', width=1, dash="dash")
        )
    ])

    fig.update_layout(
        barmode='group',
        yaxis_title='Injured Side Asymmetry(%)',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', family='Futura', size=16),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=15, family='Futura')
        )
    )

    fig.add_shape(
        dict(
            type="line",
            xref="paper",
            x0=0, x1=1,
            yref="y",
            y0=90, y1=90,
            line=dict(color="black", width=1, dash="dash")
        )
    )

    # Variabel for å sjekke om alle relative forskjeller for injured side er grønne
    all_green = True
    # Beregn injured ratio for hver metric (ved bruk av non‑injured side som referanse)
    metrics_names = list(bar_data.keys())
    injured_ratios = []
    for i, metric_name in enumerate(metrics_names):
        if injured_side == 'Right':
            ratio_injured = (right_data[i] / base_data[i]) * 100
        else:
            ratio_injured = (left_data[i] / base_data[i]) * 100
        injured_ratios.append(ratio_injured)
    # Finn den høyeste injured ratioen (non‑injured er referansen, så vanligvis 100 hvis injured er lavere)
    max_injured_ratio = max(injured_ratios) if injured_ratios else 100
    # Definer en felles y-posisjon – vi kan bruke max_injured_ratio + 10 hvis den er over 100, ellers 110
    common_y = max_injured_ratio + 10 if max_injured_ratio > 100 else 110

    # Legg til annotasjoner for injured side – bruk common_y for alle
    for i, metric_name in enumerate(metrics_names):
        if injured_side == 'Right':
            ratio = (right_data[i] / base_data[i]) * 100
        else:
            ratio = (left_data[i] / base_data[i]) * 100
        percentage_diff = ratio - 100

        if (injured_side == 'Right' and ratio == (right_data[i] / base_data[i]) * 100) or \
           (injured_side == 'Left' and ratio == (left_data[i] / base_data[i]) * 100):
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
            if color != 'green':
                all_green = False

            fig.add_annotation(
                x=metric_name,
                y=common_y,  # Bruk felles y-verdi for alle annotasjoner
                text=f"{percentage_diff:.1f}%",
                showarrow=False,
                font=dict(color=color, size=32),
                xanchor='center',
                yanchor='bottom'
            )

    # Legg til bildeoverlay i nederste høyre hjørne av figuren (ingen endring her)
    fig.update_layout(images=[
        dict(
            source=f"/assets/ASH_{pos_label}.png",
            xref="paper",
            yref="paper",
            x=1.15, y=-0.11,
            sizex=0.5, sizey=0.5,
            xanchor="right", yanchor="bottom",
            layer="above"
        )
    ])
    return fig, all_green

# ================================
# 2. Dash-app og Layout
# ================================

signal.signal(signal.SIGINT, signal_handler)

server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.config.suppress_callback_exceptions = True

# Venstre logo (allerede definert)
logo_left = html.Img(
    src='/assets/le.png',
    style={'height': '150px', 'marginRight': '20px'}
)

# Høyre logo (AlphaPWR)
alpha_logo = html.Img(
    src='/assets/AlphaPWR.png',
    style={'height': '50px', 'marginRight': '20px', 'marginTop': '15px', 'width': 'auto'}
)

# Endret input_box – knappene er stablet vertikalt
input_box = html.Div([
    html.Div([
        html.Label("Injured Side:", style={
            'fontFamily': 'Futura',
            'fontSize': '17px',
            'marginRight': '10px',
            'color': 'grey'
        }),
        dcc.Dropdown(
            id='injured-side-radio',  # Samme id slik at callback fungerer
            options=[
                {'label': 'Left', 'value': 'Left'},
                {'label': 'Right', 'value': 'Right'}
            ],
            placeholder="Select...",
            value=None,
            clearable=False,
            style={
                'fontFamily': 'Futura',
                'fontSize': '17px',
                'width': '70px',
                'border': '2px white'
            }
        )
    ], style={
        'border': '2px white',
        'padding': '10px',
        'borderRadius': '5px',
        'backgroundColor': 'white',
        'marginRight': '20px',
        'display': 'flex',
        'alignItems': 'center',
        'width': '190px',
        'height': '10px'
    }),
    html.Div([
        html.Label("Bodyweight:", style={
            'fontFamily': 'Futura',
            'fontSize': '17px',

            'marginRight': '10px',
            'color': 'grey'
        }),
        dcc.Input(
            id='bodyweight-input',
            type='number',
            placeholder='',
            style={
                'fontFamily': 'Futura',
                'marginLeft': '9px',
                'fontSize': '17px',
                'width': '55px',
                'height': '30px',
                'border': '2px white'
            }
        )
    ], style={
        'border': '2px white',
        'padding': '10px',
        'borderRadius': '5px',
        'backgroundColor': 'white',
        'width': '210px',
        'height': '10px'
    })
], style={
    'display': 'flex',
    'marginTop': '20px',
    'marginLeft': '0px',
    'flexDirection': 'column',  # Stable knappene vertikalt
    'alignItems': 'center',
    'gap': '-5px'
})

# Venstre container for logo og input_box
left_container = html.Div([
    logo_left,
    input_box
], style={
    'display': 'flex',
    'alignItems': 'flex-start'
})

# Overordnet container med left_container til venstre og alpha_logo til høyre
top_container = html.Div([
    left_container,
    alpha_logo
], style={
    'display': 'flex',
    'justifyContent': 'space-between',
    'alignItems': 'flex-start',
    'margin': '25px'
})

# Opprett boks for hver posisjon med redusert intern padding og ekstra luft mellom boksene.
def create_position_box(pos_label, graph_id, container_id):
    return html.Div([
        html.Div([
            html.H2(f"ASH test position {pos_label}", style={'textAlign': 'left',
                                                     'fontFamily': 'Avenir',
                                                     'color': 'grey',
                                                     'fontSize': '23px',
                                                     'margin': '5px 0',
                                                     'marginLeft': '20px',
                                                     'fontWeight': 'bold'})
        ]),
        dcc.Graph(id=graph_id, figure=go.Figure(), style={'height': '400px'}, config={'displayModeBar': False}),
    ], id=container_id, style=default_box_style)

spacer = html.Div("", style={'height': '50px', 'backgroundColor': 'white'})

# Legg til en spacer-div og logo på bunnen for ekstra mellomrom
bottom_logo = html.Div([
    html.Div("", style={'height': '100px', 'backgroundColor': 'white'}),
    html.Img(src='/assets/Logo.png', style={'display': 'block', 'margin': 'auto', 'height': '100px'})
])

app.layout = html.Div([
    top_container,
    html.H1("Functional Shoulder Screening", style={'textAlign': 'center',
                                                   'fontFamily': 'Avenir',
                                                   'fontSize': '35px',
                                                   'paddingTop': '50px',
                                                   'fontWeight': 'bold'}),
    spacer,                                               
    create_position_box("I", "bar-chart-I", "position-box-I"),
    spacer,
    create_position_box("Y", "bar-chart-Y", "position-box-Y"),
    spacer,
    create_position_box("T", "bar-chart-T", "position-box-T"),
    bottom_logo
], style={'backgroundColor': 'white', 'width': '100vw', 'position': 'relative'})

@app.callback(
    [Output('bar-chart-I', 'figure'),
     Output('bar-chart-Y', 'figure'),
     Output('bar-chart-T', 'figure'),
     Output('position-box-I', 'style'),
     Output('position-box-Y', 'style'),
     Output('position-box-T', 'style')],
    [Input('injured-side-radio', 'value'),
     Input('bodyweight-input', 'value')]
)
def update_graphs(injured_side, bodyweight):
    position_results = process_all_positions(injured_side)
    figs = {}
    styles = {}
    for pos in ['I', 'Y', 'T']:
        if pos in position_results:
            fig, all_green = create_bar_chart_for_position(position_results[pos], injured_side, bodyweight, pos)
            figs[pos] = fig
            if all_green:
                styles[pos] = green_box_style
            else:
                styles[pos] = default_box_style
        else:
            figs[pos] = go.Figure()
            styles[pos] = default_box_style
    return figs['I'], figs['Y'], figs['T'], styles['I'], styles['Y'], styles['T']

if __name__ == '__main__':
    port = find_free_port()
    Timer(3, open_browser, args=[port]).start()
    # Deaktiver auto-reloaderen for å unngå at Timer kjøres to ganger
    app.run_server(debug=False, use_reloader=False, port=port)
