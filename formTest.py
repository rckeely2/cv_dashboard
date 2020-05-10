import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

data_group = dbc.Form([
        dbc.FormGroup(
        [
            dbc.Label("Countries", html_for="example-email-row", width=2),
            dbc.Col(
                dbc.Input(
                    type="text", id="example-email-row", placeholder="Enter email"
                ),
                width=6,
            ),
        ],
        row=True,
        ),
        dbc.FormGroup(
        [
            dbc.Label("Variables", html_for="example-password-row", width=2),
            dbc.Col(
                dbc.Input(
                    type="text",
                    id="example-password-row",
                    placeholder="Enter password",
                ),
                width=10,
            ),
        ],
        row=True,
        )])

transform_group = dbc.FormGroup(
    [
        dbc.Label("Transformations", html_for="example-radios-row", width=3),
        dbc.Col(
            dbc.RadioItems(
                id="transform_radio",
                options=[
                    {"label": "None", "value": 0},
                    {"label": "Normalise by population", "value": 1},
                    {"label": "Percent of Total (by day)", "value": 2},
                    {
                        "label": "Indexed",
                        "value": 3,
                        "disabled": True,
                    },
                ],
            ),
            width=9,
        ),
    ],
    row=True,
)

filter_group = dbc.Row(
    [
        dbc.Col(
            dbc.FormGroup([
                dbc.Label("Cumulative Threshold", className="mr-2"),
                dbc.Input(type="numeric", placeholder="Enter email"),
            ],
            className="mr-3",
            ),
            width=4
        ),
        dbc.Col(
            dbc.FormGroup([
                dbc.Label("Daily Threshold", className="mr-2"),
                dbc.Input(type="numeric", placeholder="Enter password"),
            ],
            className="mr-3",
            ),
            width=4
        ),
        dbc.Col(
            dbc.FormGroup([
                dbc.Label("Rolling mean", className="mr-2"),
                dbc.Input(type="numeric", placeholder="Enter password"),
            ],
            className="mr-3",
            ),
            width=4
        ),
        #dbc.Button("Submit", color="primary"),
    ],

)

plot_group = dbc.FormGroup([
        dbc.Label("Axis Configuration:"),
        dbc.Checklist(
            options=[
                {"label": "Y Axis : Log", "value": 1},
                #{"label": "Option 2", "value": 2},
                #{"label": "Disabled Option", "value": 3, "disabled": True},
            ],
            value=[],
            id="switches-input",
            switch=True,
        ),
    ])

collapse = html.Div(
    [
        dbc.ButtonGroup([
            dbc.Button("Data", id="data_button", className="mb-3", color="primary",),
            dbc.Button("Filters", id="filter_button",  className="mb-3", color="primary"),
            dbc.Button("Transforms", id="transform_button", className="mb-3", color="primary",),
            dbc.Button("Plot", id="plot_button", className="mb-3", color="primary", ),
        ]),
        dbc.Collapse([
            dbc.CardHeader("Select data:"),
            dbc.Card(dbc.CardBody(data_group)),
            ],
            id="data_collapse",
        ),
        dbc.Collapse([
            dbc.CardHeader("Apply filters to selected data:"),
            dbc.Card(dbc.CardBody(filter_group)),
            ],
            id="filter_collapse",
        ),
        dbc.Collapse([
            dbc.CardHeader("Transform selected data:"),
            dbc.Card(dbc.CardBody(transform_group)),
            ],
            id="transform_collapse",
        ),
        dbc.Collapse([
            dbc.CardHeader("Configure the plot:"),
            dbc.Card(dbc.CardBody(plot_group)),
            ],
            id="plot_collapse",
        ),
    ]
)



app.layout = dbc.Container(collapse)

@app.callback(
    Output("data_collapse", "is_open"),
    [Input("data_button", "n_clicks")],
    [State("data_collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("filter_collapse", "is_open"),
    [Input("filter_button", "n_clicks")],
    [State("filter_collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("transform_collapse", "is_open"),
    [Input("transform_button", "n_clicks")],
    [State("transform_collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("plot_collapse", "is_open"),
    [Input("plot_button", "n_clicks")],
    [State("plot_collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == "__main__":
    app.run_server(port=8051, host="127.0.0.1", debug=True)
