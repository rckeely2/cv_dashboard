import dash
import dash_html_components as html
import dash_core_components as dcc

# def create_layout():
#     return

app = dash.Dash(__name__)
app.layout = html.Div(className="container",
                                children=["Hello World"])

if __name__ == "__main__":
    app.run_server(port=8050, debug=True)
