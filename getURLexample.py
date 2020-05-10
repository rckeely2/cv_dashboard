
import ast
import dash
import dash_core_components as dcc
import dash_html_components as html
from urllib.parse import urlparse, parse_qsl, urlencode
from dash.dependencies import Input, Output


app = dash.Dash()

app.config.suppress_callback_exceptions = True


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-layout')
])

def apply_default_value(params):
    def wrapper(func):
        def apply_value(*args, **kwargs):
            if 'id' in kwargs and kwargs['id'] in params:
                if ((component_ids[kwargs['id']]['value_type'] == 'numeric') or \
                    (component_ids[kwargs['id']]['component'] == 'multi_dd')):
                    kwargs['value'] = ast.literal_eval((params[kwargs['id']]))
                # elif :
                #     kwargs['value'] = ast.literal_eval((params[kwargs['id']]))
                else:
                    kwargs['value'] = params[kwargs['id']]
            return func(*args, **kwargs)
        return apply_value
    return wrapper

component_ids = {
    'single_dd' : {'component' : 'single_dd', 'value_type': 'text'},
    'multi_dd' : {'component' : 'multi_dd', 'value_type': 'text'},
    'input' : {'component' : 'input', 'value_type': 'text'},
    #'topMap' : {'component' : 'slider', 'value_type': 'numeric'},
    'range' : {'component' : 'rangeSlider', 'value_type': 'numeric'},
    'radioButton' : {'component' : 'radioButton', 'value_type': 'numeric'}
}

def build_layout(params):
    layout = [
        html.H2('URL State demo', id='state'),
        apply_default_value(params)(dcc.Dropdown)(
            id='single_dd',
            options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
            value='LA',
            multi=False
        ),
        apply_default_value(params)(dcc.Dropdown)(
            id='multi_dd',
            options=[{'label': idx, 'value': idx} for idx in ['LA', 'NYC', 'MTL']],
            value=['LA', 'NYC'],
            multi=True
        ),
        apply_default_value(params)(dcc.Input)(
            id='input',
            placeholder='Enter a value...',
            value=''
        ),
        apply_default_value(params)(dcc.RadioItems)(
            id='radioButton',
            options=[{'label': rb_label, 'value': idx} for idx, rb_label in enumerate(['LA', 'NYC', 'MTL'])],
            value=0
        ),
        # apply_default_value(params)(dcc.Slider)(
        #     id='topMap',
        #     min=0,
        #     max=9,
        #     marks={i: 'Label {}'.format(i) for i in range(10)},
        #     value=5,
        # ),
        apply_default_value(params)(dcc.RangeSlider)(
            id='range',
            min=0,
            max=9,
            marks={i: 'Label {}'.format(i) for i in range(10)},
            value=[5,10],
        ),
        html.Br(),
    ]
    return layout

# Adapted from [ https://gist.github.com/jtpio/1aeb0d850dcd537a5b244bcf5aeaa75b#file-app-py ]
# def apply_default_value(params):
#     def wrapper(func, number=False):
#         def apply_value(*args, **kwargs):
#             if 'id' in kwargs and kwargs['id'] in params:
#                 if number:
#                     kwargs[params[kwargs['id']][0]] = int(params[kwargs['id']][1])
#                     print("here")
#                 else:
#                     try:
#                         kwargs[params[kwargs['id']][0]] = params[kwargs['id']][1]
#                     except TypeError:
#                         print(kwargs)
#             return func(*args, **kwargs)
#         return apply_value
#     return wrapper
#
#
# def build_layout(params):
#     layout = [
#         html.H2('URL State demo', id='state'),
#         apply_default_value(params)(dcc.Dropdown)(
#             id='dropdown',
#             options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
#             value='LA'
#         ),
#         apply_default_value(params)(dcc.Input)(
#             id='input',
#             placeholder='Enter a value...',
#             value=''
#         ),
#         apply_default_value(params)(dcc.Slider, number=True)(
#             id='slider',
#             min=0,
#             max=9,
#             marks={i: 'Label {}'.format(i) for i in range(10)},
#             value=5,
#         ),
#         html.Br(),
#     ]
#
#     return layout

def parse_state(url):
    parse_result = urlparse(url)
    params = parse_qsl(parse_result.query)
    state = dict(params)
    return state


@app.callback(Output('page-layout', 'children'),
              inputs=[Input('url', 'href')])
def page_load(href):
    if not href:
        return []
    state = parse_state(href)
    return build_layout(state)

@app.callback(Output('url', 'search'),
              inputs=[Input(i, 'value') for i in component_ids])
def update_url_state(*values):
    state = urlencode(dict(zip(component_ids.keys(), values)))
    return f'?{state}'


if __name__ == '__main__':
    app.run_server(port=8051, host="127.0.0.1", debug=True)
