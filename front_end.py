"""
Created on April 2021
@author: YUAN YE
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

## Home page
# make a navitem for the different examples
nav_item1 = dbc.NavItem(dbc.NavLink('HOME', href='/'))
nav_item2 = dbc.NavItem(dbc.NavLink('SIGN IN', href='/page-1'))
nav_item3 = dbc.NavItem(dbc.NavLink("NEWS", href="/page-2"))
# make dropdown
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Our Technology"),
        dbc.DropdownMenuItem("Our Solutions"),
        dbc.DropdownMenuItem("Enterprise"),
        dbc.DropdownMenuItem("Developers"),
    ],
    nav=True,
    in_navbar=True,
    label="MENU",
)
# this example that adds a logo to the navbar brand
logo = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                        dbc.Col(dbc.NavbarBrand("Quant", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="https://plot.ly",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    [nav_item1, nav_item2, nav_item3, dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ],
    ),
    color="dark",
    dark=True,
    style={'margin': '0px!important'},
    sticky="top"
)

index_page = html.Div([

    logo,

    html.Img(src=app.get_asset_url('home_page.jpg'),
             style={'height': '100%', 'width': '100%',
                    'margin': 0}),
    html.Br(),
    html.Br(),

    html.H2(['Strategy', dbc.Badge("Hot", color="danger", className="ml-1")],
            style={
                'textAlign': 'center',
                'font_weight': 'bold',
                'font_family': 'microsoft sans serif'
            }),

    html.Br(),

    html.Div([
        dbc.Card([dbc.Button("Introduction", color="danger"),
                  dbc.CardBody(
                      [
                          # html.H5("Card title", className="card-title"),
                          html.P('Here, we are going to help you find the most suitable investment strategy '
                                 'considering your risk preference. The process of investment is quite clear. '
                                 'Firstly, you need to take a questionnaire to measure risk appetite. Secondly, '
                                 'we will use Natural Language Processing, Statistical indicators and Machine '
                                 'Learning techniques to construct a stock portfolio. Finally, we will adjust weights '
                                 'between stock portfolio and bond to suit your appetite. Generally speaking, '
                                 'our strategy performs well and gained 15.6 percent annualized return.',
                              className="card-text",
                              style={'color': 'black'}
                          ),
                      ]
                  ),
                  ],
                 color="light",
                 inverse=True,
                 style={"width": "58rem",
                        },
                 )], style={'display': 'flex',
                            "justifyContent": 'center',
                            }),

    html.Br(),
    html.Br(),

    html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card([dbc.Button("Strategy Results", color="danger"),
                                  dbc.CardImg(src=app.get_asset_url('s1.png'), top=True)
                                  ])
                    ),
                    dbc.Col(dbc.Card([dbc.Button("Equity Curves for 5 Strategies", color="danger"),
                                      dbc.CardImg(src=app.get_asset_url('s2.png'), top=True),
                                      ])),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card([dbc.Button("Stock Position: 80%", color="danger"),
                                      dbc.CardImg(src=app.get_asset_url('s3.png'), top=True),
                                      ])),
                    dbc.Col(dbc.Card([dbc.Button("Stock Position: 40%", color="danger"),
                                      dbc.CardImg(src=app.get_asset_url('s4.png'), top=True),
                                      ])),
                ]
            ),
        ]
    )
])


@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# questionare page
page_1_layout = html.Div([

    logo,

    html.H2(
        children='Questionnaire',
        style={
            'textAlign': 'center',
            'font_weight': 'bold'
        }
    ),
    html.Br(),

    html.H6("1. Your main source of income is: ",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb1',
        options=[
            {'label': 'No fixed income', 'value': 4},
            {'label': 'Income from financial assets such as interest, dividends, and transfer of securities',
             'value': 3},
            {'label': 'Income from renting, selling real estate and other non-financial assets',
             'value': 2},
            {'label': 'Wages, labor compensation / Income from production and operation', 'value': 1},
        ],
        value=0,
        labelStyle={'display': False}
    ),
    html.Br(),

    html.H6("2. What percentage of your annual household disposable income can be used for financial investment ("
            "excluding savings deposits)?",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb2',
        options=[
            {'label': 'Less than 10%', 'value': 5},
            {'label': '10% to 25%', 'value': 4},
            {'label': '25% to 50%', 'value': 3},
            {'label': '50%-70%', 'value': 2},
            {'label': 'More than 70%', 'value': 1}
        ],
        value=0
    ),
    html.Br(),

    html.H6("3. Do you have a large amount of outstanding debts, if so, its nature:",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb3',
        options=[
            {'label': 'Yes, short-term credit debts such as credit card debts and consumer credit', 'value': 4},
            {'label': 'Yes, long-term fixed debts such as housing mortgage loans', 'value': 3},
            {'label': 'Yes, borrowing between relatives and friends', 'value': 2},
            {'label': 'No', 'value': 1}
        ],
        value=0
    ),
    html.Br(),

    html.H6("4. The amount of assets (including financial assets and real estate) available for investment is:",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb4',
        options=[
            {'label': 'No more than RMB 500,000 yuan', 'value': 4},
            {'label': 'RMB 500,000-3 million yuan (not included)', 'value': 3},
            {'label': 'RMB 3-10 million (not included)', 'value': 2},
            {'label': 'More than RMB 10 million yuan', 'value': 1},
        ],
        value=0
    ),
    html.Br(),

    html.H6("5. Your investment experience can be summarized as:",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb5',
        options=[
            {'label': 'Limited: Except for bank current accounts and time deposits, I basically have no other '
                      'investment experience', 'value': 4},
            {'label': 'General: In addition to bank current accounts and time deposits, I have purchased wealth '
                      'management products such as funds and insurance, but I still need further guidance',
             'value': 3},
            {'label': 'Rich: I am an experienced investor, I have participated in the trading of stocks, funds and '
                      'other products, and I tend to make investment decisions by myself', 'value': 2},
            {'label': 'Very rich: I am a very experienced investor and have participated in the trading of warrants, '
                      'futures or GEM products', 'value': 1},
        ],
        value=0
    ),
    html.Br(),

    html.H6("6. Choose the most suitable description of your investment philosophy.",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb6',
        options=[
            {'label': 'I hope to get a very stable asset appreciation, even if it means a relatively low total return;',
             'value': 3},
            {'label': 'I want to strike a balance between maximizing long-term returns and minimizing volatility;',
             'value': 2},
            {'label': 'My main goal is to get the highest long-term return, even if I have to endure some very huge '
                      'short-term losses;', 'value': 1},
        ],
        value=0
    ),
    html.Br(),

    html.H6("7. How many years of experience have you invested in venture capital products such as funds, stocks, "
            "trusts, private equity securities or financial derivatives?",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb7',
        options=[
            {'label': 'no experience',
             'value': 5},
            {'label': 'Less than 2 years',
             'value': 4},
            {'label': '2 to 5 years',
             'value': 3},
            {'label': '5 to 10 years',
             'value': 2},
            {'label': 'More than 10 years',
             'value': 1},
        ],
        value=0
    ),
    html.Br(),

    html.H6("8. The period during which most of your funds used for securities investment will not be used for other "
            "purposes are:",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb8',
        options=[
            {'label': '0 to 1 year',
             'value': 4},
            {'label': '1 to 3 years',
             'value': 3},
            {'label': '3 to 5 years',
             'value': 2},
            {'label': 'more than 5 years',
             'value': 1},
        ],
        value=0
    ),
    html.Br(),

    html.H6("9. In a long period of time in the future (such as more than 5 years), what kinds of investment products "
            "do you plan to invest more than 50% of your investable assets in?",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb9',
        options=[
            {'label': 'Fixed-income investment products such as bonds, money market funds, bond funds, etc.',
             'value': 3},
            {'label': 'Hybrid funds or products such as hybrid funds, asset-backed securities, convertible bonds, etc.',
             'value': 2},
            {'label': 'High-risk financial products or services such as stocks funds, futures, options, etc.',
             'value': 1},
        ],
        value=0
    ),
    html.Br(),

    html.H6("10. When you invest in, your primary goals are:",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb10',
        options=[
            {'label': 'Keep the principal as safe as possible, don’t care about the low yield',
             'value': 4},
            {'label': 'Generate a certain amount of income and can bear certain investment risks',
             'value': 3},
            {'label': 'Generate more returns and can bear greater investment risks',
             'value': 2},
            {'label': 'Achieve substantial growth in assets and be willing to bear great investment risks',
             'value': 1},
        ],
        value=0
    ),
    html.Br(),

    html.H6("11. What do you think is the biggest investment loss you can bear?",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb11',
        options=[
            {'label': "Can't afford any loss",
             'value': 4},
            {'label': 'Certain investment loss',
             'value': 3},
            {'label': 'Large investment loss',
             'value': 2},
            {'label': 'Loss may exceed principal',
             'value': 1},
        ],
        value=0
    ),
    html.Br(),

    html.H6("12. Suppose you have money that you do not need to use in the next 5 years. Now you have two different "
            "investment options: Product A expects to get 6% annualized return, and the cumulative compound interest "
            "return is 33% after 5 years. The probability of realizing the return is very high. And the possibility "
            "of principal loss is very low. Product B expects to obtain an annualized return of 20%, "
            "with a cumulative compound interest return of 150% after 5 years, but the possibility of principal gains "
            "and losses fluctuates within 5 years. There is a 50% chance of gaining a loss of 20% or even higher. You "
            "allocate your investment assets as:",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb12',
        options=[
            {'label': "All invested in A",
             'value': 5},
            {'label': 'Most invested in A',
             'value': 4},
            {'label': 'Half of the two investments',
             'value': 3},
            {'label': 'Most invested in B',
             'value': 2},
            {'label': 'Invest all in B',
             'value': 1},
        ],
        value=0
    ),
    html.Br(),

    html.H6("13. Do you have full capacity for civil conduct (a natural person over the age of 18 or a minor over the "
            "age of 16 with your own labor income as the main source of living)?",
            style={
                'font-weight': 'bold',
            }
            ),
    dbc.RadioItems(
        id='rb13',
        options=[
            {'label': "No",
             'value': 2},
            {'label': 'Yes',
             'value': 1},
        ],
        value=0
    ),
    html.Br(),

    html.Div([
        dbc.Alert(
            id="alert-fade",
            color="primary"
        ),
    ], style={'display': 'flex', 'justifyContent': 'center'}),
    html.Br(),

    html.Div([
        dbc.Button(id='submit-button-state',
                   n_clicks=0,
                   children='Submit',
                   color="primary")
    ], style={'display': 'flex', 'justifyContent': 'center'}),
    html.Br(),
])


@app.callback(
    Output('alert-fade', 'children'),
    Input('submit-button-state', 'n_clicks'),
    State('rb1', 'value'),
    State('rb2', 'value'),
    State('rb3', 'value'),
    State('rb4', 'value'),
    State('rb5', 'value'),
    State('rb6', 'value'),
    State('rb7', 'value'),
    State('rb8', 'value'),
    State('rb9', 'value'),
    State('rb10', 'value'),
    State('rb11', 'value'),
    State('rb12', 'value'),
    State('rb13', 'value'),
)
def page_1_dropdown(n_clicks, rb1_value, rb2_value, rb3_value, rb4_value, rb5_value, rb6_value, rb7_value,
                    rb8_value, rb9_value, rb10_value, rb11_value, rb12_value, rb13_value):
    total_score = rb1_value + rb2_value + rb3_value + rb4_value + rb5_value + rb6_value + rb7_value + \
                  rb8_value + rb9_value + rb10_value + rb11_value + rb12_value + rb13_value

    return 'Your Risk Score is "{}"'.format(total_score)


def get_news(pages):
    '''
    获取东方财富网新闻列表至本地xls
    url_list是指链接列表
    '''

    url_list = []
    for page in range(1, pages + 1):
        url = "http://guba.eastmoney.com/list,cjpl_{}.html".format(page)
        url_list.append(url)

    headers = {
        'User-Agent': UserAgent(verify_ssl=False).random,
        'cookie': 'intellpositionL=815.458px; em_hq_fls=js; _qddaz=QD.ulhxhb.n2npa0.kgbvw7i6; pgv_pvi=4468324352; '
                  'qgqp_b_id=5fa78977af4078bc5c839676cc090dc3; intellpositionT=955px; '
                  'HAList=d-hk-01810%2Ca-sz-300766-%u6BCF%u65E5%u4E92%u52A8%2Ca-sh-600017-%u65E5%u7167%u6E2F%2Ca-sz'
                  '-300613-%u5BCC%u701A%u5FAE%2Ca-sz-300507-%u82CF%u5965%u4F20%u611F%2Ca-sz-300477-%u5408%u7EB5%u79D1'
                  '%u6280%2Ca-sz-300362-%u5929%u7FD4%u73AF%u5883%2Ca-sz-300232-%u6D32%u660E%u79D1%u6280%2Ca-sz-300177'
                  '-%u4E2D%u6D77%u8FBE%2Ca-sz-300175-%u6717%u6E90%u80A1%u4EFD%2Ca-sz-300147-%u9999%u96EA%u5236%u836F'
                  '%2Ca-sz-300065-%u6D77%u5170%u4FE1; st_si=56788486589893; '
                  'emshistory=%5B%22%E5%B0%8F%E7%B1%B3%E9%9B%86%E5%9B%A2-W%22%2C%22%E5%B0%8F%E7%B1%B3%E9%80%A0%E8%BD'
                  '%A6%22%2C%22%E6%AF%8F%E6%97%A5%E4%BA%92%E5%8A%A8%22%2C%22%E5%AF%8C%E7%80%9A%E5%BE%AE%22%2C%22%E5'
                  '%90%88%E7%BA%B5%E7%A7%91%E6%8A%80%22%2C%22%E5%A4%A9%E7%BF%94%E7%8E%AF%E5%A2%83%22%2C%22%E6%B4%B2'
                  '%E6%98%8E%E7%A7%91%E6%8A%80%22%2C%22%E9%A6%99%E9%9B%AA%E5%88%B6%E8%8D%AF%22%2C%22%E8%B4%9D%E8%82'
                  '%AF%E8%83%BD%E6%BA%90%22%2C%22%E9%94%A1%E4%B8%9A%E8%82%A1%E4%BB%BD%22%5D; st_asi=delete; '
                  'st_pvi=88263969806757; st_sp=2020-08-06%2017%3A23%3A17; '
                  'st_inirUrl=https%3A%2F%2Fwww.baidu.com%2Flink; st_sn=20; '
                  'st_psi=20210412095105327-117001313005-3718701417 '
    }

    total_title_list = []
    total_href_list = []
    for i in range(len(url_list)):
        url = url_list[i]
        res = requests.get(url, headers=headers)
        res.encoding = res.apparent_encoding
        html = res.text
        soup = BeautifulSoup(html, "html.parser")
        title_list = soup.select(".l3.a3")[1:]

        new_title_list = []
        new_href_list = []
        for i in title_list:
            i_new = i.select('a')[0]["title"]
            i_href = 'http://guba.eastmoney.com' + i.select('a')[0]['href']
            if (len(i_new) >= 15) and ('cjpl' in i_href):
                new_title_list.append(i_new)
                new_href_list.append(i_href)

        total_title_list.extend(new_title_list)
        total_href_list.extend(new_href_list)

    return total_title_list, total_href_list


# news page
page_2_layout = html.Div([
    logo,
    html.Br(),
    html.H1(["FINANCIAL NEWS ", dbc.Badge("New", className="ml-1", color="warning")],
            style={
                'textAlign': 'center',
                'font_weight': 'bold',
                'font_family': 'microsoft sans serif'
            }),
    html.Br(),

    html.Div([
        dbc.Row([
            dbc.Col(
                html.Div([
                    dbc.Row([
                        html.P('choose the number of pages (70 news/page)',
                               style={'whiteSpace': 'pre-line'})
                    ]),
                    dbc.Row([
                        dbc.Input(id='pages-state', type='number', value=0,
                                  style={'margin-right': '10px',
                                         'width': '60px'}),
                        dbc.Button(id='button-state', n_clicks=0, children='Confirm', color='primary'),
                    ]),
                    # html.Br(),
                    html.Br(),
                    dbc.Row([
                        html.Div(id='textarea-example-output',
                                 style={'width': '500px',
                                        'height': '665px',
                                        'overflow': 'scroll'}),
                    ]),
                ]),

            ),
            dbc.Col([
                html.Br(),
                html.Img(src=app.get_asset_url('news6.jpg'),
                         style={'height': '335px', 'width': '500px',
                                'margin-bottom': '0px'}),
                html.P('source：https://www.quanjing.com/imgbuy/QJ6732994518.html',
                       style={'font-size': '10px',
                              "font-style": "italic",
                              'margin-bottom': '10px'}),
                # html.Img(src=app.get_asset_url('news2.png'),
                #          style={'height': '300px', 'width': '410px',
                #                 'margin': 0}),
                html.Img(src=app.get_asset_url('news3.png'),
                         style={'height': '366px', 'width': '500px',
                                'margin': 0}),

                html.P('source：http://www.lemeitu.com/photo/6433.html',
                       style={'font-size': '10px',
                              "font-style": "italic",
                              'margin': 0})],

            )
        ], style={'width': '1080px',
                  })
    ],
        style={
            'display': 'flex',
            'justifyContent': 'center'
        }
    )

    # dbc.Row([dbc.Col(html.Div(id='textarea-example-output',
    #          style={'width': '500px'}), width='500px'),
    #          dbc.Col(html.Img(src=PLOTLY_LOGO, height="300px"), width='300px')])

])


@app.callback(
    Output('textarea-example-output', 'children'),
    Input('button-state', 'n_clicks'),
    State('pages-state', 'value')
)
def update_output_div(n_clicks, input1):
    total_title_list, total_href_list = get_news(input1)

    html_a_list = []
    for index in range(0, len(total_title_list)):
        html_a_list.append(html.A(total_title_list[index], href=total_href_list[index]))
        html_a_list.append(html.Br())

    return html_a_list


# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page


if __name__ == '__main__':
    app.run_server(debug=True, port=8868)
