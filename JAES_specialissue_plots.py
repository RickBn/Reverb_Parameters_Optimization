import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scripts.params_dim_reduction import get_dim_red_model, reconstruct_original_params

bands = ['125', '250', '500', '1000', '2000', '4000']

# x_colors = ['#f7fcfd','#e0ecf4','#bfd3e6','#9ebcda','#8c96c6','#8c6bb1','#88419d','#810f7c','#4d004b']
# y_colors = ['#ffffe5','#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#993404','#662506']
y_colors = ['#c6dbef','#6baed6','#2171b5','#08306b']
x_colors = ['#fdd0a2','#fd8d3c','#d94801','#7f2704']

colors_heatmap = ['#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3',
                  '#2b8cbe', '#0868ac', '#084081']

font = 'Consolas'

save_path = r'C:\Users\david\Universita\Dottorato\SONICOM\Projects\Students\Magistrale\Marco Fontana\Special issue JAES_DAFx\img'

def process_save_line_plot_coef(fig, filename, width, height, title=''):

    fig.update_layout(
        title=title,
        title_x=0.1,
        xaxis_title="Octave band [Hz]",
        yaxis_title='Absorption coef.',
        width=width,
        height=height,
        font_family=font,
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.875,
            xanchor="right",
            x=0.9975,
            borderwidth=0.5,
            orientation='v'
        ),
        autosize=False,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    fig.update_xaxes(
        #     mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        #         ticktext=bands
    )
    fig.update_yaxes(
        #     mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    fig.update_yaxes(range=[0, 1])  # , gridcolor='#000000')
    fig.update_xaxes(range=[-0.1, 6.7])
    # fig.update_xaxes(gridcolor='#000000')
    #
    # fig.update_layout(plot_bgcolor="#FFFFFF")

    fig.show()

    save_filename = f'{save_path}/{filename}'

    fig.write_html(f'{save_filename}.html')

    fig.write_image(f'{save_filename}.pdf', format='pdf', engine='orca')

if __name__ == "__main__":

    apply_dim_red = {'pts_2d': r'.\wall_coeff_dim_reduction\PCA_data\324000_iterations\2d_projection_data.csv',
                     'pts_original': r'.\wall_coeff_dim_reduction\PCA_data\324000_iterations\filters_data.csv'}
    dim_red_mdl = get_dim_red_model(dim_red_alg='pca',
                                    voronoi=False,
                                    inv_interp=True,
                                    unit_circle=False,
                                    path=apply_dim_red,
                                    points_to_remove=[]
                                    )

    n_samples = 4
    x_values = np.linspace(-0.9, 0.9, n_samples)
    y_fixed = 0

    y_values = np.linspace(-0.9, 0.9, n_samples)
    x_fixed = 0

    # fig = go.Figure()
    width = 450
    height = 165

    abs_coef_x = np.zeros((n_samples, len(bands)))
    abs_coef_y = np.zeros((n_samples, len(bands)))

    fig = go.Figure()
    for n, x in enumerate(x_values):

        abs_coef = reconstruct_original_params(dim_red_mdl, [x, y_fixed])
        abs_coef_x[n, :] = np.array(abs_coef)

        fig.add_trace(
                go.Scatter(x=bands, y=abs_coef,
                           # name=f'[{x:.2f}, {y_fixed:.2f}]',
                           name=f'$x={x:.1f}$',
                           marker_color=x_colors[n],
                           )
            )

    process_save_line_plot_coef(fig, '2d_space_sampling_lineplot_fixedY', width=width, height=height,
                                # title='$y=0$'
                                )

    fig = go.Figure()
    for n, y in enumerate(y_values):

        abs_coef = reconstruct_original_params(dim_red_mdl, [x_fixed, y])
        abs_coef_y[n, :] = np.array(abs_coef)


        fig.add_trace(
                go.Scatter(x=bands, y=abs_coef,
                           # name=f'[{x_fixed:.2f}, {y:.2f}]',
                           name=f'$y={y:.1f}$',
                           marker_color=y_colors[n],
                           )
            )

    process_save_line_plot_coef(fig, '2d_space_sampling_lineplot_fixedX', width=width, height=height,
                                # title='$x=0$'
                                )

    width = 400
    height = 400
    fig = go.Figure(data=go.Heatmap(z=abs_coef_x, zmin=0, zmax=1,
                                    x=bands,
                                    y=x_values,
                                    # xgap=1,
                                    # ygap=1,
                                    # colorscale='Teal'
                                    # reversescale=True,
                                    colorscale=colors_heatmap
                                    ))
    fig.update_layout(xaxis_title="Octave band [Hz]", yaxis_title="x",
                      width=width,
                      height=height)
    fig.show()

    save_filename = f'{save_path}/2d_space_sampling_heatmap_x'

    fig.write_html(f'{save_filename}.html')

    fig.write_image(f'{save_filename}.pdf', format='pdf', engine='orca')

    fig = go.Figure(data=go.Heatmap(z=abs_coef_y, zmin=0, zmax=1,
                                    x=bands,
                                    y=y_values,
                                    # xgap=1,
                                    # ygap=1,
                                    # colorscale='Teal'
                                    # reversescale=True,
                                    colorscale=colors_heatmap
                                    ))
    fig.update_layout(xaxis_title="Octave band [Hz]", yaxis_title="y",
                      width=width,
                      height=height)

    fig.show()

    save_filename = f'{save_path}/2d_space_sampling_heatmap_y'

    fig.write_html(f'{save_filename}.html')

    fig.write_image(f'{save_filename}.pdf', format='pdf', engine='orca')

    pass