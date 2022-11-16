import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# marker adds a slope triangle on a figure.

def marker(figure,          # matplotlib figure (plt.figure()) the triangle has to be added to.
           x_data,          # one-dimensional array.
           y_datas,         # list of one-dimensional arrays.
           position,        # float between 0 and 1 to position the triangle along the curve.
           gap,             # float to space out the triangle from the data.
           slope=1.,        # strictly positive float representing the slope of the hypotenuse of the triangle.
           loglog=False,    # boolean True if the plot is in loglog scale.
           reverse=False,   # boolean the vertical side of the triangle is on the right if True and on the left if False.
           textsize=13):    # int size of the annotated text on horizontal and vertical sides of the triangle.

    x_data = np.asarray(x_data)

    mins = []
    for y_data in y_datas:
        mins.append(y_data.min())
    ind_min = np.argmin(mins)

    anchor_1 = [0., 0.]
    anchor_2 = [0., 0.]
    
    anchor_1[0] = x_data[0]
    anchor_2[0] = x_data[-1]
    
    anchor_1[1] = y_datas[ind_min][0]
    anchor_2[1] = y_datas[ind_min][-1]
    
    if loglog:
        if reverse:
            marker_x = anchor_1[0]**position * anchor_2[0]**(1.-position)\
                * (max(x_data)/min(x_data))**(gap/10.)
            marker_y = anchor_1[1]**position * anchor_2[1]**(1.-position)\
                / (max(y_datas[ind_min])/min(y_datas[ind_min]))**(gap/10.)

            right_angle = np.asarray([marker_x, marker_y])
            x_corner = np.asarray([marker_x / (x_data[-1] / x_data[0]) ** 0.1, marker_y])
            y_corner = np.asarray([marker_x, marker_y / (x_data[-1] / x_data[0]) ** (0.1 * slope)])

            coef = 1
        else:
            marker_x = anchor_1[0]**position * anchor_2[0]**(1.-position)\
                / (max(x_data)/min(x_data))**(gap/10.)
            marker_y = anchor_1[1]**position * anchor_2[1]**(1.-position)\
                / (max(y_datas[ind_min])/min(y_datas[ind_min]))**(gap/10.)

            right_angle = np.asarray([marker_x, marker_y])
            x_corner = np.asarray([marker_x * (x_data[-1] / x_data[0]) ** 0.1, marker_y])
            y_corner = np.asarray([marker_x, marker_y * (x_data[-1] / x_data[0]) ** (0.1 * slope)])

            coef = -1
        
        pos_horizontal_annotation = right_angle ** 0.5 * x_corner ** 0.5 / (y_corner / right_angle) ** 0.3
        pos_vertical_annotation = right_angle ** 0.5 * y_corner ** 0.5 / (x_corner / right_angle) ** 0.2
            
    else:
        if reverse:
            marker_x = anchor_1[0] * position + anchor_2[0] * (1. - position)\
                    + ((np.abs(max(x_data) - min(x_data)))) * (gap/10.)
            marker_y = anchor_1[1] * position + anchor_2[1] * (1. - position)\
                    - ((np.abs(max(y_datas[ind_min]) - min(y_datas[ind_min])))) * (gap/10.)
            
            right_angle = np.asarray([marker_x, marker_y])
            x_corner = np.asarray([marker_x + (x_data[-1] - x_data[0]) * 0.1, marker_y])
            y_corner = np.asarray([marker_x, marker_y - (x_data[-1] - x_data[0]) * (0.1 * slope)])

            coef = 1
        else:
            marker_x = anchor_1[0] * position + anchor_2[0] * (1. - position)\
                    - ((np.abs(max(x_data) - min(x_data)))) * (gap/10.)
            marker_y = anchor_1[1] * position + anchor_2[1] * (1. - position)\
                    - ((np.abs(max(y_datas[ind_min]) - min(y_datas[ind_min])))) * (gap/10.)
            
            right_angle = np.asarray([marker_x, marker_y])
            x_corner = np.asarray([marker_x - (x_data[-1] - x_data[0]) * 0.1, marker_y])
            y_corner = np.asarray([marker_x, marker_y - (x_data[-1] - x_data[0]) * (0.1 * slope)])
        
            pos_horizontal_annotation = right_angle * 0.5 + x_corner * 0.5 - (np.abs(y_corner - right_angle)) * 0.3
            pos_vertical_annotation = right_angle * 0.5 + y_corner * 0.5 - (np.abs(x_corner - right_angle)) * 0.2

            coef = -1

    plt.annotate("1", xy=pos_horizontal_annotation, size=13, verticalalignment='center', horizontalalignment='center')
    plt.annotate(f"{str(np.round(coef * slope,0))}", xy=pos_vertical_annotation, size=13, verticalalignment= 'center', horizontalalignment='right')

    triangle = plt.Polygon((x_corner, right_angle, y_corner), color="lightgray")
    figure.gca().add_patch(triangle)


if __name__=="__main__":
    figure = plt.figure()

    xs = np.linspace(0.1, 1, 100)[::-1]
    slope = 2
    ys = xs ** -slope + np.random.normal(loc=0., scale=0.01, size=len(xs))

    plt.loglog(xs, ys)

    marker(figure, xs, [ys], 0.5, 0.7, slope=slope, loglog=True)

    plt.savefig("test.pdf")