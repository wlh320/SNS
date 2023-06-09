import itertools
from matplotlib.colors import hex2color

# function copy from seaborn to generate n unique_dashes
def unique_dashes(n):
    # Start with dash specs that are well distinguishable
    dashes = [
        "",
        (4, 1.5),
        (1, 1),
        (3, 1.25, 1.5, 1.25),
        (5, 1, 1, 1),
    ]
    dashes = dashes[::-1]

    # Now programatically build as many as we need
    p = 3
    while len(dashes) < n:
        # Take combinations of long and short dashes
        a = itertools.combinations_with_replacement([3, 1.25], p)
        b = itertools.combinations_with_replacement([4, 1], p)
        # Interleave the combinations, reversing one of the streams
        segment_list = itertools.chain(*zip(
            list(a)[1:-1][::-1],
            list(b)[1:-1]
        ))
        # Now insert the gaps
        for segments in segment_list:
            gap = min(segments)
            spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
            dashes.append(spec)
        p += 1
    return dashes[:n]

# color schemes Courtesy of the excellent mbostock's D3.js project
d3_10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

d3_20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

d3_20b = ['#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39', '#e7ba52', '#e7cb94', '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6']

d3_20c = ['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476', '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb', '#636363', '#969696', '#bdbdbd', '#d9d9d9']

colors10  = list(map(hex2color, d3_10))
colors20  = list(map(hex2color, d3_20))
colors20b = map(hex2color, d3_20b)
colors20c = map(hex2color, d3_20c)