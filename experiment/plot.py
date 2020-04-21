from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
import aggregators

import plotly.graph_objects as go
from datetime import datetime
from itertools import product
import numpy as np


# Intermediary representation between parsing and plotting
class ExpResult(object):
    def __init__(self, name, duration, errors, blocked):
        self.name = name
        self.errors = errors
        self.blocked = blocked
        self.duration = duration

    def __str__(self):
        return "Experiment {} took {}; errors for 35 rounds: {}".format(self.name, self.duration, self.errors)


default = DefaultExperimentConfiguration()

# OUTPUT LOG PARSING
fileName = "experiment"
file = open("experiment/{}.log".format(fileName), "r")
results = []
currentName = None
currentStart = None
currentRound = None
currentErrors = []
currentBlocked = []
for line in file:
    if "TRAINING" in line:
        currentName = line.split(': ')[1].replace('TRAINING ', '').replace('...', '')
    elif 'Round...' in line:
        if 'Round...  0' in line:
            currentStart = datetime.strptime(line.split(': ')[0], "%d/%m/%Y,%H:%M:%S")
            currentRound = 0
        else:
            currentRound += 1
    elif 'BLOCKED' in line:
        blocked = int(line.split("  ")[1])
        currentBlocked.append((currentRound, blocked))
    elif "Error Rate:" in line:
        error = float(line.split(': ')[2].replace('%', '').strip())

        currentErrors.append(error)
        if len(currentErrors) == default.rounds:
            currentEnd = datetime.strptime(line.split(': ')[0], "%d/%m/%Y,%H:%M:%S")
            results.append(ExpResult(name=currentName,
                                     errors=currentErrors,
                                     blocked=currentBlocked,
                                     duration=(currentEnd - currentStart)))
            currentErrors = []
            currentBlocked = []

# Recreating tested configurations for labeling
releaseProportion = {0.1, 0.4}
epsilon1 = {1, 0.01, 0.0001}
epsilon3 = {1, 0.01, 0.0001}
clipValues = {0.01, 0.0001}
needClip = {False, True}
needNormalise = {False, True}
aggregators = [a.__name__.replace("Aggregator", "")
               for a in aggregators.allAggregators()]

DPconfigs = list(product({True}, needClip, clipValues, epsilon1, epsilon3,
                         needNormalise, releaseProportion, aggregators))

noDPconfig = [(False, default.needClip, default.clipValue, default.epsilon1, default.epsilon3,
               default.needNormalization, default.releaseProportion, agg) for agg in aggregators]
configs = noDPconfig + DPconfigs


# Plotting given experiment results

# For incomplete log files:
configs = configs[:len(results)]  # Can be removed when there's an run result for each configuration
experiments = dict(zip(configs, results))

# Maybe one day
# jsonFile = open("{}.json".format(fileName), "w")
# jsonFile.write(json.dumps(experiments, indent=2))
# sys.exit()

# If needed FILTERING can be performed here by removing from experiments from the dictionary
casesToPlot = list(product({False, True},  # use DP
                           {False, True},  # needClip,
                           {0.01, 0.0001},  # gamma, clipValue
                           {1, 0.01, 0.0001},  # epsilon1
                           {1, 0.01, 0.0001},  # epsilon3
                           {False, True},  # need normalisation
                           {0.1, 0.4},  # release proportion
                           ["FA", "COMED", "MKRUM", "AFA"]))  # aggregator
plotBloking = False

for config in list(experiments):
    if config not in casesToPlot:
        del experiments[config]
    if plotBloking and not experiments[config].blocked:
        del experiments[config]

# Create figure
fig = go.Figure()

# Add traces, one for each slider step
for config, exp in experiments.items():
    useDP, needClip, clip, e1, e3, normalise, release, agg = config
    if useDP:
        configName = exp.name
        configName += "Q:{};".format(release)
        configName += "e1:{};".format(e1)
        configName += "e3:{};".format(e3)
        configName += "clip:{};".format(needClip)
        configName += "gamma:{};".format(clip)
        configName += "norm:{};".format(normalise)
    else:
        configName = "{} without DP;".format(exp.name)

    transparent = 'rgba(0, 0, 0, 0)'
    markerColors = np.full(default.rounds, transparent, dtype=object)
    hoverText = np.full(default.rounds, None, dtype=object)
    if exp.blocked:
        configName += "Blocked:"
        for blockRound, client in exp.blocked:
            configName += "{};".format(client)
            hoverText[blockRound] = "{} blocked {} at round {}".format(exp.name, blockRound, client)
            markerColors[blockRound] = 'firebrick'

    plot = go.Scatter(
        name=configName,
        x=np.arange(1, default.rounds),
        y=np.array(exp.errors),
        mode='lines+markers',
        marker_size=15,
        marker_symbol='x',
        marker_color=markerColors,
        text=hoverText)

    fig.add_trace(plot)

annotations = [dict(xref='paper', yref='paper', x=0.0, y=1.05,
                    xanchor='left', yanchor='bottom',
                    text='{} DP configurations; no Byzantine clients'.format(len(experiments)),
                    font=dict(family='Arial',
                              size=30,
                              color='rgba(20,20,20,0.5)'),
                    showarrow=False)]
# Title

fig.update_layout(annotations=annotations)
fig.show()
