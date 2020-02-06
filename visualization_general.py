# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:23:21 2020

@author: rbrow
"""
# standard libraries
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import re
from datetime import date, datetime, timedelta
from itertools import compress
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import matplotlib.cm as cm
import colorsys
from os import path

# local module
import fbhtml_to_df as fbparse

datadict = fbparse.fbhtml_to_df(save=True)

# df = pd.read_csv("C://Users//rbrow//Downloads//facebook-rbrown789//messages//inbox//CaitlinEliseFox_p8iZOwaGag//message_1.csv")  
# df = df.applymap(str) 

df = datadict['df']
dirnm = datadict['dirpath']
filepref = datadict['filepref']

plt.style.use('dark_background')

df['datetime'] = pd.to_datetime(df.datetimestr,format = '%b %d, %Y, %I:%M %p')


nuser = len(df.user.unique())
cmap = cm.get_cmap('tab10', nuser)
colors = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    def draw_poly_patch(self):
        # rotate theta such that the first axis is at the top
        verts = unit_poly_verts(theta + np.pi / 2)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def __init__(self, *args, **kwargs):
            super(RadarAxes, self).__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta + np.pi / 2)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts


### emoji isolating functions
def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

def sortFreqDict(freqdict):
    aux = [(key,freqdict[key]) for key in freqdict]
    sorttuple = sorted(aux , key=lambda x: x[1],reverse=True)
    return sorttuple

def return_emoji(text):
    RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    allemojis = RE_EMOJI.findall(text)    
    allemojis = "".join("".join(i) for i in allemojis)    
    return allemojis


def wc_colorgen(word=None, font_size=None, position=None,  orientation=None,
                 font_path=None, random_state=None):
    hexcols = colors
    ncols = len(hexcols)
    wtlst = [1/ncols] * ncols
    
    # sample a hex color from a multinomial
    val = np.random.multinomial(1,wtlst,1)
    val = np.argmax(val.ravel())
    col = hexcols[val]
    
    # convert to hslformat
    rgb = matplotlib.colors.hex2color(col)
    hls = colorsys.rgb_to_hls(rgb[0],rgb[1],rgb[2])
    h = hls[0]*360
    s = hls[2]*100
    l = hls[1]*100    
    out = "hsl({}, {}%, {}%)".format(h, s, l)
    
    return out


######################################################################################################

## Generate WordClouds ###
stopwords = set(STOPWORDS)
stopwords.update(["go","going","probably","will","some","back","will"])

## Overall word cloud ###
text = " ".join(i for i in df.body)
textwc = WordCloud(background_color="black", width=1000,height=600,
               stopwords=stopwords,color_func=wc_colorgen ).generate(text)

## emoji word cloud
emojisonly = return_emoji(text)
emojifreqdict = wordListToFreqDict(emojisonly)

try:
    del emojifreqdict["🏽"]
except KeyError:
    print("Key not found")

emojiwc = WordCloud(background_color="black", width=700,height=600,
               font_path = "Symbola.ttf",color_func=wc_colorgen).generate_from_frequencies(emojifreqdict)

##########################################################################################################

## Generate data for message frequency by week plot ##
df['date'] = df['datetime'].dt.date
df['dow'] = df['datetime'].dt.dayofweek
df['hour'] = df.datetime.dt.hour

counts_user_date = df.groupby(["date","user"]).count()['body'].unstack(fill_value=0)

# generate dataframe of zeros for all dates with no messages and append 
def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta

datelst = list(counts_user_date.index)
alldatelst = list(perdelta(min(datelst),max(datelst),timedelta(days=1)))
alldays = [i.day for i in alldatelst]
allmos = [i.month for i in alldatelst]
xticklocs = list(compress(alldatelst,[alldays[i]==1 and allmos[i] in [1,7] for i in range(0,len(alldatelst))]))
xticklabs = [i.strftime("%m/%y") for i in xticklocs]

## Group data by week ##
df['datemin7'] = pd.to_datetime(df['date']) - pd.to_timedelta(7, unit='d')
counts_user_wk = df.groupby([pd.Grouper(key='datemin7', freq='W-MON'),'user']).count()['body'].unstack(fill_value=0)
wkdtlst = [i.to_pydatetime().date() for i in list(counts_user_wk.index)]

## fill in weeks that have zero messages for all users ##
counts_user_wk.index = wkdtlst
counts_user_wk = counts_user_wk.rename_axis('date').rename_axis('user', axis='columns') 
allwkdtlst = list(perdelta(min(wkdtlst),max(wkdtlst),timedelta(days=7)))
nomsgwks = list(compress(allwkdtlst,  [i not in wkdtlst for i in allwkdtlst])) 
nomsgwkdf = pd.DataFrame(dict([(i,[0]*len(nomsgwks)) for i in list(counts_user_wk.columns)]),index=nomsgwks)
nomsgwkdf = nomsgwkdf.rename_axis('date').rename_axis('user', axis='columns') 
counts_user_wk = pd.concat([counts_user_wk,nomsgwkdf])

##########################################################################################################

# generate radar plot data by hour
countsbyhr = df.groupby(['hour','user']).count()['body'].unstack(fill_value=0)
hrlst = list(countsbyhr.index)
allhrlst = list(perdelta(0,24,1))

# fill in any hours that have no messages #
nomsghrs = list(compress(allhrlst,  [i not in hrlst for i in allhrlst]))
if(len(nomsghrs) > 0): 
    nomsgdf = pd.DataFrame(dict([(i,[0]*len(nomsghrs)) for i in list(countsbyhr.columns)]),index=nomsghrs)
    nomsgdf = nomsgdf.rename_axis('hour').rename_axis('user', axis='columns')
    countsbyhr = pd.concat([countsbyhr,nomsgdf])


outlst = []
for i in list(countsbyhr.columns):
    userlst = list(countsbyhr[i])[::-1]
    userlst.insert(0,userlst.pop(len(userlst)-1))
    outlst.append(userlst)

datahod = [
    ['12am','11pm','10pm','9pm','8pm','7pm','6pm','5pm','4pm','3pm','2pm','1pm',
     '12pm','11am','10am','9am','8am','7am','6am','5am','4am','3am','2am','1am'],
    ('',outlst)  ]    

N_hod = len(datahod[0])
theta_hod = radar_factory(N_hod, frame='circle')
spoke_labels_hod = datahod.pop(0)


# generate radar plot data by dow
counts_user_dow = df.groupby(["dow","user"]).count()['body'].unstack(fill_value=0)
dowlst = list(counts_user_dow.index)
alldowlst = list(perdelta(0,7,1))
nomsgdow = list(compress(alldowlst,  [i not in dowlst for i in alldowlst]))

# fill in days of week that have no messages
if(len(nomsgdow) > 0): 
    nomsgdf = pd.DataFrame(dict([(i,[0]*len(nomsgdow)) for i in list(counts_user_dow.columns)]),index=nomsgdow)
    nomsgdf = nomsgdf.rename_axis('dow').rename_axis('user', axis='columns')
    counts_user_dow = pd.concat([counts_user_dow,nomsgdf])

outlst = []
for i in list(counts_user_dow.columns):
    userlst = list(counts_user_dow[i])[::-1]
    outlst.append(userlst)

datadow = [
    ['Sunday','Saturday','Friday','Thursday','Wednesday','Tuesday','Monday'],
    ('By Day of Week',outlst)]

N_dow = len(datadow[0])
theta_dow = radar_factory(N_dow, frame='circle')
spoke_labels_dow = datadow.pop(0)


##########################################################################################################

# Word/Message Statistics
df['nwords'] = [len(i.split()) for i in df['body']]
totmess = 'Total Msgs: ' + str(df.shape[0])
totdays = 'Total Days: ' + str((max(df.date)-min(df.date)).days)
totwrds = 'Total Words: ' + str(sum(df.nwords))
msgperday = 'Msgs/Day: ' + str(round(df.shape[0]/(max(df.date)-min(df.date)).days,2))
wordpermsg = 'Words/Msg: ' + str(round(sum(df.nwords)/df.shape[0],2))
totemoji = 'Total Emojis: ' + str(len(emojisonly))

##########################################################################################################


####### Build the Plot #########

# arrange figures
globgrey = '#c2c2c2'

fig = plt.figure(figsize=(10,15))

ax0 = fig.add_axes([0,0.91,1,0.09]) # title
ax1 = fig.add_axes([0.15,0.58,0.7,0.32]) # word cloud
ax2 = fig.add_axes([0.04, 0.45, 0.93, 0.12])# lineplot
ax3 = fig.add_axes([0.03,0.21,0.45,0.17],projection="radar")# dow radar
ax4 = fig.add_axes([0.52,0.21,0.45,0.17],projection="radar")# hod radar
ax5 = fig.add_axes([0.35,0.02,0.3,0.16])# emoji cloud
ax6 = fig.add_axes([0.68,0.02,0.3,0.16])   # text summaries (1)
ax7 = fig.add_axes([0.02,0.02,0.3,0.16]) # text summaries (2) 

for ax, color in zip([ax2, ax3, ax4], [globgrey,globgrey,globgrey]):
    plt.setp(ax.spines.values(), color=color)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color)

# title
ax0.set(xlim=[0,1],ylim=[0,1])
ax0.text(0.5, 0.1,"Message Visualization",fontsize=60,horizontalalignment='center',color=globgrey)
ax0.get_xaxis().set_visible(False)
ax0.axes.get_yaxis().set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.spines['left'].set_visible(False)


# main word cloud
ax1.imshow(textwc, interpolation='bilinear')
ax1.axis('off')

# week line plot
weekplot = counts_user_wk.plot(ax=ax2,color=colors)
weekplot.tick_params(length=0)
weekplot.spines['top'].set_visible(False)
weekplot.spines['right'].set_visible(False)
leg = weekplot.legend(frameon=False)
for text in leg.get_texts():
    plt.setp(text, color = globgrey)
weekplot.set_xlabel("")
# weekplot.set_xticks(xticklocs, minor=False)
# weekplot.set_xticklabels(xticklabs,size='large')
weekplot.set_title('Message Frequency By Week', size='xx-large',horizontalalignment='center',
                   position=(0.5, 0.88),color=globgrey)

# radar 1 - day of week
ax3.set_title('By Day of Week', size='xx-large', position=(0.5, 1.15),
             horizontalalignment='center', verticalalignment='center',color=globgrey)
# ax3.set_ylim(0,1400)
# ax3.set_rgrids([200,600,1000,1400])
for d, color in zip(datadow[0][1], colors):
    ax3.plot(theta_dow, d, color=color)
    ax3.fill(theta_dow, d, facecolor=color, alpha=0.5)

ax3.set_thetagrids(np.degrees(theta_dow),spoke_labels_dow)

# radar 2 - hour of day 
ax4.set_title('By Hour of Day', size='xx-large', position=(0.5, 1.15),
             horizontalalignment='center', verticalalignment='center',color=globgrey)
# ax4.set_ylim(0,700)
# ax4.set_rgrids([100,300,500,700],angle=340)
for d, color in zip(datahod[0][1], colors):
    ax4.plot(theta_hod, d, color=color)
    ax4.fill(theta_hod, d, facecolor=color, alpha=0.5)

ax4.set_thetagrids(np.degrees(theta_hod),spoke_labels_hod)

# emoji word cloud
ax5.imshow(emojiwc,interpolation='bilinear')
ax5.axis('off')

# text (1)
fsize = 22
ax6.set(xlim=[0,1],ylim=[0,1])
ax6.axis('off')
ax6.text(0.5, 0.75, msgperday, fontsize=fsize,color=globgrey, horizontalalignment='center')
ax6.text(0.5, 0.45, wordpermsg, fontsize=fsize,color=globgrey, horizontalalignment='center')
ax6.text(0.5, 0.15, totemoji, fontsize=fsize,color=globgrey, horizontalalignment='center')


# text (2)
ax7.set(xlim=[0,1],ylim=[0,1])
ax7.axis('off')
ax7.text(0.5, 0.75, totdays, fontsize=fsize,color=globgrey, horizontalalignment='center')
ax7.text(0.5, 0.45, totmess, fontsize=fsize,color=globgrey, horizontalalignment='center')
ax7.text(0.5, 0.15, totwrds, fontsize=fsize,color=globgrey, horizontalalignment='center')

################################################################################################

# save the figure
fig.savefig(path.join(dirnm,filepref) +'.png',dpi=500)

