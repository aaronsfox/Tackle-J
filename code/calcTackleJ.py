# -*- coding: utf-8 -*-
"""

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    Python script for Big Data Bowl 2024 entry calculating Tackle-J and associated
    metrics. This script uses the associated helperFuncs.py script for visualisations.
    
    This script was run with Python 3.9.12


"""

# %% Import packages

#Import base Python packages
#Note that variablepackage versions may result in slightly different outcomes,
#or maybe not work at all!
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage, TextArea)
import seaborn as sns
import nfl_data_py as nfl
import math
import os
import pickle

#Import helper functions
from helperFuncs import createField, drawPlay

# %% Set-up

#Set a boolean value to re-analyse data or simply load from dictionary
#Default is False --- swap to True if wishing to re-run analysis
calcTackleJ = False

#Set matplotlib parameter preferences
from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 16
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'
rcParams['legend.fontsize'] = 10
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['legend.framealpha'] = 0.0
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'pdf'

#Add custom fonts for use with matplotlib
fontDir = [os.getcwd()+os.sep+os.path.join('..','fonts')]
for font in font_manager.findSystemFonts(fontDir):
    font_manager.fontManager.addfont(font)

#Import team details from nfl_data_py
teamData = nfl.import_team_desc()

#Import 2022 season roster details
rosterData = nfl.import_seasonal_rosters([2022])

# %% Create cropped headshots of players

#Note that this doesn't need to be re-run more than once
#It takes a while, so isn't really useful to run again
#Change this flag if re-processing images is desired
cropPlayerImages = False

#Check whether to crop images again
if cropPlayerImages:

    #Import and run the helper function
    from helperFuncs import cropPlayerImg
    cropPlayerImg(rosterData, os.path.join('..','img','player'))
    
# %% Download team images

#Note that this doesn't need to be re-run more than once
#It doesn't take long, but it's not really necessary to do again
#Change this flag if re-processing images is desired
getTeamImages = False

#Check whether to download team images again
if getTeamImages:

    #Import and run the helper function
    from helperFuncs import downloadTeamImages
    downloadTeamImages(teamData, os.path.join('..','img','team'))

# %% Read in data files

#Read in the games data
games = pd.read_csv(os.path.join('..','data','games.csv'))

#Read in play description data
playDescriptions = pd.read_csv(os.path.join('..','data','plays.csv'))

#Read in tackles data
tackles = pd.read_csv(os.path.join('..','data','tackles.csv'))

#Load player data 
players = pd.read_csv(os.path.join('..','data','players.csv'))

#Create a list of the set weeks tracking data is available for
weekList = ['week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7', 'week_8', 'week_9']

#Create a list to store dataframes in
trackingData = []

#Loop through weeks
for currWeek in weekList:

    #Read in the extracted tracking and model data
    trackingData.append(pd.read_csv(os.path.join('..','data',f'tracking_{currWeek}.csv')))

    #Join player positioning information onto a week's worth of tracking data 
    trackingData[-1] = trackingData[-1].merge(players.loc[:, ['nflId', 'position']], how = 'left')

# %% Identify the solo tackles to apply the Tackle-J metric to

#Set list for solo tackle Id's
soloTackleIds = []

#Loop through tackle dataframe and extract game and play Id for solo tackles
for ii in tackles.index:
    
    #Check for solo conditions
    if tackles.iloc[ii]['tackle'] == 1 and \
        tackles.iloc[ii]['assist'] == 0 and \
            tackles.iloc[ii]['pff_missedTackle'] == 0:
                #Get the game and play id
                soloTackleIds.append((tackles.iloc[ii]['gameId'], tackles.iloc[ii]['playId']))
                
# #Display number of tackles extracted
# print(f'Total number of solo tackles: {len(soloTackleIds)}')

# %% Extract kinetic energy data from solo tackle plays

#Check whether to calculate or load from file
if calcTackleJ:

    #Set up a dictionary to store the calculated kinetic energy and event data
    jData = {'gameId': [], 'playId': [], 'week': [],
             'ballCarrierId': [], 'ballCarrierPosition': [], 'ballCarrierKE': [],
             'tacklerId': [], 'tacklerPosition': [], 'tacklerKE': [],
             'contactToTackleTime': [], 'tackleEvent': [],
             'tackleJ': [], 'tackleJ_diff': [], 'tackleJ_loss': []}
    
    #Loop through solo tackles
    for ii in range(len(soloTackleIds)):
        
        #Set game and play Id to variables for ease of use
        gameId = soloTackleIds[ii][0]
        playId = soloTackleIds[ii][1]
        
        #Get the play week for the current game Id
        playWeek = games.loc[games['gameId'] == gameId,]['week'].values[0]
        
        #Get the current play description
        playDesc = playDescriptions.loc[(playDescriptions['gameId'] == gameId) &
                                        (playDescriptions['playId'] == playId),].copy()
        
        #Extract the processed data for the current play
        play = trackingData[playWeek-1].loc[(trackingData[playWeek-1]['gameId'] == gameId) &
                                            (trackingData[playWeek-1]['playId'] == playId),].copy()
        
        #There are instances where some plays don't actually have tackles
        #Avoid these with a check
        if 'tackle' in list(play['event'].unique()):
            
            # #Print out play description
            # print(playDesc['playDescription'].values[0])
            
            ##### BALL CARRIER DATA #####
            
            #Get the ball carrier Id on the play
            ballCarrierId = playDesc['ballCarrierId'].values[0]
        
            #Get ball carrier mass from the player data - convert to kg because this isn't America
            ballCarrierMass = players.loc[players['nflId'] == ballCarrierId,]['weight'].values[0] / 2.2
            
            #Get the ball carrier position
            ballCarrierPosition = players.loc[players['nflId'] == ballCarrierId,]['position'].values[0]
        
            #Get tracking data in a separate frame for ease of use
            ballCarrierTracking = play.loc[play['nflId'] == ballCarrierId,].copy()
        
            #Convert speed values to meters per second because this isn't America
            ballCarrierTracking['ms2'] = ballCarrierTracking['s'] / 1.0936132983
        
            #Add kinetic energy to the dataframe
            ballCarrierTracking['ke'] = 0.5 * ballCarrierMass * ballCarrierTracking['ms2']**2
            
            ##### TACKLER DATA #####
            
            #Get the current play tackle data
            playTackle = tackles.loc[(tackles['gameId'] == gameId) &
                                     (tackles['playId'] == playId),].copy()
        
            #Get tacklers player Id
            tacklerId = playTackle['nflId'].values[0]
        
            #Get tacklers mass from the player data - convert to kg
            tacklerMass = players.loc[players['nflId'] == tacklerId,]['weight'].values[0] / 2.2
        
            #Get the tackler position
            tacklerPosition = players.loc[players['nflId'] == tacklerId,]['position'].values[0]
        
            #Get tacklers tracking data in a separate frame for ease of use
            tacklerTracking = play.loc[play['nflId'] == tacklerId,].copy()
        
            #Convert speed values to meters per second because this isn't America
            tacklerTracking['ms2'] = tacklerTracking['s'] / 1.0936132983
        
            #Add kinetic energy to the dataframe
            tacklerTracking['ke'] = 0.5 * tacklerMass * tacklerTracking['ms2']**2
            
            #Get the tackle event (e.g. tackle, out of bounds) for the play
            if 'tackle' in list(ballCarrierTracking['event'].unique()):
                tackleEvent = 'tackle'
            elif 'out_of_bounds' in list(ballCarrierTracking['event'].unique()):
                tackleEvent = 'out_of_bounds'
                
            #Check to see whether there is a first contact event
            contactEvent = False
            if 'first_contact' in list(ballCarrierTracking['event'].unique()):
                contactEvent = True
            
            #Get kinetic energy of ball carrier at first contact or tackle (Tackle-J)
            if contactEvent:
                tackleJ = ballCarrierTracking.loc[ballCarrierTracking['event'] == 'first_contact',
                                                  ]['ke'].values[0]
            else:
                tackleJ = ballCarrierTracking.loc[ballCarrierTracking['event'] == 'tackle',
                                                  ]['ke'].values[0]      
        
            #Get differential in kinetic energy between tackler and ball carrier at first contact or tackle (Tackle-J diff)
            if contactEvent:
                tacklerKE_firstContact = tacklerTracking.loc[tacklerTracking['event'] == 'first_contact',
                                                             ]['ke'].values[0]
            else:
                tacklerKE_firstContact = tacklerTracking.loc[tacklerTracking['event'] == 'tackle',
                                                             ]['ke'].values[0]
            tackleJ_diff = tacklerKE_firstContact - tackleJ
        
            #Get the loss rate (J/sec) of kinetic energy from first contact to tackle
            #This is only calculated if there is a first contact to tackle/OoB event
            if contactEvent:
                #Get kinetic energy data
                keData = np.array((ballCarrierTracking.loc[ballCarrierTracking['event'] == 'first_contact',
                                                           ]['ke'].values[0],
                                   ballCarrierTracking.loc[ballCarrierTracking['event'] == tackleEvent,
                                                                              ]['ke'].values[0]))
                #Get the difference in time based on frames
                frameDiff = ballCarrierTracking.loc[ballCarrierTracking['event'] == tackleEvent,
                                                    ].index.values[0] - ballCarrierTracking.loc[ballCarrierTracking['event'] == 'first_contact',
                                                                                                ].index.values[0]
                timeDiff = frameDiff * 0.1
                #Calculate the loss rate
                tackleJ_loss = (keData[0] - keData[1]) / timeDiff
            else:
                tackleJ_loss = np.nan
                timeDiff = np.nan
            
            #Store data in dictionary
            #Play details
            jData['gameId'].append(gameId)
            jData['playId'].append(playId)
            jData['week'].append(playWeek)
            jData['contactToTackleTime'].append(timeDiff)
            jData['tackleEvent'].append(tackleEvent)
            #Ball carrier data
            jData['ballCarrierId'].append(ballCarrierId)
            jData['ballCarrierPosition'].append(ballCarrierPosition)
            jData['ballCarrierKE'].append(ballCarrierTracking['ke'].to_numpy())
            #Tackler data
            jData['tacklerId'].append(tacklerId)
            jData['tacklerPosition'].append(tacklerPosition)
            jData['tacklerKE'].append(tacklerTracking['ke'].to_numpy())
            #Metrics
            jData['tackleJ'].append(tackleJ)
            jData['tackleJ_diff'].append(tackleJ_diff)
            jData['tackleJ_loss'].append(tackleJ_loss)
            
    #Save to pickle file
    with open(os.path.join('..','outputs','data','jData.pkl'), 'wb') as pklFile:
        pickle.dump(jData, pklFile)
            
else:
    
    #Load data directly from file
    with open(os.path.join('..','outputs','data','jData.pkl'), 'rb') as pklFile:
        jData = pickle.load(pklFile)

#Convert dictionary to dataframe for easier analysis
jData_df = pd.DataFrame.from_dict(jData)

# %% Review some particular cases of interest

# %% Examine what the top 5 Tackle-J plays look like

#Change this flag to re-process the top plays here
reRunTopPlays = False

#Check to re-run
if reRunTopPlays:

    #Sort by Tackle-J and extract the top 5 plays
    top5_tackleJ = jData_df.sort_values(by = 'tackleJ', ascending = False).copy().iloc[0:5].reset_index(drop = True)
    
    #Create a text file to print play descriptions to
    #Check and remove file if it already exists
    if os.path.isfile(os.path.join('..','outputs','gif','topTackleJ','playDescription.txt')):
        os.remove(os.path.join('..','outputs','gif','topTackleJ','playDescription.txt'))
    #Create the text file
    playDescriptionsFile = open(os.path.join('..','outputs','gif','topTackleJ','playDescriptions.txt'), 'w')
    # playDescriptionsFile.close()
    
    #Loop through and create animations of each play
    for ii in top5_tackleJ.index:
        
        #Set game and play Id's
        gameId = top5_tackleJ.iloc[ii]['gameId']
        playId = top5_tackleJ.iloc[ii]['playId']
    
        #Get the play week for the current game Id
        playWeek = games.loc[games['gameId'] == gameId,]['week'].values[0]
    
        #Get the current play description
        playDesc = playDescriptions.loc[(playDescriptions['gameId'] == gameId) &
                                        (playDescriptions['playId'] == playId),].copy()
    
        #Extract the processed data for the current play
        play = trackingData[playWeek-1].loc[(trackingData[playWeek-1]['gameId'] == gameId) &
                                            (trackingData[playWeek-1]['playId'] == playId),].copy()
    
        #Print out play description to text file
        # with open(os.path.join('..','outputs','gif','topTackleJ','playDescriptions.txt'), 'w') as f:
        #     f.write(playDesc['playDescription'].values[0]+'\n')
        playDescriptionsFile.write(playDesc['playDescription'].values[0]+'\n')
    
        #### VISUALISE PLAY #####
    
        #Get line of scrimmage and first down mark necessary
        if play['playDirection'].unique()[0] == 'left':
            #Get line of scrimmage and then invert
            lineOfScrimmage = playDesc['absoluteYardlineNumber'].to_numpy()[0]
            firstDownMark = lineOfScrimmage - playDesc['yardsToGo'].to_numpy()[0]
        else:
            #Get line of scrimmage
            lineOfScrimmage = playDesc['absoluteYardlineNumber'].to_numpy()[0]
            firstDownMark = lineOfScrimmage + playDesc['yardsToGo'].to_numpy()[0]
            
        #Animate play to view
    
        #Set home and away team
        homeTeam = games.loc[games['gameId'] == gameId,]['homeTeamAbbr'].values[0]
        awayTeam = games.loc[games['gameId'] == gameId,]['visitorTeamAbbr'].values[0]
    
        #Create the field to plot on
        fieldFig, fieldAx = plt.subplots(figsize=(14, 6.5))
        createField(fieldFig, fieldAx, 
                    lineOfScrimmage = lineOfScrimmage, firstDownMark = firstDownMark,
                    homeTeamAbbr = homeTeam, awayTeamAbbr = awayTeam, teamData = teamData)
    
        #Identify the frame range from play
        startFrame = play['frameId'].unique().min()
        endFrame = play['frameId'].unique().max()
    
        #Set ball carrier Id
        ballCarrierId = playDesc['ballCarrierId'].values[0]
    
        #Set tackler Id
        tacklerId = tackles.loc[(tackles['gameId'] == gameId) &
                                (tackles['playId'] == playId),
                                ]['nflId'].values[0]
    
        #Run animation function
        anim = animation.FuncAnimation(fieldFig, drawPlay,
                                       frames = range(startFrame,endFrame+1), repeat = False,
                                       fargs = (homeTeam, awayTeam, teamData, play, 'pos',
                                                ballCarrierId, tacklerId,
                                                lineOfScrimmage, firstDownMark))
    
        #Write to GIF file
        gifWriter = animation.PillowWriter(fps = 60)
        anim.save(os.path.join('..','outputs','gif','topTackleJ',f'rank{ii+1}_game{gameId}_play{playId}.gif'), 
                  dpi = 150, writer = gifWriter)
        
        #Close figure after pause to avoid error
        plt.pause(1)
        plt.close()
    
    #Close play descriptions file
    playDescriptionsFile.close()
    
# %% Examine what the bottom 5 Tackle-J plays look like

#Change this flag to re-process the top plays here
reRunTopPlays = False

#Check to re-run
if reRunTopPlays:

    #Sort by Tackle-J and extract the top 5 plays
    bottom5_tackleJ = jData_df.sort_values(by = 'tackleJ', ascending = True).copy().iloc[0:5].reset_index(drop = True)
    
    #Create a text file to print play descriptions to
    #Check and remove file if it already exists
    if os.path.isfile(os.path.join('..','outputs','gif','bottomTackleJ','playDescription.txt')):
        os.remove(os.path.join('..','outputs','gif','bottomTackleJ','playDescription.txt'))
    #Create the text file
    playDescriptionsFile = open(os.path.join('..','outputs','gif','bottomTackleJ','playDescriptions.txt'), 'w')
    # playDescriptionsFile.close()
    
    #Loop through and create animations of each play
    for ii in bottom5_tackleJ.index:
        
        #Set game and play Id's
        gameId = bottom5_tackleJ.iloc[ii]['gameId']
        playId = bottom5_tackleJ.iloc[ii]['playId']
    
        #Get the play week for the current game Id
        playWeek = games.loc[games['gameId'] == gameId,]['week'].values[0]
    
        #Get the current play description
        playDesc = playDescriptions.loc[(playDescriptions['gameId'] == gameId) &
                                        (playDescriptions['playId'] == playId),].copy()
    
        #Extract the processed data for the current play
        play = trackingData[playWeek-1].loc[(trackingData[playWeek-1]['gameId'] == gameId) &
                                            (trackingData[playWeek-1]['playId'] == playId),].copy()
    
        #Print out play description to text file
        # with open(os.path.join('..','outputs','gif','bottomTackleJ','playDescriptions.txt'), 'w') as f:
        #     f.write(playDesc['playDescription'].values[0]+'\n')
        playDescriptionsFile.write(playDesc['playDescription'].values[0]+'\n')
    
        #### VISUALISE PLAY #####
    
        #Get line of scrimmage and first down mark necessary
        if play['playDirection'].unique()[0] == 'left':
            #Get line of scrimmage and then invert
            lineOfScrimmage = playDesc['absoluteYardlineNumber'].to_numpy()[0]
            firstDownMark = lineOfScrimmage - playDesc['yardsToGo'].to_numpy()[0]
        else:
            #Get line of scrimmage
            lineOfScrimmage = playDesc['absoluteYardlineNumber'].to_numpy()[0]
            firstDownMark = lineOfScrimmage + playDesc['yardsToGo'].to_numpy()[0]
            
        #Animate play to view
    
        #Set home and away team
        homeTeam = games.loc[games['gameId'] == gameId,]['homeTeamAbbr'].values[0]
        awayTeam = games.loc[games['gameId'] == gameId,]['visitorTeamAbbr'].values[0]
    
        #Create the field to plot on
        fieldFig, fieldAx = plt.subplots(figsize=(14, 6.5))
        createField(fieldFig, fieldAx, 
                    lineOfScrimmage = lineOfScrimmage, firstDownMark = firstDownMark,
                    homeTeamAbbr = homeTeam, awayTeamAbbr = awayTeam, teamData = teamData)
    
        #Identify the frame range from play
        startFrame = play['frameId'].unique().min()
        endFrame = play['frameId'].unique().max()
    
        #Set ball carrier Id
        ballCarrierId = playDesc['ballCarrierId'].values[0]
    
        #Set tackler Id
        tacklerId = tackles.loc[(tackles['gameId'] == gameId) &
                                (tackles['playId'] == playId),
                                ]['nflId'].values[0]
    
        #Run animation function
        anim = animation.FuncAnimation(fieldFig, drawPlay,
                                       frames = range(startFrame,endFrame+1), repeat = False,
                                       fargs = (homeTeam, awayTeam, teamData, play, 'pos',
                                                ballCarrierId, tacklerId,
                                                lineOfScrimmage, firstDownMark))
    
        #Write to GIF file
        gifWriter = animation.PillowWriter(fps = 60)
        anim.save(os.path.join('..','outputs','gif','bottomTackleJ',f'rank{ii+1}_game{gameId}_play{playId}.gif'), 
                  dpi = 150, writer = gifWriter)
        
        #Close figure after pause to avoid error
        plt.pause(1)
        plt.close()
    
    #Close play descriptions file
    playDescriptionsFile.close()
    
# %% Examine what the top 5 positive Tackle-J difference plays look like

#Change this flag to re-process the top plays here
reRunTopPlays = False

#Check to re-run
if reRunTopPlays:

    #Sort by Tackle-J and extract the top 5 plays
    top5_tackleJ_posDiff = jData_df.sort_values(by = 'tackleJ_diff', ascending = False).copy().iloc[0:5].reset_index(drop = True)
    
    #Create a text file to print play descriptions to
    #Check and remove file if it already exists
    if os.path.isfile(os.path.join('..','outputs','gif','topPosTackleJ_difference','playDescription.txt')):
        os.remove(os.path.join('..','outputs','gif','topPosTackleJ_difference','playDescription.txt'))
    #Create the text file
    playDescriptionsFile = open(os.path.join('..','outputs','gif','topPosTackleJ_difference','playDescriptions.txt'), 'w')
    # playDescriptionsFile.close()
    
    #Loop through and create animations of each play
    for ii in top5_tackleJ_posDiff.index:
        
        #Set game and play Id's
        gameId = top5_tackleJ_posDiff.iloc[ii]['gameId']
        playId = top5_tackleJ_posDiff.iloc[ii]['playId']
    
        #Get the play week for the current game Id
        playWeek = games.loc[games['gameId'] == gameId,]['week'].values[0]
    
        #Get the current play description
        playDesc = playDescriptions.loc[(playDescriptions['gameId'] == gameId) &
                                        (playDescriptions['playId'] == playId),].copy()
    
        #Extract the processed data for the current play
        play = trackingData[playWeek-1].loc[(trackingData[playWeek-1]['gameId'] == gameId) &
                                            (trackingData[playWeek-1]['playId'] == playId),].copy()
    
        #Print out play description to text file
        # with open(os.path.join('..','outputs','gif','topPosTackleJ_difference','playDescriptions.txt'), 'w') as f:
        #     f.write(playDesc['playDescription'].values[0]+'\n')
        playDescriptionsFile.write(playDesc['playDescription'].values[0]+'\n')
    
        #### VISUALISE PLAY #####
    
        #Get line of scrimmage and first down mark necessary
        if play['playDirection'].unique()[0] == 'left':
            #Get line of scrimmage and then invert
            lineOfScrimmage = playDesc['absoluteYardlineNumber'].to_numpy()[0]
            firstDownMark = lineOfScrimmage - playDesc['yardsToGo'].to_numpy()[0]
        else:
            #Get line of scrimmage
            lineOfScrimmage = playDesc['absoluteYardlineNumber'].to_numpy()[0]
            firstDownMark = lineOfScrimmage + playDesc['yardsToGo'].to_numpy()[0]
            
        #Animate play to view
    
        #Set home and away team
        homeTeam = games.loc[games['gameId'] == gameId,]['homeTeamAbbr'].values[0]
        awayTeam = games.loc[games['gameId'] == gameId,]['visitorTeamAbbr'].values[0]
    
        #Create the field to plot on
        fieldFig, fieldAx = plt.subplots(figsize=(14, 6.5))
        createField(fieldFig, fieldAx, 
                    lineOfScrimmage = lineOfScrimmage, firstDownMark = firstDownMark,
                    homeTeamAbbr = homeTeam, awayTeamAbbr = awayTeam, teamData = teamData)
    
        #Identify the frame range from play
        startFrame = play['frameId'].unique().min()
        endFrame = play['frameId'].unique().max()
    
        #Set ball carrier Id
        ballCarrierId = playDesc['ballCarrierId'].values[0]
    
        #Set tackler Id
        tacklerId = tackles.loc[(tackles['gameId'] == gameId) &
                                (tackles['playId'] == playId),
                                ]['nflId'].values[0]
    
        #Run animation function
        anim = animation.FuncAnimation(fieldFig, drawPlay,
                                       frames = range(startFrame,endFrame+1), repeat = False,
                                       fargs = (homeTeam, awayTeam, teamData, play, 'pos',
                                                ballCarrierId, tacklerId,
                                                lineOfScrimmage, firstDownMark))
    
        #Write to GIF file
        gifWriter = animation.PillowWriter(fps = 60)
        anim.save(os.path.join('..','outputs','gif','topPosTackleJ_difference',f'rank{ii+1}_game{gameId}_play{playId}.gif'), 
                  dpi = 150, writer = gifWriter)
        
        #Close figure after pause to avoid error
        plt.pause(1)
        plt.close()
        
    #Close play descriptions file
    playDescriptionsFile.close()
    
# %% Examine what the top 5 negative Tackle-J difference plays look like

#Change this flag to re-process the top plays here
reRunTopPlays = False

#Check to re-run
if reRunTopPlays:

    #Sort by Tackle-J and extract the top 5 plays
    top5_tackleJ_negDiff = jData_df.sort_values(by = 'tackleJ_diff', ascending = True).copy().iloc[0:5].reset_index(drop = True)
    
    #Create a text file to print play descriptions to
    #Check and remove file if it already exists
    if os.path.isfile(os.path.join('..','outputs','gif','topNegTackleJ_difference','playDescription.txt')):
        os.remove(os.path.join('..','outputs','gif','topNegTackleJ_difference','playDescription.txt'))
    #Create the text file
    playDescriptionsFile = open(os.path.join('..','outputs','gif','topNegTackleJ_difference','playDescriptions.txt'), 'w')
    # playDescriptionsFile.close()
    
    #Loop through and create animations of each play
    for ii in top5_tackleJ_negDiff.index:
        
        #Set game and play Id's
        gameId = top5_tackleJ_negDiff.iloc[ii]['gameId']
        playId = top5_tackleJ_negDiff.iloc[ii]['playId']
    
        #Get the play week for the current game Id
        playWeek = games.loc[games['gameId'] == gameId,]['week'].values[0]
    
        #Get the current play description
        playDesc = playDescriptions.loc[(playDescriptions['gameId'] == gameId) &
                                        (playDescriptions['playId'] == playId),].copy()
    
        #Extract the processed data for the current play
        play = trackingData[playWeek-1].loc[(trackingData[playWeek-1]['gameId'] == gameId) &
                                            (trackingData[playWeek-1]['playId'] == playId),].copy()
    
        #Print out play description to text file
        # with open(os.path.join('..','outputs','gif','topNegTackleJ_difference','playDescriptions.txt'), 'w') as f:
        #     f.write(playDesc['playDescription'].values[0]+'\n')
        playDescriptionsFile.write(playDesc['playDescription'].values[0]+'\n')
    
        #### VISUALISE PLAY #####
    
        #Get line of scrimmage and first down mark necessary
        if play['playDirection'].unique()[0] == 'left':
            #Get line of scrimmage and then invert
            lineOfScrimmage = playDesc['absoluteYardlineNumber'].to_numpy()[0]
            firstDownMark = lineOfScrimmage - playDesc['yardsToGo'].to_numpy()[0]
        else:
            #Get line of scrimmage
            lineOfScrimmage = playDesc['absoluteYardlineNumber'].to_numpy()[0]
            firstDownMark = lineOfScrimmage + playDesc['yardsToGo'].to_numpy()[0]
            
        #Animate play to view
    
        #Set home and away team
        homeTeam = games.loc[games['gameId'] == gameId,]['homeTeamAbbr'].values[0]
        awayTeam = games.loc[games['gameId'] == gameId,]['visitorTeamAbbr'].values[0]
    
        #Create the field to plot on
        fieldFig, fieldAx = plt.subplots(figsize=(14, 6.5))
        createField(fieldFig, fieldAx, 
                    lineOfScrimmage = lineOfScrimmage, firstDownMark = firstDownMark,
                    homeTeamAbbr = homeTeam, awayTeamAbbr = awayTeam, teamData = teamData)
    
        #Identify the frame range from play
        startFrame = play['frameId'].unique().min()
        endFrame = play['frameId'].unique().max()
    
        #Set ball carrier Id
        ballCarrierId = playDesc['ballCarrierId'].values[0]
    
        #Set tackler Id
        tacklerId = tackles.loc[(tackles['gameId'] == gameId) &
                                (tackles['playId'] == playId),
                                ]['nflId'].values[0]
    
        #Run animation function
        anim = animation.FuncAnimation(fieldFig, drawPlay,
                                       frames = range(startFrame,endFrame+1), repeat = False,
                                       fargs = (homeTeam, awayTeam, teamData, play, 'pos',
                                                ballCarrierId, tacklerId,
                                                lineOfScrimmage, firstDownMark))
    
        #Write to GIF file
        gifWriter = animation.PillowWriter(fps = 60)
        anim.save(os.path.join('..','outputs','gif','topNegTackleJ_difference',f'rank{ii+1}_game{gameId}_play{playId}.gif'), 
                  dpi = 150, writer = gifWriter)
        
        #Close figure after pause to avoid error
        plt.pause(1)
        plt.close()

    #Close play descriptions file
    playDescriptionsFile.close()
    
# %% Identify and present top Tackle-J offensive and defensive players

#Extract the average tackler Tackle-J values
averageTackleJ_tackler = jData_df.groupby('tacklerId')['tackleJ'].describe().reset_index(drop = False)

#Remove any players with less than 10 tackles
averageTackleJ_tackler = averageTackleJ_tackler.loc[averageTackleJ_tackler['count'] >= 10,]

#Extract the top 10 tackle-J tacklers
top10_tackleJ_tackler = averageTackleJ_tackler.sort_values(
    by = 'mean', ascending = False).iloc[0:10]['tacklerId'].to_list()

#Extract the average ball carrier Tackle-J values
averageTackleJ_ballCarrier = jData_df.groupby('ballCarrierId')['tackleJ'].describe().reset_index(drop = False)

#Remove any players with less than 25 carries
averageTackleJ_ballCarrier = averageTackleJ_ballCarrier.loc[averageTackleJ_ballCarrier['count'] >= 25,]

#Extract the top 10 tackle-J tacklers
top10_tackleJ_ballCarrier = averageTackleJ_ballCarrier.sort_values(
    by = 'mean', ascending = False).iloc[0:10]['ballCarrierId'].to_list()

#Print out top 10 players in tackler category
print('\nTop-10 Tackle-J Tacklers')
for ii in range(10):
    #Get details
    displayName = rosterData.loc[rosterData['gsis_it_id'] == str(top10_tackleJ_tackler[ii]),['player_name']].values[0][0]
    team = rosterData.loc[rosterData['gsis_it_id'] == str(top10_tackleJ_tackler[ii]),['team']].values[0][0]
    position = rosterData.loc[rosterData['gsis_it_id'] == str(top10_tackleJ_tackler[ii]),['position']].values[0][0]
    #Print details
    print(f'#{ii+1}: {displayName} ({team}, {position})')
    
#Print out top 10 players in ball carrier category
print('\nTop-10 Tackle-J Ball Carriers')
for ii in range(10):
    #Get details
    displayName = rosterData.loc[rosterData['gsis_it_id'] == str(top10_tackleJ_ballCarrier[ii]),['player_name']].values[0][0]
    team = rosterData.loc[rosterData['gsis_it_id'] == str(top10_tackleJ_ballCarrier[ii]),['team']].values[0][0]
    position = rosterData.loc[rosterData['gsis_it_id'] == str(top10_tackleJ_ballCarrier[ii]),['position']].values[0][0]
    #Print details
    print(f'#{ii+1}: {displayName} ({team}, {position})')

#Extract the data from top 10 players from the full dataset
jData_top10_tackleJ_tackler = jData_df.loc[jData_df['tacklerId'].isin(top10_tackleJ_tackler)]
jData_top10_tackleJ_ballCarrier = jData_df.loc[jData_df['ballCarrierId'].isin(top10_tackleJ_ballCarrier)]

#Create figure and axes
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,7),
                       sharex = False, sharey = True)

#Set subplot spacing
plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.2, top = 0.85,
                    wspace = 0.1)

#Add figure title
fig.text(0.01, 0.97,
         'Top-10 Performing Tacklers and Ball Carriers for Tackle-$J$',
         font = 'Arial', fontsize = 20,
         ha = 'left', va = 'center')

#Add descriptive text
fig.text(0.01, 0.93,
         'Players identified by highest average Tackle-$J$ in the dataset (minimum 10 tackles or 25 combined carries and receptions)',
         font = 'Arial', fontsize = 10, fontweight = 'normal',
         ha = 'left', va = 'center')

##### PLOT FOR TACKLERS #####

#Create boxplot
bp = ax[0].boxplot(
    [jData_top10_tackleJ_tackler.loc[jData_top10_tackleJ_tackler['tacklerId'] == top10_tackleJ_tackler[ii],
                                    ]['tackleJ'].to_list() for ii in range(10)],
    patch_artist = True,
    whis = [5,95],
    labels = [top10_tackleJ_tackler[ii] for ii in range(10)],
    positions = range(10),
    widths = 0.4,
    showfliers = False,
    showmeans = True, meanline = True,
    zorder = 6
    )

#Get colouring order for boxplots and points based on primary and secondary team colour
bpColOrder_tackler = []
bpColOrder2_tackler = []
for ii in range(10):
    #Get the players team
    team = rosterData.loc[rosterData['gsis_it_id'] == str(top10_tackleJ_tackler[ii]),['team']].values[0][0]
    #Get primary colour
    col = teamData.loc[teamData['team_abbr'] == team,]['team_color'].values[0]
    col2 = teamData.loc[teamData['team_abbr'] == team,]['team_color2'].values[0]
    #Append to list
    bpColOrder_tackler.append(col)
    bpColOrder2_tackler.append(col2)
    
#Adjust the boxplot colouring
    
#Caps & whiskers (2 per colour)
boxInd = 0
for bpCol in bpColOrder_tackler:
    for _ in range(2):
        bp['caps'][boxInd].set_color(bpCol)
        bp['caps'][boxInd].set_linewidth(1.5)
        bp['whiskers'][boxInd].set_color(bpCol)
        bp['whiskers'][boxInd].set_linewidth(1.5)
        boxInd += 1
    
#Boxes, means and medians (1 per colour)
boxInd = 0
for bpCol in bpColOrder_tackler:
    bp['medians'][boxInd].set_color(bpCol)
    bp['medians'][boxInd].set_linewidth(1.5)
    bp['means'][boxInd].set_color(bpCol)
    bp['means'][boxInd].set_linewidth(1.5)
    bp['boxes'][boxInd].set_facecolor('none')
    bp['boxes'][boxInd].set_edgecolor(bpCol)
    bp['boxes'][boxInd].set_linewidth(1.5)
    boxInd += 1

#Add the strip plot for points
#Hacky loop way, but allows greater flexibility in aspects when plotting
for playerId in top10_tackleJ_tackler:
    sp = sns.stripplot(x = top10_tackleJ_tackler.index(playerId),
                       y = jData_top10_tackleJ_tackler.loc[jData_top10_tackleJ_tackler['tacklerId'] == playerId,
                                                           ]['tackleJ'].to_numpy(),
                       color = bpColOrder2_tackler[top10_tackleJ_tackler.index(playerId)],
                       marker = 'o',
                       size = 6, alpha = 0.5,
                       jitter = True, dodge = False, 
                       native_scale = True, zorder = 5,
                       ax = ax[0])
    
#Set axes title
ax[0].set_title('Tacklers', fontsize = 14, fontweight = 'bold')

#Set axes spine parameters
ax[0].spines['top'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['right'].set_visible(False)

#Set ticks as zero length
ax[0].tick_params(axis = 'both', length = 0)

##### PLOT FOR BALL CARRIERS #####

#Create boxplot
bp = ax[1].boxplot(
    [jData_top10_tackleJ_ballCarrier.loc[jData_top10_tackleJ_ballCarrier['ballCarrierId'] == top10_tackleJ_ballCarrier[ii],
                                         ]['tackleJ'].to_list() for ii in range(10)],
    patch_artist = True,
    whis = [5,95],
    labels = [top10_tackleJ_ballCarrier[ii] for ii in range(10)],
    positions = range(10),
    widths = 0.4,
    showfliers = False,
    showmeans = True, meanline = True,
    zorder = 6
    )

#Get colouring order for boxplots and points based on primary and secondary team colour
bpColOrder_ballCarrier = []
bpColOrder2_ballCarrier = []
for ii in range(10):
    #Get the players team
    team = rosterData.loc[rosterData['gsis_it_id'] == str(top10_tackleJ_ballCarrier[ii]),['team']].values[0][0]
    #Get primary colour
    col = teamData.loc[teamData['team_abbr'] == team,]['team_color'].values[0]
    col2 = teamData.loc[teamData['team_abbr'] == team,]['team_color2'].values[0]
    #Append to list
    bpColOrder_ballCarrier.append(col)
    bpColOrder2_ballCarrier.append(col2)
    
#Adjust the boxplot colouring
    
#Caps & whiskers (2 per colour)
boxInd = 0
for bpCol in bpColOrder_ballCarrier:
    for _ in range(2):
        bp['caps'][boxInd].set_color(bpCol)
        bp['caps'][boxInd].set_linewidth(1.5)
        bp['whiskers'][boxInd].set_color(bpCol)
        bp['whiskers'][boxInd].set_linewidth(1.5)
        boxInd += 1
    
#Boxes, means and medians (1 per colour)
boxInd = 0
for bpCol in bpColOrder_ballCarrier:
    bp['medians'][boxInd].set_color(bpCol)
    bp['medians'][boxInd].set_linewidth(1.5)
    bp['means'][boxInd].set_color(bpCol)
    bp['means'][boxInd].set_linewidth(1.5)
    bp['boxes'][boxInd].set_facecolor('none')
    bp['boxes'][boxInd].set_edgecolor(bpCol)
    bp['boxes'][boxInd].set_linewidth(1.5)
    boxInd += 1

#Add the strip plot for points
#Hacky loop way, but allows greater flexibility in aspects when plotting
for playerId in top10_tackleJ_ballCarrier:
    sp = sns.stripplot(x = top10_tackleJ_ballCarrier.index(playerId),
                       y = jData_top10_tackleJ_ballCarrier.loc[jData_top10_tackleJ_ballCarrier['ballCarrierId'] == playerId,
                                                               ]['tackleJ'].to_numpy(),
                       color = bpColOrder2_ballCarrier[top10_tackleJ_ballCarrier.index(playerId)],
                       marker = 'o',
                       size = 6, alpha = 0.5,
                       jitter = True, dodge = False, 
                       native_scale = True, zorder = 5,
                       ax = ax[1])
    
#Set axes title
ax[1].set_title('Ball Carriers', fontsize = 14, fontweight = 'bold')

#Set axes spine parameters
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].spines['right'].set_visible(False)

#Set ticks as zero length
ax[1].tick_params(axis = 'both', length = 0)

#Add the horizontal line indicators on axes
for ii in range(1,len(ax[0].get_yticks())-1):
    ax[0].axhline(y = ax[0].get_yticks()[ii], lw = 0.5, ls = ':', c = 'dimgrey')
    ax[1].axhline(y = ax[0].get_yticks()[ii], lw = 0.5, ls = ':', c = 'dimgrey')
    
#Get y-ticks on second axis
ax[1].yaxis.set_tick_params(labelleft = True)

#Set y-axes label
ax[0].set_ylabel('Tackle-$J$ (Joules)', labelpad = 10, fontsize = 12)

#Add player and team images, and player details as x-tick labels

#Tacklers

#Remove x-tick labels
ax[0].set_xticklabels([])

#Add images and circles
for playerId in top10_tackleJ_tackler:
    
    #Load player image
    try:
        playerImg = plt.imread(os.path.join('..','img','player',f'{playerId}_cropped.png'))
    except:
        playerImg = plt.imread(os.path.join('..','img','player','NA_cropped.png'))
    
    #Create the offset image
    offsetImg = OffsetImage(playerImg, zoom = 0.04)
    offsetImg.image.axes = ax[0]
    
    #Add the image using annotation box
    aBox = AnnotationBbox(offsetImg, [top10_tackleJ_tackler.index(playerId) - (top10_tackleJ_tackler.index(playerId)*0.021), -37.5],
                          xycoords = 'data',
                          boxcoords = 'offset points',
                          bboxprops = {'lw': 0, 'fc': 'none'}
                    )
    ax[0].add_artist(aBox)
    
    #Add the circle patch
    circlePatch = patches.Ellipse((top10_tackleJ_tackler.index(playerId), -525),
                                  0.85, 550, fc = 'none', ec = bpColOrder_tackler[top10_tackleJ_tackler.index(playerId)], lw = 1.5,
                                  clip_on = False, zorder = 7)
    ax[0].add_patch(circlePatch)
    
    #Add player name
    firstName = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['player_name']].values[0][0].split(' ')[0]
    lastName = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['last_name']].values[0][0]
    ax[0].text(top10_tackleJ_tackler.index(playerId), -925, f'{firstName}\n{lastName}',
               fontsize = 8, fontweight = 'bold', ha = 'center', va = 'center',
               c = bpColOrder_tackler[top10_tackleJ_tackler.index(playerId)])
    #Add number and position details
    number = str(int(rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['jersey_number']].values[0][0]))
    position = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['position']].values[0][0]
    ax[0].text(top10_tackleJ_tackler.index(playerId), -1100, f'(#{number}, {position})',
               fontsize = 8, fontweight = 'normal', ha = 'center', va = 'center',
               c = bpColOrder_tackler[top10_tackleJ_tackler.index(playerId)])
    
    #Load team image
    team = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['team']].values[0][0]
    teamImg = plt.imread(os.path.join('..','img','team',f'{team}.png'))
    
    #Create the offset image
    offsetImg = OffsetImage(teamImg, zoom = 0.06)
    offsetImg.image.axes = ax[0]
    
    #Add the image using annotation box
    aBox = AnnotationBbox(offsetImg, [top10_tackleJ_tackler.index(playerId) - (top10_tackleJ_tackler.index(playerId)*0.021), -95],
                          xycoords = 'data',
                          boxcoords = 'offset points',
                          bboxprops = {'lw': 0, 'fc': 'none'}
                    )
    ax[0].add_artist(aBox)
    
#Ball Carriers

#Remove x-tick labels
ax[1].set_xticklabels([])

#Add images and circles
for playerId in top10_tackleJ_ballCarrier:
    
    #Load player image
    try:
        playerImg = plt.imread(os.path.join('..','img','player',f'{playerId}_cropped.png'))
    except:
        playerImg = plt.imread(os.path.join('..','img','player','NA_cropped.png'))
    
    #Create the offset image
    offsetImg = OffsetImage(playerImg, zoom = 0.04)
    offsetImg.image.axes = ax[1]
    
    #Add the image using annotation box
    aBox = AnnotationBbox(offsetImg, [top10_tackleJ_ballCarrier.index(playerId) - (top10_tackleJ_ballCarrier.index(playerId)*0.021), -37.5],
                          xycoords = 'data',
                          boxcoords = 'offset points',
                          bboxprops = {'lw': 0, 'fc': 'none'}
                    )
    ax[1].add_artist(aBox)
    
    #Add the circle patch
    circlePatch = patches.Ellipse((top10_tackleJ_ballCarrier.index(playerId), -525),
                                  0.85, 550, fc = 'none', ec = bpColOrder_ballCarrier[top10_tackleJ_ballCarrier.index(playerId)], lw = 1.5,
                                  clip_on = False, zorder = 7)
    ax[1].add_patch(circlePatch)
    
    #Add player name
    firstName = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['player_name']].values[0][0].split(' ')[0]
    lastName = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['last_name']].values[0][0]
    ax[1].text(top10_tackleJ_ballCarrier.index(playerId), -925, f'{firstName}\n{lastName}',
               fontsize = 8, fontweight = 'bold', ha = 'center', va = 'center',
               c = bpColOrder_ballCarrier[top10_tackleJ_ballCarrier.index(playerId)])
    #Add number and position details
    number = str(int(rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['jersey_number']].values[0][0]))
    position = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['position']].values[0][0]
    ax[1].text(top10_tackleJ_ballCarrier.index(playerId), -1100, f'(#{number}, {position})',
               fontsize = 8, fontweight = 'normal', ha = 'center', va = 'center',
               c = bpColOrder_ballCarrier[top10_tackleJ_ballCarrier.index(playerId)])
    
    #Load team image
    team = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['team']].values[0][0]
    teamImg = plt.imread(os.path.join('..','img','team',f'{team}.png'))
    
    #Create the offset image
    offsetImg = OffsetImage(teamImg, zoom = 0.06)
    offsetImg.image.axes = ax[1]
    
    #Add the image using annotation box
    aBox = AnnotationBbox(offsetImg, [top10_tackleJ_ballCarrier.index(playerId) - (top10_tackleJ_ballCarrier.index(playerId)*0.021), -95],
                          xycoords = 'data',
                          boxcoords = 'offset points',
                          bboxprops = {'lw': 0, 'fc': 'none'}
                    )
    ax[1].add_artist(aBox)

#Save figure
fig.savefig(os.path.join('..','outputs','figure','topPlayers_tackleJ.png'),
            format = 'png', dpi = 600)

#Close figure
plt.close('all')

# %% Identify and present top Tackle-J for offensive and defensive positions

#Extract the average tackler Tackle-J values split by position
#Sort values by mean and get order
averageTackleJ_tacklerPosition = jData_df.groupby('tacklerPosition')['tackleJ'].describe().reset_index(drop = False)
averageTackleJ_tacklerPosition = averageTackleJ_tacklerPosition.sort_values(by = 'mean', ascending = False).reset_index(drop = True)
tacklerPositionOrder = averageTackleJ_tacklerPosition['tacklerPosition'].to_list()

#Extract the average ball carrier Tackle-J values split by position
#Sort values by mean and get order
averageTackleJ_ballCarrierPosition = jData_df.groupby('ballCarrierPosition')['tackleJ'].describe().reset_index(drop = False)
averageTackleJ_ballCarrierPosition = averageTackleJ_ballCarrierPosition.sort_values(by = 'mean', ascending = False).reset_index(drop = True)
ballCarrierPositionOrder = averageTackleJ_ballCarrierPosition['ballCarrierPosition'].to_list()

#Create figure and axes
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,6),
                       sharex = False, sharey = True)

#Set subplot spacing
plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.85,
                    wspace = 0.1)

#Add figure title
fig.text(0.01, 0.97,
         'Tackle-$J$ Data for Tacklers and Ball Carriers Split by Playing Position',
         font = 'Arial', fontsize = 20,
         ha = 'left', va = 'center')

#Add descriptive text
fig.text(0.01, 0.93,
         'Playing position identified from player data in Big Data Bowl 2024 dataset',
         font = 'Arial', fontsize = 10, fontweight = 'normal',
         ha = 'left', va = 'center')

##### PLOT FOR TACKLERS #####

#Create boxplot
bp = ax[0].boxplot(
    [jData_df.loc[jData_df['tacklerPosition'] == playingPosition,
                  ]['tackleJ'].to_list() for playingPosition in tacklerPositionOrder],
    patch_artist = True,
    whis = [5,95],
    labels = tacklerPositionOrder,
    positions = range(len(tacklerPositionOrder)),
    widths = 0.4,
    showfliers = False,
    showmeans = True, meanline = True,
    zorder = 6
    )
    
#Adjust the boxplot colouring
#Tackler positions get NFL blue colour
    
#Caps & whiskers (2 per colour)
boxInd = 0
for _ in tacklerPositionOrder:
    for _ in range(2):
        bp['caps'][boxInd].set_color('#013369')
        bp['caps'][boxInd].set_linewidth(1.5)
        bp['whiskers'][boxInd].set_color('#013369')
        bp['whiskers'][boxInd].set_linewidth(1.5)
        boxInd += 1
    
#Boxes, means and medians (1 per colour)
boxInd = 0
for _ in tacklerPositionOrder:
    bp['medians'][boxInd].set_color('#013369')
    bp['medians'][boxInd].set_linewidth(1.5)
    bp['means'][boxInd].set_color('#013369')
    bp['means'][boxInd].set_linewidth(1.5)
    bp['boxes'][boxInd].set_facecolor('none')
    bp['boxes'][boxInd].set_edgecolor('#013369')
    bp['boxes'][boxInd].set_linewidth(1.5)
    boxInd += 1

#Add the strip plot for points
#Hacky loop way, but allows greater flexibility in aspects when plotting
for playingPosition in tacklerPositionOrder:
    sp = sns.stripplot(x = tacklerPositionOrder.index(playingPosition),
                       y = jData_df.loc[jData_df['tacklerPosition'] == playingPosition,
                                        ]['tackleJ'].to_numpy(),
                       color = '#013369',
                       marker = 'o',
                       size = 3, alpha = 0.1,
                       jitter = True, dodge = False, 
                       native_scale = True, zorder = 5,
                       ax = ax[0])
    
#Set axes title
ax[0].set_title('Tacklers', fontsize = 14, fontweight = 'bold')

#Set axes spine parameters
ax[0].spines['top'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['right'].set_visible(False)

#Set ticks as zero length
ax[0].tick_params(axis = 'both', length = 0)

##### PLOT FOR BALL CARRIERS #####

#Create boxplot
bp = ax[1].boxplot(
    [jData_df.loc[jData_df['ballCarrierPosition'] == playingPosition,
                  ]['tackleJ'].to_list() for playingPosition in ballCarrierPositionOrder],
    patch_artist = True,
    whis = [5,95],
    labels = ballCarrierPositionOrder,
    positions = range(len(ballCarrierPositionOrder)),
    widths = 0.2,
    showfliers = False,
    showmeans = True, meanline = True,
    zorder = 6
    )
    
#Adjust the boxplot colouring
#Tackler positions get NFL red colour
    
#Caps & whiskers (2 per colour)
boxInd = 0
for _ in ballCarrierPositionOrder:
    for _ in range(2):
        bp['caps'][boxInd].set_color('#d50a0a')
        bp['caps'][boxInd].set_linewidth(1.5)
        bp['whiskers'][boxInd].set_color('#d50a0a')
        bp['whiskers'][boxInd].set_linewidth(1.5)
        boxInd += 1
    
#Boxes, means and medians (1 per colour)
boxInd = 0
for _ in ballCarrierPositionOrder:
    bp['medians'][boxInd].set_color('#d50a0a')
    bp['medians'][boxInd].set_linewidth(1.5)
    bp['means'][boxInd].set_color('#d50a0a')
    bp['means'][boxInd].set_linewidth(1.5)
    bp['boxes'][boxInd].set_facecolor('none')
    bp['boxes'][boxInd].set_edgecolor('#d50a0a')
    bp['boxes'][boxInd].set_linewidth(1.5)
    boxInd += 1

#Add the strip plot for points
#Hacky loop way, but allows greater flexibility in aspects when plotting
for playingPosition in ballCarrierPositionOrder:
    sp = sns.stripplot(x = ballCarrierPositionOrder.index(playingPosition),
                       y = jData_df.loc[jData_df['ballCarrierPosition'] == playingPosition,
                                        ]['tackleJ'].to_numpy(),
                       color = '#d50a0a',
                       marker = 'o',
                       size = 3, alpha = 0.05,
                       jitter = True, dodge = False, 
                       native_scale = True, zorder = 5,
                       ax = ax[1])
    
#Set axes title
ax[1].set_title('Ball Carriers', fontsize = 14, fontweight = 'bold')

#Set axes spine parameters
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].spines['right'].set_visible(False)

#Set ticks as zero length
ax[1].tick_params(axis = 'both', length = 0)

#Add the horizontal line indicators on axes
for ii in range(1,len(ax[0].get_yticks())-1):
    ax[0].axhline(y = ax[0].get_yticks()[ii], lw = 0.5, ls = ':', c = 'dimgrey')
    ax[1].axhline(y = ax[0].get_yticks()[ii], lw = 0.5, ls = ':', c = 'dimgrey')
    
#Get y-ticks on second axis
ax[1].yaxis.set_tick_params(labelleft = True)

#Set y-axes label
ax[0].set_ylabel('Tackle-$J$ (Joules)', labelpad = 10, fontsize = 12)

#Save figure
fig.savefig(os.path.join('..','outputs','figure','splitByPosition_tackleJ.png'),
            format = 'png', dpi = 600)

#Close figure
plt.close('all')

# %% Identify and present top Tackle-J difference positive vs. negative players

#Extract the average tackler Tackle-J difference values
averageTackleJ_difference = jData_df.groupby('tacklerId')['tackleJ_diff'].describe().reset_index(drop = False)

#Remove any players with less than 10 tackles
averageTackleJ_difference = averageTackleJ_difference.loc[averageTackleJ_difference['count'] >= 10,]

#Extract the top 10 positive and negative tackle-J difference tacklers
pos10_tackleJ_difference = averageTackleJ_difference.sort_values(
    by = 'mean', ascending = False).iloc[0:10]['tacklerId'].to_list()
neg10_tackleJ_difference = averageTackleJ_difference.sort_values(
    by = 'mean', ascending = True).iloc[0:10]['tacklerId'].to_list()

#Print out top 10 players in positive difference category
print('\nTop-10 Positive Tackle-J Difference Tacklers')
for ii in range(10):
    #Get details
    displayName = rosterData.loc[rosterData['gsis_it_id'] == str(pos10_tackleJ_difference[ii]),['player_name']].values[0][0]
    team = rosterData.loc[rosterData['gsis_it_id'] == str(pos10_tackleJ_difference[ii]),['team']].values[0][0]
    position = rosterData.loc[rosterData['gsis_it_id'] == str(pos10_tackleJ_difference[ii]),['position']].values[0][0]
    #Print details
    print(f'#{ii+1}: {displayName} ({team}, {position})')
    
#Print out top 10 players in negative difference category
print('\nTop-10 Negative Tackle-J Difference Tacklers')
for ii in range(10):
    #Get details
    displayName = rosterData.loc[rosterData['gsis_it_id'] == str(neg10_tackleJ_difference[ii]),['player_name']].values[0][0]
    team = rosterData.loc[rosterData['gsis_it_id'] == str(neg10_tackleJ_difference[ii]),['team']].values[0][0]
    position = rosterData.loc[rosterData['gsis_it_id'] == str(neg10_tackleJ_difference[ii]),['position']].values[0][0]
    #Print details
    print(f'#{ii+1}: {displayName} ({team}, {position})')

#Extract the data from top 10 players from the full dataset
jData_pos10_tackleJ_difference = jData_df.loc[jData_df['tacklerId'].isin(pos10_tackleJ_difference)]
jData_neg10_tackleJ_difference = jData_df.loc[jData_df['tacklerId'].isin(neg10_tackleJ_difference)]

#Create figure and axes
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,7),
                       sharex = False, sharey = True)

#Set subplot spacing
plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.2, top = 0.85,
                    wspace = 0.1)

#Add figure title
fig.text(0.01, 0.97,
         'Top-10 Performing Tacklers for Positive and Negative Tackle-$J_{difference}$',
         font = 'Arial', fontsize = 20,
         ha = 'left', va = 'center')

#Add descriptive text
fig.text(0.01, 0.93,
         'Players identified by highest (i.e. positive) and lowest (i.e. negative) average Tackle-$J_{difference}$ in the dataset (minimum 10 tackles)',
         font = 'Arial', fontsize = 10, fontweight = 'normal',
         ha = 'left', va = 'center')

##### PLOT FOR HIGH TACKLE J DIFF #####

#Create boxplot
bp = ax[0].boxplot(
    [jData_pos10_tackleJ_difference.loc[jData_pos10_tackleJ_difference['tacklerId'] == pos10_tackleJ_difference[ii],
                                    ]['tackleJ_diff'].to_list() for ii in range(10)],
    patch_artist = True,
    whis = [5,95],
    labels = [pos10_tackleJ_difference[ii] for ii in range(10)],
    positions = range(10),
    widths = 0.4,
    showfliers = False,
    showmeans = True, meanline = True,
    zorder = 6
    )

#Get colouring order for boxplots and points based on primary and secondary team colour
bpColOrder_pos = []
bpColOrder2_pos = []
for ii in range(10):
    #Get the players team
    team = rosterData.loc[rosterData['gsis_it_id'] == str(pos10_tackleJ_difference[ii]),['team']].values[0][0]
    #Get primary colour
    col = teamData.loc[teamData['team_abbr'] == team,]['team_color'].values[0]
    col2 = teamData.loc[teamData['team_abbr'] == team,]['team_color2'].values[0]
    #Append to list
    bpColOrder_pos.append(col)
    bpColOrder2_pos.append(col2)
    
#Adjust the boxplot colouring
    
#Caps & whiskers (2 per colour)
boxInd = 0
for bpCol in bpColOrder_pos:
    for _ in range(2):
        bp['caps'][boxInd].set_color(bpCol)
        bp['caps'][boxInd].set_linewidth(1.5)
        bp['whiskers'][boxInd].set_color(bpCol)
        bp['whiskers'][boxInd].set_linewidth(1.5)
        boxInd += 1
    
#Boxes, means and medians (1 per colour)
boxInd = 0
for bpCol in bpColOrder_pos:
    bp['medians'][boxInd].set_color(bpCol)
    bp['medians'][boxInd].set_linewidth(1.5)
    bp['means'][boxInd].set_color(bpCol)
    bp['means'][boxInd].set_linewidth(1.5)
    bp['boxes'][boxInd].set_facecolor('none')
    bp['boxes'][boxInd].set_edgecolor(bpCol)
    bp['boxes'][boxInd].set_linewidth(1.5)
    boxInd += 1

#Add the strip plot for points
#Hacky loop way, but allows greater flexibility in aspects when plotting
for playerId in pos10_tackleJ_difference:
    sp = sns.stripplot(x = pos10_tackleJ_difference.index(playerId),
                       y = jData_pos10_tackleJ_difference.loc[jData_pos10_tackleJ_difference['tacklerId'] == playerId,
                                                              ]['tackleJ_diff'].to_numpy(),
                       color = bpColOrder2_pos[pos10_tackleJ_difference.index(playerId)],
                       marker = 'o',
                       size = 6, alpha = 0.5,
                       jitter = True, dodge = False, 
                       native_scale = True, zorder = 5,
                       ax = ax[0])
    
#Set axes title
ax[0].set_title('Highest Average Tackle-$J_{difference}$', fontsize = 14, fontweight = 'bold')

#Set axes spine parameters
ax[0].spines['top'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['right'].set_visible(False)

#Set ticks as zero length
ax[0].tick_params(axis = 'both', length = 0)

##### PLOT FOR LOW TACKLE J DIFF #####

#Create boxplot
bp = ax[1].boxplot(
    [jData_neg10_tackleJ_difference.loc[jData_neg10_tackleJ_difference['tacklerId'] == neg10_tackleJ_difference[ii],
                                    ]['tackleJ_diff'].to_list() for ii in range(10)],
    patch_artist = True,
    whis = [5,95],
    labels = [neg10_tackleJ_difference[ii] for ii in range(10)],
    positions = range(10),
    widths = 0.4,
    showfliers = False,
    showmeans = True, meanline = True,
    zorder = 6
    )

#Get colouring order for boxplots and points based on primary and secondary team colour
bpColOrder_neg = []
bpColOrder2_neg = []
for ii in range(10):
    #Get the players team
    team = rosterData.loc[rosterData['gsis_it_id'] == str(neg10_tackleJ_difference[ii]),['team']].values[0][0]
    #Get primary colour
    col = teamData.loc[teamData['team_abbr'] == team,]['team_color'].values[0]
    col2 = teamData.loc[teamData['team_abbr'] == team,]['team_color2'].values[0]
    #Append to list
    bpColOrder_neg.append(col)
    bpColOrder2_neg.append(col2)
    
#Adjust the boxplot colouring
    
#Caps & whiskers (2 per colour)
boxInd = 0
for bpCol in bpColOrder_neg:
    for _ in range(2):
        bp['caps'][boxInd].set_color(bpCol)
        bp['caps'][boxInd].set_linewidth(1.5)
        bp['whiskers'][boxInd].set_color(bpCol)
        bp['whiskers'][boxInd].set_linewidth(1.5)
        boxInd += 1
    
#Boxes, means and medians (1 per colour)
boxInd = 0
for bpCol in bpColOrder_neg:
    bp['medians'][boxInd].set_color(bpCol)
    bp['medians'][boxInd].set_linewidth(1.5)
    bp['means'][boxInd].set_color(bpCol)
    bp['means'][boxInd].set_linewidth(1.5)
    bp['boxes'][boxInd].set_facecolor('none')
    bp['boxes'][boxInd].set_edgecolor(bpCol)
    bp['boxes'][boxInd].set_linewidth(1.5)
    boxInd += 1

#Add the strip plot for points
#Hacky loop way, but allows greater flexibility in aspects when plotting
for playerId in neg10_tackleJ_difference:
    sp = sns.stripplot(x = neg10_tackleJ_difference.index(playerId),
                       y = jData_neg10_tackleJ_difference.loc[jData_neg10_tackleJ_difference['tacklerId'] == playerId,
                                                              ]['tackleJ_diff'].to_numpy(),
                       color = bpColOrder2_neg[neg10_tackleJ_difference.index(playerId)],
                       marker = 'o',
                       size = 6, alpha = 0.5,
                       jitter = True, dodge = False, 
                       native_scale = True, zorder = 5,
                       ax = ax[1])
    
#Set axes title
ax[1].set_title('Lowest Average Tackle-$J_{difference}$', fontsize = 14, fontweight = 'bold')

#Set axes spine parameters
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].spines['right'].set_visible(False)

#Set ticks as zero length
ax[1].tick_params(axis = 'both', length = 0)

#Add the horizontal line indicators on axes
for ii in range(1,len(ax[0].get_yticks())-1):
    ax[0].axhline(y = ax[0].get_yticks()[ii], lw = 0.5, ls = ':', c = 'dimgrey')
    ax[1].axhline(y = ax[0].get_yticks()[ii], lw = 0.5, ls = ':', c = 'dimgrey')
    
#Get y-ticks on second axis
ax[1].yaxis.set_tick_params(labelleft = True)

#Set y-axes label
ax[0].set_ylabel('Tackle-$J_{difference}$ (Joules)', labelpad = 5, fontsize = 12)

#Add player and team images, and player details as x-tick labels

#High Average Tackle-J Diff

#Remove x-tick labels
ax[0].set_xticklabels([])

#Add images and circles
for playerId in pos10_tackleJ_difference:
    
    #Load player image
    try:
        playerImg = plt.imread(os.path.join('..','img','player',f'{playerId}_cropped.png'))
    except:
        playerImg = plt.imread(os.path.join('..','img','player','NA_cropped.png'))
    
    #Create the offset image
    offsetImg = OffsetImage(playerImg, zoom = 0.04)
    offsetImg.image.axes = ax[0]
    
    #Add the image using annotation box
    aBox = AnnotationBbox(offsetImg, [pos10_tackleJ_difference.index(playerId) - (pos10_tackleJ_difference.index(playerId)*0.021), (-37.5 - 140)],
                          xycoords = 'data',
                          boxcoords = 'offset points',
                          bboxprops = {'lw': 0, 'fc': 'none'}
                    )
    ax[0].add_artist(aBox)
    
    #Add the circle patch
    circlePatch = patches.Ellipse((pos10_tackleJ_difference.index(playerId), -3200),
                                  0.85, 675, fc = 'none', ec = bpColOrder_pos[pos10_tackleJ_difference.index(playerId)], lw = 1.5,
                                  clip_on = False, zorder = 7)
    ax[0].add_patch(circlePatch)
    
    #Add player name
    firstName = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['player_name']].values[0][0].split(' ')[0]
    lastName = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['last_name']].values[0][0]
    ax[0].text(pos10_tackleJ_difference.index(playerId), -3725, f'{firstName}\n{lastName}',
               fontsize = 8, fontweight = 'bold', ha = 'center', va = 'center',
               c = bpColOrder_pos[pos10_tackleJ_difference.index(playerId)])
    #Add number and position details
    number = str(int(rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['jersey_number']].values[0][0]))
    position = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['position']].values[0][0]
    ax[0].text(pos10_tackleJ_difference.index(playerId), -3950, f'(#{number}, {position})',
               fontsize = 8, fontweight = 'normal', ha = 'center', va = 'center',
               c = bpColOrder_pos[pos10_tackleJ_difference.index(playerId)])
    
    #Load team image
    team = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['team']].values[0][0]
    teamImg = plt.imread(os.path.join('..','img','team',f'{team}.png'))
    
    #Create the offset image
    offsetImg = OffsetImage(teamImg, zoom = 0.05)
    offsetImg.image.axes = ax[0]
    
    #Add the image using annotation box
    aBox = AnnotationBbox(offsetImg, [pos10_tackleJ_difference.index(playerId) - (pos10_tackleJ_difference.index(playerId)*0.021), (-95 - 140)],
                          xycoords = 'data',
                          boxcoords = 'offset points',
                          bboxprops = {'lw': 0, 'fc': 'none'}
                    )
    ax[0].add_artist(aBox)
    
#Low Average Tackle-J Diff

#Remove x-tick labels
ax[1].set_xticklabels([])

#Add images and circles
for playerId in neg10_tackleJ_difference:
    
    #Load player image
    try:
        playerImg = plt.imread(os.path.join('..','img','player',f'{playerId}_cropped.png'))
    except:
        playerImg = plt.imread(os.path.join('..','img','player','NA_cropped.png'))
    
    #Create the offset image
    offsetImg = OffsetImage(playerImg, zoom = 0.04)
    offsetImg.image.axes = ax[0]
    
    #Add the image using annotation box
    aBox = AnnotationBbox(offsetImg, [neg10_tackleJ_difference.index(playerId) - (neg10_tackleJ_difference.index(playerId)*0.021), (-37.5 - 140)],
                          xycoords = 'data',
                          boxcoords = 'offset points',
                          bboxprops = {'lw': 0, 'fc': 'none'}
                    )
    ax[1].add_artist(aBox)
    
    #Add the circle patch
    circlePatch = patches.Ellipse((neg10_tackleJ_difference.index(playerId), -3200),
                                  0.85, 675, fc = 'none', ec = bpColOrder_neg[neg10_tackleJ_difference.index(playerId)], lw = 1.5,
                                  clip_on = False, zorder = 7)
    ax[1].add_patch(circlePatch)
    
    #Add player name
    firstName = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['player_name']].values[0][0].split(' ')[0]
    lastName = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['last_name']].values[0][0]
    ax[1].text(neg10_tackleJ_difference.index(playerId), -3725, f'{firstName}\n{lastName}',
               fontsize = 8, fontweight = 'bold', ha = 'center', va = 'center',
               c = bpColOrder_neg[neg10_tackleJ_difference.index(playerId)])
    #Add number and position details
    number = str(int(rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['jersey_number']].values[0][0]))
    position = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['position']].values[0][0]
    ax[1].text(neg10_tackleJ_difference.index(playerId), -3950, f'(#{number}, {position})',
               fontsize = 8, fontweight = 'normal', ha = 'center', va = 'center',
               c = bpColOrder_neg[neg10_tackleJ_difference.index(playerId)])
    
    #Load team image
    team = rosterData.loc[rosterData['gsis_it_id'] == str(playerId),['team']].values[0][0]
    teamImg = plt.imread(os.path.join('..','img','team',f'{team}.png'))
    
    #Create the offset image
    offsetImg = OffsetImage(teamImg, zoom = 0.05)
    offsetImg.image.axes = ax[0]
    
    #Add the image using annotation box
    aBox = AnnotationBbox(offsetImg, [neg10_tackleJ_difference.index(playerId) - (neg10_tackleJ_difference.index(playerId)*0.021), (-95 - 140)],
                          xycoords = 'data',
                          boxcoords = 'offset points',
                          bboxprops = {'lw': 0, 'fc': 'none'}
                    )
    ax[1].add_artist(aBox)

#Save figure
fig.savefig(os.path.join('..','outputs','figure','topPlayers_tackleJ_difference.png'),
            format = 'png', dpi = 600)

#Close figure
plt.close('all')

# %% ----- End of calcTackleJ.py ----- %% #