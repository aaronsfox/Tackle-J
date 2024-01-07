# -*- coding: utf-8 -*-
"""

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    Python helper functions for Big Data Bowl 2024 entry calculating Tackle-J.
    The functions within this script are called in the main calcTackleJ script.
    
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
import nfl_data_py as nfl
import math
import os
from PIL import Image, ImageChops, ImageDraw
import requests

#Add custom fonts for use with matplotlib
fontDir = [os.getcwd()+os.sep+os.path.join('..','fonts')]
for font in font_manager.findSystemFonts(fontDir):
    font_manager.fontManager.addfont(font)

# %% Function for creating a static NFL field

"""

Note that a figure size of (14, 6.5) seems to work best with the font sizes
Other sizes haven't been tested.

Sample usage:

#Create figure
fieldFig, fieldAx = plt.subplots(figsize=(14, 6.5))
# createField(fieldFig, fieldAx)
createField(fieldFig, fieldAx,
            lineOfScrimmage = 33, firstDownMark = 27,
            homeTeamAbbr = 'WAS', awayTeamAbbr = 'NO', teamData = teamData)

"""

def createField(fig, ax,
                lineOfScrimmage = None, firstDownMark = None,
                homeTeamAbbr = None, awayTeamAbbr = None, teamData = None):
    
    #Adjust subplot positioning for field size
    plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95)
    
    #Set axes and figure facecolour to match field
    ax.set_facecolor('#29a500')
    fig.set_facecolor('#29a500')

    #Set axes according to space around field definition
    ax.set_xlim([-1,121])
    ax.set_ylim([-0.5,53.8])
    
    #Create the green field
    field = patches.Rectangle((0, 0), 120, 53.3,
                              linewidth = 2, edgecolor = '#ffffff',
                              facecolor = '#29a500', zorder = 0)
    ax.add_patch(field)
    
    #Create the end zone markings
    #Rectangles
    #Check for home, away and team data
    if not [xx for xx in (homeTeamAbbr, awayTeamAbbr, teamData) if xx is None]:
        #Use team colourings and names
        #Use team colour 1 for home team and team colour 2 for away team
        #Home team patch
        homeZone = patches.Rectangle((0, 0), 10, 53.3,
                                     linewidth = 2, edgecolor = '#ffffff',
                                     facecolor = teamData.loc[teamData['team_abbr'] == homeTeamAbbr,['team_color']].values[0][0],
                                     zorder = 1)
        ax.add_patch(homeZone)
        #Home team text
        ax.text(5, 53.3 / 2, teamData.loc[teamData['team_abbr'] == homeTeamAbbr,['team_nick']].values[0][0].upper(),
                va = 'center', ha = 'center', fontsize = 30, fontweight = 'bold', rotation = 90,
                color = teamData.loc[teamData['team_abbr'] == homeTeamAbbr,['team_color2']].values[0][0],
                zorder = 2)
        #Away team patch
        awayZone = patches.Rectangle((110, 0), 10, 53.3,
                                     linewidth = 2, edgecolor = '#ffffff',
                                     facecolor = teamData.loc[teamData['team_abbr'] == awayTeamAbbr,['team_color2']].values[0][0],
                                     zorder = 1)
        ax.add_patch(awayZone)
        #Away team text
        ax.text(115, 53.3 / 2, teamData.loc[teamData['team_abbr'] == awayTeamAbbr,['team_nick']].values[0][0].upper(),
                va = 'center', ha = 'center', fontsize = 30, fontweight = 'bold', rotation = 90,
                color = teamData.loc[teamData['team_abbr'] == awayTeamAbbr,['team_color']].values[0][0],
                zorder = 2)
    else:
        #Use standard colourings and names
        homeZone = patches.Rectangle((0, 0), 10, 53.3,
                                     linewidth = 2, edgecolor = '#ffffff',
                                     facecolor = '#013369', zorder = 1)
        ax.add_patch(homeZone)
        awayZone = patches.Rectangle((110, 0), 10, 53.3,
                                     linewidth = 2, edgecolor = '#ffffff',
                                     facecolor = '#d50a0a', zorder = 1)
        ax.add_patch(awayZone)
        #Text
        ax.text(5, 53.3 / 2, 'HOME', va = 'center', ha = 'center',
                fontsize = 30, fontweight = 'bold', rotation = 90, color = 'white', zorder = 2)
        ax.text(115, 53.3 / 2, 'AWAY', va = 'center', ha = 'center',
                fontsize = 30, fontweight = 'bold', rotation = 270, color = 'white', zorder = 2)
    
    #Plot the line markers
    for lineMark in np.arange(10,120,10):
        plt.plot([lineMark, lineMark], [0, 53.3],
                 linewidth = 2, color = '#ffffff', zorder = 2)
    
    #Plot field numbers
    for fieldNo in range(20, 110, 10):
        #Get number for appropriate side of field
        if fieldNo > 50:
            dispNo = int(120-fieldNo)
        else:
            dispNo = int(fieldNo)
        #Split display number into the first and second digit
        dispStr1 = str(int(dispNo/10 - 1))
        dispStr2 = '0'
        #Display number on field
        #First digit, bottom of field
        ax.text(fieldNo, 4, dispStr1, ha = 'right', font = 'Clarendon',
                fontsize = 22, fontweight = 'bold', color = 'white', zorder = 1)
        #Second digit, bottom of field
        ax.text(fieldNo, 4, dispStr2, ha = 'left', font = 'Clarendon',
                fontsize = 22, fontweight = 'bold', color = 'white', zorder = 1)
        #First digit, top of field
        ax.text(fieldNo, 53.3 - 4, dispStr1, ha = 'left', font = 'Clarendon',
                fontsize = 22, fontweight = 'bold', color = 'white', rotation = 180, zorder = 1)
        #Second digit, bottom of field
        ax.text(fieldNo, 53.3 - 4, dispStr2, ha = 'right', font = 'Clarendon',
                fontsize = 22, fontweight = 'bold', color = 'white', rotation = 180, zorder = 1)
    
    #Plot hash marks
    for hashMark in range(11, 110):
        ax.plot([hashMark, hashMark], [0.4, 0.7], color = 'white', zorder = 1)
        ax.plot([hashMark, hashMark], [53.0, 52.5], color = 'white', zorder = 1)
        ax.plot([hashMark, hashMark], [23, 23.66], color = 'white', zorder = 1)
        ax.plot([hashMark, hashMark], [29.66, 30.33], color = 'white', zorder = 1)
        
    #Plot line of scrimmage if provided
    if lineOfScrimmage is not None:
        plt.plot([lineOfScrimmage, lineOfScrimmage], [0, 53.3],
                 linewidth = 2, color = 'yellow', zorder = 2)
        
    #Plot first down marker if provided
    if firstDownMark is not None:
        plt.plot([firstDownMark, firstDownMark], [0, 53.3],
                 linewidth = 2, color = 'red', zorder = 2)
    
    #Turn off 
    plt.axis('off')

# %% Function to use as input for Matplotlib animate function to draw a play frame

def drawPlay(frame, homeTeam, awayTeam, teamData, play, playerLabels,
             ballCarrierId, tacklerId,
             lineOfScrimmage = None, firstDownMark = None):
        
    #Get the current figure and axes to use
    fieldFig = plt.gcf()
    fieldAx = plt.gca()
    
    #Clear axes for subsequent frame
    fieldAx.clear()
    
    #Redraw field
    createField(fieldFig, fieldAx, lineOfScrimmage, firstDownMark,
                homeTeam, awayTeam, teamData)    
    
    #Extract data for frame
    trackingDataFrame = play.loc[play['frameId'] == frame,]
    
    #Loop through teams to plot
    for team in [homeTeam, awayTeam]:
        
        #Set plotting colours
        #Conditional to flip colour scheme
        if team == homeTeam:
            primaryCol = teamData.loc[teamData['team_abbr'] == team,['team_color']].values[0][0]
            secondaryCol = teamData.loc[teamData['team_abbr'] == team,['team_color2']].values[0][0]
        elif team == awayTeam:
            primaryCol = teamData.loc[teamData['team_abbr'] == team,['team_color2']].values[0][0]
            secondaryCol = teamData.loc[teamData['team_abbr'] == team,['team_color']].values[0][0]

        #Extract data to plot
        trackingDataTeam = trackingDataFrame.loc[trackingDataFrame['club'] == team, ]
        
        #Create unique position column
        if playerLabels == 'pos':
            trackingDataTeam['pos_unique'] = (trackingDataTeam['position']
                                              .add(trackingDataTeam
                                                   .groupby('position', as_index=False)
                                                   .cumcount()
                                                   .add(1)
                                                   .dropna()
                                                   .astype(str)
                                                   .str.replace('.0','', regex=False)
                                                   .str.replace('0','', regex=False)))
            #Create the final position unique variable by removing any single listed positions (e.g. like QB)
            pos_final = []
            for pInd in trackingDataTeam.index:
                if np.sum(trackingDataTeam['position'] == trackingDataTeam.loc[pInd]['position']) == 1:
                    pos_final.append(trackingDataTeam.loc[pInd]['position'])
                else:
                    pos_final.append(trackingDataTeam.loc[pInd]['pos_unique'])
            trackingDataTeam['pos_final'] = pos_final
        
        #Plot the current teams data
        for ii in trackingDataTeam.index:
            
            #Get the parameters for the current player
            xPos = trackingDataTeam.loc[ii]['x']
            yPos = trackingDataTeam.loc[ii]['y']
            speed = trackingDataTeam.loc[ii]['s']
            direction = trackingDataTeam.loc[ii]['dir']
            orientation = trackingDataTeam.loc[ii]['o']
            jerseyNo = str(int(trackingDataTeam.loc[ii]['jerseyNumber']))
            posLabel = trackingDataTeam.loc[ii]['pos_final']
            
            #Create the circle patch for the player
            playerPatch = patches.Circle((xPos, yPos), radius = 1,
                                         facecolor = primaryCol, linewidth = 1, edgecolor = secondaryCol,
                                         zorder = 5, clip_on = False)
            fieldAx.add_patch(playerPatch)
            #Add the jersey number or position
            if playerLabels == 'pos':
                fieldAx.text(xPos, yPos, posLabel, ha = 'center', va = 'center',
                             fontsize = 5.5, fontweight = 'bold', color = secondaryCol, zorder = 7)
            else:
                fieldAx.text(xPos, yPos, jerseyNo, ha = 'center', va = 'center',
                             fontsize = 7, fontweight = 'bold', color = secondaryCol, zorder = 7)
                    
            #Draw arrow indicating direction of movement
            #Calculate dx, dy for line based on direction
            #Convert angle relative to x-axis & calculate vector components
            convertedDir = 90 - direction
            dx = speed * np.cos(np.deg2rad(convertedDir))
            dy = speed * np.sin(np.deg2rad(convertedDir))
            #Draw scaled arrow indicating movement speed
            dirArrow = patches.FancyArrowPatch((xPos, yPos), (xPos+dx, yPos+dy),
                                               edgecolor = primaryCol, facecolor = primaryCol,
                                               mutation_scale = 10, lw = 1.5,
                                               zorder = 3, clip_on = False)
            fieldAx.add_patch(dirArrow)
            
            #Draw arrow indicating orientation
            #This is an arrow that sits underneath the player circle and just pokes out
            #to indicate the direction they're looking - hence the arrow length is basically
            #to scale of the players circle patch - which just happens to be 1, so we just
            #assume dx and dy to come from a vector magnitude of 2
            #Convert angle relative to x-axis to calculate vector components
            convertedOri = 90 - orientation
            dx = 2 * np.cos(np.deg2rad(convertedOri))
            dy = 2 * np.sin(np.deg2rad(convertedOri))
            #Draw scaled arrow to indicate orientation
            oriArrow = patches.FancyArrowPatch((xPos, yPos), (xPos+dx, yPos+dy),
                                               edgecolor = secondaryCol, facecolor = secondaryCol,
                                               mutation_scale = 10, lw = 1,
                                               zorder = 3, clip_on = False)
            fieldAx.add_patch(oriArrow)
    
    #Get max speed of relevant player traces for mapping
    if ballCarrierId is not None and tacklerId is not None:
        #Get the absolute max from both players
        maxSpeed = play.loc[play['nflId'].isin([ballCarrierId,tacklerId]),]['s'].to_numpy().max()
    elif ballCarrierId is not None and tacklerId is None:
        #Get the absolute max from ball carrier
        maxSpeed = play.loc[play['nflId'] == ballCarrierId,]['s'].to_numpy().max()
    elif ballCarrierId is None and tacklerId is not None:
        #Get the absolute max from ball carrier
        maxSpeed = play.loc[play['nflId'] == tacklerId,]['s'].to_numpy().max()
        
    #Add the ball carriers path
    if ballCarrierId is not None:
    
        #Extract the ball carriers data up to the current frame
        ballCarrierTrack = play.loc[(play['nflId'] == ballCarrierId) &
                                    (play['frameId'] <= frame),]
        
        #Extract XY positions to numpy arrays
        ballCarrierX = ballCarrierTrack['x'].to_numpy()
        ballCarrierY = ballCarrierTrack['y'].to_numpy()
        
        #Set the plot colour
        if ballCarrierTrack['club'].unique()[0] == homeTeam:
            traceCol = teamData.loc[teamData['team_abbr'] == homeTeam,['team_color2']].values[0][0]
        elif ballCarrierTrack['club'].unique()[0] == awayTeam:
            traceCol = teamData.loc[teamData['team_abbr'] == awayTeam,['team_color']].values[0][0]
            
        # #Plot the trace
        # fieldAx.plot(ballCarrierX, ballCarrierY, c = traceCol, ls = '-', lw = 2.5, zorder = 4, clip_on = False)
        
        #Plot the trace as line segments to specify variable alpha
        #See: https://stackoverflow.com/questions/61478675/line-plot-that-continuously-varies-transparency-matplotlib
        
        #Create the alphas and line segments
        alphas = ballCarrierTrack['s'].to_numpy() / maxSpeed
        points = np.vstack((ballCarrierX, ballCarrierY)).T.reshape(-1, 1, 2)
        segments = np.hstack((points[:-1], points[1:]))
        
        #Create and plot the line collection
        lineCollection = LineCollection(segments, alpha = alphas, color = traceCol,
                                        ls = '-', lw = 2.5,
                                        zorder = 4, clip_on = False)
        ballCarrierLine = fieldAx.add_collection(lineCollection)
    
    #Add the tacklers path
    if tacklerId is not None:
        
        #Extract the tacklers data up to the current frame
        tacklerTrack = play.loc[(play['nflId'] == tacklerId) &
                                (play['frameId'] <= frame),]
        
        #Extract XY positions to numpy arrays
        tacklerX = tacklerTrack['x'].to_numpy()
        tacklerY = tacklerTrack['y'].to_numpy()
        
        #Set the plot colour
        if tacklerTrack['club'].unique()[0] == homeTeam:
            traceCol = teamData.loc[teamData['team_abbr'] == homeTeam,['team_color2']].values[0][0]
        elif tacklerTrack['club'].unique()[0] == awayTeam:
            traceCol = teamData.loc[teamData['team_abbr'] == awayTeam,['team_color']].values[0][0]
            
        # #Plot the trace
        # fieldAx.plot(tacklerX, tacklerY, c = traceCol, ls = '-', lw = 2.5, zorder = 4, clip_on = False)
        
        #Plot the trace as line segments to specify variable alpha
        #See: https://stackoverflow.com/questions/61478675/line-plot-that-continuously-varies-transparency-matplotlib
        
        #Create the alphas and line segments
        alphas = tacklerTrack['s'].to_numpy() / maxSpeed
        points = np.vstack((tacklerX, tacklerY)).T.reshape(-1, 1, 2)
        segments = np.hstack((points[:-1], points[1:]))
        
        #Create and plot the line collection
        lineCollection = LineCollection(segments, alpha = alphas, color = traceCol,
                                        ls = '-', lw = 2.5,
                                        zorder = 4, clip_on = False)
        tacklerLine = fieldAx.add_collection(lineCollection)
    
    #Add the ball position
    
    #Extract the ball data for frame
    trackingDataBall = trackingDataFrame.loc[trackingDataFrame['club'] == 'football', ]
    
    #Add ball to axes
    ball = patches.Ellipse((trackingDataBall['x'].to_numpy()[0],trackingDataBall['y'].to_numpy()[0]),
                           2, 1, angle = 0, facecolor = '#825736', edgecolor = '#ffffff',
                           zorder = 6, clip_on = False)
    fieldAx.add_patch(ball)
    
    #Add ball tracking trace
    
    #Extract the tacklers data up to the current frame
    ballTrack = play.loc[(play['club'] == 'football') &
                         (play['frameId'] <= frame),]
    
    #Extract XY positions to numpy arrays
    ballX = ballTrack['x'].to_numpy()
    ballY = ballTrack['y'].to_numpy()
        
    #Plot the trace
    fieldAx.plot(ballX, ballY, c = '#ffffff', ls = ':', lw = 2.5, zorder = 4, clip_on = False)

# %% Crop player headshot images and save to file for later use

def cropPlayerImg(rosterData, outputFolder):
    
    #Loop through players in roster data
    for ii in rosterData.index:
        
        #Check to see if headshot URL exists
        if rosterData.iloc[ii]['headshot_url'] is not None:
        
            #Download the image    
            imgData = requests.get(rosterData.iloc[ii]['headshot_url']).content
            
            #Get the player nfl Id from roster data
            playerId = rosterData.iloc[ii]['gsis_it_id']
            
            #Write the image data to file
            imgFile = open(os.path.join(outputFolder,f'{playerId}.png'),'wb')
            imgFile.write(imgData) 
            imgFile.close()
            
            #Open image to crop
            im = Image.open(os.path.join(outputFolder,f'{playerId}.png')).convert('RGBA')
            
            #Get image size
            w = im.size[0]
            h = im.size[1]
            
            #Create mask
            mask = Image.new('L', (w,h), 0)
            
            #Draw the ellipse
            ImageDraw.Draw(mask).ellipse([((w/2)-(h/2),0),((w/2)+(h/2),h)], fill = 255)
            
            #Set the transparency on the mask
            mask = ImageChops.darker(mask, im.split()[-1])
            
            #Add mask to image
            im.putalpha(mask)
            
            #Save image
            im.save(os.path.join(outputFolder,f'{playerId}_cropped.png'))
            
# %% Download team logo images and save to file for later use

def downloadTeamImages(teamData, outputFolder):
    
    #Loop through players in roster data
    for ii in teamData.index:
        
        #Download the image    
        imgData = requests.get(teamData.iloc[ii]['team_logo_espn']).content
            
        #Get the team abbreviation from team data
        teamAbbr = teamData.iloc[ii]['team_abbr']
        
        #Write the image data to file
        imgFile = open(os.path.join(outputFolder,f'{teamAbbr}.png'),'wb')
        imgFile.write(imgData) 
        imgFile.close()

# %% ----- End of helperFuncs.py ----- %% #