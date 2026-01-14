#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.4),
    on Wed Jan 14 16:54:19 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.4'
expName = 'WMpsilocybin_v3'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/sub-%s/%s_%s_%s' % (expInfo['participant'], expInfo['participant'], expName, expInfo['date']) 
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/calebjerinic-brodeur/Dropbox (ASU)/My Mac (Calebs-MacBook-Pro.local)/Documents/GitHub/WM_Psilocybin/WMpsilocybin_v3_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "init" ---
    # Run 'Begin Experiment' code from code_init
    import os as _os
    
    # CREATE SUBJECT-SPECIFIC FOLDERS FOR DESIGN MATRICES
    # assign participant number from psychopy input
    participant_number = expInfo['participant']
    
    # set directory (i.e., experiment working directory)
    expDir = _os.path.dirname(_os.path.abspath(__file__))
    
    # create a subject specific folder to save all lists
    list_foldername = _os.path.join(expDir, 'subject_designs', f'sub-{participant_number}')
    
    # phase folders to create
    phase_subFolder = ['practice', 'main']
    
    # create the subject folder if it does not already exist
    if not _os.path.exists(list_foldername):
        _os.makedirs(list_foldername)
        print(f'List folder successfully created for sub-{participant_number}!')
    else:
        print(f'List folder already exists for sub-{participant_number}...')
    
    # create each phase subfolder within the subject's folder
    for subfolder in phase_subFolder:
        phase_folder_path = _os.path.join(list_foldername, subfolder)
        
        # create the phase folder
        if not _os.path.exists(phase_folder_path):
            _os.makedirs(phase_folder_path)
            print(f'Created phase folder: {subfolder}')
        else:
            print(f'Phase folder {subfolder} already exists...')
    
    
    # CREATE SUBJECT-SPECIFIC FOLDERS FOR DATA
    # create main folder
    data_foldername = _os.path.join(expDir, 'data', f'sub-{participant_number}')
    
    if not _os.path.exists(data_foldername):
        _os.makedirs(data_foldername)
        print(f'Data folder successfully created for sub-{participant_number}!')
    else:
        print(f'Data folder alreadt exists for sub-{participant_number}...')
    
    # --- Initialize components for Routine "WELCOME_screen" ---
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='Welcome to the experiment!\n\nIn this portion of the experiment we will ask you to remember the location of 5 colored dots over a breif period, and then report the position of each dot on the screen.\n\nThe following slides will go through more detailed instructions…\n\n<press SPACE to continue>',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    welcome_keyResp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "ITI_instructScreen" ---
    iti_instructText = visual.TextStim(win=win, name='iti_instructText',
        text='Each trial will start with a 1 second fixation period. You should maintain your focus on the small, black square in the middle of the screen throughout the entire experiment.\n\nThe 1 second fixation period will look like this:',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    iti_instructImage = visual.ImageStim(
        win=win,
        name='iti_instructImage', 
        image='instruction_images/iti_instruction.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.2), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    iti_continueText = visual.TextStim(win=win, name='iti_continueText',
        text='<press SPACE to continue>',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    itiInstruct_keyResp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "memArray_instructScreen" ---
    memArray_instructText = visual.TextStim(win=win, name='memArray_instructText',
        text='Next, you will briefly be shown a visual array of 5 colored dots at different spatial locations on the screen.\n\nThe memory array will look something like this:',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    memArray_instructImage = visual.ImageStim(
        win=win,
        name='memArray_instructImage', 
        image='instruction_images/memArray_instruction.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.2), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    memArray_continueText = visual.TextStim(win=win, name='memArray_continueText',
        text='<press SPACE to continue>',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    memArray_keyResp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "delayPeriod_instructScreen" ---
    delayPeriod_instructText = visual.TextStim(win=win, name='delayPeriod_instructText',
        text='After you are briefly shown the memory array, there will be brief 1 second delay. During this delay, we want you to remember the spatail positions of the 5 colored circles you just saw.\n\nThe delay period screen will look like this:',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    delayPeriod_instructImage = visual.ImageStim(
        win=win,
        name='delayPeriod_instructImage', 
        image='instruction_images/delay_instruction.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.2), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    delayPeriod_continueText = visual.TextStim(win=win, name='delayPeriod_continueText',
        text='<press SPACE to continue>',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    delayPeriod_keyResp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "wholeReport_instructScreen" ---
    wholeReport_instructText = visual.TextStim(win=win, name='wholeReport_instructText',
        text='At the end of each trial, we will ask you to use the computer mouse to report the remembered position of each colored dot you memorized. To do this, you will click on one of the colors below the response circle to choose which colored dot position you will report. After you select the color, you will use the mouse to drag the colored dot to the remembered position. After you make your response, the color option below will become transparent. This indicates you already reported the position for that colored dot. You will follow the same response procedure until the remembered positions of all colored dots have been reported.\n\nThe response procedure will look something like this:',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    wholeReport_instructImage = visual.ImageStim(
        win=win,
        name='wholeReport_instructImage', 
        image='instruction_images/report_instruction.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.2), draggable=False, size=(0.5, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    wholeReport_continueText = visual.TextStim(win=win, name='wholeReport_continueText',
        text='<press SPACE to continue>',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    wholeReport_keyResp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Practice_startScreen" ---
    practiceStart_text = visual.TextStim(win=win, name='practiceStart_text',
        text='We will start with a few practice trials so you understand the structure of the experiment and how to make response.\n\n<press SPACE to continue>',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    practiceStart_keyResp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "prePhase_blank" ---
    prePhase_fix = visual.Rect(
        win=win, name='prePhase_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "ITI" ---
    ITI_fix = visual.Rect(
        win=win, name='ITI_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "mem_array" ---
    mem_fix = visual.Rect(
        win=win, name='mem_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=-1.0, interpolate=True)
    memCircle = visual.ShapeStim(
        win=win, name='memCircle',
        size=(0.45, 0.45), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='gray', fillColor=None,
        opacity=None, depth=-2.0, interpolate=True)
    dot0 = visual.ShapeStim(
        win=win, name='dot0',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    dot1 = visual.ShapeStim(
        win=win, name='dot1',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    dot2 = visual.ShapeStim(
        win=win, name='dot2',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    dot3 = visual.ShapeStim(
        win=win, name='dot3',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    dot4 = visual.ShapeStim(
        win=win, name='dot4',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-7.0, interpolate=True)
    
    # --- Initialize components for Routine "delay" ---
    delay_fix = visual.Rect(
        win=win, name='delay_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "whole_report" ---
    report_fix = visual.Rect(
        win=win, name='report_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=-1.0, interpolate=True)
    reportCircle = visual.ShapeStim(
        win=win, name='reportCircle',
        size=(0.45, 0.45), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-2.0, interpolate=True)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    cursorDot = visual.ShapeStim(
        win=win, name='cursorDot',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "Practice_doneScreen" ---
    practiceDone_text = visual.TextStim(win=win, name='practiceDone_text',
        text='All done with the practice trials!\n\nIf you have an questions, please ask the experimenter now…\n\n<press SPACE when you are ready to continue>',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    practiceDone_keyResp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Main_startScreen" ---
    mainStart_text = visual.TextStim(win=win, name='mainStart_text',
        text='We are about to start the main portion of the experiment.\n\nPlease ask the experimenter any final questions you may have…\n\n<press SPACE when you are ready to start>',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mainStart_keyResp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "prePhase_blank" ---
    prePhase_fix = visual.Rect(
        win=win, name='prePhase_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "ITI" ---
    ITI_fix = visual.Rect(
        win=win, name='ITI_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "mem_array" ---
    mem_fix = visual.Rect(
        win=win, name='mem_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=-1.0, interpolate=True)
    memCircle = visual.ShapeStim(
        win=win, name='memCircle',
        size=(0.45, 0.45), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='gray', fillColor=None,
        opacity=None, depth=-2.0, interpolate=True)
    dot0 = visual.ShapeStim(
        win=win, name='dot0',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    dot1 = visual.ShapeStim(
        win=win, name='dot1',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    dot2 = visual.ShapeStim(
        win=win, name='dot2',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    dot3 = visual.ShapeStim(
        win=win, name='dot3',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    dot4 = visual.ShapeStim(
        win=win, name='dot4',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-7.0, interpolate=True)
    
    # --- Initialize components for Routine "delay" ---
    delay_fix = visual.Rect(
        win=win, name='delay_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "whole_report" ---
    report_fix = visual.Rect(
        win=win, name='report_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=-1.0, interpolate=True)
    reportCircle = visual.ShapeStim(
        win=win, name='reportCircle',
        size=(0.45, 0.45), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-2.0, interpolate=True)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    cursorDot = visual.ShapeStim(
        win=win, name='cursorDot',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "END_screen" ---
    end_screen = visual.Rect(
        win=win, name='end_screen',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='red', fillColor='red',
        opacity=None, depth=0.0, interpolate=True)
    end_text = visual.TextStim(win=win, name='end_text',
        text='Congratulations, you are all done with this portion of the experiment!\n\n<press SPACE to exit>',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    end_keyResp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "init" ---
    # create an object to store info about Routine init
    init = data.Routine(
        name='init',
        components=[],
    )
    init.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_init
    import os as _os
    import csv
    import numpy as np
    
    # -----------------------------
    # SETTINGS
    # -----------------------------
    N_COLORS_PER_TRIAL = 5
    N_TRIALS_MAIN = 125
    N_TRIALS_PRACTICE = 4
    
    COLOR_POOL_9 = ["red", "green", "blue", "yellow", "magenta",
                    "cyan", "orange", "white", "black"]
    
    MIN_SEP_DEG = 25
    RECENT_WINDOW = 12
    RECENT_PENALTY = 1.5
    
    # fixed circle geometry
    CIRCLE_CENTER_X = 0.0
    CIRCLE_CENTER_Y = 0.0
    CIRCLE_RADIUS = 0.45 / 2.0  # 0.225
    
    # -----------------------------
    # RNG: stable per participant
    # -----------------------------
    participant_number = expInfo['participant']
    seed = int(''.join([c for c in str(participant_number) if c.isdigit()]) or 0) + 12345
    rng = np.random.default_rng(seed)
    
    # -----------------------------
    # Helpers
    # -----------------------------
    def circ_dist(a, b):
        return abs(np.angle(np.exp(1j * (a - b))))
    
    def sample_angles_bilateral(n, rng, min_sep_deg=25, max_tries=50000):
        min_sep = np.deg2rad(min_sep_deg)
        n_left = n // 2
        n_right = n - n_left
    
        def rand_angle_left():
            return rng.uniform(np.pi/2, 3*np.pi/2)
    
        def rand_angle_right():
            return rng.uniform(-np.pi/2, np.pi/2) % (2*np.pi)
    
        angles = []
        tries = 0
    
        for _ in range(n_left):
            while tries < max_tries:
                tries += 1
                a = rand_angle_left()
                if all(circ_dist(a, b) >= min_sep for b in angles):
                    angles.append(a)
                    break
    
        for _ in range(n_right):
            while tries < max_tries:
                tries += 1
                a = rand_angle_right()
                if all(circ_dist(a, b) >= min_sep for b in angles):
                    angles.append(a)
                    break
    
        angles = np.array(angles)
        rng.shuffle(angles)
        return angles
    
    def rad_to_deg_360(a):
        return (np.degrees(a) + 360) % 360
    
    def make_balanced_color_trials(n_trials, colors, k_per_trial, rng,
                                   recent_window=10, recent_penalty=1.0,
                                   max_restarts=200):
        n_colors = len(colors)
        total_slots = n_trials * k_per_trial
        base = total_slots // n_colors
        rem = total_slots % n_colors
    
        target = np.full(n_colors, base)
        if rem:
            target[rng.choice(n_colors, rem, replace=False)] += 1
    
        for _ in range(max_restarts):
            remaining = target.copy()
            trials = []
            recent = []
    
            ok = True
            for _ in range(n_trials):
                recent_counts = np.zeros(n_colors)
                for prev in recent[-recent_window:]:
                    for idx in prev:
                        recent_counts[idx] += 1
    
                weights = remaining / (1 + recent_penalty * recent_counts)
                avail = np.where(remaining > 0)[0]
    
                if len(avail) < k_per_trial:
                    ok = False
                    break
    
                probs = weights[avail]
                probs = probs / probs.sum() if probs.sum() > 0 else None
                chosen = rng.choice(avail, k_per_trial, replace=False, p=probs)
    
                for idx in chosen:
                    remaining[idx] -= 1
                    if remaining[idx] < 0:
                        ok = False
                        break
    
                trials.append(chosen.tolist())
                recent.append(chosen.tolist())
    
            if ok and remaining.sum() == 0:
                return [[colors[i] for i in trial] for trial in trials]
    
        raise RuntimeError("Could not generate balanced color trials.")
    
    # -----------------------------
    # CSV writer
    # -----------------------------
    def write_design_csv(path, rows):
        fieldnames = ["trial", "phase"]
    
        for i in range(N_COLORS_PER_TRIAL):
            fieldnames += [
                f"item{i}_color",
                f"item{i}_x", f"item{i}_y",
                f"item{i}_angle_rad",
                f"item{i}_angle_deg",
            ]
    
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    
    # -----------------------------
    # Paths
    # -----------------------------
    expDir = _os.path.dirname(_os.path.abspath(__file__))
    base_dir = _os.path.join(expDir, "subject_designs", f"sub-{participant_number}")
    practice_dir = _os.path.join(base_dir, "practice")
    main_dir = _os.path.join(base_dir, "main")
    _os.makedirs(practice_dir, exist_ok=True)
    _os.makedirs(main_dir, exist_ok=True)
    
    practice_csv = _os.path.join(practice_dir, "practice_design.csv")
    main_csv = _os.path.join(main_dir, "main_design.csv")
    
    # -----------------------------
    # PRACTICE
    # -----------------------------
    practice_colors = make_balanced_color_trials(
        N_TRIALS_PRACTICE, COLOR_POOL_9, N_COLORS_PER_TRIAL, rng,
        RECENT_WINDOW, RECENT_PENALTY
    )
    
    practice_rows = []
    for t, cols in enumerate(practice_colors):
        ang = sample_angles_bilateral(N_COLORS_PER_TRIAL, rng, MIN_SEP_DEG)
        xy = np.c_[CIRCLE_RADIUS*np.cos(ang), CIRCLE_RADIUS*np.sin(ang)]
    
        row = {"trial": t, "phase": "practice"}
        for i in range(N_COLORS_PER_TRIAL):
            row[f"item{i}_color"] = cols[i]
            row[f"item{i}_x"] = float(xy[i, 0])
            row[f"item{i}_y"] = float(xy[i, 1])
            row[f"item{i}_angle_rad"] = float(ang[i])
            row[f"item{i}_angle_deg"] = float(rad_to_deg_360(ang[i]))
    
        practice_rows.append(row)
    
    write_design_csv(practice_csv, practice_rows)
    
    # -----------------------------
    # MAIN
    # -----------------------------
    main_colors = make_balanced_color_trials(
        N_TRIALS_MAIN, COLOR_POOL_9, N_COLORS_PER_TRIAL, rng,
        RECENT_WINDOW, RECENT_PENALTY
    )
    
    main_rows = []
    for t, cols in enumerate(main_colors):
        ang = sample_angles_bilateral(N_COLORS_PER_TRIAL, rng, MIN_SEP_DEG)
        xy = np.c_[CIRCLE_RADIUS*np.cos(ang), CIRCLE_RADIUS*np.sin(ang)]
    
        row = {"trial": t, "phase": "main"}
        for i in range(N_COLORS_PER_TRIAL):
            row[f"item{i}_color"] = cols[i]
            row[f"item{i}_x"] = float(xy[i, 0])
            row[f"item{i}_y"] = float(xy[i, 1])
            row[f"item{i}_angle_rad"] = float(ang[i])
            row[f"item{i}_angle_deg"] = float(rad_to_deg_360(ang[i]))
    
        main_rows.append(row)
    
    write_design_csv(main_csv, main_rows)
    
    condFilePractice = practice_csv
    condFileMain = main_csv
    
    # store start times for init
    init.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    init.tStart = globalClock.getTime(format='float')
    init.status = STARTED
    thisExp.addData('init.started', init.tStart)
    init.maxDuration = None
    # keep track of which components have finished
    initComponents = init.components
    for thisComponent in init.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "init" ---
    thisExp.currentRoutine = init
    init.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=init,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            init.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if init.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in init.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "init" ---
    for thisComponent in init.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for init
    init.tStop = globalClock.getTime(format='float')
    init.tStopRefresh = tThisFlipGlobal
    thisExp.addData('init.stopped', init.tStop)
    # Run 'End Routine' code from code_init
    # READ-IN AND ASSIGN PRACTICE AND MAIN DESIGN MATRICES
    import os as _os
    
    participant_number = expInfo['participant']
    
    expDir = _os.path.dirname(_os.path.abspath(__file__))
    
    practice_list = _os.path.join(expDir, 'subject_designs', f'sub-{participant_number}', 'practice', 'practice_design.csv')
    
    main_list = _os.path.join(expDir, 'subject_designs', f'sub-{participant_number}', 'main', 'main_design.csv')
    thisExp.nextEntry()
    # the Routine "init" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "WELCOME_screen" ---
    # create an object to store info about Routine WELCOME_screen
    WELCOME_screen = data.Routine(
        name='WELCOME_screen',
        components=[welcome_text, welcome_keyResp],
    )
    WELCOME_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for welcome_keyResp
    welcome_keyResp.keys = []
    welcome_keyResp.rt = []
    _welcome_keyResp_allKeys = []
    # store start times for WELCOME_screen
    WELCOME_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    WELCOME_screen.tStart = globalClock.getTime(format='float')
    WELCOME_screen.status = STARTED
    thisExp.addData('WELCOME_screen.started', WELCOME_screen.tStart)
    WELCOME_screen.maxDuration = None
    # keep track of which components have finished
    WELCOME_screenComponents = WELCOME_screen.components
    for thisComponent in WELCOME_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "WELCOME_screen" ---
    thisExp.currentRoutine = WELCOME_screen
    WELCOME_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
            # update params
            pass
        
        # *welcome_keyResp* updates
        waitOnFlip = False
        
        # if welcome_keyResp is starting this frame...
        if welcome_keyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_keyResp.frameNStart = frameN  # exact frame index
            welcome_keyResp.tStart = t  # local t and not account for scr refresh
            welcome_keyResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_keyResp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_keyResp.started')
            # update status
            welcome_keyResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(welcome_keyResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(welcome_keyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if welcome_keyResp.status == STARTED and not waitOnFlip:
            theseKeys = welcome_keyResp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _welcome_keyResp_allKeys.extend(theseKeys)
            if len(_welcome_keyResp_allKeys):
                welcome_keyResp.keys = _welcome_keyResp_allKeys[-1].name  # just the last key pressed
                welcome_keyResp.rt = _welcome_keyResp_allKeys[-1].rt
                welcome_keyResp.duration = _welcome_keyResp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=WELCOME_screen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            WELCOME_screen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if WELCOME_screen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in WELCOME_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "WELCOME_screen" ---
    for thisComponent in WELCOME_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for WELCOME_screen
    WELCOME_screen.tStop = globalClock.getTime(format='float')
    WELCOME_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('WELCOME_screen.stopped', WELCOME_screen.tStop)
    # check responses
    if welcome_keyResp.keys in ['', [], None]:  # No response was made
        welcome_keyResp.keys = None
    thisExp.addData('welcome_keyResp.keys',welcome_keyResp.keys)
    if welcome_keyResp.keys != None:  # we had a response
        thisExp.addData('welcome_keyResp.rt', welcome_keyResp.rt)
        thisExp.addData('welcome_keyResp.duration', welcome_keyResp.duration)
    thisExp.nextEntry()
    # the Routine "WELCOME_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ITI_instructScreen" ---
    # create an object to store info about Routine ITI_instructScreen
    ITI_instructScreen = data.Routine(
        name='ITI_instructScreen',
        components=[iti_instructText, iti_instructImage, iti_continueText, itiInstruct_keyResp],
    )
    ITI_instructScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for itiInstruct_keyResp
    itiInstruct_keyResp.keys = []
    itiInstruct_keyResp.rt = []
    _itiInstruct_keyResp_allKeys = []
    # store start times for ITI_instructScreen
    ITI_instructScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ITI_instructScreen.tStart = globalClock.getTime(format='float')
    ITI_instructScreen.status = STARTED
    thisExp.addData('ITI_instructScreen.started', ITI_instructScreen.tStart)
    ITI_instructScreen.maxDuration = None
    # keep track of which components have finished
    ITI_instructScreenComponents = ITI_instructScreen.components
    for thisComponent in ITI_instructScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ITI_instructScreen" ---
    thisExp.currentRoutine = ITI_instructScreen
    ITI_instructScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *iti_instructText* updates
        
        # if iti_instructText is starting this frame...
        if iti_instructText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            iti_instructText.frameNStart = frameN  # exact frame index
            iti_instructText.tStart = t  # local t and not account for scr refresh
            iti_instructText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(iti_instructText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'iti_instructText.started')
            # update status
            iti_instructText.status = STARTED
            iti_instructText.setAutoDraw(True)
        
        # if iti_instructText is active this frame...
        if iti_instructText.status == STARTED:
            # update params
            pass
        
        # *iti_instructImage* updates
        
        # if iti_instructImage is starting this frame...
        if iti_instructImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            iti_instructImage.frameNStart = frameN  # exact frame index
            iti_instructImage.tStart = t  # local t and not account for scr refresh
            iti_instructImage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(iti_instructImage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'iti_instructImage.started')
            # update status
            iti_instructImage.status = STARTED
            iti_instructImage.setAutoDraw(True)
        
        # if iti_instructImage is active this frame...
        if iti_instructImage.status == STARTED:
            # update params
            pass
        
        # *iti_continueText* updates
        
        # if iti_continueText is starting this frame...
        if iti_continueText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            iti_continueText.frameNStart = frameN  # exact frame index
            iti_continueText.tStart = t  # local t and not account for scr refresh
            iti_continueText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(iti_continueText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'iti_continueText.started')
            # update status
            iti_continueText.status = STARTED
            iti_continueText.setAutoDraw(True)
        
        # if iti_continueText is active this frame...
        if iti_continueText.status == STARTED:
            # update params
            pass
        
        # *itiInstruct_keyResp* updates
        waitOnFlip = False
        
        # if itiInstruct_keyResp is starting this frame...
        if itiInstruct_keyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            itiInstruct_keyResp.frameNStart = frameN  # exact frame index
            itiInstruct_keyResp.tStart = t  # local t and not account for scr refresh
            itiInstruct_keyResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(itiInstruct_keyResp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'itiInstruct_keyResp.started')
            # update status
            itiInstruct_keyResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(itiInstruct_keyResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(itiInstruct_keyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if itiInstruct_keyResp.status == STARTED and not waitOnFlip:
            theseKeys = itiInstruct_keyResp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _itiInstruct_keyResp_allKeys.extend(theseKeys)
            if len(_itiInstruct_keyResp_allKeys):
                itiInstruct_keyResp.keys = _itiInstruct_keyResp_allKeys[-1].name  # just the last key pressed
                itiInstruct_keyResp.rt = _itiInstruct_keyResp_allKeys[-1].rt
                itiInstruct_keyResp.duration = _itiInstruct_keyResp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=ITI_instructScreen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ITI_instructScreen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ITI_instructScreen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ITI_instructScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ITI_instructScreen" ---
    for thisComponent in ITI_instructScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ITI_instructScreen
    ITI_instructScreen.tStop = globalClock.getTime(format='float')
    ITI_instructScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ITI_instructScreen.stopped', ITI_instructScreen.tStop)
    # check responses
    if itiInstruct_keyResp.keys in ['', [], None]:  # No response was made
        itiInstruct_keyResp.keys = None
    thisExp.addData('itiInstruct_keyResp.keys',itiInstruct_keyResp.keys)
    if itiInstruct_keyResp.keys != None:  # we had a response
        thisExp.addData('itiInstruct_keyResp.rt', itiInstruct_keyResp.rt)
        thisExp.addData('itiInstruct_keyResp.duration', itiInstruct_keyResp.duration)
    thisExp.nextEntry()
    # the Routine "ITI_instructScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "memArray_instructScreen" ---
    # create an object to store info about Routine memArray_instructScreen
    memArray_instructScreen = data.Routine(
        name='memArray_instructScreen',
        components=[memArray_instructText, memArray_instructImage, memArray_continueText, memArray_keyResp],
    )
    memArray_instructScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for memArray_keyResp
    memArray_keyResp.keys = []
    memArray_keyResp.rt = []
    _memArray_keyResp_allKeys = []
    # store start times for memArray_instructScreen
    memArray_instructScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    memArray_instructScreen.tStart = globalClock.getTime(format='float')
    memArray_instructScreen.status = STARTED
    thisExp.addData('memArray_instructScreen.started', memArray_instructScreen.tStart)
    memArray_instructScreen.maxDuration = None
    # keep track of which components have finished
    memArray_instructScreenComponents = memArray_instructScreen.components
    for thisComponent in memArray_instructScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "memArray_instructScreen" ---
    thisExp.currentRoutine = memArray_instructScreen
    memArray_instructScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *memArray_instructText* updates
        
        # if memArray_instructText is starting this frame...
        if memArray_instructText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            memArray_instructText.frameNStart = frameN  # exact frame index
            memArray_instructText.tStart = t  # local t and not account for scr refresh
            memArray_instructText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(memArray_instructText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'memArray_instructText.started')
            # update status
            memArray_instructText.status = STARTED
            memArray_instructText.setAutoDraw(True)
        
        # if memArray_instructText is active this frame...
        if memArray_instructText.status == STARTED:
            # update params
            pass
        
        # *memArray_instructImage* updates
        
        # if memArray_instructImage is starting this frame...
        if memArray_instructImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            memArray_instructImage.frameNStart = frameN  # exact frame index
            memArray_instructImage.tStart = t  # local t and not account for scr refresh
            memArray_instructImage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(memArray_instructImage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'memArray_instructImage.started')
            # update status
            memArray_instructImage.status = STARTED
            memArray_instructImage.setAutoDraw(True)
        
        # if memArray_instructImage is active this frame...
        if memArray_instructImage.status == STARTED:
            # update params
            pass
        
        # *memArray_continueText* updates
        
        # if memArray_continueText is starting this frame...
        if memArray_continueText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            memArray_continueText.frameNStart = frameN  # exact frame index
            memArray_continueText.tStart = t  # local t and not account for scr refresh
            memArray_continueText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(memArray_continueText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'memArray_continueText.started')
            # update status
            memArray_continueText.status = STARTED
            memArray_continueText.setAutoDraw(True)
        
        # if memArray_continueText is active this frame...
        if memArray_continueText.status == STARTED:
            # update params
            pass
        
        # *memArray_keyResp* updates
        waitOnFlip = False
        
        # if memArray_keyResp is starting this frame...
        if memArray_keyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            memArray_keyResp.frameNStart = frameN  # exact frame index
            memArray_keyResp.tStart = t  # local t and not account for scr refresh
            memArray_keyResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(memArray_keyResp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'memArray_keyResp.started')
            # update status
            memArray_keyResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(memArray_keyResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(memArray_keyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if memArray_keyResp.status == STARTED and not waitOnFlip:
            theseKeys = memArray_keyResp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _memArray_keyResp_allKeys.extend(theseKeys)
            if len(_memArray_keyResp_allKeys):
                memArray_keyResp.keys = _memArray_keyResp_allKeys[-1].name  # just the last key pressed
                memArray_keyResp.rt = _memArray_keyResp_allKeys[-1].rt
                memArray_keyResp.duration = _memArray_keyResp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=memArray_instructScreen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            memArray_instructScreen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if memArray_instructScreen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in memArray_instructScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "memArray_instructScreen" ---
    for thisComponent in memArray_instructScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for memArray_instructScreen
    memArray_instructScreen.tStop = globalClock.getTime(format='float')
    memArray_instructScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('memArray_instructScreen.stopped', memArray_instructScreen.tStop)
    # check responses
    if memArray_keyResp.keys in ['', [], None]:  # No response was made
        memArray_keyResp.keys = None
    thisExp.addData('memArray_keyResp.keys',memArray_keyResp.keys)
    if memArray_keyResp.keys != None:  # we had a response
        thisExp.addData('memArray_keyResp.rt', memArray_keyResp.rt)
        thisExp.addData('memArray_keyResp.duration', memArray_keyResp.duration)
    thisExp.nextEntry()
    # the Routine "memArray_instructScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "delayPeriod_instructScreen" ---
    # create an object to store info about Routine delayPeriod_instructScreen
    delayPeriod_instructScreen = data.Routine(
        name='delayPeriod_instructScreen',
        components=[delayPeriod_instructText, delayPeriod_instructImage, delayPeriod_continueText, delayPeriod_keyResp],
    )
    delayPeriod_instructScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for delayPeriod_keyResp
    delayPeriod_keyResp.keys = []
    delayPeriod_keyResp.rt = []
    _delayPeriod_keyResp_allKeys = []
    # store start times for delayPeriod_instructScreen
    delayPeriod_instructScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    delayPeriod_instructScreen.tStart = globalClock.getTime(format='float')
    delayPeriod_instructScreen.status = STARTED
    thisExp.addData('delayPeriod_instructScreen.started', delayPeriod_instructScreen.tStart)
    delayPeriod_instructScreen.maxDuration = None
    # keep track of which components have finished
    delayPeriod_instructScreenComponents = delayPeriod_instructScreen.components
    for thisComponent in delayPeriod_instructScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "delayPeriod_instructScreen" ---
    thisExp.currentRoutine = delayPeriod_instructScreen
    delayPeriod_instructScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *delayPeriod_instructText* updates
        
        # if delayPeriod_instructText is starting this frame...
        if delayPeriod_instructText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            delayPeriod_instructText.frameNStart = frameN  # exact frame index
            delayPeriod_instructText.tStart = t  # local t and not account for scr refresh
            delayPeriod_instructText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(delayPeriod_instructText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'delayPeriod_instructText.started')
            # update status
            delayPeriod_instructText.status = STARTED
            delayPeriod_instructText.setAutoDraw(True)
        
        # if delayPeriod_instructText is active this frame...
        if delayPeriod_instructText.status == STARTED:
            # update params
            pass
        
        # *delayPeriod_instructImage* updates
        
        # if delayPeriod_instructImage is starting this frame...
        if delayPeriod_instructImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            delayPeriod_instructImage.frameNStart = frameN  # exact frame index
            delayPeriod_instructImage.tStart = t  # local t and not account for scr refresh
            delayPeriod_instructImage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(delayPeriod_instructImage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'delayPeriod_instructImage.started')
            # update status
            delayPeriod_instructImage.status = STARTED
            delayPeriod_instructImage.setAutoDraw(True)
        
        # if delayPeriod_instructImage is active this frame...
        if delayPeriod_instructImage.status == STARTED:
            # update params
            pass
        
        # *delayPeriod_continueText* updates
        
        # if delayPeriod_continueText is starting this frame...
        if delayPeriod_continueText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            delayPeriod_continueText.frameNStart = frameN  # exact frame index
            delayPeriod_continueText.tStart = t  # local t and not account for scr refresh
            delayPeriod_continueText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(delayPeriod_continueText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'delayPeriod_continueText.started')
            # update status
            delayPeriod_continueText.status = STARTED
            delayPeriod_continueText.setAutoDraw(True)
        
        # if delayPeriod_continueText is active this frame...
        if delayPeriod_continueText.status == STARTED:
            # update params
            pass
        
        # *delayPeriod_keyResp* updates
        waitOnFlip = False
        
        # if delayPeriod_keyResp is starting this frame...
        if delayPeriod_keyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            delayPeriod_keyResp.frameNStart = frameN  # exact frame index
            delayPeriod_keyResp.tStart = t  # local t and not account for scr refresh
            delayPeriod_keyResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(delayPeriod_keyResp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'delayPeriod_keyResp.started')
            # update status
            delayPeriod_keyResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(delayPeriod_keyResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(delayPeriod_keyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if delayPeriod_keyResp.status == STARTED and not waitOnFlip:
            theseKeys = delayPeriod_keyResp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _delayPeriod_keyResp_allKeys.extend(theseKeys)
            if len(_delayPeriod_keyResp_allKeys):
                delayPeriod_keyResp.keys = _delayPeriod_keyResp_allKeys[-1].name  # just the last key pressed
                delayPeriod_keyResp.rt = _delayPeriod_keyResp_allKeys[-1].rt
                delayPeriod_keyResp.duration = _delayPeriod_keyResp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=delayPeriod_instructScreen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            delayPeriod_instructScreen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if delayPeriod_instructScreen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in delayPeriod_instructScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "delayPeriod_instructScreen" ---
    for thisComponent in delayPeriod_instructScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for delayPeriod_instructScreen
    delayPeriod_instructScreen.tStop = globalClock.getTime(format='float')
    delayPeriod_instructScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('delayPeriod_instructScreen.stopped', delayPeriod_instructScreen.tStop)
    # check responses
    if delayPeriod_keyResp.keys in ['', [], None]:  # No response was made
        delayPeriod_keyResp.keys = None
    thisExp.addData('delayPeriod_keyResp.keys',delayPeriod_keyResp.keys)
    if delayPeriod_keyResp.keys != None:  # we had a response
        thisExp.addData('delayPeriod_keyResp.rt', delayPeriod_keyResp.rt)
        thisExp.addData('delayPeriod_keyResp.duration', delayPeriod_keyResp.duration)
    thisExp.nextEntry()
    # the Routine "delayPeriod_instructScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "wholeReport_instructScreen" ---
    # create an object to store info about Routine wholeReport_instructScreen
    wholeReport_instructScreen = data.Routine(
        name='wholeReport_instructScreen',
        components=[wholeReport_instructText, wholeReport_instructImage, wholeReport_continueText, wholeReport_keyResp],
    )
    wholeReport_instructScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for wholeReport_keyResp
    wholeReport_keyResp.keys = []
    wholeReport_keyResp.rt = []
    _wholeReport_keyResp_allKeys = []
    # store start times for wholeReport_instructScreen
    wholeReport_instructScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    wholeReport_instructScreen.tStart = globalClock.getTime(format='float')
    wholeReport_instructScreen.status = STARTED
    thisExp.addData('wholeReport_instructScreen.started', wholeReport_instructScreen.tStart)
    wholeReport_instructScreen.maxDuration = None
    # keep track of which components have finished
    wholeReport_instructScreenComponents = wholeReport_instructScreen.components
    for thisComponent in wholeReport_instructScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "wholeReport_instructScreen" ---
    thisExp.currentRoutine = wholeReport_instructScreen
    wholeReport_instructScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *wholeReport_instructText* updates
        
        # if wholeReport_instructText is starting this frame...
        if wholeReport_instructText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            wholeReport_instructText.frameNStart = frameN  # exact frame index
            wholeReport_instructText.tStart = t  # local t and not account for scr refresh
            wholeReport_instructText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(wholeReport_instructText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'wholeReport_instructText.started')
            # update status
            wholeReport_instructText.status = STARTED
            wholeReport_instructText.setAutoDraw(True)
        
        # if wholeReport_instructText is active this frame...
        if wholeReport_instructText.status == STARTED:
            # update params
            pass
        
        # *wholeReport_instructImage* updates
        
        # if wholeReport_instructImage is starting this frame...
        if wholeReport_instructImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            wholeReport_instructImage.frameNStart = frameN  # exact frame index
            wholeReport_instructImage.tStart = t  # local t and not account for scr refresh
            wholeReport_instructImage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(wholeReport_instructImage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'wholeReport_instructImage.started')
            # update status
            wholeReport_instructImage.status = STARTED
            wholeReport_instructImage.setAutoDraw(True)
        
        # if wholeReport_instructImage is active this frame...
        if wholeReport_instructImage.status == STARTED:
            # update params
            pass
        
        # *wholeReport_continueText* updates
        
        # if wholeReport_continueText is starting this frame...
        if wholeReport_continueText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            wholeReport_continueText.frameNStart = frameN  # exact frame index
            wholeReport_continueText.tStart = t  # local t and not account for scr refresh
            wholeReport_continueText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(wholeReport_continueText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'wholeReport_continueText.started')
            # update status
            wholeReport_continueText.status = STARTED
            wholeReport_continueText.setAutoDraw(True)
        
        # if wholeReport_continueText is active this frame...
        if wholeReport_continueText.status == STARTED:
            # update params
            pass
        
        # *wholeReport_keyResp* updates
        waitOnFlip = False
        
        # if wholeReport_keyResp is starting this frame...
        if wholeReport_keyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            wholeReport_keyResp.frameNStart = frameN  # exact frame index
            wholeReport_keyResp.tStart = t  # local t and not account for scr refresh
            wholeReport_keyResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(wholeReport_keyResp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'wholeReport_keyResp.started')
            # update status
            wholeReport_keyResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(wholeReport_keyResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(wholeReport_keyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if wholeReport_keyResp.status == STARTED and not waitOnFlip:
            theseKeys = wholeReport_keyResp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _wholeReport_keyResp_allKeys.extend(theseKeys)
            if len(_wholeReport_keyResp_allKeys):
                wholeReport_keyResp.keys = _wholeReport_keyResp_allKeys[-1].name  # just the last key pressed
                wholeReport_keyResp.rt = _wholeReport_keyResp_allKeys[-1].rt
                wholeReport_keyResp.duration = _wholeReport_keyResp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=wholeReport_instructScreen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            wholeReport_instructScreen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if wholeReport_instructScreen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in wholeReport_instructScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "wholeReport_instructScreen" ---
    for thisComponent in wholeReport_instructScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for wholeReport_instructScreen
    wholeReport_instructScreen.tStop = globalClock.getTime(format='float')
    wholeReport_instructScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('wholeReport_instructScreen.stopped', wholeReport_instructScreen.tStop)
    # check responses
    if wholeReport_keyResp.keys in ['', [], None]:  # No response was made
        wholeReport_keyResp.keys = None
    thisExp.addData('wholeReport_keyResp.keys',wholeReport_keyResp.keys)
    if wholeReport_keyResp.keys != None:  # we had a response
        thisExp.addData('wholeReport_keyResp.rt', wholeReport_keyResp.rt)
        thisExp.addData('wholeReport_keyResp.duration', wholeReport_keyResp.duration)
    thisExp.nextEntry()
    # the Routine "wholeReport_instructScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Practice_startScreen" ---
    # create an object to store info about Routine Practice_startScreen
    Practice_startScreen = data.Routine(
        name='Practice_startScreen',
        components=[practiceStart_text, practiceStart_keyResp],
    )
    Practice_startScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for practiceStart_keyResp
    practiceStart_keyResp.keys = []
    practiceStart_keyResp.rt = []
    _practiceStart_keyResp_allKeys = []
    # store start times for Practice_startScreen
    Practice_startScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Practice_startScreen.tStart = globalClock.getTime(format='float')
    Practice_startScreen.status = STARTED
    thisExp.addData('Practice_startScreen.started', Practice_startScreen.tStart)
    Practice_startScreen.maxDuration = None
    # keep track of which components have finished
    Practice_startScreenComponents = Practice_startScreen.components
    for thisComponent in Practice_startScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Practice_startScreen" ---
    thisExp.currentRoutine = Practice_startScreen
    Practice_startScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practiceStart_text* updates
        
        # if practiceStart_text is starting this frame...
        if practiceStart_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practiceStart_text.frameNStart = frameN  # exact frame index
            practiceStart_text.tStart = t  # local t and not account for scr refresh
            practiceStart_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practiceStart_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practiceStart_text.started')
            # update status
            practiceStart_text.status = STARTED
            practiceStart_text.setAutoDraw(True)
        
        # if practiceStart_text is active this frame...
        if practiceStart_text.status == STARTED:
            # update params
            pass
        
        # *practiceStart_keyResp* updates
        waitOnFlip = False
        
        # if practiceStart_keyResp is starting this frame...
        if practiceStart_keyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practiceStart_keyResp.frameNStart = frameN  # exact frame index
            practiceStart_keyResp.tStart = t  # local t and not account for scr refresh
            practiceStart_keyResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practiceStart_keyResp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practiceStart_keyResp.started')
            # update status
            practiceStart_keyResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(practiceStart_keyResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(practiceStart_keyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if practiceStart_keyResp.status == STARTED and not waitOnFlip:
            theseKeys = practiceStart_keyResp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _practiceStart_keyResp_allKeys.extend(theseKeys)
            if len(_practiceStart_keyResp_allKeys):
                practiceStart_keyResp.keys = _practiceStart_keyResp_allKeys[-1].name  # just the last key pressed
                practiceStart_keyResp.rt = _practiceStart_keyResp_allKeys[-1].rt
                practiceStart_keyResp.duration = _practiceStart_keyResp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Practice_startScreen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Practice_startScreen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Practice_startScreen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Practice_startScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Practice_startScreen" ---
    for thisComponent in Practice_startScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Practice_startScreen
    Practice_startScreen.tStop = globalClock.getTime(format='float')
    Practice_startScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Practice_startScreen.stopped', Practice_startScreen.tStop)
    # check responses
    if practiceStart_keyResp.keys in ['', [], None]:  # No response was made
        practiceStart_keyResp.keys = None
    thisExp.addData('practiceStart_keyResp.keys',practiceStart_keyResp.keys)
    if practiceStart_keyResp.keys != None:  # we had a response
        thisExp.addData('practiceStart_keyResp.rt', practiceStart_keyResp.rt)
        thisExp.addData('practiceStart_keyResp.duration', practiceStart_keyResp.duration)
    thisExp.nextEntry()
    # the Routine "Practice_startScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "prePhase_blank" ---
    # create an object to store info about Routine prePhase_blank
    prePhase_blank = data.Routine(
        name='prePhase_blank',
        components=[prePhase_fix],
    )
    prePhase_blank.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for prePhase_blank
    prePhase_blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prePhase_blank.tStart = globalClock.getTime(format='float')
    prePhase_blank.status = STARTED
    thisExp.addData('prePhase_blank.started', prePhase_blank.tStart)
    prePhase_blank.maxDuration = None
    # keep track of which components have finished
    prePhase_blankComponents = prePhase_blank.components
    for thisComponent in prePhase_blank.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "prePhase_blank" ---
    thisExp.currentRoutine = prePhase_blank
    prePhase_blank.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *prePhase_fix* updates
        
        # if prePhase_fix is starting this frame...
        if prePhase_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prePhase_fix.frameNStart = frameN  # exact frame index
            prePhase_fix.tStart = t  # local t and not account for scr refresh
            prePhase_fix.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prePhase_fix, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prePhase_fix.started')
            # update status
            prePhase_fix.status = STARTED
            prePhase_fix.setAutoDraw(True)
        
        # if prePhase_fix is active this frame...
        if prePhase_fix.status == STARTED:
            # update params
            pass
        
        # if prePhase_fix is stopping this frame...
        if prePhase_fix.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > prePhase_fix.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                prePhase_fix.tStop = t  # not accounting for scr refresh
                prePhase_fix.tStopRefresh = tThisFlipGlobal  # on global time
                prePhase_fix.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prePhase_fix.stopped')
                # update status
                prePhase_fix.status = FINISHED
                prePhase_fix.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=prePhase_blank,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            prePhase_blank.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if prePhase_blank.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in prePhase_blank.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prePhase_blank" ---
    for thisComponent in prePhase_blank.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prePhase_blank
    prePhase_blank.tStop = globalClock.getTime(format='float')
    prePhase_blank.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prePhase_blank.stopped', prePhase_blank.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if prePhase_blank.maxDurationReached:
        routineTimer.addTime(-prePhase_blank.maxDuration)
    elif prePhase_blank.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    practice_trials = data.TrialHandler2(
        name='practice_trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(practice_list), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(practice_trials)  # add the loop to the experiment
    thisPractice_trial = practice_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
    if thisPractice_trial != None:
        for paramName in thisPractice_trial:
            globals()[paramName] = thisPractice_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_trial in practice_trials:
        practice_trials.status = STARTED
        if hasattr(thisPractice_trial, 'status'):
            thisPractice_trial.status = STARTED
        currentLoop = practice_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
        if thisPractice_trial != None:
            for paramName in thisPractice_trial:
                globals()[paramName] = thisPractice_trial[paramName]
        
        # --- Prepare to start Routine "ITI" ---
        # create an object to store info about Routine ITI
        ITI = data.Routine(
            name='ITI',
            components=[ITI_fix],
        )
        ITI.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for ITI
        ITI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        ITI.tStart = globalClock.getTime(format='float')
        ITI.status = STARTED
        thisExp.addData('ITI.started', ITI.tStart)
        ITI.maxDuration = None
        # keep track of which components have finished
        ITIComponents = ITI.components
        for thisComponent in ITI.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ITI" ---
        thisExp.currentRoutine = ITI
        ITI.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_trial, 'status') and thisPractice_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ITI_fix* updates
            
            # if ITI_fix is starting this frame...
            if ITI_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ITI_fix.frameNStart = frameN  # exact frame index
                ITI_fix.tStart = t  # local t and not account for scr refresh
                ITI_fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ITI_fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ITI_fix.started')
                # update status
                ITI_fix.status = STARTED
                ITI_fix.setAutoDraw(True)
            
            # if ITI_fix is active this frame...
            if ITI_fix.status == STARTED:
                # update params
                pass
            
            # if ITI_fix is stopping this frame...
            if ITI_fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ITI_fix.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    ITI_fix.tStop = t  # not accounting for scr refresh
                    ITI_fix.tStopRefresh = tThisFlipGlobal  # on global time
                    ITI_fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ITI_fix.stopped')
                    # update status
                    ITI_fix.status = FINISHED
                    ITI_fix.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=ITI,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                ITI.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if ITI.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in ITI.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ITI" ---
        for thisComponent in ITI.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for ITI
        ITI.tStop = globalClock.getTime(format='float')
        ITI.tStopRefresh = tThisFlipGlobal
        thisExp.addData('ITI.stopped', ITI.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if ITI.maxDurationReached:
            routineTimer.addTime(-ITI.maxDuration)
        elif ITI.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "mem_array" ---
        # create an object to store info about Routine mem_array
        mem_array = data.Routine(
            name='mem_array',
            components=[mem_fix, memCircle, dot0, dot1, dot2, dot3, dot4],
        )
        mem_array.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from resetArray_code
        # --- hard reset dot objects each trial (array routine) ---
        dots = [dot0, dot1, dot2, dot3, dot4]
        for d in dots:
            d.setAutoDraw(True)     # if you want them drawn in this routine
            d.opacity = 1.0         # ALWAYS reset opacity
            d.ori = 0               # just in case
        
        dot0.setFillColor(item0_color)
        dot0.setPos((item0_x, item0_y))
        dot0.setLineColor(item0_color)
        dot1.setFillColor(item1_color)
        dot1.setPos((item1_x, item1_y))
        dot1.setLineColor(item1_color)
        dot2.setFillColor(item2_color)
        dot2.setPos((item2_x, item2_y))
        dot2.setLineColor(item2_color)
        dot3.setFillColor(item3_color)
        dot3.setPos((item3_x, item3_y))
        dot3.setLineColor(item3_color)
        dot4.setFillColor(item4_color)
        dot4.setPos((item4_x, item4_y))
        dot4.setLineColor(item4_color)
        # store start times for mem_array
        mem_array.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        mem_array.tStart = globalClock.getTime(format='float')
        mem_array.status = STARTED
        thisExp.addData('mem_array.started', mem_array.tStart)
        mem_array.maxDuration = None
        # keep track of which components have finished
        mem_arrayComponents = mem_array.components
        for thisComponent in mem_array.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "mem_array" ---
        thisExp.currentRoutine = mem_array
        mem_array.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.25:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_trial, 'status') and thisPractice_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *mem_fix* updates
            
            # if mem_fix is starting this frame...
            if mem_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mem_fix.frameNStart = frameN  # exact frame index
                mem_fix.tStart = t  # local t and not account for scr refresh
                mem_fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mem_fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mem_fix.started')
                # update status
                mem_fix.status = STARTED
                mem_fix.setAutoDraw(True)
            
            # if mem_fix is active this frame...
            if mem_fix.status == STARTED:
                # update params
                pass
            
            # if mem_fix is stopping this frame...
            if mem_fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > mem_fix.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    mem_fix.tStop = t  # not accounting for scr refresh
                    mem_fix.tStopRefresh = tThisFlipGlobal  # on global time
                    mem_fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'mem_fix.stopped')
                    # update status
                    mem_fix.status = FINISHED
                    mem_fix.setAutoDraw(False)
            
            # *memCircle* updates
            
            # if memCircle is starting this frame...
            if memCircle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                memCircle.frameNStart = frameN  # exact frame index
                memCircle.tStart = t  # local t and not account for scr refresh
                memCircle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(memCircle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'memCircle.started')
                # update status
                memCircle.status = STARTED
                memCircle.setAutoDraw(True)
            
            # if memCircle is active this frame...
            if memCircle.status == STARTED:
                # update params
                pass
            
            # if memCircle is stopping this frame...
            if memCircle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > memCircle.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    memCircle.tStop = t  # not accounting for scr refresh
                    memCircle.tStopRefresh = tThisFlipGlobal  # on global time
                    memCircle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'memCircle.stopped')
                    # update status
                    memCircle.status = FINISHED
                    memCircle.setAutoDraw(False)
            
            # *dot0* updates
            
            # if dot0 is starting this frame...
            if dot0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot0.frameNStart = frameN  # exact frame index
                dot0.tStart = t  # local t and not account for scr refresh
                dot0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot0.started')
                # update status
                dot0.status = STARTED
                dot0.setAutoDraw(True)
            
            # if dot0 is active this frame...
            if dot0.status == STARTED:
                # update params
                pass
            
            # if dot0 is stopping this frame...
            if dot0.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot0.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    dot0.tStop = t  # not accounting for scr refresh
                    dot0.tStopRefresh = tThisFlipGlobal  # on global time
                    dot0.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot0.stopped')
                    # update status
                    dot0.status = FINISHED
                    dot0.setAutoDraw(False)
            
            # *dot1* updates
            
            # if dot1 is starting this frame...
            if dot1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot1.frameNStart = frameN  # exact frame index
                dot1.tStart = t  # local t and not account for scr refresh
                dot1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot1.started')
                # update status
                dot1.status = STARTED
                dot1.setAutoDraw(True)
            
            # if dot1 is active this frame...
            if dot1.status == STARTED:
                # update params
                pass
            
            # if dot1 is stopping this frame...
            if dot1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot1.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    dot1.tStop = t  # not accounting for scr refresh
                    dot1.tStopRefresh = tThisFlipGlobal  # on global time
                    dot1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot1.stopped')
                    # update status
                    dot1.status = FINISHED
                    dot1.setAutoDraw(False)
            
            # *dot2* updates
            
            # if dot2 is starting this frame...
            if dot2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot2.frameNStart = frameN  # exact frame index
                dot2.tStart = t  # local t and not account for scr refresh
                dot2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot2.started')
                # update status
                dot2.status = STARTED
                dot2.setAutoDraw(True)
            
            # if dot2 is active this frame...
            if dot2.status == STARTED:
                # update params
                pass
            
            # if dot2 is stopping this frame...
            if dot2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot2.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    dot2.tStop = t  # not accounting for scr refresh
                    dot2.tStopRefresh = tThisFlipGlobal  # on global time
                    dot2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot2.stopped')
                    # update status
                    dot2.status = FINISHED
                    dot2.setAutoDraw(False)
            
            # *dot3* updates
            
            # if dot3 is starting this frame...
            if dot3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot3.frameNStart = frameN  # exact frame index
                dot3.tStart = t  # local t and not account for scr refresh
                dot3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot3.started')
                # update status
                dot3.status = STARTED
                dot3.setAutoDraw(True)
            
            # if dot3 is active this frame...
            if dot3.status == STARTED:
                # update params
                pass
            
            # if dot3 is stopping this frame...
            if dot3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot3.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    dot3.tStop = t  # not accounting for scr refresh
                    dot3.tStopRefresh = tThisFlipGlobal  # on global time
                    dot3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot3.stopped')
                    # update status
                    dot3.status = FINISHED
                    dot3.setAutoDraw(False)
            
            # *dot4* updates
            
            # if dot4 is starting this frame...
            if dot4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot4.frameNStart = frameN  # exact frame index
                dot4.tStart = t  # local t and not account for scr refresh
                dot4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot4.started')
                # update status
                dot4.status = STARTED
                dot4.setAutoDraw(True)
            
            # if dot4 is active this frame...
            if dot4.status == STARTED:
                # update params
                pass
            
            # if dot4 is stopping this frame...
            if dot4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot4.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    dot4.tStop = t  # not accounting for scr refresh
                    dot4.tStopRefresh = tThisFlipGlobal  # on global time
                    dot4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot4.stopped')
                    # update status
                    dot4.status = FINISHED
                    dot4.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=mem_array,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                mem_array.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if mem_array.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in mem_array.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "mem_array" ---
        for thisComponent in mem_array.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for mem_array
        mem_array.tStop = globalClock.getTime(format='float')
        mem_array.tStopRefresh = tThisFlipGlobal
        thisExp.addData('mem_array.stopped', mem_array.tStop)
        # Run 'End Routine' code from resetArray_code
        for d in dots:
            d.setAutoDraw(False)
        
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if mem_array.maxDurationReached:
            routineTimer.addTime(-mem_array.maxDuration)
        elif mem_array.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.250000)
        
        # --- Prepare to start Routine "delay" ---
        # create an object to store info about Routine delay
        delay = data.Routine(
            name='delay',
            components=[delay_fix],
        )
        delay.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for delay
        delay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        delay.tStart = globalClock.getTime(format='float')
        delay.status = STARTED
        thisExp.addData('delay.started', delay.tStart)
        delay.maxDuration = None
        # keep track of which components have finished
        delayComponents = delay.components
        for thisComponent in delay.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "delay" ---
        thisExp.currentRoutine = delay
        delay.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_trial, 'status') and thisPractice_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *delay_fix* updates
            
            # if delay_fix is starting this frame...
            if delay_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                delay_fix.frameNStart = frameN  # exact frame index
                delay_fix.tStart = t  # local t and not account for scr refresh
                delay_fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(delay_fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'delay_fix.started')
                # update status
                delay_fix.status = STARTED
                delay_fix.setAutoDraw(True)
            
            # if delay_fix is active this frame...
            if delay_fix.status == STARTED:
                # update params
                pass
            
            # if delay_fix is stopping this frame...
            if delay_fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > delay_fix.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    delay_fix.tStop = t  # not accounting for scr refresh
                    delay_fix.tStopRefresh = tThisFlipGlobal  # on global time
                    delay_fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'delay_fix.stopped')
                    # update status
                    delay_fix.status = FINISHED
                    delay_fix.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=delay,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                delay.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if delay.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in delay.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "delay" ---
        for thisComponent in delay.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for delay
        delay.tStop = globalClock.getTime(format='float')
        delay.tStopRefresh = tThisFlipGlobal
        thisExp.addData('delay.stopped', delay.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if delay.maxDurationReached:
            routineTimer.addTime(-delay.maxDuration)
        elif delay.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "whole_report" ---
        # create an object to store info about Routine whole_report
        whole_report = data.Routine(
            name='whole_report',
            components=[report_fix, reportCircle, mouse, cursorDot],
        )
        whole_report.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from wholeReport_code
        import numpy as np  # OK here if you haven't already imported in Begin Experiment
        
        N_DOTS = 5
        
        # Pull trial-specific values from Builder condition columns
        trial_colors = [item0_color, item1_color, item2_color, item3_color, item4_color]
        trial_xy = [
            (float(item0_x), float(item0_y)),
            (float(item1_x), float(item1_y)),
            (float(item2_x), float(item2_y)),
            (float(item3_x), float(item3_y)),
            (float(item4_x), float(item4_y)),
        ]
        trial_angles = np.array([np.arctan2(y, x) for (x, y) in trial_xy], dtype=float)
        
        # Use the same dot stimuli objects as options
        dots = [dot0, dot1, dot2, dot3, dot4]
        
        for d in dots:
            d.setAutoDraw(True)
        
        # Set option colors to THIS TRIAL’S colors (not DOT_COLORS)
        for i, d in enumerate(dots):
            d.fillColor = trial_colors[i]
            d.lineColor = trial_colors[i]
            d.opacity = 1.0
        
        # lay options below report circle
        cx, cy = reportCircle.pos
        r = float(reportCircle.size[0]) / 2.0
        y_opt = cy - (r * 1.4)
        x_offsets = np.linspace(-r*1.2, r*1.2, N_DOTS)
        for i, d in enumerate(dots):
            d.pos = (cx + float(x_offsets[i]), float(y_opt))
        
        # cursor preview
        cursorDot.setAutoDraw(True)
        cursorDot.opacity = 0.0
        
        reported = [False] * N_DOTS
        resp_angles = [None] * N_DOTS
        phase = "choose_color"
        selected_idx = None
        
        mouse.clickReset()
        
        # timing + order
        click_order = []
        click_rt = []
        report_start = core.getTime()
        
        select_time = None
        decision_rt = []
        
        # by-order outputs
        resp_by_order_colors = []
        resp_by_order_angles = []
        true_by_order_angles = []
        ang_error_by_order = []
        
        resp_by_order_deg = []
        true_by_order_deg = []
        ang_error_by_order_deg = []
        
        def rad_to_deg(a):
            return (np.degrees(a) + 360) % 360
        
        def circ_dist(a, b):
            return abs(np.angle(np.exp(1j * (a - b))))
        
        def circ_dist_deg(a, b):
            d = rad_to_deg(a) - rad_to_deg(b)
            d = (d + 180) % 360 - 180
            return abs(d)
        
        # setup some python lists for storing info about the mouse
        mouse.x = []
        mouse.y = []
        mouse.leftButton = []
        mouse.midButton = []
        mouse.rightButton = []
        mouse.time = []
        gotValidClick = False  # until a click is received
        # store start times for whole_report
        whole_report.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        whole_report.tStart = globalClock.getTime(format='float')
        whole_report.status = STARTED
        thisExp.addData('whole_report.started', whole_report.tStart)
        whole_report.maxDuration = None
        # keep track of which components have finished
        whole_reportComponents = whole_report.components
        for thisComponent in whole_report.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "whole_report" ---
        thisExp.currentRoutine = whole_report
        whole_report.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_trial, 'status') and thisPractice_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from wholeReport_code
            # Grey out used colors
            for i, d in enumerate(dots):
                d.opacity = 0.25 if reported[i] else 1.0
            
            # Cursor preview snapped to circumference
            if phase == "choose_location" and selected_idx is not None:
                cursorDot.opacity = 1.0
                cursorDot.fillColor = trial_colors[selected_idx]
                cursorDot.lineColor = trial_colors[selected_idx]
            
                mx, my = mouse.getPos()
                cx, cy = reportCircle.pos
                r = float(reportCircle.size[0]) / 2.0
            
                dx, dy = mx - cx, my - cy
                norm = (dx*dx + dy*dy) ** 0.5
            
                if norm > 1e-6:
                    snap_x = cx + r * dx / norm
                    snap_y = cy + r * dy / norm
                else:
                    snap_x = cx + r
                    snap_y = cy
            
                cursorDot.pos = (float(snap_x), float(snap_y))
            else:
                cursorDot.opacity = 0.0
            
            # click handling
            buttons = mouse.getPressed()
            if buttons[0]:
                mouse.clickReset()
            
                if phase == "choose_color":
                    for i, d in enumerate(dots):
                        if (not reported[i]) and d.contains(mouse):
                            selected_idx = i
                            select_time = core.getTime()
                            phase = "choose_location"
                            break
            
                elif phase == "choose_location":
                    if reportCircle.contains(mouse):
            
                        sx, sy = cursorDot.pos
                        cx, cy = reportCircle.pos
                        ang = float(np.arctan2(sy - cy, sx - cx))
            
                        resp_angles[selected_idx] = ang
                        reported[selected_idx] = True
            
                        click_order.append(int(selected_idx))
                        click_rt.append(float(core.getTime() - report_start))
            
                        if select_time is None:
                            decision_rt.append(None)
                        else:
                            decision_rt.append(float(core.getTime() - select_time))
                        select_time = None
            
                        true_ang = float(trial_angles[selected_idx])
            
                        resp_by_order_colors.append(trial_colors[selected_idx])
                        resp_by_order_angles.append(float(ang))
                        true_by_order_angles.append(true_ang)
            
                        ang_error_by_order.append(float(circ_dist(ang, true_ang)))
            
                        resp_deg = float(rad_to_deg(ang))
                        true_deg = float(rad_to_deg(true_ang))
                        resp_by_order_deg.append(resp_deg)
                        true_by_order_deg.append(true_deg)
                        ang_error_by_order_deg.append(float(circ_dist_deg(ang, true_ang)))
            
                        selected_idx = None
                        phase = "choose_color"
            
                        if all(reported):
                            continueRoutine = False
            
            
            # *report_fix* updates
            
            # if report_fix is starting this frame...
            if report_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                report_fix.frameNStart = frameN  # exact frame index
                report_fix.tStart = t  # local t and not account for scr refresh
                report_fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(report_fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'report_fix.started')
                # update status
                report_fix.status = STARTED
                report_fix.setAutoDraw(True)
            
            # if report_fix is active this frame...
            if report_fix.status == STARTED:
                # update params
                pass
            
            # *reportCircle* updates
            
            # if reportCircle is starting this frame...
            if reportCircle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reportCircle.frameNStart = frameN  # exact frame index
                reportCircle.tStart = t  # local t and not account for scr refresh
                reportCircle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reportCircle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reportCircle.started')
                # update status
                reportCircle.status = STARTED
                reportCircle.setAutoDraw(True)
            
            # if reportCircle is active this frame...
            if reportCircle.status == STARTED:
                # update params
                pass
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse.started', t)
                # update status
                mouse.status = STARTED
                mouse.mouseClock.reset()
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            if mouse.status == STARTED:  # only update if started and not finished!
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse.getPos()
                        mouse.x.append(float(x))
                        mouse.y.append(float(y))
                        buttons = mouse.getPressed()
                        mouse.leftButton.append(buttons[0])
                        mouse.midButton.append(buttons[1])
                        mouse.rightButton.append(buttons[2])
                        mouse.time.append(mouse.mouseClock.getTime())
            
            # *cursorDot* updates
            
            # if cursorDot is starting this frame...
            if cursorDot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cursorDot.frameNStart = frameN  # exact frame index
                cursorDot.tStart = t  # local t and not account for scr refresh
                cursorDot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cursorDot, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cursorDot.started')
                # update status
                cursorDot.status = STARTED
                cursorDot.setAutoDraw(True)
            
            # if cursorDot is active this frame...
            if cursorDot.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=whole_report,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                whole_report.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if whole_report.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in whole_report.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "whole_report" ---
        for thisComponent in whole_report.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for whole_report
        whole_report.tStop = globalClock.getTime(format='float')
        whole_report.tStopRefresh = tThisFlipGlobal
        thisExp.addData('whole_report.stopped', whole_report.tStop)
        # Run 'End Routine' code from wholeReport_code
        for d in dots:
            d.setAutoDraw(False)
        cursorDot.setAutoDraw(False)
        
        thisExp.addData("resp_angles", resp_angles)
        thisExp.addData("true_angles", [float(a) for a in trial_angles])
        thisExp.addData("colors", trial_colors)
        
        thisExp.addData("click_order", click_order)
        thisExp.addData("click_rt", click_rt)
        thisExp.addData("decision_rt", decision_rt)
        
        thisExp.addData("resp_by_order_colors", resp_by_order_colors)
        thisExp.addData("resp_by_order_angles", resp_by_order_angles)
        thisExp.addData("true_by_order_angles", true_by_order_angles)
        thisExp.addData("ang_error_by_order", ang_error_by_order)
        
        thisExp.addData("resp_by_order_deg", resp_by_order_deg)
        thisExp.addData("true_by_order_deg", true_by_order_deg)
        thisExp.addData("ang_error_by_order_deg", ang_error_by_order_deg)
        
        # store data for practice_trials (TrialHandler)
        practice_trials.addData('mouse.x', mouse.x)
        practice_trials.addData('mouse.y', mouse.y)
        practice_trials.addData('mouse.leftButton', mouse.leftButton)
        practice_trials.addData('mouse.midButton', mouse.midButton)
        practice_trials.addData('mouse.rightButton', mouse.rightButton)
        practice_trials.addData('mouse.time', mouse.time)
        # the Routine "whole_report" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPractice_trial as finished
        if hasattr(thisPractice_trial, 'status'):
            thisPractice_trial.status = FINISHED
        # if awaiting a pause, pause now
        if practice_trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            practice_trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practice_trials'
    practice_trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Practice_doneScreen" ---
    # create an object to store info about Routine Practice_doneScreen
    Practice_doneScreen = data.Routine(
        name='Practice_doneScreen',
        components=[practiceDone_text, practiceDone_keyResp],
    )
    Practice_doneScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for practiceDone_keyResp
    practiceDone_keyResp.keys = []
    practiceDone_keyResp.rt = []
    _practiceDone_keyResp_allKeys = []
    # store start times for Practice_doneScreen
    Practice_doneScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Practice_doneScreen.tStart = globalClock.getTime(format='float')
    Practice_doneScreen.status = STARTED
    thisExp.addData('Practice_doneScreen.started', Practice_doneScreen.tStart)
    Practice_doneScreen.maxDuration = None
    # keep track of which components have finished
    Practice_doneScreenComponents = Practice_doneScreen.components
    for thisComponent in Practice_doneScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Practice_doneScreen" ---
    thisExp.currentRoutine = Practice_doneScreen
    Practice_doneScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practiceDone_text* updates
        
        # if practiceDone_text is starting this frame...
        if practiceDone_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practiceDone_text.frameNStart = frameN  # exact frame index
            practiceDone_text.tStart = t  # local t and not account for scr refresh
            practiceDone_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practiceDone_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practiceDone_text.started')
            # update status
            practiceDone_text.status = STARTED
            practiceDone_text.setAutoDraw(True)
        
        # if practiceDone_text is active this frame...
        if practiceDone_text.status == STARTED:
            # update params
            pass
        
        # *practiceDone_keyResp* updates
        waitOnFlip = False
        
        # if practiceDone_keyResp is starting this frame...
        if practiceDone_keyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practiceDone_keyResp.frameNStart = frameN  # exact frame index
            practiceDone_keyResp.tStart = t  # local t and not account for scr refresh
            practiceDone_keyResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practiceDone_keyResp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practiceDone_keyResp.started')
            # update status
            practiceDone_keyResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(practiceDone_keyResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(practiceDone_keyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if practiceDone_keyResp.status == STARTED and not waitOnFlip:
            theseKeys = practiceDone_keyResp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _practiceDone_keyResp_allKeys.extend(theseKeys)
            if len(_practiceDone_keyResp_allKeys):
                practiceDone_keyResp.keys = _practiceDone_keyResp_allKeys[-1].name  # just the last key pressed
                practiceDone_keyResp.rt = _practiceDone_keyResp_allKeys[-1].rt
                practiceDone_keyResp.duration = _practiceDone_keyResp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Practice_doneScreen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Practice_doneScreen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Practice_doneScreen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Practice_doneScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Practice_doneScreen" ---
    for thisComponent in Practice_doneScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Practice_doneScreen
    Practice_doneScreen.tStop = globalClock.getTime(format='float')
    Practice_doneScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Practice_doneScreen.stopped', Practice_doneScreen.tStop)
    # check responses
    if practiceDone_keyResp.keys in ['', [], None]:  # No response was made
        practiceDone_keyResp.keys = None
    thisExp.addData('practiceDone_keyResp.keys',practiceDone_keyResp.keys)
    if practiceDone_keyResp.keys != None:  # we had a response
        thisExp.addData('practiceDone_keyResp.rt', practiceDone_keyResp.rt)
        thisExp.addData('practiceDone_keyResp.duration', practiceDone_keyResp.duration)
    thisExp.nextEntry()
    # the Routine "Practice_doneScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Main_startScreen" ---
    # create an object to store info about Routine Main_startScreen
    Main_startScreen = data.Routine(
        name='Main_startScreen',
        components=[mainStart_text, mainStart_keyResp],
    )
    Main_startScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for mainStart_keyResp
    mainStart_keyResp.keys = []
    mainStart_keyResp.rt = []
    _mainStart_keyResp_allKeys = []
    # store start times for Main_startScreen
    Main_startScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Main_startScreen.tStart = globalClock.getTime(format='float')
    Main_startScreen.status = STARTED
    thisExp.addData('Main_startScreen.started', Main_startScreen.tStart)
    Main_startScreen.maxDuration = None
    # keep track of which components have finished
    Main_startScreenComponents = Main_startScreen.components
    for thisComponent in Main_startScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Main_startScreen" ---
    thisExp.currentRoutine = Main_startScreen
    Main_startScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *mainStart_text* updates
        
        # if mainStart_text is starting this frame...
        if mainStart_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mainStart_text.frameNStart = frameN  # exact frame index
            mainStart_text.tStart = t  # local t and not account for scr refresh
            mainStart_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mainStart_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'mainStart_text.started')
            # update status
            mainStart_text.status = STARTED
            mainStart_text.setAutoDraw(True)
        
        # if mainStart_text is active this frame...
        if mainStart_text.status == STARTED:
            # update params
            pass
        
        # *mainStart_keyResp* updates
        waitOnFlip = False
        
        # if mainStart_keyResp is starting this frame...
        if mainStart_keyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mainStart_keyResp.frameNStart = frameN  # exact frame index
            mainStart_keyResp.tStart = t  # local t and not account for scr refresh
            mainStart_keyResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mainStart_keyResp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'mainStart_keyResp.started')
            # update status
            mainStart_keyResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(mainStart_keyResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(mainStart_keyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if mainStart_keyResp.status == STARTED and not waitOnFlip:
            theseKeys = mainStart_keyResp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _mainStart_keyResp_allKeys.extend(theseKeys)
            if len(_mainStart_keyResp_allKeys):
                mainStart_keyResp.keys = _mainStart_keyResp_allKeys[-1].name  # just the last key pressed
                mainStart_keyResp.rt = _mainStart_keyResp_allKeys[-1].rt
                mainStart_keyResp.duration = _mainStart_keyResp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Main_startScreen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Main_startScreen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Main_startScreen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Main_startScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Main_startScreen" ---
    for thisComponent in Main_startScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Main_startScreen
    Main_startScreen.tStop = globalClock.getTime(format='float')
    Main_startScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Main_startScreen.stopped', Main_startScreen.tStop)
    # check responses
    if mainStart_keyResp.keys in ['', [], None]:  # No response was made
        mainStart_keyResp.keys = None
    thisExp.addData('mainStart_keyResp.keys',mainStart_keyResp.keys)
    if mainStart_keyResp.keys != None:  # we had a response
        thisExp.addData('mainStart_keyResp.rt', mainStart_keyResp.rt)
        thisExp.addData('mainStart_keyResp.duration', mainStart_keyResp.duration)
    thisExp.nextEntry()
    # the Routine "Main_startScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "prePhase_blank" ---
    # create an object to store info about Routine prePhase_blank
    prePhase_blank = data.Routine(
        name='prePhase_blank',
        components=[prePhase_fix],
    )
    prePhase_blank.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for prePhase_blank
    prePhase_blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prePhase_blank.tStart = globalClock.getTime(format='float')
    prePhase_blank.status = STARTED
    thisExp.addData('prePhase_blank.started', prePhase_blank.tStart)
    prePhase_blank.maxDuration = None
    # keep track of which components have finished
    prePhase_blankComponents = prePhase_blank.components
    for thisComponent in prePhase_blank.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "prePhase_blank" ---
    thisExp.currentRoutine = prePhase_blank
    prePhase_blank.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *prePhase_fix* updates
        
        # if prePhase_fix is starting this frame...
        if prePhase_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prePhase_fix.frameNStart = frameN  # exact frame index
            prePhase_fix.tStart = t  # local t and not account for scr refresh
            prePhase_fix.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prePhase_fix, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prePhase_fix.started')
            # update status
            prePhase_fix.status = STARTED
            prePhase_fix.setAutoDraw(True)
        
        # if prePhase_fix is active this frame...
        if prePhase_fix.status == STARTED:
            # update params
            pass
        
        # if prePhase_fix is stopping this frame...
        if prePhase_fix.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > prePhase_fix.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                prePhase_fix.tStop = t  # not accounting for scr refresh
                prePhase_fix.tStopRefresh = tThisFlipGlobal  # on global time
                prePhase_fix.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prePhase_fix.stopped')
                # update status
                prePhase_fix.status = FINISHED
                prePhase_fix.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=prePhase_blank,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            prePhase_blank.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if prePhase_blank.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in prePhase_blank.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prePhase_blank" ---
    for thisComponent in prePhase_blank.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prePhase_blank
    prePhase_blank.tStop = globalClock.getTime(format='float')
    prePhase_blank.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prePhase_blank.stopped', prePhase_blank.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if prePhase_blank.maxDurationReached:
        routineTimer.addTime(-prePhase_blank.maxDuration)
    elif prePhase_blank.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    main_trials = data.TrialHandler2(
        name='main_trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(main_list), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(main_trials)  # add the loop to the experiment
    thisMain_trial = main_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMain_trial.rgb)
    if thisMain_trial != None:
        for paramName in thisMain_trial:
            globals()[paramName] = thisMain_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisMain_trial in main_trials:
        main_trials.status = STARTED
        if hasattr(thisMain_trial, 'status'):
            thisMain_trial.status = STARTED
        currentLoop = main_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisMain_trial.rgb)
        if thisMain_trial != None:
            for paramName in thisMain_trial:
                globals()[paramName] = thisMain_trial[paramName]
        
        # --- Prepare to start Routine "ITI" ---
        # create an object to store info about Routine ITI
        ITI = data.Routine(
            name='ITI',
            components=[ITI_fix],
        )
        ITI.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for ITI
        ITI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        ITI.tStart = globalClock.getTime(format='float')
        ITI.status = STARTED
        thisExp.addData('ITI.started', ITI.tStart)
        ITI.maxDuration = None
        # keep track of which components have finished
        ITIComponents = ITI.components
        for thisComponent in ITI.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ITI" ---
        thisExp.currentRoutine = ITI
        ITI.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisMain_trial, 'status') and thisMain_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ITI_fix* updates
            
            # if ITI_fix is starting this frame...
            if ITI_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ITI_fix.frameNStart = frameN  # exact frame index
                ITI_fix.tStart = t  # local t and not account for scr refresh
                ITI_fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ITI_fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ITI_fix.started')
                # update status
                ITI_fix.status = STARTED
                ITI_fix.setAutoDraw(True)
            
            # if ITI_fix is active this frame...
            if ITI_fix.status == STARTED:
                # update params
                pass
            
            # if ITI_fix is stopping this frame...
            if ITI_fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ITI_fix.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    ITI_fix.tStop = t  # not accounting for scr refresh
                    ITI_fix.tStopRefresh = tThisFlipGlobal  # on global time
                    ITI_fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ITI_fix.stopped')
                    # update status
                    ITI_fix.status = FINISHED
                    ITI_fix.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=ITI,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                ITI.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if ITI.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in ITI.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ITI" ---
        for thisComponent in ITI.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for ITI
        ITI.tStop = globalClock.getTime(format='float')
        ITI.tStopRefresh = tThisFlipGlobal
        thisExp.addData('ITI.stopped', ITI.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if ITI.maxDurationReached:
            routineTimer.addTime(-ITI.maxDuration)
        elif ITI.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "mem_array" ---
        # create an object to store info about Routine mem_array
        mem_array = data.Routine(
            name='mem_array',
            components=[mem_fix, memCircle, dot0, dot1, dot2, dot3, dot4],
        )
        mem_array.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from resetArray_code
        # --- hard reset dot objects each trial (array routine) ---
        dots = [dot0, dot1, dot2, dot3, dot4]
        for d in dots:
            d.setAutoDraw(True)     # if you want them drawn in this routine
            d.opacity = 1.0         # ALWAYS reset opacity
            d.ori = 0               # just in case
        
        dot0.setFillColor(item0_color)
        dot0.setPos((item0_x, item0_y))
        dot0.setLineColor(item0_color)
        dot1.setFillColor(item1_color)
        dot1.setPos((item1_x, item1_y))
        dot1.setLineColor(item1_color)
        dot2.setFillColor(item2_color)
        dot2.setPos((item2_x, item2_y))
        dot2.setLineColor(item2_color)
        dot3.setFillColor(item3_color)
        dot3.setPos((item3_x, item3_y))
        dot3.setLineColor(item3_color)
        dot4.setFillColor(item4_color)
        dot4.setPos((item4_x, item4_y))
        dot4.setLineColor(item4_color)
        # store start times for mem_array
        mem_array.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        mem_array.tStart = globalClock.getTime(format='float')
        mem_array.status = STARTED
        thisExp.addData('mem_array.started', mem_array.tStart)
        mem_array.maxDuration = None
        # keep track of which components have finished
        mem_arrayComponents = mem_array.components
        for thisComponent in mem_array.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "mem_array" ---
        thisExp.currentRoutine = mem_array
        mem_array.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.25:
            # if trial has changed, end Routine now
            if hasattr(thisMain_trial, 'status') and thisMain_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *mem_fix* updates
            
            # if mem_fix is starting this frame...
            if mem_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mem_fix.frameNStart = frameN  # exact frame index
                mem_fix.tStart = t  # local t and not account for scr refresh
                mem_fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mem_fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mem_fix.started')
                # update status
                mem_fix.status = STARTED
                mem_fix.setAutoDraw(True)
            
            # if mem_fix is active this frame...
            if mem_fix.status == STARTED:
                # update params
                pass
            
            # if mem_fix is stopping this frame...
            if mem_fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > mem_fix.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    mem_fix.tStop = t  # not accounting for scr refresh
                    mem_fix.tStopRefresh = tThisFlipGlobal  # on global time
                    mem_fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'mem_fix.stopped')
                    # update status
                    mem_fix.status = FINISHED
                    mem_fix.setAutoDraw(False)
            
            # *memCircle* updates
            
            # if memCircle is starting this frame...
            if memCircle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                memCircle.frameNStart = frameN  # exact frame index
                memCircle.tStart = t  # local t and not account for scr refresh
                memCircle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(memCircle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'memCircle.started')
                # update status
                memCircle.status = STARTED
                memCircle.setAutoDraw(True)
            
            # if memCircle is active this frame...
            if memCircle.status == STARTED:
                # update params
                pass
            
            # if memCircle is stopping this frame...
            if memCircle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > memCircle.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    memCircle.tStop = t  # not accounting for scr refresh
                    memCircle.tStopRefresh = tThisFlipGlobal  # on global time
                    memCircle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'memCircle.stopped')
                    # update status
                    memCircle.status = FINISHED
                    memCircle.setAutoDraw(False)
            
            # *dot0* updates
            
            # if dot0 is starting this frame...
            if dot0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot0.frameNStart = frameN  # exact frame index
                dot0.tStart = t  # local t and not account for scr refresh
                dot0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot0.started')
                # update status
                dot0.status = STARTED
                dot0.setAutoDraw(True)
            
            # if dot0 is active this frame...
            if dot0.status == STARTED:
                # update params
                pass
            
            # if dot0 is stopping this frame...
            if dot0.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot0.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    dot0.tStop = t  # not accounting for scr refresh
                    dot0.tStopRefresh = tThisFlipGlobal  # on global time
                    dot0.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot0.stopped')
                    # update status
                    dot0.status = FINISHED
                    dot0.setAutoDraw(False)
            
            # *dot1* updates
            
            # if dot1 is starting this frame...
            if dot1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot1.frameNStart = frameN  # exact frame index
                dot1.tStart = t  # local t and not account for scr refresh
                dot1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot1.started')
                # update status
                dot1.status = STARTED
                dot1.setAutoDraw(True)
            
            # if dot1 is active this frame...
            if dot1.status == STARTED:
                # update params
                pass
            
            # if dot1 is stopping this frame...
            if dot1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot1.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    dot1.tStop = t  # not accounting for scr refresh
                    dot1.tStopRefresh = tThisFlipGlobal  # on global time
                    dot1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot1.stopped')
                    # update status
                    dot1.status = FINISHED
                    dot1.setAutoDraw(False)
            
            # *dot2* updates
            
            # if dot2 is starting this frame...
            if dot2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot2.frameNStart = frameN  # exact frame index
                dot2.tStart = t  # local t and not account for scr refresh
                dot2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot2.started')
                # update status
                dot2.status = STARTED
                dot2.setAutoDraw(True)
            
            # if dot2 is active this frame...
            if dot2.status == STARTED:
                # update params
                pass
            
            # if dot2 is stopping this frame...
            if dot2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot2.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    dot2.tStop = t  # not accounting for scr refresh
                    dot2.tStopRefresh = tThisFlipGlobal  # on global time
                    dot2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot2.stopped')
                    # update status
                    dot2.status = FINISHED
                    dot2.setAutoDraw(False)
            
            # *dot3* updates
            
            # if dot3 is starting this frame...
            if dot3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot3.frameNStart = frameN  # exact frame index
                dot3.tStart = t  # local t and not account for scr refresh
                dot3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot3.started')
                # update status
                dot3.status = STARTED
                dot3.setAutoDraw(True)
            
            # if dot3 is active this frame...
            if dot3.status == STARTED:
                # update params
                pass
            
            # if dot3 is stopping this frame...
            if dot3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot3.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    dot3.tStop = t  # not accounting for scr refresh
                    dot3.tStopRefresh = tThisFlipGlobal  # on global time
                    dot3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot3.stopped')
                    # update status
                    dot3.status = FINISHED
                    dot3.setAutoDraw(False)
            
            # *dot4* updates
            
            # if dot4 is starting this frame...
            if dot4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot4.frameNStart = frameN  # exact frame index
                dot4.tStart = t  # local t and not account for scr refresh
                dot4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot4.started')
                # update status
                dot4.status = STARTED
                dot4.setAutoDraw(True)
            
            # if dot4 is active this frame...
            if dot4.status == STARTED:
                # update params
                pass
            
            # if dot4 is stopping this frame...
            if dot4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot4.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    dot4.tStop = t  # not accounting for scr refresh
                    dot4.tStopRefresh = tThisFlipGlobal  # on global time
                    dot4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot4.stopped')
                    # update status
                    dot4.status = FINISHED
                    dot4.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=mem_array,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                mem_array.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if mem_array.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in mem_array.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "mem_array" ---
        for thisComponent in mem_array.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for mem_array
        mem_array.tStop = globalClock.getTime(format='float')
        mem_array.tStopRefresh = tThisFlipGlobal
        thisExp.addData('mem_array.stopped', mem_array.tStop)
        # Run 'End Routine' code from resetArray_code
        for d in dots:
            d.setAutoDraw(False)
        
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if mem_array.maxDurationReached:
            routineTimer.addTime(-mem_array.maxDuration)
        elif mem_array.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.250000)
        
        # --- Prepare to start Routine "delay" ---
        # create an object to store info about Routine delay
        delay = data.Routine(
            name='delay',
            components=[delay_fix],
        )
        delay.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for delay
        delay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        delay.tStart = globalClock.getTime(format='float')
        delay.status = STARTED
        thisExp.addData('delay.started', delay.tStart)
        delay.maxDuration = None
        # keep track of which components have finished
        delayComponents = delay.components
        for thisComponent in delay.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "delay" ---
        thisExp.currentRoutine = delay
        delay.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisMain_trial, 'status') and thisMain_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *delay_fix* updates
            
            # if delay_fix is starting this frame...
            if delay_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                delay_fix.frameNStart = frameN  # exact frame index
                delay_fix.tStart = t  # local t and not account for scr refresh
                delay_fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(delay_fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'delay_fix.started')
                # update status
                delay_fix.status = STARTED
                delay_fix.setAutoDraw(True)
            
            # if delay_fix is active this frame...
            if delay_fix.status == STARTED:
                # update params
                pass
            
            # if delay_fix is stopping this frame...
            if delay_fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > delay_fix.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    delay_fix.tStop = t  # not accounting for scr refresh
                    delay_fix.tStopRefresh = tThisFlipGlobal  # on global time
                    delay_fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'delay_fix.stopped')
                    # update status
                    delay_fix.status = FINISHED
                    delay_fix.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=delay,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                delay.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if delay.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in delay.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "delay" ---
        for thisComponent in delay.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for delay
        delay.tStop = globalClock.getTime(format='float')
        delay.tStopRefresh = tThisFlipGlobal
        thisExp.addData('delay.stopped', delay.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if delay.maxDurationReached:
            routineTimer.addTime(-delay.maxDuration)
        elif delay.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "whole_report" ---
        # create an object to store info about Routine whole_report
        whole_report = data.Routine(
            name='whole_report',
            components=[report_fix, reportCircle, mouse, cursorDot],
        )
        whole_report.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from wholeReport_code
        import numpy as np  # OK here if you haven't already imported in Begin Experiment
        
        N_DOTS = 5
        
        # Pull trial-specific values from Builder condition columns
        trial_colors = [item0_color, item1_color, item2_color, item3_color, item4_color]
        trial_xy = [
            (float(item0_x), float(item0_y)),
            (float(item1_x), float(item1_y)),
            (float(item2_x), float(item2_y)),
            (float(item3_x), float(item3_y)),
            (float(item4_x), float(item4_y)),
        ]
        trial_angles = np.array([np.arctan2(y, x) for (x, y) in trial_xy], dtype=float)
        
        # Use the same dot stimuli objects as options
        dots = [dot0, dot1, dot2, dot3, dot4]
        
        for d in dots:
            d.setAutoDraw(True)
        
        # Set option colors to THIS TRIAL’S colors (not DOT_COLORS)
        for i, d in enumerate(dots):
            d.fillColor = trial_colors[i]
            d.lineColor = trial_colors[i]
            d.opacity = 1.0
        
        # lay options below report circle
        cx, cy = reportCircle.pos
        r = float(reportCircle.size[0]) / 2.0
        y_opt = cy - (r * 1.4)
        x_offsets = np.linspace(-r*1.2, r*1.2, N_DOTS)
        for i, d in enumerate(dots):
            d.pos = (cx + float(x_offsets[i]), float(y_opt))
        
        # cursor preview
        cursorDot.setAutoDraw(True)
        cursorDot.opacity = 0.0
        
        reported = [False] * N_DOTS
        resp_angles = [None] * N_DOTS
        phase = "choose_color"
        selected_idx = None
        
        mouse.clickReset()
        
        # timing + order
        click_order = []
        click_rt = []
        report_start = core.getTime()
        
        select_time = None
        decision_rt = []
        
        # by-order outputs
        resp_by_order_colors = []
        resp_by_order_angles = []
        true_by_order_angles = []
        ang_error_by_order = []
        
        resp_by_order_deg = []
        true_by_order_deg = []
        ang_error_by_order_deg = []
        
        def rad_to_deg(a):
            return (np.degrees(a) + 360) % 360
        
        def circ_dist(a, b):
            return abs(np.angle(np.exp(1j * (a - b))))
        
        def circ_dist_deg(a, b):
            d = rad_to_deg(a) - rad_to_deg(b)
            d = (d + 180) % 360 - 180
            return abs(d)
        
        # setup some python lists for storing info about the mouse
        mouse.x = []
        mouse.y = []
        mouse.leftButton = []
        mouse.midButton = []
        mouse.rightButton = []
        mouse.time = []
        gotValidClick = False  # until a click is received
        # store start times for whole_report
        whole_report.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        whole_report.tStart = globalClock.getTime(format='float')
        whole_report.status = STARTED
        thisExp.addData('whole_report.started', whole_report.tStart)
        whole_report.maxDuration = None
        # keep track of which components have finished
        whole_reportComponents = whole_report.components
        for thisComponent in whole_report.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "whole_report" ---
        thisExp.currentRoutine = whole_report
        whole_report.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisMain_trial, 'status') and thisMain_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from wholeReport_code
            # Grey out used colors
            for i, d in enumerate(dots):
                d.opacity = 0.25 if reported[i] else 1.0
            
            # Cursor preview snapped to circumference
            if phase == "choose_location" and selected_idx is not None:
                cursorDot.opacity = 1.0
                cursorDot.fillColor = trial_colors[selected_idx]
                cursorDot.lineColor = trial_colors[selected_idx]
            
                mx, my = mouse.getPos()
                cx, cy = reportCircle.pos
                r = float(reportCircle.size[0]) / 2.0
            
                dx, dy = mx - cx, my - cy
                norm = (dx*dx + dy*dy) ** 0.5
            
                if norm > 1e-6:
                    snap_x = cx + r * dx / norm
                    snap_y = cy + r * dy / norm
                else:
                    snap_x = cx + r
                    snap_y = cy
            
                cursorDot.pos = (float(snap_x), float(snap_y))
            else:
                cursorDot.opacity = 0.0
            
            # click handling
            buttons = mouse.getPressed()
            if buttons[0]:
                mouse.clickReset()
            
                if phase == "choose_color":
                    for i, d in enumerate(dots):
                        if (not reported[i]) and d.contains(mouse):
                            selected_idx = i
                            select_time = core.getTime()
                            phase = "choose_location"
                            break
            
                elif phase == "choose_location":
                    if reportCircle.contains(mouse):
            
                        sx, sy = cursorDot.pos
                        cx, cy = reportCircle.pos
                        ang = float(np.arctan2(sy - cy, sx - cx))
            
                        resp_angles[selected_idx] = ang
                        reported[selected_idx] = True
            
                        click_order.append(int(selected_idx))
                        click_rt.append(float(core.getTime() - report_start))
            
                        if select_time is None:
                            decision_rt.append(None)
                        else:
                            decision_rt.append(float(core.getTime() - select_time))
                        select_time = None
            
                        true_ang = float(trial_angles[selected_idx])
            
                        resp_by_order_colors.append(trial_colors[selected_idx])
                        resp_by_order_angles.append(float(ang))
                        true_by_order_angles.append(true_ang)
            
                        ang_error_by_order.append(float(circ_dist(ang, true_ang)))
            
                        resp_deg = float(rad_to_deg(ang))
                        true_deg = float(rad_to_deg(true_ang))
                        resp_by_order_deg.append(resp_deg)
                        true_by_order_deg.append(true_deg)
                        ang_error_by_order_deg.append(float(circ_dist_deg(ang, true_ang)))
            
                        selected_idx = None
                        phase = "choose_color"
            
                        if all(reported):
                            continueRoutine = False
            
            
            # *report_fix* updates
            
            # if report_fix is starting this frame...
            if report_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                report_fix.frameNStart = frameN  # exact frame index
                report_fix.tStart = t  # local t and not account for scr refresh
                report_fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(report_fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'report_fix.started')
                # update status
                report_fix.status = STARTED
                report_fix.setAutoDraw(True)
            
            # if report_fix is active this frame...
            if report_fix.status == STARTED:
                # update params
                pass
            
            # *reportCircle* updates
            
            # if reportCircle is starting this frame...
            if reportCircle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reportCircle.frameNStart = frameN  # exact frame index
                reportCircle.tStart = t  # local t and not account for scr refresh
                reportCircle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reportCircle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reportCircle.started')
                # update status
                reportCircle.status = STARTED
                reportCircle.setAutoDraw(True)
            
            # if reportCircle is active this frame...
            if reportCircle.status == STARTED:
                # update params
                pass
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse.started', t)
                # update status
                mouse.status = STARTED
                mouse.mouseClock.reset()
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            if mouse.status == STARTED:  # only update if started and not finished!
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse.getPos()
                        mouse.x.append(float(x))
                        mouse.y.append(float(y))
                        buttons = mouse.getPressed()
                        mouse.leftButton.append(buttons[0])
                        mouse.midButton.append(buttons[1])
                        mouse.rightButton.append(buttons[2])
                        mouse.time.append(mouse.mouseClock.getTime())
            
            # *cursorDot* updates
            
            # if cursorDot is starting this frame...
            if cursorDot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cursorDot.frameNStart = frameN  # exact frame index
                cursorDot.tStart = t  # local t and not account for scr refresh
                cursorDot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cursorDot, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cursorDot.started')
                # update status
                cursorDot.status = STARTED
                cursorDot.setAutoDraw(True)
            
            # if cursorDot is active this frame...
            if cursorDot.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=whole_report,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                whole_report.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if whole_report.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in whole_report.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "whole_report" ---
        for thisComponent in whole_report.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for whole_report
        whole_report.tStop = globalClock.getTime(format='float')
        whole_report.tStopRefresh = tThisFlipGlobal
        thisExp.addData('whole_report.stopped', whole_report.tStop)
        # Run 'End Routine' code from wholeReport_code
        for d in dots:
            d.setAutoDraw(False)
        cursorDot.setAutoDraw(False)
        
        thisExp.addData("resp_angles", resp_angles)
        thisExp.addData("true_angles", [float(a) for a in trial_angles])
        thisExp.addData("colors", trial_colors)
        
        thisExp.addData("click_order", click_order)
        thisExp.addData("click_rt", click_rt)
        thisExp.addData("decision_rt", decision_rt)
        
        thisExp.addData("resp_by_order_colors", resp_by_order_colors)
        thisExp.addData("resp_by_order_angles", resp_by_order_angles)
        thisExp.addData("true_by_order_angles", true_by_order_angles)
        thisExp.addData("ang_error_by_order", ang_error_by_order)
        
        thisExp.addData("resp_by_order_deg", resp_by_order_deg)
        thisExp.addData("true_by_order_deg", true_by_order_deg)
        thisExp.addData("ang_error_by_order_deg", ang_error_by_order_deg)
        
        # store data for main_trials (TrialHandler)
        main_trials.addData('mouse.x', mouse.x)
        main_trials.addData('mouse.y', mouse.y)
        main_trials.addData('mouse.leftButton', mouse.leftButton)
        main_trials.addData('mouse.midButton', mouse.midButton)
        main_trials.addData('mouse.rightButton', mouse.rightButton)
        main_trials.addData('mouse.time', mouse.time)
        # the Routine "whole_report" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisMain_trial as finished
        if hasattr(thisMain_trial, 'status'):
            thisMain_trial.status = FINISHED
        # if awaiting a pause, pause now
        if main_trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            main_trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'main_trials'
    main_trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "END_screen" ---
    # create an object to store info about Routine END_screen
    END_screen = data.Routine(
        name='END_screen',
        components=[end_screen, end_text, end_keyResp],
    )
    END_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for end_keyResp
    end_keyResp.keys = []
    end_keyResp.rt = []
    _end_keyResp_allKeys = []
    # store start times for END_screen
    END_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    END_screen.tStart = globalClock.getTime(format='float')
    END_screen.status = STARTED
    thisExp.addData('END_screen.started', END_screen.tStart)
    END_screen.maxDuration = None
    # keep track of which components have finished
    END_screenComponents = END_screen.components
    for thisComponent in END_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "END_screen" ---
    thisExp.currentRoutine = END_screen
    END_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 20.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_screen* updates
        
        # if end_screen is starting this frame...
        if end_screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_screen.frameNStart = frameN  # exact frame index
            end_screen.tStart = t  # local t and not account for scr refresh
            end_screen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_screen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_screen.started')
            # update status
            end_screen.status = STARTED
            end_screen.setAutoDraw(True)
        
        # if end_screen is active this frame...
        if end_screen.status == STARTED:
            # update params
            pass
        
        # if end_screen is stopping this frame...
        if end_screen.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_screen.tStartRefresh + 20-frameTolerance:
                # keep track of stop time/frame for later
                end_screen.tStop = t  # not accounting for scr refresh
                end_screen.tStopRefresh = tThisFlipGlobal  # on global time
                end_screen.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_screen.stopped')
                # update status
                end_screen.status = FINISHED
                end_screen.setAutoDraw(False)
        
        # *end_text* updates
        
        # if end_text is starting this frame...
        if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_text.frameNStart = frameN  # exact frame index
            end_text.tStart = t  # local t and not account for scr refresh
            end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_text.started')
            # update status
            end_text.status = STARTED
            end_text.setAutoDraw(True)
        
        # if end_text is active this frame...
        if end_text.status == STARTED:
            # update params
            pass
        
        # if end_text is stopping this frame...
        if end_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_text.tStartRefresh + 20-frameTolerance:
                # keep track of stop time/frame for later
                end_text.tStop = t  # not accounting for scr refresh
                end_text.tStopRefresh = tThisFlipGlobal  # on global time
                end_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_text.stopped')
                # update status
                end_text.status = FINISHED
                end_text.setAutoDraw(False)
        
        # *end_keyResp* updates
        waitOnFlip = False
        
        # if end_keyResp is starting this frame...
        if end_keyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_keyResp.frameNStart = frameN  # exact frame index
            end_keyResp.tStart = t  # local t and not account for scr refresh
            end_keyResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_keyResp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_keyResp.started')
            # update status
            end_keyResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_keyResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_keyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if end_keyResp is stopping this frame...
        if end_keyResp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_keyResp.tStartRefresh + 20-frameTolerance:
                # keep track of stop time/frame for later
                end_keyResp.tStop = t  # not accounting for scr refresh
                end_keyResp.tStopRefresh = tThisFlipGlobal  # on global time
                end_keyResp.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_keyResp.stopped')
                # update status
                end_keyResp.status = FINISHED
                end_keyResp.status = FINISHED
        if end_keyResp.status == STARTED and not waitOnFlip:
            theseKeys = end_keyResp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _end_keyResp_allKeys.extend(theseKeys)
            if len(_end_keyResp_allKeys):
                end_keyResp.keys = _end_keyResp_allKeys[-1].name  # just the last key pressed
                end_keyResp.rt = _end_keyResp_allKeys[-1].rt
                end_keyResp.duration = _end_keyResp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=END_screen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            END_screen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if END_screen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in END_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "END_screen" ---
    for thisComponent in END_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for END_screen
    END_screen.tStop = globalClock.getTime(format='float')
    END_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('END_screen.stopped', END_screen.tStop)
    # check responses
    if end_keyResp.keys in ['', [], None]:  # No response was made
        end_keyResp.keys = None
    thisExp.addData('end_keyResp.keys',end_keyResp.keys)
    if end_keyResp.keys != None:  # we had a response
        thisExp.addData('end_keyResp.rt', end_keyResp.rt)
        thisExp.addData('end_keyResp.duration', end_keyResp.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if END_screen.maxDurationReached:
        routineTimer.addTime(-END_screen.maxDuration)
    elif END_screen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-20.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
