#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.4),
    on Tue Jan 13 10:27:53 2026
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
expName = 'WMpsilocybin_v2'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
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
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/calebjerinic-brodeur/Dropbox (ASU)/My Mac (Calebs-MacBook-Pro.local)/Documents/GitHub/WM_Psilocybin/WMpsilocybin_v2_lastrun.py',
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
    
    # --- Initialize components for Routine "ITI" ---
    ITI_fix = visual.Rect(
        win=win, name='ITI_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "mem_array" ---
    # Run 'Begin Experiment' code from placeDotsOnCircle
    import numpy as np
    
    def circ_dist(a, b):
        return np.abs(np.angle(np.exp(1j * (a - b))))
    
    def sample_angles_bilateral(n, rng, min_sep_deg=25, max_tries=50000):
        min_sep = np.deg2rad(min_sep_deg)
        n_left = n // 2
        n_right = n - n_left
    
        def rand_angle_left():
            return rng.uniform(np.pi/2, 3*np.pi/2)
    
        def rand_angle_right():
            a = rng.uniform(-np.pi/2, np.pi/2)
            return a % (2*np.pi)
    
        angles = []
        tries = 0
    
        for _ in range(n_left):
            while tries < max_tries:
                tries += 1
                a = rand_angle_left()
                if all(circ_dist(a, b) >= min_sep for b in angles):
                    angles.append(a); break
            else:
                raise RuntimeError("Failed sampling LEFT angles; lower min_sep_deg.")
    
        for _ in range(n_right):
            while tries < max_tries:
                tries += 1
                a = rand_angle_right()
                if all(circ_dist(a, b) >= min_sep for b in angles):
                    angles.append(a); break
            else:
                raise RuntimeError("Failed sampling RIGHT angles; lower min_sep_deg.")
    
        angles = np.array(angles)
        rng.shuffle(angles)
        return angles
    
    # stable RNG per participant
    sub = expInfo.get('participant', '0')
    seed = int(''.join([c for c in str(sub) if c.isdigit()]) or 0) + 12345
    rng = np.random.default_rng(seed)
    
    N_DOTS = 5
    DOT_COLORS = ["red", "green", "blue", "yellow", "magenta"]
    
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
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    dot1 = visual.ShapeStim(
        win=win, name='dot1',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    dot2 = visual.ShapeStim(
        win=win, name='dot2',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    dot3 = visual.ShapeStim(
        win=win, name='dot3',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    dot4 = visual.ShapeStim(
        win=win, name='dot4',
        size=(0.02, 0.02), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-7.0, interpolate=True)
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "delay" ---
    delay_fix = visual.Rect(
        win=win, name='delay_fix',
        width=(0.01, 0.01)[0], height=(0.01, 0.01)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "report" ---
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
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=5.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
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
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
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
            components=[mem_fix, memCircle, dot0, dot1, dot2, dot3, dot4, key_resp],
        )
        mem_array.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from placeDotsOnCircle
        dots = [dot0, dot1, dot2, dot3, dot4]  # NOW this is safe
        
        r = float(memCircle.size[0]) / 2.0
        cx, cy = memCircle.pos
        
        angles = sample_angles_bilateral(N_DOTS, rng, min_sep_deg=25)
        dot_pos = np.c_[cx + r*np.cos(angles), cy + r*np.sin(angles)]
        
        for i, d in enumerate(dots):
            d.pos = dot_pos[i].tolist()
            d.fillColor = DOT_COLORS[i]
            d.lineColor = DOT_COLORS[i]
        
        # store for later routines
        trial_dot_pos = dot_pos
        trial_dot_colors = DOT_COLORS[:N_DOTS]
        trial_angles = angles
        
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
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
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
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
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
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
        # Run 'End Routine' code from placeDotsOnCircle
        thisExp.addData("dot_pos", trial_dot_pos.tolist())
        thisExp.addData("dot_colors", trial_dot_colors)
        thisExp.addData("angles", trial_angles.tolist())
        
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
            trials.addData('key_resp.duration', key_resp.duration)
        # the Routine "mem_array" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
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
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
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
        
        # --- Prepare to start Routine "report" ---
        # create an object to store info about Routine report
        report = data.Routine(
            name='report',
            components=[report_fix, reportCircle, mouse, cursorDot],
        )
        report.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        # Use the same dot stimuli from the array
        dots = [dot0, dot1, dot2, dot3, dot4]
        
        # Make sure they're actually drawn in this routine
        for d in dots:
            d.setAutoDraw(True)
        
        # Make sure they keep their colors (in case something reset them)
        for i, d in enumerate(dots):
            d.fillColor = DOT_COLORS[i]
            d.lineColor = DOT_COLORS[i]
            d.opacity = 1.0
        
        # Place them BELOW the reportCircle in a row, scaled to the circle size
        cx, cy = reportCircle.pos
        r = float(reportCircle.size[0]) / 2.0
        
        y_opt = cy - (r * 1.4)          # below circle
        x_offsets = np.linspace(-r*1.2, r*1.2, N_DOTS)
        
        for i, d in enumerate(dots):
            d.pos = (cx + float(x_offsets[i]), float(y_opt))
        
        # ---- Cursor preview dot setup ----
        cursorDot.setAutoDraw(True)   # draw it in this routine
        cursorDot.opacity = 0.0       # hidden until a color is selected
        
        # Whole-report state
        reported = [False] * N_DOTS
        resp_angles = [None] * N_DOTS
        phase = "choose_color"
        selected_idx = None
        
        mouse.clickReset()
        
        # ---- per-click order + RT tracking ----
        click_order = []              # indices of colors in the order they were placed
        click_rt = []                 # RT (s) for each placement relative to report onset
        report_start = core.getTime() # requires: from psychopy import core (Builder usually has it)
        
        # ---- decision-time RT (select option -> place on circle) ----
        select_time = None            # timestamp when a color option is selected
        decision_rt = []              # decision time per placed item (s)
        
        # ---- by-order (no indexing later) ----
        resp_by_order_colors = []
        resp_by_order_angles = []
        true_by_order_angles = []
        
        # ---- angular error (by-order) ----
        # circular distance in radians on (-pi, pi] mapped to absolute distance in [0, pi]
        def circ_dist(a, b):
            return abs(np.angle(np.exp(1j * (a - b))))
        
        ang_error_by_order = []       # abs angular error (radians) in report order
        
        # ---- NEW: degree versions (analysis-friendly) ----
        resp_by_order_deg = []
        true_by_order_deg = []
        ang_error_by_order_deg = []
        
        def rad_to_deg(a):
            """Map radians to [0, 360)."""
            return (np.degrees(a) + 360) % 360
        
        def circ_dist_deg(a, b):
            """Circular distance in degrees, returned in [0, 180]."""
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
        # store start times for report
        report.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        report.tStart = globalClock.getTime(format='float')
        report.status = STARTED
        thisExp.addData('report.started', report.tStart)
        report.maxDuration = None
        # keep track of which components have finished
        reportComponents = report.components
        for thisComponent in report.components:
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
        
        # --- Run Routine "report" ---
        thisExp.currentRoutine = report
        report.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code
            # Grey out used colors
            for i, d in enumerate(dots):
                d.opacity = 0.25 if reported[i] else 1.0
            
            # ---- Cursor preview dot behavior (SNAP TO CIRCUMFERENCE) ----
            if phase == "choose_location" and selected_idx is not None:
                cursorDot.opacity = 1.0
                cursorDot.fillColor = DOT_COLORS[selected_idx]
                cursorDot.lineColor = DOT_COLORS[selected_idx]
            
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
            
            # ---- click handling ----
            buttons = mouse.getPressed()
            if buttons[0]:
                mouse.clickReset()
            
                if phase == "choose_color":
                    # click a dot option
                    for i, d in enumerate(dots):
                        if (not reported[i]) and d.contains(mouse):
                            selected_idx = i
                            select_time = core.getTime()   # start decision timer here
                            phase = "choose_location"
                            break
            
                elif phase == "choose_location":
                    # place at the snapped position (optionally require click "near" circle)
                    if reportCircle.contains(mouse):
            
                        sx, sy = cursorDot.pos  # snapped position
                        cx, cy = reportCircle.pos
                        ang = float(np.arctan2(sy - cy, sx - cx))
            
                        resp_angles[selected_idx] = ang
                        reported[selected_idx] = True
            
                        # record order + RT since report onset
                        click_order.append(int(selected_idx))
                        click_rt.append(float(core.getTime() - report_start))
            
                        # record decision RT (select option -> place)
                        if select_time is None:
                            decision_rt.append(None)
                        else:
                            decision_rt.append(float(core.getTime() - select_time))
                        select_time = None  # reset for next item
            
                        # ---- save values in REPORT ORDER (no indexing later) ----
                        true_ang = float(trial_angles[selected_idx])
            
                        resp_by_order_colors.append(DOT_COLORS[selected_idx])
                        resp_by_order_angles.append(float(ang))
                        true_by_order_angles.append(true_ang)
            
                        # radians error
                        ang_error_by_order.append(float(circ_dist(ang, true_ang)))
            
                        # ---- NEW: degree versions ----
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
                    currentRoutine=report,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                report.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if report.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in report.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "report" ---
        for thisComponent in report.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for report
        report.tStop = globalClock.getTime(format='float')
        report.tStopRefresh = tThisFlipGlobal
        thisExp.addData('report.stopped', report.tStop)
        # Run 'End Routine' code from code
        # Optionally stop drawing dots if you don't want them to persist
        for d in dots:
            d.setAutoDraw(False)
        
        # stop drawing cursor dot
        cursorDot.setAutoDraw(False)
        
        thisExp.addData("resp_angles", resp_angles)
        thisExp.addData("true_angles", [float(a) for a in trial_angles])
        thisExp.addData("colors", DOT_COLORS)
        
        # save order + RT per click
        thisExp.addData("click_order", click_order)
        thisExp.addData("click_rt", click_rt)
        
        # save decision-time RT per click (select -> place)
        thisExp.addData("decision_rt", decision_rt)
        
        # save report-ordered versions (analysis-friendly)
        thisExp.addData("resp_by_order_colors", resp_by_order_colors)
        thisExp.addData("resp_by_order_angles", resp_by_order_angles)
        thisExp.addData("true_by_order_angles", true_by_order_angles)
        
        # save angular error per item (radians) in report order
        thisExp.addData("ang_error_by_order", ang_error_by_order)
        
        # ---- NEW: save degree-based measures ----
        thisExp.addData("resp_by_order_deg", resp_by_order_deg)
        thisExp.addData("true_by_order_deg", true_by_order_deg)
        thisExp.addData("ang_error_by_order_deg", ang_error_by_order_deg)
        # store data for trials (TrialHandler)
        trials.addData('mouse.x', mouse.x)
        trials.addData('mouse.y', mouse.y)
        trials.addData('mouse.leftButton', mouse.leftButton)
        trials.addData('mouse.midButton', mouse.midButton)
        trials.addData('mouse.rightButton', mouse.rightButton)
        trials.addData('mouse.time', mouse.time)
        # the Routine "report" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
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
