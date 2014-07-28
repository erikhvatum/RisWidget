# The MIT License (MIT)
#
# Copyright (c) 2014 WUSTL ZPLAB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors: Erik Hvatum

from PyQt5 import Qt
import skimage.io as skio
import time

def MakeFlipbook(risWidget, images=None, imageFileNames=None):
    '''In addition to being useful in its own right, this function is intended as an example of convenient
    GUI utility creation on the fly that builds upon more permanent code without requiring its modification.'''

    if images is None and imageFileNames is None or images is not None and imageFileNames is not None:
        Exception('Either images or imageFileNames argument must be provided (but not both).')

    flipBook = Qt.QDialog()

    if images is not None:
        flipBook.images = images
    else:
        flipBook.images = []
        for imageFileName in imageFileNames:
            flipBook.images.append(skio.imread(str(imageFileName)))

    frameCount = len(flipBook.images)

    if frameCount <= 0:
        Exception('No images provided...')

    layout = Qt.QVBoxLayout()
    flipBook.setLayout(layout)

    playButton = Qt.QPushButton("play");
    playButton.setCheckable(True);
    layout.addWidget(playButton)

    fpsLayout = Qt.QHBoxLayout()
    layout.addLayout(fpsLayout)

    fpsLabel = Qt.QLabel("fps limit: ")
    fpsLayout.addWidget(fpsLabel)
    fpsEdit = Qt.QLineEdit()
    fpsEdit.setText('5')
    fpsEdit.setValidator(Qt.QDoubleValidator(0.1, 1e9, 6))
    fpsLayout.addWidget(fpsEdit)

    frameIndexLayout = Qt.QHBoxLayout()
    layout.addLayout(frameIndexLayout)
    frameIndexLabel = Qt.QLabel("image #: ")
    frameIndexLayout.addWidget(frameIndexLabel)
    frameIndexSpinBox = Qt.QSpinBox()
    frameIndexSpinBox.setRange(0, frameCount - 1)
    frameIndexSpinBox.setSingleStep(1)
    frameIndexLayout.addWidget(frameIndexSpinBox)

    scrubSlider = Qt.QSlider(Qt.Qt.Horizontal)
    scrubSlider.setMinimum(0)
    scrubSlider.setMaximum(frameCount - 1)
    scrubSlider.setTickInterval(1)
    scrubSlider.setTickPosition(Qt.QSlider.TicksBothSides)
    scrubSlider.setTracking(True)
    layout.addWidget(scrubSlider)
    wasPlayingBeforeScrub = False

    nextFrameTimer = Qt.QTimer(flipBook)
    nextFrameTimer.setSingleShot(True)

    frameTime = None
    frameIndex = 0

    isPlaying = False

    def showNextFrame():
        t = time.time()
        showFrame(frameIndex + 1)
        nowTime = time.time()
        showTime = nowTime - t
        waitForNext = frameTime - showTime
        if waitForNext < 0:
            waitForNext = 0
        nextFrameTimer.start(waitForNext * 1000)

    def showFrame(frameIndex_):
        nonlocal frameIndex
        if frameIndex != frameIndex_:
            frameIndex = frameIndex_
            if frameIndex < 0 or frameIndex >= frameCount:
                frameIndex = 0

            scrubSlider.setValue(frameIndex)
            frameIndexSpinBox.setValue(frameIndex)
            risWidget.showImage(flipBook.images[frameIndex])
    
    def playClicked(checked):
        nonlocal frameTime
        nonlocal isPlaying
        if checked:
            isPlaying = True
            playButton.setText('stop')
            frameTime = 1 / float(fpsEdit.text())
            nextFrameTimer.start(frameTime * 1000)
        else:
            nextFrameTimer.stop()
            isPlaying = False
            playButton.setText('play')

    def scrubSliderPressed():
        nonlocal isPlaying
        nonlocal wasPlayingBeforeScrub
        if isPlaying:
            nextFrameTimer.stop()
            playButton.setText('(paused while slider selected)')
            wasPlayingBeforeScrub = True
            isPlaying = False
        else:
            wasPlayingBeforeScrub = False

    def scrubSliderReleased():
        nonlocal frameTime
        nonlocal isPlaying
        if wasPlayingBeforeScrub:
            playButton.setText('stop')
            isPlaying = True
            frameTime = 1 / float(fpsEdit.text())
            nextFrameTimer.start(frameTime * 1000)
        else:
            playButton.setText('play')

    playButton.clicked.connect(playClicked)
    nextFrameTimer.timeout.connect(showNextFrame, Qt.Qt.QueuedConnection)
    frameIndexSpinBox.valueChanged.connect(showFrame)
    scrubSlider.valueChanged.connect(showFrame)
    scrubSlider.sliderPressed.connect(scrubSliderPressed)
    scrubSlider.sliderReleased.connect(scrubSliderReleased)

    flipBook.show()
    return flipBook
