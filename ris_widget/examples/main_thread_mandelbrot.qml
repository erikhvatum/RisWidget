import QtQuick 2.4
import QtQuick.Controls 1.4
import QtQuick.Extras 1.4
import QtQuick.Layouts 1.2

Item {
    id: mandelbrotItem
    objectName: 'mandelbrotItem'
    property var mandelbrot
    enabled: false
    width: 400
    height: 400

    GridLayout {
        anchors.margins: 5
        anchors.fill: parent
        columnSpacing: 5
        rowSpacing: 5
        columns: 2

        Label {
            id: iterationsLabel
            text: 'Iterations: '
        }
        SpinBox {
            id: iterationsSpinbox
            minimumValue: 1
            maximumValue: 65535
        }

        Label {
            id: currentIterationLabel
            text: 'Current iteration: '
        }
        Label {
            id: currentIterationValueLabel
            text: ''
        }

        Button {
            id: runButton
            text: 'Run'
            checkable: true
            checked: false
            Layout.columnSpan: 2
            Layout.fillWidth: true
        }

        Item {
            id: filler
            Layout.columnSpan: 2
            Layout.fillHeight: true
        }
    }

    states : [
        State {
            name: 'has_mandelbrot'
            when: mandelbrot !== undefined
            PropertyChanges { target: iterationsLabel; enabled: !mandelbrot.isRunning }
            PropertyChanges { target: iterationsSpinbox; enabled: !mandelbrot.isRunning; value: mandelbrot.iterationCount }
            PropertyChanges { target: currentIterationLabel; enabled: mandelbrot.isRunning }
            PropertyChanges { target: currentIterationValueLabel; enabled: mandelbrot.isRunning }
            PropertyChanges {
                target: runButton
                onClicked: {
                    mandelbrot.isRunning = runButton.checked
                }
            }
            PropertyChanges {
                target: mandelbrot
                iterationCount: iterationsSpinbox.value
                onCurrentIterationChanged: {
                    currentIterationValueLabel.text = mandelbrot.currentIteration.toString()
                }
                onIsRunningChanged: {
                    runButton.checked = mandelbrot.isRunning
                }
            }
            PropertyChanges { target: mandelbrotItem; enabled: true }
        }
    ]
}
