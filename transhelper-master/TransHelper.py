from PySide import QtGui

class CentralWidget(QtGui.QWidget):
    
    def __init__(self, parent=None):
        super(CentralWidget, self).__init__(parent)
        self.input = QtGui.QPlainTextEdit()
        self.output = QtGui.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.button = QtGui.QPushButton("Convert")
        
        vLayout = QtGui.QVBoxLayout()
        
        hLayout = QtGui.QHBoxLayout()
        hLayout.addWidget(self.input)
        hLayout.addWidget(self.output)
        
        vLayout.addLayout(hLayout)
        vLayout.addWidget(self.button)
        
        self.setLayout(vLayout)
        self.button.clicked.connect(self.convert)
        
        self.incursor = self.input.textCursor()
        self.indoc    = self.input.document()
        self.outcursor = self.output.textCursor()
        self.outdoc    = self.output.document()
        
    def convert(self):
        self.incursor.movePosition(QtGui.QTextCursor.Start)
        inText = self.input.toPlainText()
        outText = inText.replace('\n', ' ')
        outText = outText.replace('- ', '')
        self.output.clear()
        self.outcursor.insertText(outText)
        #obj = self.input.text()
        #self.output.setText(obj)

class _MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(_MainWindow, self).__init__(parent)        
        self.setWindowTitle("TransHelper")
        self.setCentralWidget(CentralWidget(self))


if __name__ == '__main__':
    app = QtGui.QApplication.instance()
    
    MainWindow = _MainWindow()
    MainWindow.show()