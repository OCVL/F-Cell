from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QSize, Signal
import advancedconfig
import wizard_creator
import import_generation


if __name__ == "__main__":
    app = QApplication([])

    window = wizard_creator.MainWizard()
    window.setWindowTitle("MEAO Configuration JSON File Generator")
    window.setMinimumSize(1000, 600)
    window.resize(600, 500)
    window.show()
    app.exec()