import sys
from PySide6.QtWidgets import QApplication
import wizard_creator

if __name__ == "__main__":
    app = QApplication(sys.argv + ['-platform', 'windows:darkmode=2'])
    app.setStyle('Fusion')

    window = wizard_creator.MainWizard()
    window.setWindowTitle("MEAO Configuration JSON File Generator")
    window.setMinimumSize(1000, 600)
    window.resize(600, 500)
    window.show()
    app.exec()