import os
import re

from PySide6.QtGui import QAction, QPixmap, QFontMetrics, QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QToolButton,
                               QLabel, QComboBox, QTextEdit, QLineEdit, QPushButton, QRadioButton, QHBoxLayout,
                               QButtonGroup, QCheckBox, QSizePolicy, QDialog,
                               QDialogButtonBox, QFileDialog, QSplitter, QListWidget,
                               QInputDialog, QMenu, QMessageBox, QFrame, QGridLayout)
from PySide6.QtCore import Qt, QSize, Signal

from PySide6.QtWidgets import QWidget, QHBoxLayout, QCheckBox

class OptionalField(QWidget):
    def __init__(self, widget, default_checked=True):
        super().__init__()

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(default_checked)

        self.inner_widget = widget
        self.inner_widget.setEnabled(default_checked)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.inner_widget)
        layout.setAlignment(Qt.AlignLeft)

        # Disable/enable inner widget based on checkbox state
        self.checkbox.toggled.connect(self.inner_widget.setEnabled)

    def is_checked(self):
        return self.checkbox.isChecked()

    def get_widget(self):
        return self.inner_widget


class ColorMapSelector(QWidget):
    def __init__(self, parent=None, cmap_def=None):
        super().__init__(parent)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(15)

        # Widget to display the selected colormap name
        self.chosen_label = QLabel(f"{cmap_def}" if cmap_def else "None")
        self.chosen_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Center vertically

        # Widget to display the selected colormap image
        self.chosen_image = QLabel()
        self.chosen_image.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Center vertically
        self.chosen_image.setFixedSize(180, 25)

        self.select_button = QPushButton("Select Colormap")
        self.select_button.clicked.connect(self.show_selector)

        # Add widgets with proper alignment
        self.layout.addWidget(self.chosen_label)
        self.layout.addWidget(self.chosen_image)
        self.layout.addWidget(self.select_button)
        self.layout.addStretch()  # Push everything to the left

        # Set the overall widget alignment
        self.setMaximumHeight(35)  # Constrain height to match other form elements

        self.current_selection = cmap_def
        if cmap_def:
            self.update_selection(cmap_def)

    def show_selector(self):
        """Show the list editing dialog"""
        dialog = ColorMapSelectorDialog(self.current_selection, self)
        if dialog.exec() == QDialog.Accepted:
            selection = dialog.get_selection()
            if selection:
                self.current_selection = selection
                self.update_selection(selection)

    def update_selection(self, selection):
        """Update the label and image with the selected colormap"""
        self.chosen_label.setText(selection)

        # Load and display the corresponding image
        image_path = f"images/colormap_options/{selection.lower()}.png"
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            # Scale the pixmap to fit the label while maintaining aspect ratio
            self.chosen_image.setPixmap(pixmap.scaled(
                self.chosen_image.width(),
                self.chosen_image.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def get_value(self):
        return self.current_selection.lower() if self.current_selection else None

    def set_value(self, val):
        """Set the colormap value and update the UI"""
        self.current_selection = val
        if val:
            self.update_selection(val)
        else:
            # Handle case where val is None or empty
            self.chosen_label.setText("None")
            self.chosen_image.clear()  # Clear the image


class ColorMapSelectorDialog(QDialog):
    def __init__(self, current_selection=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Colormap")
        self.setModal(True)
        self.setFixedSize(550, 200)  # Slightly wider to accommodate alignment

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(8)  # Slightly increased spacing for better grouping

        # Create a button group to manage exclusive selection
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        # List of available colormaps
        self.colormaps = ["Cividis", "Inferno", "Magma", "Plasma", "Viridis"]

        # Find the maximum text width to align radio buttons
        font_metrics = QFontMetrics(self.font())
        max_text_width = max(font_metrics.horizontalAdvance(cmap) for cmap in self.colormaps) + 25

        # Create a grid layout for perfect alignment
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(15)  # Space between radio button and image
        grid_layout.setVerticalSpacing(5)  # Space between rows

        for row, cmap in enumerate(self.colormaps):
            # Create radio button with fixed width
            radio = QRadioButton(cmap)
            radio.setFixedWidth(max_text_width)  # Ensure all radio buttons same width
            if cmap == current_selection:
                radio.setChecked(True)
            self.button_group.addButton(radio)

            # Load and display colormap image
            image_path = f"images/colormap_options/{cmap.lower()}.png"
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                image_label = QLabel()
                image_label.setPixmap(pixmap.scaled(300, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                image_label.setFixedSize(300, 30)  # Ensure all images same size

            grid_layout.addWidget(radio, row, 0)
            grid_layout.addWidget(image_label, row, 1)

        layout.addLayout(grid_layout)
        layout.addStretch()  # Push buttons to bottom

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_selection(self):
        """Return the selected colormap name"""
        selected_button = self.button_group.checkedButton()
        return selected_button.text() if selected_button else None

class ListEditorWidget(QWidget):
    itemsChanged = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)

        # Main widget layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Label to display the list
        self.label = QLabel("null")
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Button to open the editor
        self.edit_button = QPushButton("Edit List")
        self.edit_button.clicked.connect(self.show_editor)

        # Add widgets to layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.edit_button)

        # Initialize the list
        self.items = []

    def show_editor(self):
        """Show the list editing dialog"""
        dialog = ListEditorDialog(self.items, self)
        if dialog.exec() == QDialog.Accepted:
            self.items = dialog.get_items()
            self.update_label()
            self.itemsChanged.emit()

    def update_label(self):
        if not self.items:
            self.label.setText("null")
        else:
            self.label.setText(', '.join(self.items))

    def get_list(self):
        return self.items if self.items else None

    def set_value(self, val):
        self.items = val
        self.update_label()
        self.itemsChanged.emit()

class ListEditorDialog(QDialog):
    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit List")
        self.setModal(True)

        # Layout
        layout = QVBoxLayout(self)

        # List widget to display items
        self.list_widget = QListWidget()
        self.list_widget.addItems(items)

        # Add/remove controls
        control_layout = QHBoxLayout()

        self.new_item_input = QLineEdit()
        self.new_item_input.setPlaceholderText("New item")

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_item)

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.remove_item)

        control_layout.addWidget(self.new_item_input)
        control_layout.addWidget(add_button)
        control_layout.addWidget(remove_button)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Add widgets to layout
        layout.addWidget(self.list_widget)
        layout.addLayout(control_layout)
        layout.addWidget(button_box)

    def add_item(self):
        """Add a new item to the list"""
        text = self.new_item_input.text().strip()
        if text:
            self.list_widget.addItem(text)
            self.new_item_input.clear()
            self.list_widget.scrollToBottom()

    def remove_item(self):
        """Remove the currently selected item"""
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            self.list_widget.takeItem(current_row)

    def get_items(self):
        """Return all items as a list of strings"""
        return [self.list_widget.item(i).text()
                for i in range(self.list_widget.count())]


class OpenFolder(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(30)
        main_layout.setAlignment(Qt.AlignLeft)

        self.textbox = FreetextBox("null")
        self.textbox.setFixedWidth(250)
        self.button = QPushButton("Select Output Folder")
        self.button.setFixedWidth(150)

        main_layout.addWidget(self.textbox)
        main_layout.addWidget(self.button)

        self.button.clicked.connect(self.select_folder)

        self.folder_path = ""

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder_path:  # If user didn't cancel
            self.folder_path = folder_path
            self.textbox.set_text(folder_path)
        else:
            self.folder_path = ""
            self.textbox.set_text("null")

    def get_text(self):
        return None if self.textbox.get_text() == "null" else self.textbox.get_text()

    def set_text(self, val):
        self.textbox.set_text(val)


class FormatElementsEditor(QDialog):
    """Dialog for editing filename format using predefined elements"""
    copyRequested = Signal(str, str, str)

    def __init__(self, current_format=None, parent=None, queryloc=False, section_name=None, format_key=None, enable_copy=True):
        super().__init__(parent)
        self.setWindowTitle("Format Editor")
        self.setGeometry(500, 500, 650, 500)

        self.original_elements = [
            ":.1", "Day", "Eye", "FOV_Height", "FOV_Width",
            "IDnum", "LocX", "LocY", "Modality", "Month",
            "VidNum", "Year"
        ]
        if queryloc: self.original_elements.append("QueryLoc")
        self.available_elements = self.original_elements.copy()

        self.section_name = section_name
        self.format_key = format_key

        self.copy_button = QPushButton("Copy to All in Section")
        self.copy_button.clicked.connect(self.copy_to_all)

        # === MAIN VERTICAL LAYOUT ===
        window_layout = QVBoxLayout(self)

        # === PREVIEW AT TOP (horizontal layout) ===
        preview_layout = QHBoxLayout()
        self.preview_label = QLabel("Preview:")
        self.preview_label.setStyleSheet("font-weight: bold; padding: 6px;")
        self.preview_display = QLabel("")
        self.preview_display.setStyleSheet("""
            QLabel {
                padding: 6px;
                color: #333;
                min-height: 40px;
            }
        """)

        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(self.preview_display)
        if enable_copy:
            preview_layout.addWidget(self.copy_button)
        preview_layout.setAlignment(Qt.AlignLeft)
        window_layout.addLayout(preview_layout)

        # === MAIN BODY LAYOUT (Available + Buttons + Selected) ===
        main_layout = QHBoxLayout()
        window_layout.addLayout(main_layout)

        # === LEFT PANEL - Available Elements ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_label = QLabel("Available Elements")
        left_label.setStyleSheet("font-weight: bold; padding: 6px;")
        left_layout.addWidget(left_label)

        self.available_list = QListWidget()
        self.available_list.addItems(self.available_elements)
        # Enable hover events for tooltips
        self.available_list.setMouseTracking(True)
        self.available_list.itemEntered.connect(self.update_tooltip)
        left_layout.addWidget(self.available_list)
        main_layout.addWidget(left_panel)

        # === MIDDLE PANEL - Buttons ===
        button_panel = QWidget()
        button_layout = QVBoxLayout(button_panel)
        button_layout.setAlignment(Qt.AlignCenter)

        # Add some spacing at the top to center buttons vertically
        button_layout.addStretch()

        self.add_element_button = QPushButton("Add >>")
        self.remove_element_button = QPushButton("<< Remove")
        self.set_width_button = QPushButton("Set Width")

        # Make buttons a consistent size
        for btn in (self.add_element_button, self.remove_element_button, self.set_width_button):
            btn.setMinimumWidth(120)
            button_layout.addWidget(btn)

        # Connect button signals
        self.add_element_button.clicked.connect(self.add_selected_element)
        self.remove_element_button.clicked.connect(self.remove_selected_element)
        self.set_width_button.clicked.connect(self.set_element_width)

        button_layout.addStretch()
        main_layout.addWidget(button_panel)

        # === RIGHT PANEL - Selected Elements ===
        right_panel = QWidget()
        right_main_layout = QHBoxLayout(right_panel)

        # Left side of right panel - the list
        right_left_panel = QWidget()
        right_layout = QVBoxLayout(right_left_panel)
        right_label = QLabel("Selected Format Elements")
        right_label.setStyleSheet("font-weight: bold; padding: 6px;")
        right_layout.addWidget(right_label)

        self.selected_list = QListWidget()
        self.selected_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.selected_list.customContextMenuRequested.connect(self.show_context_menu)
        self.selected_list.setMouseTracking(True)
        self.selected_list.itemEntered.connect(self.update_tooltip)
        right_layout.addWidget(self.selected_list)
        right_main_layout.addWidget(right_left_panel)

        # Right side of right panel - vertical buttons
        right_button_panel = QWidget()
        right_button_layout = QVBoxLayout(right_button_panel)
        right_button_layout.setAlignment(Qt.AlignCenter)

        # Add stretch to center the buttons vertically
        right_button_layout.addStretch()

        self.move_up_button = QPushButton("Move Up")
        self.move_down_button = QPushButton("Move Down")
        self.add_separator_button = QPushButton("Add Static Text")
        self.clear_button = QPushButton("Clear All")

        # Make buttons consistent size
        for btn in (self.move_up_button, self.move_down_button, self.add_separator_button, self.clear_button):
            btn.setMinimumWidth(120)
            right_button_layout.addWidget(btn)

        right_button_layout.addStretch()
        right_main_layout.addWidget(right_button_panel)

        main_layout.addWidget(right_panel)

        # Make the left and right panels equal width, smaller than before
        main_layout.setStretchFactor(left_panel, 1)
        main_layout.setStretchFactor(button_panel, 0)
        main_layout.setStretchFactor(right_panel, 1)

        # Connect move button signals
        self.move_up_button.clicked.connect(self.move_element_up)
        self.move_down_button.clicked.connect(self.move_element_down)
        self.add_separator_button.clicked.connect(self.add_separator)
        self.clear_button.clicked.connect(self.clear_all_elements)

        # === TOOLTIP BELOW BOTH LISTS ===
        self.tooltip_label = QLabel("Hover over an available element to see its description")
        self.tooltip_label.setWordWrap(True)
        self.tooltip_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                border-radius: 6px;
                background-color: #f0f0f0;
                padding: 6px;
                font-style: italic;
                color: #333;
                min-height: 40px;
            }
        """)
        window_layout.addWidget(self.tooltip_label)

        # === ACCEPT / CANCEL ===
        button_row = QHBoxLayout()
        button_row.addStretch()

        self.accept_button = QPushButton("Accept")
        self.cancel_button = QPushButton("Cancel")
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        button_row.addWidget(self.accept_button)
        button_row.addWidget(self.cancel_button)
        window_layout.addLayout(button_row)

        # Connect double-click events
        self.available_list.itemDoubleClicked.connect(self.handle_double_click_available)
        self.selected_list.itemDoubleClicked.connect(self.handle_double_click_selected)

        self.available_list.leaveEvent = self.leave_event_available
        self.selected_list.leaveEvent = self.leave_event_selected

        # Connect list change signals
        self.selected_list.model().rowsInserted.connect(self.update_preview)
        self.selected_list.model().rowsRemoved.connect(self.update_preview)
        self.selected_list.model().dataChanged.connect(self.update_preview)

        # Parse the current format if provided
        if current_format:
            self.parse_format(current_format)

        # Update the preview initially
        self.update_preview()

    def copy_to_all(self):
        reply = QMessageBox.question(
            self,
            "Confirm Copy",
            "Are you sure you copy this format to all others in this section?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            format_string = self.get_format_string()
            self.copyRequested.emit(self.section_name, self.format_key, format_string)
            self.accept()

    def clear_all_elements(self):
        """Clear all selected elements and return format elements to available list"""
        # Process each item in the selected list
        items_to_return = []

        for i in range(self.selected_list.count()):
            item_text = self.selected_list.item(i).text()

            # Only return format elements (not static text) to available list
            if item_text.startswith("{") and item_text.endswith("}") and not item_text.startswith("{Added Text:"):
                element = item_text[1:-1]  # Remove the braces

                # Special handling for :.1 element
                if element == ":.1":
                    items_to_return.append(element)
                else:
                    # Get base element name (without width specification)
                    base_element = element.split(':')[0] if ':' in element else element
                    if base_element in self.original_elements:
                        items_to_return.append(base_element)

        # Clear the selected list
        self.selected_list.clear()

        # Add returned elements back to available list
        for element in items_to_return:
            if element not in self.available_elements:
                self.available_elements.append(element)

        # Keep list sorted and unique
        self.available_elements = sorted(list(set(self.available_elements)))
        self.available_list.clear()
        self.available_list.addItems(self.available_elements)

        # Update the preview
        self.update_preview()

    def set_element_width(self):
        """Set the width for the selected element"""
        current_item = self.selected_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection",
                                    "Please select an element from the Selected Format Elements list first.")
            return

        item_text = current_item.text()

        # Check if it's a format element (not a separator)
        if not item_text.startswith("{") or not item_text.endswith("}") or item_text.startswith("{Added Text:"):
            QMessageBox.information(self, "Invalid Selection",
                                    "Width can only be set for format elements, not text separators.")
            return

        # Extract the element name and any existing width
        element = item_text[1:-1]  # Remove braces
        current_width = None

        # Check if element already has a width specification
        if ':' in element:
            parts = element.split(':')
            element_name = parts[0]
            width_part = parts[1]
            if '.' in width_part:
                current_width = width_part.split('.')[1]
            else:
                current_width = width_part
        else:
            element_name = element

        # Ask user for new width
        width, ok = QInputDialog.getText(self, "Set Element Width",
                                         f"Enter width for {element_name}:",
                                         text=current_width if current_width else "")
        if ok and width:
            try:
                # Validate it's a number
                width_int = int(width)
                if width_int == 0:
                    # Remove width specification entirely
                    new_element = f"{{{element_name}}}"
                else:
                    # Update the element with new width
                    new_element = f"{{{element_name}:.{width}}}"
                current_item.setText(new_element)
                self.update_preview()
            except ValueError:
                QMessageBox.warning(self, "Invalid Width", "Width must be a number")

    def parse_format(self, format_string):
        """Parse an existing format string into elements"""
        # First reset available elements
        self.available_elements = self.original_elements.copy()
        self.available_list.clear()
        self.available_list.addItems(self.available_elements)

        # Parse the string by looking for elements enclosed in brackets
        remaining = format_string

        # Process the format string
        while remaining:
            # Looking for elements in brackets like {IDnum}
            start_idx = remaining.find('{')
            if start_idx == -1:
                # No more elements in brackets, add remaining as separator if not empty
                if remaining:
                    self.selected_list.addItem(f"{{Added Text: {remaining}}}")
                break

            # Add any text before the bracket as a separator
            if start_idx > 0:
                separator = remaining[:start_idx]
                self.selected_list.addItem(f"{{Added Text: {separator}}}")

            # Find closing bracket
            end_idx = remaining.find('}', start_idx)
            if end_idx == -1:
                # No closing bracket, add the rest as a separator
                self.selected_list.addItem(f"{{Added Text: {remaining}}}")
                break

            # Extract the element name
            element = remaining[start_idx + 1:end_idx]
            if element in self.available_elements or element == ":.1":
                self.selected_list.addItem(f"{{{element}}}")
                # Remove from available elements
                if element in self.available_elements:
                    self.available_elements.remove(element)
                elif element == ":.1" and ":.1" in self.available_elements:
                    self.available_elements.remove(":.1")
                self.available_list.clear()
                self.available_list.addItems(self.available_elements)
            else:
                # Not a recognized element, treat as a separator with brackets
                self.selected_list.addItem(f"{{Added Text: {{{element}}}}}")

            # Continue with the remaining string
            remaining = remaining[end_idx + 1:]

    def get_used_elements(self):
        """Return a set of elements already in use"""
        used_elements = set()
        for i in range(self.selected_list.count()):
            item_text = self.selected_list.item(i).text()
            if item_text.startswith("{") and item_text.endswith("}") and not item_text.startswith("{Added Text:"):
                element = item_text[1:-1]  # Remove the braces
                # Get base element name (without width specification)
                base_element = element.split(':')[0] if ':' in element else element
                if base_element in self.original_elements or element in self.original_elements:
                    used_elements.add(element)
        return used_elements

    def add_selected_element(self):
        """Add the selected element from available list to selected list"""
        if not self.available_list.currentItem():
            QMessageBox.information(self, "No Selection",
                                    "Please select an element from the Available Elements list first.")
            return

        element = self.available_list.currentItem().text()
        self.selected_list.addItem(f"{{{element}}}")

        # Remove from available list
        row = self.available_list.currentRow()
        self.available_list.takeItem(row)
        self.available_elements.remove(element)

        self.update_preview()

    def remove_selected_element(self):
        """Remove the selected element from the selected list"""
        if not self.selected_list.currentItem():
            QMessageBox.information(self, "No Selection",
                                    "Please select an element from the Selected Format Elements list first.")
            return

        item = self.selected_list.currentItem()
        item_text = item.text()

        # Only return to available list if it's a format element (not a separator)
        if item_text.startswith("{") and item_text.endswith("}") and not item_text.startswith("{Added Text:"):
            element = item_text[1:-1]  # Remove the braces

            # Special handling for :.1 element
            if element == ":.1":
                self.available_elements.append(element)
            else:
                # Get base element name (without width specification)
                base_element = element.split(':')[0] if ':' in element else element
                if base_element in self.original_elements:
                    # Add back to available list
                    self.available_elements.append(base_element)

            # Keep list sorted and unique
            self.available_elements = sorted(list(set(self.available_elements)))
            self.available_list.clear()
            self.available_list.addItems(self.available_elements)

        row = self.selected_list.currentRow()
        self.selected_list.takeItem(row)
        self.update_preview()

    def handle_double_click_available(self, item):
        """Handle double-click on available list item"""
        element = item.text()
        self.selected_list.addItem(f"{{{element}}}")

        # Remove from available list
        row = self.available_list.row(item)
        self.available_list.takeItem(row)
        self.available_elements.remove(element)

        self.update_preview()

    def handle_double_click_selected(self, item):
        """Handle double-click on selected list item"""
        item_text = item.text()

        # Only return to available list if it's a format element (not a separator)
        if item_text.startswith("{") and item_text.endswith("}") and not item_text.startswith("{Added Text:"):
            element = item_text[1:-1]  # Remove the braces

            # Special handling for :.1 element
            if element == ":.1":
                self.available_elements.append(element)
            else:
                # Get base element name (without width specification)
                base_element = element.split(':')[0] if ':' in element else element
                if base_element in self.original_elements:
                    # Add back to available list
                    self.available_elements.append(base_element)

            # Keep list sorted and unique
            self.available_elements = sorted(list(set(self.available_elements)))
            self.available_list.clear()
            self.available_list.addItems(self.available_elements)

        row = self.selected_list.row(item)
        self.selected_list.takeItem(row)
        self.update_preview()

    def move_element_up(self):
        current_row = self.selected_list.currentRow()
        if current_row <= 0:
            return

        item = self.selected_list.takeItem(current_row)
        self.selected_list.insertItem(current_row - 1, item)
        self.selected_list.setCurrentRow(current_row - 1)
        self.update_preview()

    def move_element_down(self):
        """Move the selected element down in the selected list"""
        current_row = self.selected_list.currentRow()
        if current_row < 0 or current_row >= self.selected_list.count() - 1:
            return

        item = self.selected_list.takeItem(current_row)
        self.selected_list.insertItem(current_row + 1, item)
        self.selected_list.setCurrentRow(current_row + 1)
        self.update_preview()

    def add_separator(self):
        """Add a separator between elements"""
        text, ok = QInputDialog.getText(self, "Add text",
                                        "Enter text:")
        if ok:
            # Get selected position or append to end
            current_row = self.selected_list.currentRow()
            if current_row >= 0:
                self.selected_list.insertItem(current_row + 1, f"{{Added Text: {text}}}")
                self.selected_list.setCurrentRow(current_row + 1)
            else:
                self.selected_list.addItem(f"{{Added Text: {text}}}")
                self.selected_list.setCurrentRow(self.selected_list.count() - 1)
            self.update_preview()

    def update_preview(self):
        """Update the preview label with the current format string with different styling for elements and static text."""
        preview_html = ""
        # preview_html = "<html><body style='white-space: pre;'>

        for i in range(self.selected_list.count()):
            item_text = self.selected_list.item(i).text()

            if item_text.startswith("{Added Text: ") and item_text.endswith("}"):
                # Static text - display in blue and bold
                separator = item_text[13:-1]
                preview_html += f"{separator}"
                # preview_html += f"<span style='color: #095591; font-weight: bold;'>{separator}</span>"
            else:
                # Format element - display in dark green with slight italic
                preview_html += f"{item_text}"
                # preview_html += f"<span style='color: #958e7e; '>{item_text}</span>"

        preview_html += ""
        # preview_html += "</body></html>"
        self.preview_display.setText(preview_html)

    def update_tooltip(self, item):
        """Update tooltip box when hovering over available or selected elements."""
        tooltips = {
            ":.1": "Format specifier for decimal precision (e.g., :.1 for 1 decimal place)",
            "Day": "Calendar day (1-31)",
            "Eye": "Eye designation (OD/OS for right/left eye)",
            "FOV_Height": "Field of view height in pixels",
            "FOV_Width": "Field of view width in pixels",
            "IDnum": "Unique identifier number for the image",
            "LocX": "X-coordinate location",
            "LocY": "Y-coordinate location",
            "Modality": "Imaging modality (e.g., OCT, Fundus)",
            "Month": "Calendar month in numerical format (1-12)",
            "VidNum": "Video number identifier",
            "Year": "Calendar year (4 digits)",
            "{Added Text:": "Static text that will appear literally in the filename"
        }

        if not item:
            self.tooltip_label.setText("Hover over an element to see its description")
            return

        item_text = item.text()

        # Handle selected list items (which might have formatting)
        if item_text.startswith("{") and item_text.endswith("}"):
            if item_text.startswith("{Added Text:"):
                # Static text separator
                self.tooltip_label.setText("Static text that will appear literally in the filename")
            else:
                # Format element - extract the base name
                element = item_text[1:-1]  # Remove braces
                if ':' in element:
                    element = element.split(':')[0]  # Get base element before width spec

                # Get the tooltip or fall back to available elements tooltip
                tooltip = tooltips.get(element, tooltips.get(item_text, "No description available"))
                self.tooltip_label.setText(f"{element}: {tooltip}")
        else:
            # Available list item
            tooltip = tooltips.get(item_text, "No description available")
            self.tooltip_label.setText(f"{item_text}: {tooltip}")

    def show_context_menu(self, position):
        """Show context menu for the selected list items"""
        menu = QMenu()
        edit_action = QAction("Edit", self)
        remove_action = QAction("Remove", self)

        menu.addAction(edit_action)
        menu.addAction(remove_action)

        # Get the item at the position
        item = self.selected_list.itemAt(position)
        if not item:
            return

        # Connect actions
        action = menu.exec(self.selected_list.mapToGlobal(position))

        if action == edit_action:
            item_text = item.text()
            # Check if it's a separator
            if item_text.startswith("{Added Text: ") and item_text.endswith("}"):
                old_text = item_text[13:-1]  # Extract text between "{Added Text: " and "}"
                new_text, ok = QInputDialog.getText(self, "Edit Added Text",
                                                    "Edit added text:", text=old_text)
                if ok:
                    item.setText(f"{{Added Text: {new_text}}}")
        elif action == remove_action:
            self.remove_selected_element()

        self.update_preview()

    def get_format_string(self):
        """Return the complete format string as plain text without HTML formatting"""
        format_string = ""

        for i in range(self.selected_list.count()):
            item_text = self.selected_list.item(i).text()

            if item_text.startswith("{Added Text: ") and item_text.endswith("}"):
                # Static text - just add the text part
                separator = item_text[13:-1]
                format_string += separator
            else:
                # Format element - add as is
                format_string += item_text

        return format_string

    def get_formatted_preview(self):
        """Return the HTML formatted version for display"""
        return self.preview_display.text()

    def leave_event_available(self, event):
        """Clear tooltip when mouse leaves available list"""
        self.tooltip_label.setText("Hover over an element to see its description")

    def leave_event_selected(self, event):
        """Clear tooltip when mouse leaves selected list"""
        self.tooltip_label.setText("Hover over an element to see its description")


class FormatEditorWidget(QWidget):
    """A reusable widget that combines a format preview label and edit button"""

    formatChanged = Signal(str)
    copyToAllRequested = Signal(str, str, str)

    def __init__(self, label_text, default_format="", parent=None, queryloc=False, section_name=None, format_key=None):
        super().__init__(parent)
        self.default_format = default_format
        self.current_format = default_format  # Initialize with default
        self.return_text = default_format  # Initialize return text with default
        self.queryloc = queryloc
        self.section_name = section_name
        self.format_key = format_key

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create label for the format string
        self.format_label = QLabel(default_format)
        self.layout.addWidget(self.format_label)

        # Create edit button
        self.edit_button = QPushButton("Edit...")
        self.edit_button.clicked.connect(self._open_format_editor)  # Connect to the method
        self.layout.addWidget(self.edit_button)

        # Store the label text for the editor dialog
        self.label_text = label_text

    def _open_format_editor(self):
        """Open the format editor dialog"""
        dialog = FormatElementsEditor(self.current_format, self, self.queryloc, self.section_name, self.format_key)

        dialog.copyRequested.connect(self.relay_copy_request)

        if dialog.exec() == QDialog.Accepted:
            # Store the plain text version for JSON
            self.return_text = dialog.get_format_string()
            # Store the formatted version for display
            self.current_format = dialog.get_formatted_preview()
            # Update the display
            self.format_label.setText(self.current_format)
            self.formatChanged.emit(self.return_text)  # Emit the plain text version

    def relay_copy_request(self, section, key, fmt):
        self.copyToAllRequested.emit(section, key, fmt)

    def set_value(self, format_string):
        """Set both the display and plain text versions"""
        self.current_format = format_string
        self.return_text = format_string  # Assuming input is plain text
        self.format_label.setText(format_string)

    def reset_to_default(self):
        """Reset the format to the original default"""
        self.set_value(self.default_format)

    def get_value(self):
        return self.return_text


class GroupByFormatEditorWidget(FormatEditorWidget):
    """Specialized format editor for groupby parameter that gets elements from format strings"""

    def __init__(self, image_format, video_format, mask_format, label_text, default_format="", parent=None):
        super().__init__(label_text, default_format, parent)

        # Set format strings, using defaults if empty/None
        self.image_format = image_format if image_format else None
        self.video_format = video_format if video_format else None
        self.mask_format = mask_format if mask_format else None

        # Initialize with default elements from formats
        self.available_elements = self.get_available_elements()

        # Initialize with empty format if default is "null"
        if self.current_format == "null":
            self.current_format = ""
            self.format_label.setText("")

    def get_available_elements(self):
        """Dynamically get elements from the provided format strings, preserving width specifications"""
        elements = set()

        def extract_elements(format_str):
            if not format_str or format_str == "null":  # Skip if format is "null"
                return set()

            elements = set()
            start = 0
            while True:
                # Look for {element} patterns
                start_idx = format_str.find('{', start)
                if start_idx == -1:
                    break

                end_idx = format_str.find('}', start_idx)
                if end_idx == -1:
                    break

                # Extract the full element including any format specifiers
                element = format_str[start_idx + 1:end_idx]
                # Only add if it's not static text (Added Text: prefix)
                if not element.startswith("Added Text:"):
                    elements.add(element)

                start = end_idx + 1
            return elements

        # Get elements from each format string, preserving their full specifications
        elements.update(extract_elements(self.image_format))
        elements.update(extract_elements(self.video_format))
        elements.update(extract_elements(self.mask_format))

        # Remove any empty strings that might have been added
        elements.discard('')


        return sorted(elements)

    def update_format_sources(self, image_format, video_format, mask_format):
        self.image_format = image_format if image_format else None
        self.video_format = video_format if video_format else None
        self.mask_format = mask_format if mask_format else None
        self.available_elements = self.get_available_elements()

        current_fields = re.findall(r"{(.*?)}", self.current_format) if self.current_format else []

        invalid = [field for field in current_fields if field not in self.available_elements]
        if invalid:
            self.current_format = ""
            self.return_text = "null"
            self.format_label.setText("")
            self.formatChanged.emit("null")

    def _open_format_editor(self):
        """Override parent method to use dynamic elements"""
        # Update available elements first, preserving any width specifications
        self.available_elements = self.get_available_elements()

        # If current format is "null", pass empty string to the dialog
        current_format = None if self.current_format == "null" else self.current_format
        dialog = FormatElementsEditor(current_format, self, enable_copy=False)

        # Override the dialog's available elements with our dynamic ones
        dialog.original_elements = self.available_elements.copy()
        dialog.available_elements = self.available_elements.copy()

        # Clear and repopulate the available elements list with dynamic elements
        dialog.available_list.clear()
        dialog.available_list.addItems(dialog.available_elements)

        if dialog.exec() == QDialog.Accepted:
            new_format= dialog.get_format_string()
            # Store empty string as "null" if the format is empty
            self.return_text = "null" if not new_format else new_format
            self.current_format = "null" if not new_format else new_format
            self.format_label.setText("" if not new_format else new_format)
            # Emit the change signal
            self.formatChanged.emit(self.return_text)

    def set_value(self, val):
        """Set the format string value and update the UI"""
        if not val or val == "null":
            self.current_format = "null"
            self.return_text = "null"
            self.format_label.setText("")
        else:
            self.current_format = val
            self.return_text = val
            self.format_label.setText(val)

class TrueFalseSelector(QWidget):
    def __init__(self, default_value=None, parent=None):
        super().__init__(parent)

        # Create the layout
        layout = QHBoxLayout()
        layout.setSpacing(30)

        # Create checkbox
        self.checkbox = QCheckBox("")

        # Set default value
        if default_value:
            self.checkbox.setChecked(True)
        else:
            self.checkbox.setChecked(False)

        # Add to layout
        layout.addWidget(self.checkbox)
        layout.addStretch()  # Push checkbox to the left

        self.setLayout(layout)

    def get_value(self):
        return self.checkbox.isChecked()

    def set_value(self, val):
        if val is True:
            self.checkbox.setChecked(True)
        else:
            self.checkbox.setChecked(False)


class AffineRigidSelector(QWidget):
    def __init__(self, default_value=None, parent=None):
        super().__init__(parent)

        # Create the layout
        layout = QHBoxLayout()
        layout.setSpacing(30)

        # Create radio buttons
        self.true_button = QRadioButton("Affine")
        self.false_button = QRadioButton("Rigid")

        # Create button group
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.true_button, 1)  # ID 1 for True
        self.button_group.addButton(self.false_button, 0)  # ID 0 for False
        if default_value:
            self.true_button.setChecked(True)
        elif not default_value:
            self.false_button.setChecked(True)

        # Add to layout
        layout.addWidget(self.true_button)
        layout.addWidget(self.false_button)
        layout.addStretch()  # Push buttons to the left

        self.setLayout(layout)

    def get_value(self):
        return "affine" if self.true_button.isChecked() else "rigid"

    def set_value(self, val):
        if val == "affine":
            self.true_button.setChecked(True)
        else:
            self.false_button.setChecked(True)

class DropdownMenu(QWidget):
    def __init__(self, default="", options=None, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.comboBox = QComboBox(self)
        self.comboBox.setFixedWidth(100)

        if isinstance(options, str):
            options = [options]

        self.comboBox.addItem(default)
        if options:
            for option in options:
                self.comboBox.addItem(option)

        layout.addWidget(self.comboBox)

    def get_value(self):
        return self.comboBox.currentText()

    def set_value(self, val):
        if val is None:
            return
        if self.comboBox.findText(val) == -1:
            self.comboBox.addItem(val)
        self.comboBox.setCurrentText(val)

    def update_options(self, options, keep_selected=True):
        current_val = self.get_value() if keep_selected else None

        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        for option in options:
            self.comboBox.addItem(option)
        if keep_selected and current_val in options:
            self.comboBox.setCurrentText(current_val)
        self.comboBox.blockSignals(False)


class CollapsibleSection(QWidget):
    def __init__(self, title="", default=None, parent=None):
        super().__init__(parent)
        self._title = title

        # Create checkbox
        self.enable_checkbox = QCheckBox()

        if default:
            self.enable_checkbox.setChecked(True)
        else:
            self.enable_checkbox.setChecked(False)
        self.enable_checkbox.stateChanged.connect(self._update_enabled_state)
        self.enable_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Create toggle button with arrow icon
        self.toggle_button = QToolButton(self)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setStyleSheet("""
            QToolButton {
                border: none;
                text-align: left;
                padding: 5px;
                color: black;
            }
            QToolButton:checked {
                background-color: #f0f0f0;
            }
            QToolButton:disabled {
                color: gray;
            }
        """)

        # Set up arrow icon
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setIconSize(QSize(12, 12))
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.clicked.connect(self.toggle_content)

        # Header widget containing checkbox and toggle button
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setSpacing(30)
        header_layout.setSpacing(8)  # Add a small spacing between checkbox and button
        header_layout.addWidget(self.enable_checkbox)
        header_layout.addWidget(self.toggle_button, 1)  # Allow toggle button to expand
        header_layout.setAlignment(self.enable_checkbox, Qt.AlignCenter)  # Center align the checkbox vertically
        header_layout.setAlignment(self.toggle_button, Qt.AlignLeft)  # Left align the toggle button

        # Content area
        self.content_area = QWidget()
        self.content_area.setVisible(False)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(header_widget)
        layout.addWidget(self.content_area)

        # Update initial state
        self._update_enabled_state()

    def title(self):
        """Returns the section title"""
        return self._title

    def toggle_content(self):
        if not self.enable_checkbox.isChecked():
            self.toggle_button.setChecked(False)
            return

        # Toggle visibility of content area
        is_checked = self.toggle_button.isChecked()
        self.content_area.setVisible(is_checked)

        # Change arrow direction
        self.toggle_button.setArrowType(Qt.DownArrow if is_checked else Qt.RightArrow)

    def _update_enabled_state(self):
        """Update the enabled state based on checkbox"""
        enabled = self.enable_checkbox.isChecked()

        # Enable/disable the toggle button
        self.toggle_button.setEnabled(enabled)

        # If disabled, ensure the content is collapsed
        if not enabled:
            if self.toggle_button.isChecked():
                self.toggle_button.setChecked(False)
                self.content_area.setVisible(False)
            self.toggle_button.setArrowType(Qt.RightArrow)

    def set_content_layout(self, layout):
        # Clear existing layout if any
        if self.content_area.layout():
            old_layout = self.content_area.layout()
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        # Set new layout
        self.content_area.setLayout(layout)

    def is_enabled(self):
        return self.enable_checkbox.isChecked()


class FreetextBox(QWidget):
    def __init__(self, title=None, multi_line=False, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if multi_line:
            # Multi-line text input
            self.text_input = QTextEdit(self)
            self.text_input.setPlaceholderText(title)
        else:
            # Single-line text input
            self.text_input = QLineEdit(self)
            self.text_input.setPlaceholderText(title)

        layout.addWidget(self.text_input)

    def set_text(self, text):
        self.text_input.setText(text)

    def get_text(self):
        if isinstance(self.text_input, QTextEdit):
            text = self.text_input.toPlainText()
        else:
            text = self.text_input.text()

        # Return placeholder text if the field is empty
        return text if text else self.text_input.placeholderText()


class SaveasExtensionsEditor(QDialog):
    """Dialog for editing filename format using predefined elements"""

    def __init__(self, current_format=None, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Format Editor")
        self.setGeometry(500, 500, 750, 400)

        # Available elements
        self.available_elements = [
            ".png", ".tiff", ".svg", ".ps", ".pgf", ".pgm"
        ]

        # Create the layout
        main_layout = QHBoxLayout(self)

        # Create a splitter to allow resizing the panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Available elements
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        left_label = QLabel("Available Elements")
        left_layout.addWidget(left_label)

        self.available_list = QListWidget()
        self.available_list.addItems(self.available_elements)
        left_layout.addWidget(self.available_list)

        # Center panel with buttons
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.addStretch()

        self.add_button = QPushButton("Add >>")
        self.add_button.clicked.connect(self.add_selected_element)
        center_layout.addWidget(self.add_button)

        self.remove_button = QPushButton("<< Remove")
        self.remove_button.clicked.connect(self.remove_selected_element)
        center_layout.addWidget(self.remove_button)

        center_layout.addStretch()

        # Right panel - Selected elements
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        right_label = QLabel("Selected Format Elements")
        right_layout.addWidget(right_label)

        self.selected_list = QListWidget()
        self.selected_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.selected_list.customContextMenuRequested.connect(self.show_context_menu)
        right_layout.addWidget(self.selected_list)

        # Preview layout
        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(30)
        preview_layout.addWidget(QLabel("Preview:"))
        self.preview_text = QLineEdit()
        self.preview_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_text)
        right_layout.addLayout(preview_layout)

        # Accept/Cancel buttons
        action_buttons = QHBoxLayout()
        self.accept_button = QPushButton("Accept")
        self.accept_button.clicked.connect(self.accept)
        action_buttons.addWidget(self.accept_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        action_buttons.addWidget(self.cancel_button)

        right_layout.addLayout(action_buttons)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)

        # Set the initial sizes of the splitter
        splitter.setSizes([200, 100, 450])

        # Connect double-click events
        self.available_list.itemDoubleClicked.connect(self.handle_double_click_available)
        self.selected_list.itemDoubleClicked.connect(self.handle_double_click_selected)

        # Connect list change signals
        self.selected_list.model().rowsInserted.connect(self.update_preview)
        self.selected_list.model().rowsRemoved.connect(self.update_preview)
        self.selected_list.model().dataChanged.connect(self.update_preview)

        # Parse the current format if provided
        if current_format:
            self.parse_format(current_format)

        # Update the preview initially
        self.update_preview()

    def parse_format(self, format_string):
        """Parse an existing format string into elements"""
        # Remove brackets and quotes if present
        if format_string.startswith('[') and format_string.endswith(']'):
            format_string = format_string[1:-1]

        # Split the string by commas and remove quotes and whitespace
        elements = []
        for elem in format_string.split(","):
            elem = elem.strip()
            # Remove both single and double quotes, handling nested quotes
            while (elem.startswith('"') and elem.endswith('"')) or (elem.startswith("'") and elem.endswith("'")):
                elem = elem[1:-1]
            if elem and elem in self.available_elements:
                elements.append(elem)

        # Add the elements to the selected list (without quotes)
        for elem in elements:
            self.selected_list.addItem(elem)

    def get_used_elements(self):
        """Return a set of elements already in use"""
        used_elements = set()
        for i in range(self.selected_list.count()):
            item_text = self.selected_list.item(i).text()
            if item_text in self.available_elements:
                used_elements.add(item_text)
        return used_elements

    def add_selected_element(self):
        """Add the selected element from available list to selected list"""
        if self.available_list.currentItem():
            element = self.available_list.currentItem().text()
            used_elements = self.get_used_elements()
            if element not in used_elements:
                self.selected_list.addItem(element)
                self.update_preview()

    def remove_selected_element(self):
        """Remove the selected element from the selected list"""
        if self.selected_list.currentItem():
            row = self.selected_list.currentRow()
            self.selected_list.takeItem(row)
            self.update_preview()

    def handle_double_click_available(self, item):
        """Handle double-click on available list item"""
        element = item.text()
        used_elements = self.get_used_elements()
        if element not in used_elements:
            self.selected_list.addItem(element)
            self.update_preview()

    def handle_double_click_selected(self, item):
        """Handle double-click on selected list item"""
        row = self.selected_list.row(item)
        self.selected_list.takeItem(row)
        self.update_preview()

    def update_preview(self):
        """Update the preview of the final format string"""
        elements = []
        for i in range(self.selected_list.count()):
            item_text = self.selected_list.item(i).text()
            elements.append(f'"{item_text}"')

        format_string = "[" + ", ".join(elements) + "]"
        self.preview_text.setText(format_string)

    def show_context_menu(self, position):
        """Show context menu for the selected list items"""
        menu = QMenu()
        remove_action = QAction("Remove", self)

        menu.addAction(remove_action)

        # Get the item at the position
        item = self.selected_list.itemAt(position)
        if not item:
            return

        # Connect actions
        action = menu.exec(self.selected_list.mapToGlobal(position))

        if action == remove_action:
            self.selected_list.takeItem(self.selected_list.row(item))
            self.update_preview()

    def get_format_string(self):
        """Return the complete format string"""
        return self.preview_text.text()


class SaveasExtensionsEditorWidget(QWidget):
    """A reusable widget that combines a format preview label and edit button"""

    formatChanged = Signal(str)

    def __init__(self, label_text, default_format="", parent=None):
        super().__init__(parent)

        self.default_format = default_format  # Store the default format
        self.current_format = ""

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create label for the format string
        self.format_label = QLabel(default_format)
        self.layout.addWidget(self.format_label)

        # Create edit button
        self.edit_button = QPushButton("Edit...")
        self.edit_button.clicked.connect(self.open_format_editor)
        self.layout.addWidget(self.edit_button)

        # Store the label text for the editor dialog
        self.label_text = label_text

    def open_format_editor(self):
        """Open the format editor dialog"""
        dialog = SaveasExtensionsEditor(self.current_format, self)

        if dialog.exec() == QDialog.Accepted:
            # Get the new format and update both current and display
            self.current_format = dialog.get_format_string()
            self.format_label.setText(self.current_format)
            self.formatChanged.emit(self.current_format)

    def get_format(self):
        """Get the current format string"""
        return self.current_format

    def set_value(self, format_string):
        """Set the format string"""
        self.current_format = format_string
        self.format_label.setText(format_string)

    def reset_to_default(self):
        """Reset the format to the original default"""
        self.set_value(self.default_format)

    def get_value(self):
        """Return the list of elements without quotes around the brackets"""
        format_string = self.format_label.text()

        # If empty or just brackets, return empty list
        if not format_string or format_string == "[]":
            return []

        # Remove outer brackets
        if format_string.startswith('[') and format_string.endswith(']'):
            format_string = format_string[1:-1]

        # Split by comma and clean up each element
        if format_string.strip():
            elements = []
            for elem in format_string.split(","):
                elem = elem.strip()
                # Remove both single and double quotes, handling nested quotes
                while (elem.startswith('"') and elem.endswith('"')) or (elem.startswith("'") and elem.endswith("'")):
                    elem = elem[1:-1]
                if elem:  # Only add non-empty elements
                    elements.append(elem)
            return elements
        else:
            return []


class AlignmentModalitySelector(QWidget):
    def __init__(self, modalities_list_creator, default_value="null", parent=None):
        super().__init__(parent)

        self.modalities_list_creator = modalities_list_creator
        self.default_value = default_value

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.comboBox = QComboBox(self)
        self.comboBox.setFixedWidth(120)
        self.comboBox.adjustSize()
        self.update_options()
        layout.addWidget(self.comboBox)

        # Connect to modalities list changes
        self.modalities_list_creator.edit_button.clicked.connect(self.update_options)

    def update_options(self):
        """Update the combo box options based on current modalities list"""
        current_text = self.comboBox.currentText()
        self.comboBox.clear()

        # Add null option first
        self.comboBox.addItem("null")

        # Add modalities from the list
        modalities = self.modalities_list_creator.get_list() or []
        for modality in modalities:
            self.comboBox.addItem(modality)

        # Try to restore previous selection if it still exists
        if current_text in [self.comboBox.itemText(i) for i in range(self.comboBox.count())]:
            self.comboBox.setCurrentText(current_text)
        else:
            self.comboBox.setCurrentText(self.default_value)

    def get_value(self):
        """Get the currently selected modality"""
        text = self.comboBox.currentText()
        return None if text == "null" else text

    def set_value(self, value):
        """Set the current modality"""
        self.comboBox.setCurrentText(value if value else "null")

class freeNumber(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        self.textbox = QLineEdit(self)
        double_validator = QDoubleValidator(-999.9, 999.9, 2)
        self.textbox.setValidator(double_validator)
        main_layout.addWidget(self.textbox)

    def set_text(self, text):
        self.textbox.setText(text)

    def get_text(self):
        if isinstance(self.textbox, QTextEdit):
            text = self.textbox.toPlainText()
        else:
            text = self.textbox.text()

        # Return placeholder text if the field is empty
        if text == "null" or "":
            return None
        elif text:
            return text

class rangeSelector(QWidget):
    def __init__(self, def_min = None, def_max = None, parent=None):
        super().__init__(parent)

        main_layout = QHBoxLayout(self)
        self.min_box = QLineEdit(self)
        self.min_box.setFixedWidth(125)
        self.max_box = QLineEdit(self)
        self.max_box.setFixedWidth(125)
        double_validator = QDoubleValidator(-999.99, 999.99, 2)
        self.min_box.setValidator(double_validator)
        self.max_box.setValidator(double_validator)
        main_layout.addWidget(self.min_box)
        main_layout.addWidget(self.max_box)
        main_layout.setAlignment(Qt.AlignLeft)

    def get_value(self):
        return [float(self.min_box.text()), float(self.max_box.text())]

    def set_value(self, value):
        # Parse the string input like "[-1, 2]" to extract numbers
        if isinstance(value, str):
            # Remove brackets and split by comma
            value_str = value.strip('[]')
            parts = value_str.split(',')
            min_val = float(parts[0].strip())
            max_val = float(parts[1].strip())
        else:
            # Handle case where value is already a list/tuple
            min_val = float(value[0])
            max_val = float(value[1])

        # Set the text boxes with the numeric values as strings
        self.min_box.setText(str(min_val))
        self.max_box.setText(str(max_val))