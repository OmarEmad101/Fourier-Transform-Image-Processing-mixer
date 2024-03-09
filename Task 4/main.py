import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from pyqtgraph import PlotWidget
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ui, _ = loadUiType('khaled.ui')

class MainApp(QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)
        self.component_Combo_Box = self.findChild(QComboBox, 'component_ComboBox')
        self.mixing_fts=[]
        self.radiobutton_out_1 = self.findChild(QRadioButton, 'radiobutton_out_1')
        self.radiobutton_out_2 = self.findChild(QRadioButton, 'radiobutton_out_2')
        self.radiobutton_out_1.setChecked(True)
        self.radiobutton_out_1.toggled.connect(self.radio_button_out_toggled)
        self.mag_phase_radioButton = self.findChild(QRadioButton, 'radioButton_mag_phase')
        self.real_imag_radioButton = self.findChild(QRadioButton, 'radioButton_real_imag')
        self.mag_phase_radioButton.setChecked(True)
        self.mag_phase_radioButton.toggled.connect(self.radio_button_mode_toggled)

        # Widgets
        self.before_labels = {}
        for i in range(4):
            self.before_labels[f'label_{i+1}_before'] = LabelBefore(parent=self, label_found_child=self.findChild(QLabel, f'label_{i+1}_before'),label_index=i+1)
        self.output_labels = {}
        for i in range(2):
            self.output_labels[f'label_{i + 1}_output'] = OutputLabel(parent=self,label_found_child=self.findChild(QLabel,f'label_{i + 1}_output'),label_index=i + 1)

    def clear_attributes(self,label):
        for attr in ['magnitudes', 'phases', 'real', 'imaginary']:
            getattr(label, attr).clear()

    def append_data(self,output_label, before_label, attribute):
        for i in range(4):
            before_attr = getattr(before_label[f'label_{i + 1}_before'], f'weightd_{attribute}')
            if before_attr is not None:
                getattr(output_label, attribute).append(before_attr)

    def send_ft(self):
        for j in range(2):
            label = self.output_labels[f'label_{j + 1}_output']
            self.clear_attributes(label)
            if self.mag_phase_radioButton.isChecked():
                self.append_data(label, self.before_labels, 'magnitudes')
                self.append_data(label, self.before_labels, 'phases')
            elif self.real_imag_radioButton.isChecked():
                self.append_data(label, self.before_labels, 'real')
                self.append_data(label, self.before_labels, 'imaginary')
        self.radio_button_out_toggled()

    def radio_button_mode_toggled(self):
        if self.mag_phase_radioButton.isChecked():
            self.left_label.setText('Magnitude')
            self.right_label.setText('Phase')
        elif self.real_imag_radioButton.isChecked():
            self.left_label.setText('Real')
            self.right_label.setText('Imaginary')

    def radio_button_out_toggled(self):
        selected_label_index = int(self.radiobutton_out_2.isChecked()) + 1
        if self.mag_phase_radioButton.isChecked():
            result_magnitudes = np.sum(self.output_labels[f'label_{selected_label_index}_output'].magnitudes, axis=0)
            result_phases = np.sum(self.output_labels[f'label_{selected_label_index}_output'].phases, axis=0)
            self.output_labels[f'label_{selected_label_index}_output'].ifft2_convert_to_img(1, result_magnitudes,
                                                                                            result_phases)
        elif self.real_imag_radioButton.isChecked():
            result_real = np.sum(self.output_labels[f'label_{selected_label_index}_output'].real, axis=0)
            result_imaginary = np.sum(self.output_labels[f'label_{selected_label_index}_output'].imaginary, axis=0)
            self.output_labels[f'label_{selected_label_index}_output'].ifft2_convert_to_img(2, result_real,
                                                                                            result_imaginary)


class ImageConverter:
    @staticmethod
    def numpy_to_pixmap(array):
        array = (array - array.min()) / (array.max() - array.min()) * 255
        array = array.astype(np.uint8)

        # Check if the array is 2D or 3D
        if len(array.shape) == 2:
            # For 2D arrays
            height, width = array.shape
            bytes_per_line = width
            img = QImage(array.data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(array.shape) == 3:
            # For 3D arrays
            height, width, channel = array.shape
            bytes_per_line = 3 * width
            img = QImage(array.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            print("Unsupported array shape.")
            return None

        return QPixmap.fromImage(img)

class LabelBefore(QMainWindow):
    def __init__(self, parent=None, label_found_child=QLabel,label_index = 0):
        super(LabelBefore, self).__init__(parent)
        self.baba = parent
        self.label = label_found_child
        self.label.setAlignment(Qt.AlignCenter)
        self.label_index=label_index
        self.after_label = parent.findChild(QLabel,f'label_{label_index}_after')
        self.after_label.setAlignment(Qt.AlignCenter)
        self.slider = parent.findChild(QSlider,f'slider_port_{label_index}')
        self.slider.setValue(99)
        self.phase_slider = parent.findChild(QSlider, f'phase_slider_port_{label_index}')
        self.phase_slider.setValue(99)
        self.component_Combo_Box = parent.findChild(QComboBox,'component_ComboBox')
        self.component_Combo_Box.setCurrentIndex(0)
        self.component_Combo_Box.currentIndexChanged.connect(self.component_to_show)
        self.slider.valueChanged.connect(lambda value : self.magnitude_slider_changed(value))
        self.phase_slider.valueChanged.connect(lambda value: self.phase_slider_changed(value))
        self.initialise_label()
        #intialisations
        self.brightness = 0
        self.contrast = 1.0
        self.original_image = None
        self.original= None
        self.image_array = None
        self.img_ft = None
        self.weightd_magnitudes = None
        self.weightd_phases = None
        self.weightd_real = None
        self.weightd_imaginary = None
        self.img_array = None
        self.fourier_new = None
        self.magnitude = None
        self.phase = None
        self.real_part = None
        self.imaginary_part = None
        self.resized_image_array= None
        self.component_dictionary={}
        self.magnitude_slider_value=100
        self.phase_slider_value=100
        # rectangle slider
        self.left_label = parent.findChild(QLabel, 'left_label')
        self.right_label = parent.findChild(QLabel, 'right_label')
        self.rectangle_size_slider = parent.findChild(QSlider, 'horizontalSlider')
        self.rectangle_size_slider.valueChanged.connect(self.update_rectangle_size)
        self.inner_radioButton = parent.findChild(QRadioButton, 'radioButton')
        self.outer_radioButton = parent.findChild(QRadioButton, 'radioButton_2')

    def magnitude_slider_changed(self,value):
        self.magnitude_slider_value=value
        self.calc_weighted_componenets()

    def phase_slider_changed(self,value):
        self.phase_slider_value = value
        self.calc_weighted_componenets()


    def update_rectangle_size(self, value):
       #Applying Region
        fourier= np.copy(self.img_ft)
        center = np.array(fourier.shape) // 2
        region= int(value * min(center) / 100)
        if self.inner_radioButton.isChecked():
            filter_mask = np.zeros_like(fourier)
            filter_mask[tuple(slice(c - region, c + region + 1) for c in center)] = 1
        elif self.outer_radioButton.isChecked():
            filter_mask = np.ones_like(fourier)
            filter_mask[tuple(slice(c - region, c + region + 1) for c in center)] = 0
        fourier *= filter_mask
        self.fourier_new= fourier
        self.calc_components()
        #Drawing The rectangle
        size_percentage = value / 100.0
        max_size_width = self.original.width()
        max_size_height = self.original.height()
        rect_width = int(max_size_width * size_percentage)
        rect_height = int(max_size_height * size_percentage)
        new_pixmap = QPixmap(self.original)
        rect_x = (new_pixmap.width() - rect_width) // 2
        rect_y = (new_pixmap.height() - rect_height) // 2
        painter = QPainter(new_pixmap)
        overlay_color = QColor(255, 255, 255, 100)
        painter.fillRect(QRect(rect_x, rect_y, rect_width, rect_height), overlay_color)
        painter.end()
        scaled_pixmap = new_pixmap.scaled(self.after_label.size(), Qt.KeepAspectRatio)
        self.after_label.setPixmap(scaled_pixmap)

    def component_to_show(self,index):
        selected_component = self.component_Combo_Box.itemText(index)
        selected_component=(np.log((self.component_dictionary[selected_component]) ** 2 + 1))
        magnitude_pixmap = ImageConverter.numpy_to_pixmap(selected_component)
        self.after_label.setPixmap(magnitude_pixmap.scaled(self.after_label.size(), Qt.KeepAspectRatio))
        self.original = magnitude_pixmap.copy()
        self.update_rectangle_size(self.rectangle_size_slider.value())

    def qimage_to_numpy(self, qimg):
        # Convert to 32-bit ARGB format for consistent results
        qimg = qimg.convertToFormat(QImage.Format_ARGB32)
        # Get image data as bytes
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        img_array = np.array(ptr).reshape((qimg.height(), qimg.width(), 4))
        return img_array

    def calc_components(self):
        if self.resized_image_array is None:
            return
        # Compute the Fourier Transform
        self.img_ft = fftpack.fft2(self.resized_image_array)
        self.img_ft = fftpack.fftshift(self.img_ft)
        # Calculate components
        self.magnitude = np.abs(self.img_ft)
        self.phase = np.angle(self.img_ft)
        self.real_part = np.real(self.img_ft)
        self.imaginary_part = np.imag(self.img_ft)
        self.calc_weighted_componenets()

    def calc_weighted_componenets(self):
        copy_magnitude=np.copy(self.magnitude)
        copy_phase = np.copy(self.phase)
        copy_real=np.copy(self.real_part)
        copy_imaginary=np.copy(self.imaginary_part)
        if (self.fourier_new is not None):
            copy_magnitude = np.abs(self.fourier_new)
            copy_phase = np.angle(self.fourier_new)
            copy_real = np.real(self.fourier_new)
            copy_imaginary = np.imag(self.fourier_new)
        copy_phase *= (self.phase_slider_value/100)
        copy_magnitude *= (self.magnitude_slider_value/100)
        copy_real *= (self.magnitude_slider_value/100)
        copy_imaginary *= (self.phase_slider_value/100)
        self.weightd_magnitudes = copy_magnitude
        self.weightd_phases = copy_phase
        self.weightd_real = copy_real
        self.weightd_imaginary = copy_imaginary
        self.baba.send_ft()


    def initialise_label(self):
        self.label.mouseDoubleClickEvent = self.on_double_click
        self.label.mousePressEvent = self.on_mouse_press
        self.label.mouseMoveEvent = self.on_mouse_move

    def on_double_click(self, event):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.img = Image.open(file_path).convert('L')  # Convert to grayscale
            # Convert the image to a NumPy array
            self.img_array = np.array(self.img)
            original_pixmap = QPixmap(file_path)
            self.original_image = original_pixmap.toImage()
            self.image_array = self.qimage_to_numpy(self.original_image)
            if self.original_image.isNull():
                print("Error loading image.")
                return
            self.update_display()

    def get_resized_image_array(self,label):
        pixmap = label.pixmap()
        if pixmap:
            # Convert QPixmap to QImage
            image = pixmap.toImage()
            # Convert QImage to NumPy array
            width, height = image.width(), image.height()
            arr = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    pixel_value = image.pixel(x, y)
                    arr[y, x] = np.uint8(pixel_value & 0xFF)  # Convert to 8-bit unsigned integer
            return arr
        else:
            return None
    def set_all_to_smallest_size(self):
        sizes = [label.label.pixmap().size() for label in self.parent().before_labels.values() if label.label.pixmap()]
        if sizes:
            smallest_size = min(sizes, key=lambda x: x.width() * x.height())
            for label in self.parent().before_labels.values():
                if label.label.pixmap():
                    label.label.setPixmap(label.label.pixmap().scaled(smallest_size.width(), smallest_size.height(),Qt.IgnoreAspectRatio))
                    # Get the resized image as a NumPy array
                    label.resized_image_array = self.get_resized_image_array(label.label)
                    label.calc_components()

        else:
            print("No labels with pixmaps found.")
        self.component_dictionary = {"FT Magnitude": (self.magnitude), "FT Phase": (self.phase), "FT Real": (self.real_part), "FT Imaginary": (self.imaginary_part)}
        self.component_to_show(self.component_Combo_Box.currentIndex())
        self.rectangle_size_slider.setValue(0)
        self.update_rectangle_size(0)

    def on_mouse_press(self, event):
        if event.button() == Qt.RightButton:
            self.reset_brightness(event)
        else:
            self.start_pos = event.pos()

    def reset_brightness(self, event):
        self.brightness = 0
        self.contrast = 1.0
        self.update_display()
        self.calc_components()

    def on_mouse_move(self, event):
        if hasattr(self, 'start_pos'):
            delta = event.pos() - self.start_pos
            self.brightness += delta.y()
            self.contrast += delta.x() / 100.0  # Adjust the sensitivity
            # Ensure contrast is not less than 0
            self.contrast = max(0.01, self.contrast)
            self.update_display()
            self.start_pos = event.pos()

    def update_display(self):
        if hasattr(self, 'original_image'):
            adjusted_image = self.adjust_brightness_contrast(self.original_image, self.brightness, self.contrast)
            adjusted_pixmap = QPixmap.fromImage(adjusted_image)
            self.label.setPixmap(adjusted_pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
            self.set_all_to_smallest_size()

    def adjust_brightness_contrast(self, image, brightness, contrast):
        # Convert QImage to NumPy array
        buffer = image.bits().asstring(image.width() * image.height() * 4)
        img_array = np.frombuffer(buffer, dtype=np.uint8).reshape((image.height(), image.width(), 4))
        # Ensure the array has more than one dimension
        if len(img_array.shape) > 1:
            # Ensure the array has three dimensions
            if img_array.shape[2] == 4:
                # Extract RGB values and convert to grayscale
                gray_array = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2GRAY)
                gray_array = cv2.convertScaleAbs(gray_array, alpha=contrast, beta=brightness)
                adjusted_image = QImage(gray_array.data, gray_array.shape[1], gray_array.shape[0],gray_array.strides[0], QImage.Format_Grayscale8)
                return adjusted_image


class OutputLabel(QMainWindow):
    def __init__(self, parent=None, label_found_child=QLabel, label_index=0):
        super(OutputLabel, self).__init__(parent)
        self.output_label = label_found_child
        self.output_label.setAlignment(Qt.AlignCenter)
        self.after_output_label = parent.findChild(QLabel, f'label_{label_index}_output_after')
        # intialisations
        self.magnitudes=[]
        self.phases=[]
        self.real = []
        self.imaginary = []
        self.weighted_fts= None
        self.output_image_array = None

    def ifft2_convert_to_img(self,index,magnitudes_real_array,phases_imag_array):
        if index==1:
            fft_array=magnitudes_real_array * np.exp(1j * phases_imag_array)
        elif index==2:
            fft_array=magnitudes_real_array + 1j * phases_imag_array
        output_image_array = np.round(np.real(fftpack.ifft2(fftpack.ifftshift(fft_array))))
        magnitude_pixmap = ImageConverter.numpy_to_pixmap(output_image_array)
        self.output_label.setPixmap(magnitude_pixmap.scaled(self.output_label.size(), Qt.KeepAspectRatio))

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()