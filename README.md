# Fourier Transform Mixer

## Overview

This desktop program is designed to illustrate the relative importance of magnitude and phase components while emphasizing the frequencies' contributions to a signal. Although demonstrated using 2D images, the concepts apply universally to any signal. The software offers several features:

## Features

### Image Viewers

- **Open and View**: Capability to open and view four grayscale images, with each displayed in a separate "viewport".
  - **Grayscale Conversion**: Automatically converts colored images to grayscale upon opening.
  - **Unified Sizes**: Ensures all opened images are displayed at the same size, adjusted to the smallest size among them.
  - **FT Components**: Each image viewport displays two components:
    1. Image itself.
    2. User-selectable Fourier Transform (FT) component, including FT Magnitude, FT Phase, FT Real, and FT Imaginary.
  - **Easy Browse**: Images can be changed by double-clicking on the respective viewer.

### Output Ports

- **Two Output Viewports**: Mixer results can be displayed in one of two output viewports, identical to input image viewports.
- **User Control**: Users can determine which output viewport displays the mixer result.

### Brightness/Contrast

- **Adjustments**: Users can modify image brightness/contrast via mouse dragging within any image viewport.
- **Applicable to Components**: Brightness/contrast adjustments can be applied to any of the four components.

### Components Mixer

- **Customizable Weights**: Mixer output is the inverse Fourier transform (ifft) of a weighted average of the FT of the input four images.
  - **Slider Controls**: Users can customize weights for each image FT via sliders.
  - **Intuitive Interface**: Ensures user-friendliness for customizing weights, especially for two components like magnitude and phase.

### Regions Mixer

- **Region Selection**: Users can pick regions from each FT component for output—inner region (low frequencies) or outer region (high frequencies).
  - **Selectable Regions**: Rectangles drawn on each FT allow users to include inner or outer regions, with selected region highlighted.
  - **Customizable Size**: Size (or percentage) of region rectangle is adjustable via slider or resize handles.

### Realtime Mixing

- **Progress Indication**: Lengthy ifft operations are accompanied by a progress bar to indicate the process.
- **Cancellation**: If user initiates new mixing settings while a previous operation is ongoing, the program cancels the prior operation and starts the new one.

![Screenshot 1](/Task%204/Screenshots/1.png)
![Screenshot 1](/Task%204/Screenshots/2.png)
![Screenshot 1](/Task%204/Screenshots/3.png)
![Screenshot 1](/Task%204/Screenshots/4.png)

## Contributors <a name = "Contributors"></a>

<table>
  <tr>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/OmarEmad101">
          <img src="https://github.com/OmarEmad101.png" width="100px" alt="@OmarEmad101">
          <br>
          <sub><b>Omar Emad</b></sub>
        </a>
      </div>
    </td>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/Omarnbl">
          <img src="https://github.com/Omarnbl.png" width="100px" alt="@Omarnbl">
          <br>
          <sub><b>Omar Nabil</b></sub>
        </a>
      </div>
    </td>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/KhaledBadr07">
          <img src="https://github.com/KhaledBadr07.png" width="100px" alt="@KhaledBadr07">
          <br>
          <sub><b>Khaled Badr</b></sub>
        </a>
      </div>
    </td>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/merna-abdelmoez">
          <img src="https://github.com/merna-abdelmoez.png" width="100px" alt="@merna-abdelmoez">
          <br>
          <sub><b>Mirna Abdelmoez</b></sub>
        </a>
      </div>
    </td>
  </tr>
</table>
## Acknowledgments

**This project was supervised by Dr. Tamer Basha & Eng. Abdallah Darwish, who provided invaluable guidance and expertise throughout its development as a part of the Digital Signal Processing course at Cairo University Faculty of Engineering.**

<div style="text-align: right">
    <img src="https://imgur.com/Wk4nR0m.png" alt="Cairo University Logo" width="100" style="border-radius: 50%;"/>
</div>
