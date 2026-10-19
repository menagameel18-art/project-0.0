"""
Three-Dimensional Volume Rendering Viewport using VTK.
Provides interactive GPU/CPU ray-casting, clinical color/opacity transfer presets,
anatomical orientation markers, and camera controls.
"""
from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolButton,
    QComboBox, QFrame
)

from imaging.volume_processor import VolumeProcessor


class Volume3DViewer(QWidget):
    """VTK 3D Volume Rendering viewport with clinical transfer function presets."""
    activated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processor: Optional[VolumeProcessor] = None
        self.is_active = False

        # VTK Rendering Pipeline components
        self.renderer: Optional[vtk.vtkRenderer] = None
        self.volume_mapper: Optional[vtk.vtkSmartVolumeMapper] = None
        self.volume_property: Optional[vtk.vtkVolumeProperty] = None
        self.volume_actor: Optional[vtk.vtkVolume] = None
        self.color_tf: Optional[vtk.vtkColorTransferFunction] = None
        self.opacity_tf: Optional[vtk.vtkPiecewiseFunction] = None
        self.axes_widget: Optional[vtk.vtkOrientationMarkerWidget] = None

        self._setup_ui()
        self._init_vtk()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)

        # Header Bar
        self.header_bar = QWidget()
        self.header_bar.setFixedHeight(28)
        self.header_bar.setStyleSheet("background-color: #1e1e1e; border-bottom: 1px solid #333333;")
        h_layout = QHBoxLayout(self.header_bar)
        h_layout.setContentsMargins(8, 0, 8, 0)
        h_layout.setSpacing(6)

        title_lbl = QLabel("3D VOLUME RENDERING")
        title_lbl.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 11px;")
        h_layout.addWidget(title_lbl)

        h_layout.addStretch()

        # Preset Dropdown
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["CT Bone", "CT Soft Tissue", "CT Lung", "MIP (Maximum Intensity)", "MR / Grayscale"])
        self.combo_preset.setFixedHeight(20)
        self.combo_preset.setStyleSheet("""
            QComboBox {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                font-size: 10px;
                padding: 0 4px;
            }
        """)
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        h_layout.addWidget(self.combo_preset)

        # Quality Dropdown
        self.combo_quality = QComboBox()
        self.combo_quality.addItems(["Normal Quality", "High Quality", "Fast / Draft"])
        self.combo_quality.setFixedHeight(20)
        self.combo_quality.setStyleSheet("""
            QComboBox {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                font-size: 10px;
                padding: 0 4px;
            }
        """)
        self.combo_quality.currentIndexChanged.connect(self._on_quality_changed)
        h_layout.addWidget(self.combo_quality)

        # Reset Camera Button
        self.btn_reset_cam = QToolButton()
        self.btn_reset_cam.setText("Reset Cam")
        self.btn_reset_cam.setToolTip("Reset 3D Camera")
        self.btn_reset_cam.setFixedHeight(20)
        self.btn_reset_cam.setStyleSheet("""
            QToolButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                font-size: 10px;
                padding: 0 4px;
            }
            QToolButton:hover {
                background-color: #383838;
                border-color: #007acc;
            }
        """)
        self.btn_reset_cam.clicked.connect(self.reset_camera)
        h_layout.addWidget(self.btn_reset_cam)

        layout.addWidget(self.header_bar)

        # VTK Widget Container
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.vtkWidget.setStyleSheet("background-color: #0a0a0c;")
        layout.addWidget(self.vtkWidget, stretch=1)

        # Bottom Info Bar
        self.footer_bar = QWidget()
        self.footer_bar.setFixedHeight(32)
        self.footer_bar.setStyleSheet("background-color: #1a1a1a; border-top: 1px solid #2a2a2a;")
        f_layout = QHBoxLayout(self.footer_bar)
        f_layout.setContentsMargins(8, 0, 8, 0)
        f_layout.setSpacing(8)

        self.lbl_info = QLabel("Volume: No Data Loaded")
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        f_layout.addWidget(self.lbl_info)

        f_layout.addStretch()

        hint_lbl = QLabel("Rotate: Left-Drag | Zoom: Right-Drag / Wheel | Pan: Shift+Left / Middle")
        hint_lbl.setStyleSheet("color: #666666; font-size: 10px;")
        f_layout.addWidget(hint_lbl)

        layout.addWidget(self.footer_bar)
        self.set_active(False)

    def _init_vtk(self):
        """Initializes VTK renderer, volume mapper, transfer functions, and interactor."""
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.04, 0.04, 0.06)  # Deep radiology navy-black
        self.renderer.SetBackground2(0.12, 0.12, 0.16)
        self.renderer.SetGradientBackground(True)

        ren_win = self.vtkWidget.GetRenderWindow()
        ren_win.AddRenderer(self.renderer)

        self.volume_mapper = vtk.vtkSmartVolumeMapper()
        # Initialize mapper with a dummy volume so input port 0 is always connected
        dummy = vtk.vtkImageData()
        dummy.SetDimensions(1, 1, 1)
        dummy.AllocateScalars(vtk.VTK_FLOAT, 1)
        self.volume_mapper.SetInputData(dummy)

        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.SetAmbient(0.2)
        self.volume_property.SetDiffuse(0.8)
        self.volume_property.SetSpecular(0.25)
        self.volume_property.SetSpecularPower(15.0)

        self.color_tf = vtk.vtkColorTransferFunction()
        self.opacity_tf = vtk.vtkPiecewiseFunction()

        self.volume_property.SetColor(self.color_tf)
        self.volume_property.SetScalarOpacity(self.opacity_tf)

        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(self.volume_mapper)
        self.volume_actor.SetProperty(self.volume_property)
        self.renderer.AddVolume(self.volume_actor)

        # Set standard trackball camera style
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtkWidget.SetInteractorStyle(style)
        self.vtkWidget.Initialize()

        # 3D Coordinate axes orientation marker in bottom-left
        try:
            axes = vtk.vtkAxesActor()
            axes.SetTotalLength(25, 25, 25)
            self.axes_widget = vtk.vtkOrientationMarkerWidget()
            self.axes_widget.SetOrientationMarker(axes)
            self.axes_widget.SetInteractor(self.vtkWidget)
            self.axes_widget.SetViewport(0.0, 0.0, 0.22, 0.22)
            self.axes_widget.SetEnabled(1)
            self.axes_widget.InteractiveOff()
        except Exception:
            pass

    def set_processor(self, processor: VolumeProcessor):
        """Connects new volume data, updates VTK pipeline, and renders."""
        self.processor = processor
        if not processor or not processor.has_data():
            self.lbl_info.setText("Volume: No Data Loaded")
            return

        vtk_img = processor.get_vtk_image_data()
        if not vtk_img:
            return

        self.volume_mapper.SetInputData(vtk_img)

        # Apply appropriate initial preset based on modality
        if processor.metadata.is_ct:
            self.combo_preset.setCurrentText("CT Bone")
            self._apply_ct_bone_preset()
        else:
            self.combo_preset.setCurrentText("MR / Grayscale")
            self._apply_mr_preset()

        g = processor.geometry
        self.lbl_info.setText(
            f"Dimensions: {g.num_x}x{g.num_y}x{g.num_z} | Spacing: {g.dx:.2f}x{g.dy:.2f}x{g.dz:.2f} mm"
        )

        self.reset_camera()

    def reset_camera(self):
        """Resets camera to show the complete volume centered in patient space."""
        if not self.renderer:
            return
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        # View from anterior-superior perspective
        cam.Elevation(15)
        cam.Azimuth(-20)
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()

    def _on_preset_changed(self, idx: int):
        preset = self.combo_preset.currentText()
        if preset == "CT Bone":
            self._apply_ct_bone_preset()
        elif preset == "CT Soft Tissue":
            self._apply_ct_soft_tissue_preset()
        elif preset == "CT Lung":
            self._apply_ct_lung_preset()
        elif preset == "MIP (Maximum Intensity)":
            self._apply_mip_preset()
        elif preset == "MR / Grayscale":
            self._apply_mr_preset()

        self.vtkWidget.GetRenderWindow().Render()

    def _apply_ct_bone_preset(self):
        self.volume_mapper.SetBlendModeToComposite()
        self.volume_property.ShadeOn()

        self.opacity_tf.RemoveAllPoints()
        self.opacity_tf.AddPoint(-1000, 0.0)
        self.opacity_tf.AddPoint(100, 0.0)
        self.opacity_tf.AddPoint(220, 0.12)
        self.opacity_tf.AddPoint(450, 0.55)
        self.opacity_tf.AddPoint(1200, 0.90)

        self.color_tf.RemoveAllPoints()
        self.color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        self.color_tf.AddRGBPoint(120, 0.65, 0.45, 0.35)
        self.color_tf.AddRGBPoint(300, 0.88, 0.82, 0.70)
        self.color_tf.AddRGBPoint(800, 0.96, 0.94, 0.88)
        self.color_tf.AddRGBPoint(1500, 1.0, 1.0, 1.0)

    def _apply_ct_soft_tissue_preset(self):
        self.volume_mapper.SetBlendModeToComposite()
        self.volume_property.ShadeOn()

        self.opacity_tf.RemoveAllPoints()
        self.opacity_tf.AddPoint(-1000, 0.0)
        self.opacity_tf.AddPoint(-120, 0.0)
        self.opacity_tf.AddPoint(-40, 0.08)
        self.opacity_tf.AddPoint(40, 0.35)
        self.opacity_tf.AddPoint(150, 0.70)
        self.opacity_tf.AddPoint(800, 0.90)

        self.color_tf.RemoveAllPoints()
        self.color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        self.color_tf.AddRGBPoint(-80, 0.25, 0.05, 0.05)   # Fat
        self.color_tf.AddRGBPoint(40, 0.80, 0.30, 0.25)    # Muscle/Parenchyma
        self.color_tf.AddRGBPoint(120, 0.95, 0.60, 0.45)   # Enhanced organ
        self.color_tf.AddRGBPoint(600, 0.98, 0.95, 0.90)   # Bone

    def _apply_ct_lung_preset(self):
        self.volume_mapper.SetBlendModeToComposite()
        self.volume_property.ShadeOn()

        self.opacity_tf.RemoveAllPoints()
        self.opacity_tf.AddPoint(-1000, 0.0)
        self.opacity_tf.AddPoint(-920, 0.0)
        self.opacity_tf.AddPoint(-750, 0.30)
        self.opacity_tf.AddPoint(-400, 0.10)
        self.opacity_tf.AddPoint(-100, 0.0)
        self.opacity_tf.AddPoint(400, 0.40)

        self.color_tf.RemoveAllPoints()
        self.color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        self.color_tf.AddRGBPoint(-850, 0.20, 0.60, 0.90)  # Airways/Lung
        self.color_tf.AddRGBPoint(-600, 0.40, 0.85, 1.00)
        self.color_tf.AddRGBPoint(400, 0.95, 0.92, 0.85)   # Rib cage outline

    def _apply_mip_preset(self):
        self.volume_mapper.SetBlendModeToMaximumIntensity()
        self.volume_property.ShadeOff()

        self.opacity_tf.RemoveAllPoints()
        self.opacity_tf.AddPoint(-1000, 0.0)
        self.opacity_tf.AddPoint(50, 0.0)
        self.opacity_tf.AddPoint(300, 0.6)
        self.opacity_tf.AddPoint(1000, 1.0)

        self.color_tf.RemoveAllPoints()
        self.color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        self.color_tf.AddRGBPoint(100, 0.4, 0.4, 0.4)
        self.color_tf.AddRGBPoint(500, 0.8, 0.8, 0.8)
        self.color_tf.AddRGBPoint(1200, 1.0, 1.0, 1.0)

    def _apply_mr_preset(self):
        self.volume_mapper.SetBlendModeToComposite()
        self.volume_property.ShadeOn()

        val_min = 0.0
        val_max = 500.0
        if self.processor and self.processor.volume is not None:
            val_min = float(np.percentile(self.processor.volume, 5))
            val_max = float(np.percentile(self.processor.volume, 98))

        span = max(10.0, val_max - val_min)
        p1 = val_min + span * 0.15
        p2 = val_min + span * 0.45
        p3 = val_min + span * 0.85

        self.opacity_tf.RemoveAllPoints()
        self.opacity_tf.AddPoint(val_min, 0.0)
        self.opacity_tf.AddPoint(p1, 0.0)
        self.opacity_tf.AddPoint(p2, 0.35)
        self.opacity_tf.AddPoint(p3, 0.80)

        self.color_tf.RemoveAllPoints()
        self.color_tf.AddRGBPoint(val_min, 0.0, 0.0, 0.0)
        self.color_tf.AddRGBPoint(p1, 0.2, 0.2, 0.2)
        self.color_tf.AddRGBPoint(p2, 0.6, 0.6, 0.6)
        self.color_tf.AddRGBPoint(p3, 0.95, 0.95, 0.95)

    def _on_quality_changed(self, idx: int):
        txt = self.combo_quality.currentText()
        if "High" in txt:
            self.volume_mapper.SetSampleDistance(0.5)
        elif "Normal" in txt:
            self.volume_mapper.SetSampleDistance(1.0)
        else:
            self.volume_mapper.SetSampleDistance(2.0)
        self.vtkWidget.GetRenderWindow().Render()

    def set_active(self, active: bool):
        self.is_active = active
        if active:
            self.setStyleSheet("border: 2px solid #00aaff;")
        else:
            self.setStyleSheet("border: 1px solid #2d2d2d;")

    def mousePressEvent(self, event):
        self.activated.emit("3D_VOLUME")
        super().mousePressEvent(event)
